"""Thin async Ollama client.

Only the two endpoints we need: ``/api/tags`` to check availability and
``/api/chat`` for structured generation. No SDK dependency - the API is three
fields and a JSON body, and pinning an SDK here would be more trouble than the
code it saves.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Reasoning models (qwen3 and friends) wrap their scratchpad in these tags.
# Ollama strips them when `think` is supported, but not every build does.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


class OllamaError(RuntimeError):
    """Ollama was unreachable, errored, or returned something unusable."""


class OllamaClient:
    def __init__(
        self,
        base_url: str,
        model: str,
        timeout: float = 90.0,
        keep_alive: str = "0",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        # Ollama keeps a model resident for 5 minutes by default. On a
        # single-GPU machine that means an 8B model sits on 5GB of VRAM while
        # the diffusion model tries to load 7GB into the same 12GB card, and
        # the image generation fails with an out-of-memory error. "0" unloads
        # immediately after composing the prompt, which costs a second of
        # reload per request and buys back the VRAM that actually matters.
        self.keep_alive = keep_alive

    def _keep_alive_value(self) -> int | str:
        """Ollama accepts seconds as a number or a duration string like "5m".

        A bare "0" string is *not* a valid duration and gets ignored, leaving
        the 5-minute default in place - which is precisely the case we care
        about. Send plain digits as an integer.
        """
        text = str(self.keep_alive).strip()
        if text.lstrip("-").isdigit():
            return int(text)
        return text

    async def reachable(self) -> bool:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=5.0) as client:
                response = await client.get("/api/tags")
                return response.status_code == 200
        except (httpx.HTTPError, OSError):
            return False

    async def installed_models(self) -> list[dict[str, Any]]:
        """Models already pulled on the Ollama host, with their sizes in bytes."""
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=10.0) as client:
                response = await client.get("/api/tags")
                response.raise_for_status()
                payload = response.json()
        except (httpx.HTTPError, ValueError, OSError) as exc:
            raise OllamaError(f"Could not list Ollama models: {exc}") from exc
        return [
            {"name": m.get("name", ""), "size": int(m.get("size") or 0)}
            for m in payload.get("models", [])
            if m.get("name")
        ]

    async def available_models(self) -> list[str]:
        return [m["name"] for m in await self.installed_models()]

    async def chat_json(
        self,
        system: str,
        user: str,
        schema: dict[str, Any],
        *,
        seed: int | None = None,
        temperature: float = 0.7,
    ) -> dict[str, Any]:
        """Chat completion constrained to ``schema``, returned as a dict.

        ``seed`` is passed through to Ollama so repeated calls with the same
        inputs produce the same persona - without it, "deterministic avatars"
        would only be deterministic until the prompt cache was cleared.
        """
        body: dict[str, Any] = {
            "model": self.model,
            "stream": False,
            "format": schema,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "options": {"temperature": temperature},
            "keep_alive": self._keep_alive_value(),
            # Thinking output is wasted tokens here; the task is a template fill.
            "think": False,
        }
        if seed is not None:
            body["options"]["seed"] = seed % (2**31)

        content = await self._post_chat(body)
        cleaned = _THINK_RE.sub("", content).strip()
        if not cleaned:
            raise OllamaError("Ollama returned an empty response.")
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise OllamaError(f"Ollama did not return valid JSON: {cleaned[:200]!r}") from exc
        if not isinstance(parsed, dict):
            raise OllamaError(f"Expected a JSON object, got {type(parsed).__name__}.")
        return parsed

    async def _post_chat(self, body: dict[str, Any]) -> str:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=self.timeout) as client:
                response = await client.post("/api/chat", json=body)
                if response.status_code == 400 and "think" in body:
                    # Older builds, and non-reasoning models, reject `think`.
                    retry = {k: v for k, v in body.items() if k != "think"}
                    response = await client.post("/api/chat", json=retry)
                response.raise_for_status()
                payload = response.json()
        except httpx.HTTPStatusError as exc:
            detail = exc.response.text[:300]
            raise OllamaError(f"Ollama returned HTTP {exc.response.status_code}: {detail}") from exc
        except (httpx.HTTPError, OSError) as exc:
            raise OllamaError(
                f"Could not reach Ollama at {self.base_url}: {exc}. "
                "From inside Docker this must be http://host.docker.internal:11434 "
                "and the host daemon must listen on 0.0.0.0 (see docs/TROUBLESHOOTING.md)."
            ) from exc
        except ValueError as exc:
            raise OllamaError(f"Ollama returned a non-JSON body: {exc}") from exc

        message = payload.get("message") or {}
        content = message.get("content")
        if not isinstance(content, str):
            raise OllamaError(f"Unexpected Ollama response shape: {str(payload)[:200]}")
        return content
