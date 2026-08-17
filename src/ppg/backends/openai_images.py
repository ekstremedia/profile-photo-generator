"""Any OpenAI-compatible ``/v1/images/generations`` endpoint.

This exists for one specific reason: Ollama's image generation (Z-Image-Turbo,
FLUX.2 klein) is exposed at exactly that path. It currently runs on macOS with
MLX only, but when Linux and Windows support lands, generating through Ollama
becomes a two-line configuration change:

    PPG_BACKEND=openai_images
    PPG_OPENAI_BASE_URL=http://host.docker.internal:11434/v1
    PPG_OPENAI_MODEL=x/z-image-turbo

The same adapter also points at a hosted provider if someone would rather not
run a GPU at all.

Caveat worth knowing: the OpenAI images schema has no negative prompt and no
seed. Determinism and negative prompting are therefore best-effort here - the
negative prompt is folded into the positive one as an avoidance clause, and
the seed is sent as a non-standard field that compliant servers ignore.
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Any

import httpx
from PIL import Image

from ppg.backends.base import BackendError, RenderSpec
from ppg.config import Settings

logger = logging.getLogger(__name__)


class OpenAIImagesBackend:
    name = "openai_images"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.base_url = settings.openai_base_url.rstrip("/")
        self._loaded = False

    @property
    def model_id(self) -> str:
        return self.settings.openai_model

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    async def generate(self, spec: RenderSpec) -> Image.Image:
        prompt = spec.full_prompt
        if spec.full_negative_prompt:
            prompt = f"{prompt}. Avoid: {spec.full_negative_prompt}"

        body: dict[str, Any] = {
            "model": self.settings.openai_model,
            "prompt": prompt,
            "n": 1,
            "size": f"{spec.width}x{spec.height}",
            "response_format": "b64_json",
            "seed": spec.seed % (2**31),
        }
        headers = {}
        if self.settings.openai_api_key:
            headers["Authorization"] = f"Bearer {self.settings.openai_api_key}"

        payload = await self._post(body, headers)
        return self._decode(payload)

    async def _post(self, body: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        try:
            async with httpx.AsyncClient(base_url=self.base_url, timeout=300.0) as client:
                response = await client.post("/images/generations", json=body, headers=headers)
                if response.status_code in (400, 422) and "seed" in body:
                    # Strict OpenAI-schema servers reject unknown fields.
                    retry = {k: v for k, v in body.items() if k != "seed"}
                    response = await client.post("/images/generations", json=retry, headers=headers)
                response.raise_for_status()
                return response.json()
        except httpx.HTTPStatusError as exc:
            raise BackendError(
                f"{self.base_url}/images/generations returned HTTP "
                f"{exc.response.status_code}: {exc.response.text[:300]}"
            ) from exc
        except (httpx.HTTPError, OSError) as exc:
            raise BackendError(f"Could not reach {self.base_url}: {exc}") from exc
        except ValueError as exc:
            raise BackendError(f"Image endpoint returned a non-JSON body: {exc}") from exc

    @staticmethod
    def _decode(payload: dict[str, Any]) -> Image.Image:
        items = payload.get("data") or []
        if not items:
            raise BackendError(f"Image endpoint returned no images: {str(payload)[:200]}")
        item = items[0]

        if b64 := item.get("b64_json"):
            try:
                raw = base64.b64decode(b64)
            except (ValueError, TypeError) as exc:
                raise BackendError("Image endpoint returned undecodable base64.") from exc
            return Image.open(io.BytesIO(raw)).convert("RGB")

        if url := item.get("url"):
            try:
                response = httpx.get(url, timeout=120.0)
                response.raise_for_status()
            except httpx.HTTPError as exc:
                raise BackendError(
                    f"Could not fetch the generated image from {url}: {exc}"
                ) from exc
            return Image.open(io.BytesIO(response.content)).convert("RGB")

        raise BackendError(f"Unrecognised image response shape: {str(item)[:200]}")
