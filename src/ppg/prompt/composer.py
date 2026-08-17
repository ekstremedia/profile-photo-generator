"""Turning attributes into a prompt.

Two implementations, one interface:

* :class:`TemplateComposer` - pure Python, no network, no model. This is the
  default path and it produces complete, good prompts on its own. **Ollama is
  not required to run this project.**
* :class:`LLMComposer` - if an Ollama server happens to be reachable, a small
  local model writes the descriptive middle of the prompt instead, plus a
  richer persona. It phrases things a lookup table cannot: a 68-year-old
  trawler skipper gets weathered skin and a worn collar rather than whatever
  the vocabulary rolled.

:class:`AutoComposer` uses the LLM when it is there and silently falls back
when it is not, so the same configuration works on a machine with Ollama and a
machine without.

In every case the framing and the realism cue block are appended by us, not by
the model. Those are what make the output photorealistic, so they are not left
to chance.
"""

from __future__ import annotations

import logging
import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Literal, Protocol

import yaml
from pydantic import BaseModel, Field, ValidationError

from ppg.attributes.sampler import Attributes
from ppg.config import Settings
from ppg.prompt.ollama_client import OllamaClient, OllamaError
from ppg.prompt.templates import (
    DETAIL_AXES,
    age_descriptors,
    build_negative,
    build_prompt,
)
from ppg.schemas import Persona

logger = logging.getLogger(__name__)

NAMES_PATH = Path(__file__).with_name("names.yaml")

# Small, widely available, reliable at constrained JSON. Checked in order when
# PPG_OLLAMA_MODEL is left at "auto". Anything on this list is a 1-3GB pull.
PREFERRED_MODELS: tuple[str, ...] = (
    "llama3.2:3b",
    "qwen3:4b",
    "qwen3:1.7b",
    "gemma3:4b",
    "llama3.2:1b",
    "mistral:7b",
    "llama3.1:8b",
    "qwen3:8b",
)


@dataclass
class ComposeResult:
    """A prompt pair ready to hand to an image backend.

    One prompt, not two halves: both of SDXL's text encoders receive the same
    text. See ``templates.py`` for the measurement behind that.
    """

    prompt: str
    negative_prompt: str
    persona: Persona | None
    source: Literal["llm", "template"]


class PromptComposer(Protocol):
    name: str

    async def compose(
        self,
        attrs: Attributes,
        seed: int,
        *,
        extra: str = "",
        negative_extra: str = "",
        plain_framing: bool = False,
    ) -> ComposeResult: ...


# ---------------------------------------------------------------------------
# Template composer
# ---------------------------------------------------------------------------


@lru_cache
def _names_data() -> dict:
    with NAMES_PATH.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh) or {}


class TemplateComposer:
    """Deterministic, dependency-free prompt and persona construction."""

    name = "template"

    async def compose(
        self,
        attrs: Attributes,
        seed: int,
        *,
        extra: str = "",
        negative_extra: str = "",
        plain_framing: bool = False,
    ) -> ComposeResult:
        return ComposeResult(
            prompt=build_prompt(attrs, extra=extra, plain_framing=plain_framing),
            negative_prompt=build_negative(negative_extra, age=attrs.age, implied=attrs.negatives),
            persona=self.persona(attrs, seed),
            source="template",
        )

    def persona(self, attrs: Attributes, seed: int) -> Persona | None:
        data = _names_data()
        pools = data.get("pools", {})
        mapping = data.get("map", {})
        pool_key = mapping.get(attrs.values.get("ethnicity", ""), "anglophone")
        pool = pools.get(pool_key) or pools.get("anglophone")
        if not pool:
            return None

        # Offset the seed so the persona does not correlate with the sampler's
        # own draw order - otherwise every "first" attribute set gets the
        # "first" name.
        rnd = random.Random(seed ^ 0x9E3779B97F4A7C15)

        sex = attrs.values.get("sex", "female")
        given_pool = pool.get(sex) or pool.get("female") or []
        if not given_pool:
            return None
        given = rnd.choice(given_pool)
        family = rnd.choice(pool.get("family", ["Doe"]))
        city = rnd.choice(pool.get("cities", [])) if pool.get("cities") else None

        occupation = attrs.phrases.get("profession", "").strip() or "unspecified"
        bio = None
        templates = data.get("bio_templates", [])
        if templates and city:
            bio = rnd.choice(templates).format(occupation=occupation, city=city)

        return Persona(
            name=f"{given} {family}",
            age=attrs.age,
            occupation=occupation,
            city=city,
            country=None,
            bio=bio,
        )


# ---------------------------------------------------------------------------
# LLM composer
# ---------------------------------------------------------------------------


class _LLMOutput(BaseModel):
    """What we ask Ollama for. Deliberately small - just the parts a language
    model is better at than a lookup table."""

    description: str = Field(
        min_length=20,
        max_length=400,
        description="Comma-separated visual description of the person.",
    )
    persona: Persona


SYSTEM_PROMPT = """\
You write prompts for a photorealistic text-to-image model, and matching \
fictional persona data.

You will be given attributes of a person who does not exist. Return JSON with \
two fields:

- "description": a single line of comma-separated English visual descriptors. \
Cover hair, facial hair, eyewear, expression and clothing, then add one or two \
small concrete details that make the person feel specific and lived-in (a \
chipped tooth, sun lines around the eyes, a worn collar, paint under the \
fingernails). Strictly 25-40 words - the text encoder truncates beyond that \
and the tail is lost. No sentences, no narrative. Do NOT restate their age, \
ancestry, skin tone or sex, and do NOT mention lighting, background, camera \
settings or image quality: all of those are added separately and repeating \
them wastes the token budget.
- "persona": a fictional name that is plausible for the stated ancestry, the \
same age as given, their occupation, a city, and a one-line bio.

Hard rules:
- Anything listed as "already in the prompt" is fixed. Do not restate it and \
do not describe anything that contradicts it. If a red scarf is already \
stated, do not put the person in a polo neck.
- Attributes marked [REQUIRED] were chosen deliberately. Everything you write \
must be consistent with them.
- Match the stated age. If the age note says the person is very elderly, the \
details you invent must reflect that - thin white hair, not gray-brown; \
liver-spotted hands, not smooth ones.
- The person must not exist. Never use the name of a real or identifiable \
person, living or dead, and never describe someone as resembling one.
- Keep the description internally consistent. An outdoor manual worker does \
not have a fresh manicure; a 70-year-old does not have a teenager's skin.
- Never infer ancestry from occupation, or occupation, class, wealth or \
setting from ancestry or skin tone. Treat those attributes as independent.
- Describe an ordinary clothed person photographed for a profile picture. \
Nothing sexual, nothing violent.
- Output only the JSON object.\
"""


class LLMComposer:
    """Uses a local Ollama model to write the descriptive core of the prompt."""

    name = "llm"

    def __init__(self, client: OllamaClient) -> None:
        self.client = client
        self._fallback = TemplateComposer()

    async def compose(
        self,
        attrs: Attributes,
        seed: int,
        *,
        extra: str = "",
        negative_extra: str = "",
        plain_framing: bool = False,
    ) -> ComposeResult:
        user = self._user_message(attrs, plain_framing=plain_framing, extra=extra)
        schema = _LLMOutput.model_json_schema()

        last_error: Exception | None = None
        for attempt in range(2):
            try:
                raw = await self.client.chat_json(
                    system=SYSTEM_PROMPT,
                    user=user,
                    schema=schema,
                    # Same seed every time -> same persona every time. Without
                    # this, `by-seed` avatars would drift between restarts.
                    seed=seed + attempt,
                    temperature=0.7,
                )
                parsed = _LLMOutput.model_validate(raw)
                break
            except (OllamaError, ValidationError) as exc:
                last_error = exc
                logger.warning("Prompt composer attempt %d failed: %s", attempt + 1, exc)
        else:
            raise OllamaError(str(last_error))

        # The model's contribution is *filler*: it is appended after everything
        # the caller asked for, and is the first thing dropped when the budget
        # runs out. Identity, age and pinned attributes are assembled by
        # `build_prompt` and never depend on the model getting it right.
        description = parsed.description.strip().rstrip(",")
        for echo in (
            attrs.phrases.get("skin_tone", ""),
            attrs.phrases.get("ethnicity", ""),
        ):
            # Models restate these often enough that it is worth removing the
            # echo rather than shipping "light olive skin, light olive skin".
            description = _drop_phrase(description, echo)

        persona = parsed.persona
        # The model is asked for the given age but does not always comply.
        if persona.age != attrs.age:
            persona = persona.model_copy(update={"age": attrs.age})

        return ComposeResult(
            prompt=build_prompt(
                attrs, extra=extra, detail=description, plain_framing=plain_framing
            ),
            negative_prompt=build_negative(negative_extra, age=attrs.age, implied=attrs.negatives),
            persona=persona,
            source="llm",
        )

    @staticmethod
    def _user_message(attrs: Attributes, *, plain_framing: bool, extra: str = "") -> str:
        # Age, ancestry, sex and skin tone are still listed even though the
        # model must not restate them: they are needed for the persona and to
        # keep the details it does write coherent with the person.
        appearance, _ = age_descriptors(attrs.age)
        lines = [f"age: {attrs.age}" + (f" ({appearance})" if appearance else "")]

        skip = {"camera", "lighting", "background"}
        for axis, phrase in attrs.phrases.items():
            if axis in skip or not phrase:
                continue
            marker = " [REQUIRED]" if axis in attrs.pinned else ""
            lines.append(f"{axis}: {phrase}{marker}")

        # Anything already written into the prompt by hand is listed so the
        # model does not repeat it or, worse, describe something that
        # contradicts it - a requested red scarf plus an invented tweed collar
        # produces neither.
        stated = [
            attrs.phrases[axis]
            for axis in DETAIL_AXES
            if axis in attrs.pinned and attrs.phrases.get(axis)
        ]
        if extra:
            stated.append(extra)
        if stated:
            lines.append("already in the prompt, do not repeat or contradict: " + "; ".join(stated))
        if plain_framing:
            lines.append("framing: plain neutral portrait, no styling")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Auto composer
# ---------------------------------------------------------------------------


class AutoComposer:
    """LLM when available, template otherwise. Never raises for either reason."""

    name = "auto"

    def __init__(self, llm: LLMComposer, template: TemplateComposer) -> None:
        self.llm = llm
        self.template = template

    async def compose(
        self,
        attrs: Attributes,
        seed: int,
        *,
        extra: str = "",
        negative_extra: str = "",
        plain_framing: bool = False,
    ) -> ComposeResult:
        try:
            return await self.llm.compose(
                attrs,
                seed,
                extra=extra,
                negative_extra=negative_extra,
                plain_framing=plain_framing,
            )
        except OllamaError as exc:
            logger.info("Falling back to the template composer: %s", exc)
            return await self.template.compose(
                attrs,
                seed,
                extra=extra,
                negative_extra=negative_extra,
                plain_framing=plain_framing,
            )


def _drop_phrase(text: str, phrase: str) -> str:
    """Remove a comma-delimited clause equal to ``phrase`` from ``text``."""
    if not phrase:
        return text
    target = phrase.strip().lower()
    kept = [
        clause
        for clause in (c.strip() for c in text.split(","))
        if clause and clause.lower() != target
    ]
    return ", ".join(kept)


async def resolve_ollama_model(client: OllamaClient, configured: str) -> str | None:
    """Decide which Ollama model to use.

    ``auto`` (the default) picks whatever the user already has, preferring a
    small one. This is the difference between "install a 5GB LLM before you can
    use this" and "it uses your existing models if you have any". Returns
    ``None`` when Ollama has no models at all.
    """
    if configured and configured != "auto":
        return configured
    try:
        installed = await client.installed_models()
    except OllamaError:
        return None
    if not installed:
        return None

    names = {m["name"] for m in installed}
    for preferred in PREFERRED_MODELS:
        if preferred in names:
            return preferred
        # Tolerate quantisation/tuning suffixes on the *same* tag, e.g.
        # "llama3.2:3b-instruct-q4_K_M" for "llama3.2:3b". Matching on the bare
        # family name instead would let a request for qwen3:4b select qwen3:8b,
        # which is twice the VRAM and defeats the point of the preference list.
        matches = sorted(n for n in names if n.startswith(f"{preferred}-"))
        if matches:
            return matches[0]
    # Nothing recognised - use the smallest installed model.
    return min(installed, key=lambda m: m["size"] or 1 << 62)["name"]


def build_composer(settings: Settings, model_override: str | None = None) -> PromptComposer:
    """Construct the composer implied by ``PPG_COMPOSER``."""
    if settings.composer == "template":
        return TemplateComposer()

    client = OllamaClient(
        base_url=settings.ollama_base_url,
        model=model_override or settings.ollama_model,
        timeout=settings.ollama_timeout,
        keep_alive=settings.ollama_keep_alive,
    )
    llm = LLMComposer(client)
    if settings.composer == "llm":
        return llm
    return AutoComposer(llm, TemplateComposer())


async def build_composer_auto(settings: Settings) -> tuple[PromptComposer, str | None]:
    """Pick a composer, probing Ollama once at startup.

    Returns the composer and the Ollama model it settled on (``None`` when
    running without Ollama). Probing here rather than per request means a
    machine with no Ollama pays one 5-second timeout at boot, not on every
    generation.
    """
    if settings.composer == "template":
        return TemplateComposer(), None

    client = OllamaClient(settings.ollama_base_url, settings.ollama_model, timeout=5.0)
    model = await resolve_ollama_model(client, settings.ollama_model)

    if model is None:
        if settings.composer == "llm":
            logger.error(
                "PPG_COMPOSER=llm but no Ollama model is available at %s",
                settings.ollama_base_url,
            )
        else:
            logger.info(
                "No Ollama model available at %s - using template prompts. "
                "This is fine; Ollama only adds wording variety.",
                settings.ollama_base_url,
            )
        return TemplateComposer(), None

    logger.info("Prompt composer: Ollama model %s at %s", model, settings.ollama_base_url)
    return build_composer(settings, model_override=model), model
