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
    FRAMING,
    FRAMING_PLAIN,
    REALISM_CUES,
    build_negative,
    build_style,
    build_subject,
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
    """Prompts ready to hand to an image backend.

    Held as four halves rather than two strings because SDXL has two text
    encoders with independent 77-token budgets. ``subject`` describes the
    person, ``style`` the photographic treatment; the combined ``prompt``
    property is what gets stored, displayed and embedded in image metadata.
    See ``templates.py`` for why.
    """

    subject: str
    style: str
    negative_subject: str
    negative_style: str
    persona: Persona | None
    source: Literal["llm", "template"]

    @property
    def prompt(self) -> str:
        return f"{self.subject}, {self.style}"

    @property
    def negative_prompt(self) -> str:
        return f"{self.negative_subject}, {self.negative_style}"


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
        negative_subject, negative_style = build_negative(negative_extra)
        return ComposeResult(
            subject=build_subject(attrs, extra=extra, plain_framing=plain_framing),
            style=build_style(attrs),
            negative_subject=negative_subject,
            negative_style=negative_style,
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
        user = self._user_message(attrs, plain_framing=plain_framing)
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

        framing = FRAMING_PLAIN if plain_framing else FRAMING
        # 40 words plus the identity clause lands just under the 77-token CLIP
        # budget. Going higher starts clipping the tail of the description.
        description = _cap_words(parsed.description.strip().rstrip(","), 40)

        # The identity clause is assembled here rather than left to the model,
        # and placed immediately after the framing. Two reasons: SDXL weights
        # earlier tokens more heavily, and a model asked to write the whole
        # description tends to soften or drop the ancestry term - which shows
        # up as a batch of visibly varied attributes rendering as the same
        # handful of faces.
        identity = " ".join(
            part
            for part in (
                f"{attrs.age} year old",
                attrs.phrases.get("ethnicity", ""),
                attrs.phrases.get("sex", "person"),
            )
            if part
        )
        skin = attrs.phrases.get("skin_tone", "")
        # Models restate the skin tone often enough that it is worth removing
        # the echo rather than shipping "light olive skin, light olive skin".
        description = _drop_phrase(description, skin)
        subject_parts = [framing, identity, skin, description]
        if extra:
            subject_parts.append(extra)

        # Lighting and background come from the sampled attributes rather than
        # the model: they belong in the style half, and the model is told not
        # to mention them.
        scene = [attrs.phrases.get(axis, "") for axis in ("lighting", "background", "camera")]
        style = ", ".join([*(s for s in scene if s), REALISM_CUES])
        negative_subject, negative_style = build_negative(negative_extra)

        persona = parsed.persona
        # The model is asked for the given age but does not always comply.
        if persona.age != attrs.age:
            persona = persona.model_copy(update={"age": attrs.age})

        return ComposeResult(
            subject=", ".join(p for p in subject_parts if p),
            style=style,
            negative_subject=negative_subject,
            negative_style=negative_style,
            persona=persona,
            source="llm",
        )

    @staticmethod
    def _user_message(attrs: Attributes, *, plain_framing: bool) -> str:
        # Age, ancestry, sex and skin tone are still listed even though the
        # model must not restate them: they are needed for the persona and to
        # keep the details it does write coherent with the person.
        lines = [f"age: {attrs.age}"]
        skip = {"camera", "lighting", "background"}
        for axis, phrase in attrs.phrases.items():
            if axis in skip or not phrase:
                continue
            lines.append(f"{axis}: {phrase}")
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


def _cap_words(text: str, limit: int) -> str:
    """Hard cap on the model's description length.

    The system prompt asks for 30-45 words; small models sometimes ignore that
    and write a paragraph, whose tail then gets silently cut by the text
    encoder. Trimming here at least loses whole clauses rather than half a
    word, and keeps the important, earlier descriptors.
    """
    words = text.split()
    if len(words) <= limit:
        return text
    trimmed = " ".join(words[:limit])
    # Prefer to end on a clause boundary.
    cut = trimmed.rfind(",")
    return trimmed[:cut] if cut > len(trimmed) // 2 else trimmed


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
