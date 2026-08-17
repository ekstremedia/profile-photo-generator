"""Content rules.

Scope, stated plainly:

* This project generates **people who do not exist**. Requests that try to
  steer it at a real, identifiable person are rejected.
* Free-text fields are the only untrusted input - the attribute axes are
  drawn from a curated vocabulary and cannot express anything problematic.
* Ages are clamped to a configured range. Generating faces below that range
  requires an explicit opt-in and drops all styling controls.

None of this is a content classifier and it does not pretend to be. It is a
narrow, auditable filter on the text that reaches the model, plus an
instruction to the prompt-composing LLM. Operators who need stronger
guarantees should put a real moderation layer in front of the API.
"""

from __future__ import annotations

import re

from ppg.config import Settings


class SafetyError(ValueError):
    """Raised when a request is refused. Surfaces as HTTP 422."""


# Sexual content. The default model is NSFW-capable and the diffusers safety
# checker is deliberately off (it false-positives constantly on ordinary
# portraits), so the block happens here, on the way in.
_SEXUAL_TERMS = {
    "nude",
    "nudes",
    "naked",
    "nsfw",
    "topless",
    "shirtless",
    "lingerie",
    "underwear",
    "panties",
    "bra",
    "erotic",
    "erotica",
    "porn",
    "porno",
    "pornographic",
    "explicit",
    "seductive",
    "provocative",
    "sensual",
    "fetish",
    "bdsm",
    "bondage",
    "cleavage",
    "bikini",
    "swimsuit",
    "nipple",
    "nipples",
    "genitals",
    "sexual",
    "sexy",
    "aroused",
    "orgasm",
    "hentai",
    "onlyfans",
    "camgirl",
    "stripper",
}

# Age descriptors are not accepted as free text - the `age` parameter is the
# only way to influence age, and it is bounds-checked.
_MINOR_TERMS = {
    "child",
    "children",
    "kid",
    "kids",
    "toddler",
    "infant",
    "baby",
    "babies",
    "minor",
    "underage",
    "preteen",
    "pre-teen",
    "teen",
    "teenage",
    "teenager",
    "schoolgirl",
    "schoolboy",
    "highschooler",
    "loli",
    "shota",
    "young girl",
    "young boy",
    "little girl",
    "little boy",
    "12 year",
    "13 year",
    "14 year",
    "15 year",
    "16 year",
    "17 year",
    "years old",
}

# Markers that a request is reaching for a specific real person. Not a
# celebrity database - a database would give false confidence. These are the
# phrasings people actually use, plus the LLM composer refuses by instruction.
_REAL_PERSON_MARKERS = {
    "celebrity",
    "celebrities",
    "famous",
    "lookalike",
    "look-alike",
    "look alike",
    "deepfake",
    "deep fake",
    "face swap",
    "faceswap",
    "resembling",
    "looks like",
    "in the likeness of",
    "portrait of the",
    "president",
    "prime minister",
    "the actor",
    "the actress",
    "the singer",
    "the rapper",
    "the model ",
    "influencer named",
    "youtuber",
    "streamer",
    "public figure",
    "politician",
    "royal family",
    "instagram star",
}

_WORD_RE = re.compile(r"[a-z0-9'-]+")


def _tokens(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def _matched(text: str, phrases: set[str]) -> list[str]:
    """Find blocked entries, matching multi-word phrases as substrings."""
    lowered = text.lower()
    words = _tokens(lowered)
    hits = []
    for phrase in phrases:
        if " " in phrase or "-" in phrase:
            if phrase in lowered:
                hits.append(phrase)
        elif phrase in words:
            hits.append(phrase)
    return sorted(hits)


def check_free_text(text: str | None, *, field: str = "prompt_extra") -> str:
    """Validate a caller-supplied free-text fragment.

    Returns the trimmed text, or raises :class:`SafetyError`.
    """
    if not text:
        return ""
    cleaned = " ".join(text.split())
    if len(cleaned) > 500:
        raise SafetyError(f"{field} is too long (max 500 characters).")

    if hits := _matched(cleaned, _SEXUAL_TERMS):
        raise SafetyError(
            f"{field} rejected: this service generates profile photos, not sexual "
            f"content. Blocked term(s): {', '.join(hits)}."
        )
    if hits := _matched(cleaned, _MINOR_TERMS):
        raise SafetyError(
            f"{field} rejected: age cannot be set through free text. Use the `age` "
            f"parameter, which is bounds-checked. Blocked term(s): {', '.join(hits)}."
        )
    if hits := _matched(cleaned, _REAL_PERSON_MARKERS):
        raise SafetyError(
            f"{field} rejected: this service only generates people who do not exist. "
            f"It will not imitate a real or identifiable person. "
            f"Blocked term(s): {', '.join(hits)}."
        )
    return cleaned


def clamp_age(age: int | None, settings: Settings) -> int | None:
    """Clamp a requested age into the configured range.

    Returns ``None`` when no age was requested (the sampler then draws one).
    """
    if age is None:
        return None
    floor = 0 if settings.allow_minors else settings.min_age
    ceiling = settings.max_age
    if age < floor:
        if not settings.allow_minors:
            raise SafetyError(
                f"age {age} is below the minimum of {settings.min_age}. "
                "Set PPG_ALLOW_MINORS=true if you genuinely need younger faces; "
                "doing so also forces plain portrait framing."
            )
        age = floor
    return max(floor, min(ceiling, age))


def minor_mode(age: int, settings: Settings) -> bool:
    """True when the resolved age is below the normal floor.

    In this mode the composer is restricted to a plain, neutral school-portrait
    description and all styling overrides are dropped.
    """
    return settings.allow_minors and age < settings.min_age
