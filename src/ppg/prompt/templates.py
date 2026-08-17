"""Prompt construction without an LLM.

This is both the fallback path (Ollama unreachable) and the reference for what
the LLM composer is expected to produce. Keeping it working matters: the whole
service must degrade gracefully to "still generates good faces, just less
varied wording".

Two things here are less obvious than they look.

**Why these cue words.** Diffusion models drift towards retouched stock-photo
faces unless you explicitly ask for the imperfections that make a photograph
read as real: pores, asymmetry, grain, a physical lens. The negative prompt
then pushes away from the illustration/CGI basin.

**Why the prompt is split in two.** CLIP truncates at 77 tokens, and a full
description plus the realism block comfortably exceeds that - so the realism
cues, which are at the end, get silently dropped, which is exactly the part
that was doing the work. SDXL has *two* text encoders (CLIP-L and OpenCLIP-G)
and diffusers exposes the second as `prompt_2`, each with its own 77-token
budget. Putting the person in one and the photographic treatment in the other
doubles the usable length for free, with no extra dependency and no weighting
library. The same split applies to the negative prompt, which was also over
budget.
"""

from __future__ import annotations

from ppg.attributes.sampler import Attributes

# --- subject half (encoder 1) ---------------------------------------------

# Framing. Kept separate so `minor_mode` can swap it for something plainer.
FRAMING = "head and shoulders portrait, centred, looking at the camera"
FRAMING_PLAIN = "plain head and shoulders school portrait, centred, looking at the camera, neutral"

NEGATIVE_SUBJECT = (
    "deformed, disfigured, face asymmetry, deformed eyes, crossed eyes, "
    "extra fingers, extra limbs, bad anatomy, bad proportions, long neck, "
    "doll, mannequin, plastic skin, waxy skin, airbrushed, heavy makeup, beauty filter"
)

# --- style half (encoder 2) -----------------------------------------------

REALISM_CUES = (
    "candid headshot photograph, natural skin texture with visible pores and fine lines, "
    "subtle facial asymmetry, unretouched, sharp focus on the eyes, "
    "shallow depth of field, gentle film grain, colour photograph"
)

NEGATIVE_STYLE = (
    "cartoon, anime, illustration, painting, drawing, sketch, 3d render, cgi, "
    "video game character, oversaturated, overexposed, instagram filter, "
    "watermark, signature, text, logo, username, blurry, out of focus, "
    "low quality, jpeg artifacts, black and white, monochrome"
)


def _age_phrase(age: int) -> str:
    return f"{age} year old"


def build_subject(attrs: Attributes, *, extra: str = "", plain_framing: bool = False) -> str:
    """The person: framing, identity, features, clothing. Goes to encoder 1."""
    sex = attrs.phrases.get("sex", "person")
    ethnicity = attrs.phrases.get("ethnicity", "")
    skin = attrs.phrases.get("skin_tone", "")
    profession = attrs.phrases.get("profession", "")

    subject_bits = [_age_phrase(attrs.age), ethnicity, sex]
    subject = " ".join(b for b in subject_bits if b)
    if profession:
        subject = f"{subject}, working as a {profession}"

    detail_axes = ("hair", "facial_hair", "glasses", "expression", "clothing")
    details = [attrs.phrases[a] for a in detail_axes if attrs.phrases.get(a)]

    parts = [
        FRAMING_PLAIN if plain_framing else FRAMING,
        subject,
        skin,
        ", ".join(details),
    ]
    if extra:
        parts.append(extra)
    return ", ".join(p for p in parts if p)


def build_style(attrs: Attributes) -> str:
    """Lighting, background, lens and the realism block. Goes to encoder 2."""
    scene = [attrs.phrases.get(axis, "") for axis in ("lighting", "background", "camera")]
    return ", ".join([*(s for s in scene if s), REALISM_CUES])


def build_negative(extra: str = "") -> tuple[str, str]:
    """Return ``(negative_subject, negative_style)``.

    Caller-supplied additions join the subject half, since they are almost
    always about the person rather than the rendering style.
    """
    subject = f"{NEGATIVE_SUBJECT}, {extra}" if extra else NEGATIVE_SUBJECT
    return subject, NEGATIVE_STYLE


def build_template_prompt(
    attrs: Attributes,
    *,
    extra: str = "",
    plain_framing: bool = False,
) -> str:
    """The whole prompt as one string.

    Used for display, storage and image metadata. Rendering uses the two
    halves separately - see :func:`build_subject` and :func:`build_style`.
    """
    subject = build_subject(attrs, extra=extra, plain_framing=plain_framing)
    return f"{subject}, {build_style(attrs)}"


# Kept for backwards compatibility with anything importing the old name.
BASE_NEGATIVE = f"{NEGATIVE_SUBJECT}, {NEGATIVE_STYLE}"
