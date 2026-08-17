"""Prompt construction without an LLM.

This is both the fallback path (Ollama unreachable) and the reference for what
the LLM composer is expected to produce. Keeping it working matters: the whole
service must degrade gracefully to "still generates good faces, just less
varied wording".

Three things here are less obvious than they look.

**Why these cue words.** Diffusion models drift towards retouched stock-photo
faces unless you explicitly ask for the imperfections that make a photograph
read as real: pores, asymmetry, grain, a physical lens. The negative prompt
then pushes away from the illustration/CGI basin.

**Why one prompt and not two.** SDXL has two text encoders and diffusers
exposes the second as ``prompt_2``, which looks like a free doubling of the
77-token budget. It is not. The pooled conditioning embedding comes from the
*second* encoder alone, so sending it only photographic style starves the model
of subject information. Measured on RealVisXL with an identical seed: a prompt
asking for a 90-year-old man in a red scarf with wire glasses rendered the
scarf and glasses when the full prompt went to both encoders, and dropped both
when the subject was confined to encoder one. So both encoders get the same
text, and the token budget is managed by keeping the prompt short.

**Why the ordering is explicit.** CLIP truncates at 77 tokens and discards the
*tail*. Whatever the caller actually asked for therefore has to come early, and
the filler the vocabulary rolled has to come last. This is not a
micro-optimisation: before it, a request for a red scarf sat behind forty words
of improvised tweed and never reached the model at all.
"""

from __future__ import annotations

from ppg.attributes.sampler import Attributes

# Roughly the number of comma-separated words that survive CLIP's 77 tokens.
# Deliberately conservative - punctuation and long words both cost extra, and
# overshooting fails silently.
PROMPT_WORD_BUDGET = 55
NEGATIVE_WORD_BUDGET = 55

# Framing. Kept separate so `minor_mode` can swap it for something plainer.
FRAMING = "head and shoulders portrait, looking at the camera"
FRAMING_PLAIN = "plain head and shoulders school portrait, looking at the camera, neutral"

# Short on purpose: this block is reserved out of the budget, so every word
# here is a word the subject cannot use.
REALISM_CUES = (
    "candid photo, natural skin texture with visible pores, unretouched, "
    "sharp focus on the eyes, shallow depth of field, film grain"
)

# Split only so the anatomy terms can be prioritised over the style terms when
# the negative prompt has to be trimmed.
NEGATIVE_SUBJECT = (
    "deformed, disfigured, deformed eyes, bad anatomy, bad proportions, "
    "plastic skin, waxy skin, airbrushed, heavy makeup, beauty filter, doll"
)
NEGATIVE_STYLE = (
    "cartoon, anime, illustration, painting, 3d render, cgi, oversaturated, "
    "watermark, text, logo, blurry, low quality, black and white"
)

DETAIL_AXES = ("hair", "facial_hair", "glasses", "expression", "clothing")
SCENE_AXES = ("lighting", "background", "camera")

# A bare "90 year old" barely moves SDXL, which is trained on a corpus heavily
# skewed towards attractive thirty-somethings and will happily render a
# well-preserved sixty. The visible consequences of the age have to be spelled
# out, and the model pushed away from youth in the negative prompt.
_AGE_LOOKS: tuple[tuple[int, str, str], ...] = (
    # (upper bound inclusive, positive descriptors, negative descriptors)
    (25, "youthful, smooth unlined skin", "elderly, wrinkles, grey hair"),
    (39, "", "elderly, deep wrinkles, grey hair"),
    (54, "middle aged, faint lines", "very old, deep wrinkles, white hair"),
    (64, "older, lined face, greying hair", "young, youthful, smooth skin"),
    (79, "elderly, wrinkled skin, grey hair", "young, youthful, smooth taut skin"),
    (
        200,
        "very elderly, deeply wrinkled skin, age spots, sparse white hair, sunken cheeks",
        "young, youthful, middle aged, smooth taut skin, dark hair",
    ),
)


def age_descriptors(age: int) -> tuple[str, str]:
    """Return ``(positive, negative)`` appearance cues for an age."""
    for bound, positive, negative in _AGE_LOOKS:
        if age <= bound:
            return positive, negative
    return "", ""


def word_count(text: str) -> int:
    return len(text.split())


def assemble(required: list[str], filler: list[str], budget: int) -> str:
    """Join ``required`` in order, then as much of ``filler`` as still fits.

    ``required`` is everything the caller asked for plus the cues that make the
    output photorealistic. It is never trimmed - if it alone exceeds the
    budget, the caller has asked for more than CLIP can hold and the tail of
    their own request is what gets cut, which is at least predictable.
    """
    kept = [part for part in required if part]
    remaining = budget - sum(word_count(part) for part in kept)

    for part in filler:
        if not part:
            continue
        cost = word_count(part)
        if cost <= remaining:
            kept.append(part)
            remaining -= cost
    return ", ".join(kept)


def split_details(attrs: Attributes) -> tuple[list[str], list[str]]:
    """Split the appearance axes into ``(caller-pinned, randomly drawn)``."""
    pinned = [attrs.phrases[a] for a in DETAIL_AXES if a in attrs.pinned and attrs.phrases.get(a)]
    sampled = [
        attrs.phrases[a] for a in DETAIL_AXES if a not in attrs.pinned and attrs.phrases.get(a)
    ]
    return pinned, sampled


def identity_clause(attrs: Attributes) -> str:
    """Age, ancestry and sex - the part the model must not miss."""
    positive, _ = age_descriptors(attrs.age)
    core = " ".join(
        part
        for part in (
            f"{attrs.age} year old",
            attrs.phrases.get("ethnicity", ""),
            attrs.phrases.get("sex", "person"),
        )
        if part
    )
    return f"{core}, {positive}" if positive else core


def scene_parts(attrs: Attributes) -> list[str]:
    return [attrs.phrases[a] for a in SCENE_AXES if attrs.phrases.get(a)]


def build_prompt(
    attrs: Attributes,
    *,
    extra: str = "",
    detail: str = "",
    plain_framing: bool = False,
) -> str:
    """Assemble the full prompt.

    ``detail`` is generated description (from the LLM composer, if any) and is
    treated as filler. Everything in ``required`` below either came from the
    caller or is load-bearing for photorealism.
    """
    pinned_details, sampled_details = split_details(attrs)
    profession = attrs.phrases.get("profession", "")

    required = [
        FRAMING_PLAIN if plain_framing else FRAMING,
        identity_clause(attrs),
        extra,
        *pinned_details,
        attrs.phrases.get("skin_tone", ""),
        REALISM_CUES,
    ]
    filler = [
        detail,
        f"working as a {profession}" if profession and "profession" in attrs.pinned else "",
        *sampled_details,
        *scene_parts(attrs),
    ]
    return assemble(required, filler, PROMPT_WORD_BUDGET)


def build_negative(
    extra: str = "",
    age: int | None = None,
    implied: list[str] | None = None,
) -> str:
    """Assemble the negative prompt.

    Three sources beyond the baseline: caller additions, the age-specific terms
    (pushing away from "youthful, smooth skin" is what actually makes a
    ninety-year-old look ninety), and ``implied`` - terms attached to chosen
    options that describe an absence, such as "no glasses", which has no useful
    positive phrasing.
    """
    required = [NEGATIVE_SUBJECT]
    if age is not None:
        _positive, negative = age_descriptors(age)
        required.append(negative)
    required.extend(implied or [])
    if extra:
        required.append(extra)
    return assemble(required, [NEGATIVE_STYLE], NEGATIVE_WORD_BUDGET)
