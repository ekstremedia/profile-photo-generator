"""Seeded attribute sampling.

Two guarantees this module exists to provide:

1. **Determinism** - the same seed and the same pinned attributes always yield
   the same attribute set, on any machine, forever. That is what makes
   ``/v1/avatars/by-seed/<email>`` stable.
2. **Coherence** - a 19-year-old is not a retired school principal, and a
   drawn ancestry gets a plausible skin tone. Incoherent prompts produce
   uncanny faces, so the rules in ``vocab.yaml`` are applied here.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

VOCAB_PATH = Path(__file__).with_name("vocab.yaml")

# Sampling order. Axes drawn earlier constrain the ones drawn later, so this
# order is load-bearing - do not shuffle it casually.
SAMPLE_ORDER: tuple[str, ...] = (
    "sex",
    "ethnicity",
    "skin_tone",
    "profession",
    "hair",
    "facial_hair",
    "glasses",
    "expression",
    "clothing",
    "background",
    "lighting",
    "camera",
)

# Weight multiplier applied to an option that a rule discourages. Not zero:
# grey hair at 40 happens, it just should not be common.
DISCOURAGE_FACTOR = 0.05


@dataclass(frozen=True)
class Option:
    value: str
    weight: float = 1.0
    prompt: str | None = None
    label: str | None = None
    min: int | None = None
    max: int | None = None

    @property
    def phrase(self) -> str:
        """How this option appears in an image prompt. May be empty."""
        if self.prompt is not None:
            return self.prompt
        return self.value.replace("_", " ")

    @property
    def display(self) -> str:
        return self.label or self.value.replace("_", " ")


@dataclass
class Attributes:
    """A fully resolved person description."""

    values: dict[str, str] = field(default_factory=dict)
    age: int = 30
    age_range: str = "25-34"
    phrases: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = dict(self.values)
        out["age"] = self.age
        out["age_range"] = self.age_range
        return out

    def phrase_list(self) -> list[str]:
        """Non-empty prompt fragments, in sampling order."""
        return [self.phrases[a] for a in SAMPLE_ORDER if self.phrases.get(a)]


def seed_from_key(key: str) -> int:
    """Map any string to a stable 63-bit seed.

    blake2b rather than ``hash()`` because Python's string hash is salted per
    process - using it would silently break determinism across restarts.
    """
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") & ((1 << 63) - 1)


def normalise_seed(
    seed: int | str | None, rnd: random.Random | None = None
) -> tuple[int, str | None]:
    """Return ``(seed_int, seed_key)``.

    A string seed is hashed and reported back as ``seed_key`` so callers can
    see what produced the face. ``None`` draws a fresh random seed.
    """
    if seed is None:
        source = rnd or random.SystemRandom()
        return source.randrange(1 << 63), None
    if isinstance(seed, int):
        return seed & ((1 << 63) - 1), None
    text = str(seed).strip()
    if text.isdigit():
        return int(text) & ((1 << 63) - 1), None
    return seed_from_key(text), text


def _parse_option(raw: Any) -> Option:
    if isinstance(raw, str):
        return Option(value=raw)
    if isinstance(raw, dict):
        return Option(
            value=str(raw["value"]),
            weight=float(raw.get("weight", 1.0)),
            prompt=raw.get("prompt"),
            label=raw.get("label"),
            min=raw.get("min"),
            max=raw.get("max"),
        )
    raise TypeError(f"Unsupported vocabulary entry: {raw!r}")


class Vocabulary:
    """Parsed ``vocab.yaml``."""

    def __init__(self, data: dict[str, Any]) -> None:
        self.version: int = int(data.get("version", 1))
        self.axes: dict[str, list[Option]] = {
            axis: [_parse_option(entry) for entry in entries]
            for axis, entries in data.get("axes", {}).items()
        }
        self.rules: dict[str, Any] = data.get("rules", {})
        self.skin_tone_affinity: dict[str, list[str]] = data.get("skin_tone_affinity", {})
        self._index: dict[str, dict[str, Option]] = {
            axis: {opt.value: opt for opt in opts} for axis, opts in self.axes.items()
        }

    @classmethod
    def load(cls, path: Path = VOCAB_PATH) -> Vocabulary:
        with path.open("r", encoding="utf-8") as fh:
            return cls(yaml.safe_load(fh) or {})

    def option(self, axis: str, value: str) -> Option:
        """Look up an option, tolerating free-form values.

        Callers are allowed to ask for a profession the vocabulary has never
        heard of ("puffin researcher"). We pass it straight through rather than
        rejecting it - the diffusion model does not care that it is not in a
        YAML file.
        """
        found = self._index.get(axis, {}).get(value)
        if found is not None:
            return found
        # Also accept the human-readable form ("marine biologist").
        slug = value.strip().lower().replace(" ", "_").replace("-", "_")
        found = self._index.get(axis, {}).get(slug)
        if found is not None:
            return found
        return Option(value=value.strip(), prompt=value.strip().replace("_", " "))

    def age_bucket(self, age: int) -> str:
        for opt in self.axes.get("age_range", []):
            if opt.min is not None and opt.max is not None and opt.min <= age <= opt.max:
                return opt.value
        return f"{age}-{age}"


@lru_cache
def get_vocabulary() -> Vocabulary:
    return Vocabulary.load()


class Sampler:
    def __init__(self, vocab: Vocabulary | None = None) -> None:
        self.vocab = vocab or get_vocabulary()

    # -- internals ------------------------------------------------------
    def _pick(
        self,
        rnd: random.Random,
        axis: str,
        *,
        allowed: list[Option] | None = None,
        weight_overrides: dict[str, float] | None = None,
    ) -> Option:
        options = allowed if allowed is not None else self.vocab.axes.get(axis, [])
        if not options:
            return Option(value="", prompt="")
        weights = []
        for opt in options:
            w = opt.weight
            if weight_overrides and opt.value in weight_overrides:
                w *= weight_overrides[opt.value]
            weights.append(max(w, 1e-9))
        return rnd.choices(options, weights=weights, k=1)[0]

    def _age(
        self,
        rnd: random.Random,
        pinned: dict[str, str],
        exact_age: int | None,
        min_age: int,
        max_age: int,
    ) -> tuple[int, str]:
        if exact_age is not None:
            age = max(min_age, min(max_age, int(exact_age)))
            return age, self.vocab.age_bucket(age)

        if "age_range" in pinned:
            opt = self.vocab.option("age_range", pinned["age_range"])
            lo, hi = opt.min, opt.max
            if lo is None or hi is None:
                # Free-form "30-45" style input.
                parts = [p for p in pinned["age_range"].replace("_", "-").split("-") if p.isdigit()]
                lo, hi = (int(parts[0]), int(parts[-1])) if parts else (min_age, max_age)
        else:
            opt = self._pick(rnd, "age_range")
            lo, hi = opt.min or min_age, opt.max or max_age

        lo = max(lo, min_age)
        hi = min(hi, max_age)
        if lo > hi:
            lo, hi = min_age, max_age
        age = rnd.randint(lo, hi)
        return age, self.vocab.age_bucket(age)

    def _skin_tone_options(self, ethnicity: str) -> list[Option]:
        allowed = self.vocab.skin_tone_affinity.get(ethnicity)
        if not allowed:
            return self.vocab.axes.get("skin_tone", [])
        pool = [o for o in self.vocab.axes.get("skin_tone", []) if o.value in allowed]
        return pool or self.vocab.axes.get("skin_tone", [])

    def _profession_options(self, age: int) -> list[Option]:
        rules = self.vocab.rules.get("profession", {})
        min_age = rules.get("min_age", {})
        max_age = rules.get("max_age", {})
        pool = []
        for opt in self.vocab.axes.get("profession", []):
            if age < min_age.get(opt.value, 0):
                continue
            if age > max_age.get(opt.value, 200):
                continue
            pool.append(opt)
        return pool or self.vocab.axes.get("profession", [])

    # -- public ---------------------------------------------------------
    def sample(
        self,
        seed: int,
        pinned: dict[str, str] | None = None,
        *,
        exact_age: int | None = None,
        min_age: int = 18,
        max_age: int = 90,
    ) -> Attributes:
        pinned = dict(pinned or {})
        rnd = random.Random(seed)
        result = Attributes()

        # 1. sex
        sex = self.vocab.option("sex", pinned["sex"]) if "sex" in pinned else self._pick(rnd, "sex")

        # 2. age - drawn early because several axes depend on it
        age, age_range = self._age(rnd, pinned, exact_age, min_age, max_age)
        result.age = age
        result.age_range = age_range

        # 3. ethnicity, then a plausible skin tone for it
        ethnicity = (
            self.vocab.option("ethnicity", pinned["ethnicity"])
            if "ethnicity" in pinned
            else self._pick(rnd, "ethnicity")
        )
        if "skin_tone" in pinned:
            skin_tone = self.vocab.option("skin_tone", pinned["skin_tone"])
        else:
            skin_tone = self._pick(
                rnd, "skin_tone", allowed=self._skin_tone_options(ethnicity.value)
            )

        # 4. profession, constrained by age
        if "profession" in pinned:
            profession = self.vocab.option("profession", pinned["profession"])
        else:
            profession = self._pick(rnd, "profession", allowed=self._profession_options(age))

        chosen: dict[str, Option] = {
            "sex": sex,
            "ethnicity": ethnicity,
            "skin_tone": skin_tone,
            "profession": profession,
        }

        # 5. hair, with age-aware discouragement
        hair_rules = self.vocab.rules.get("hair", {}).get("discourage_below_age", {})
        hair_overrides = {v: DISCOURAGE_FACTOR for v, floor in hair_rules.items() if age < floor}
        chosen["hair"] = (
            self.vocab.option("hair", pinned["hair"])
            if "hair" in pinned
            else self._pick(rnd, "hair", weight_overrides=hair_overrides)
        )

        # 6. facial hair, gated on sex
        if "facial_hair" in pinned:
            chosen["facial_hair"] = self.vocab.option("facial_hair", pinned["facial_hair"])
        else:
            fh_rule = self.vocab.rules.get("facial_hair", {})
            allowed_sexes = fh_rule.get("only_if", {}).get("sex", [])
            if allowed_sexes and sex.value not in allowed_sexes:
                chosen["facial_hair"] = self.vocab.option(
                    "facial_hair", fh_rule.get("otherwise", "clean_shaven")
                )
            else:
                fh_age = self.vocab.rules.get("facial_hair_age", {}).get("discourage_below_age", {})
                overrides = {v: DISCOURAGE_FACTOR for v, floor in fh_age.items() if age < floor}
                chosen["facial_hair"] = self._pick(rnd, "facial_hair", weight_overrides=overrides)

        # 7. everything else is an unconditional draw
        for axis in ("glasses", "expression", "clothing", "background", "lighting", "camera"):
            chosen[axis] = (
                self.vocab.option(axis, pinned[axis]) if axis in pinned else self._pick(rnd, axis)
            )

        for axis in SAMPLE_ORDER:
            opt = chosen[axis]
            result.values[axis] = opt.value
            result.phrases[axis] = opt.phrase
        return result

    def strata(
        self, n: int, seed: int, pinned: dict[str, str] | None = None
    ) -> list[dict[str, str]]:
        """Build ``n`` attribute pins spread evenly across the main visual axes.

        Naive random sampling clusters on whatever the vocabulary weights
        favour, so a batch of 50 comes out looking samey. Here we walk a
        shuffled cross product of sex x age bucket x ethnicity instead, which
        makes a contact sheet actually look like a crowd.
        """
        pinned = dict(pinned or {})
        rnd = random.Random(seed)

        def pool(axis: str) -> list[str]:
            if axis in pinned:
                return [pinned[axis]]
            return [o.value for o in self.vocab.axes.get(axis, [])] or [""]

        sexes = [s for s in pool("sex") if s != "androgynous"] or pool("sex")
        ages = pool("age_range")
        ethnicities = pool("ethnicity")

        # Shuffle each axis independently, then walk all three in lockstep.
        #
        # The obvious approach - shuffle the full cross product and take the
        # first n - is uniformly random, which for small n happily returns six
        # people from the same continent. Walking each shuffled axis in turn
        # guarantees that consecutive avatars differ on every axis and that no
        # value repeats until its pool is exhausted, so a batch of six looks
        # like six different people rather than six rolls of a die.
        for axis_pool in (sexes, ages, ethnicities):
            rnd.shuffle(axis_pool)

        out: list[dict[str, str]] = []
        for i in range(n):
            combo = {
                "sex": sexes[i % len(sexes)],
                "age_range": ages[i % len(ages)],
                "ethnicity": ethnicities[i % len(ethnicities)],
            }
            combo.update(pinned)
            out.append(combo)
        return out
