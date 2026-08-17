"""The generation pipeline.

One request becomes one avatar through a fixed sequence:

    validate -> resolve seed -> sample attributes -> compose prompt
             -> content hash -> cache lookup -> render -> write files -> index

Two caches sit in that chain and they cache different things for different
reasons:

* the **prompt cache** keeps the exact wording a seed first produced, so an
  LLM-composed avatar stays reproducible even though the LLM is not;
* the **image cache** is the content hash - identical inputs never render
  twice, which matters because rendering costs seconds and cache hits cost
  microseconds.
"""

from __future__ import annotations

import json
import logging
import shutil
import time
from datetime import UTC, datetime
from typing import Any

from ppg import __version__
from ppg.attributes.sampler import Attributes, Sampler, normalise_seed
from ppg.backends.base import ImageBackend, RenderSpec
from ppg.config import Settings
from ppg.prompt.composer import ComposeResult, PromptComposer
from ppg.safety import SafetyError, check_free_text, clamp_age, minor_mode
from ppg.schemas import AvatarRequest, AvatarResult, BatchRequest, Persona
from ppg.store.db import Database
from ppg.store.files import avatar_dir, compute_hash, has_variants
from ppg.store.imaging import build_metadata, write_variants

logger = logging.getLogger(__name__)


class AvatarService:
    def __init__(
        self,
        settings: Settings,
        db: Database,
        backend: ImageBackend,
        composer: PromptComposer,
    ) -> None:
        self.settings = settings
        self.db = db
        self.backend = backend
        self.composer = composer
        self.sampler = Sampler()

    # -- request resolution ---------------------------------------------
    def _resolve(
        self, request: AvatarRequest
    ) -> tuple[Attributes, int, str | None, str, str, bool]:
        extra = check_free_text(request.prompt_extra, field="prompt_extra")
        # The negative prompt is an avoidance list. Blocking words there would
        # be backwards - "nude" in a negative prompt is exactly what we want -
        # so it only gets a length check.
        negative_extra = " ".join((request.negative_extra or "").split())[:500]

        age = clamp_age(request.age, self.settings)
        seed_int, seed_key = normalise_seed(request.seed)

        attrs = self.sampler.sample(
            seed_int,
            request.pinned_attributes(),
            exact_age=age,
            min_age=0 if self.settings.allow_minors else self.settings.min_age,
            max_age=self.settings.max_age,
        )
        return (
            attrs,
            seed_int,
            seed_key,
            extra,
            negative_extra,
            minor_mode(attrs.age, self.settings),
        )

    def _prompt_cache_key(
        self, attrs: Attributes, seed: int, extra: str, negative_extra: str, plain: bool
    ) -> str:
        return compute_hash(
            {
                "kind": "prompt",
                "attributes": attrs.to_dict(),
                "seed": seed,
                "extra": extra,
                "negative_extra": negative_extra,
                "plain": plain,
                "composer": self.settings.composer,
            }
        )

    async def _compose(
        self, attrs: Attributes, seed: int, extra: str, negative_extra: str, plain: bool
    ) -> ComposeResult:
        key = self._prompt_cache_key(attrs, seed, extra, negative_extra, plain)
        if cached := self.db.get_prompt(key):
            return ComposeResult(
                subject=cached["subject"],
                style=cached["style"],
                negative_subject=cached["negative_subject"],
                negative_style=cached["negative_style"],
                persona=Persona(**cached["persona"]) if cached["persona"] else None,
                source=cached["source"],
            )

        result = await self.composer.compose(
            attrs,
            seed,
            extra=extra,
            negative_extra=negative_extra,
            plain_framing=plain,
        )
        self.db.put_prompt(
            key,
            result.subject,
            result.style,
            result.negative_subject,
            result.negative_style,
            result.persona.model_dump() if result.persona else None,
            result.source,
        )
        return result

    def precheck(self, request: AvatarRequest) -> None:
        """Run the input rules without generating anything.

        Called by the API before a request is queued, so a refused request
        fails immediately with 422 instead of surfacing minutes later as a
        failed job.
        """
        check_free_text(request.prompt_extra, field="prompt_extra")
        clamp_age(request.age, self.settings)
        largest = max(self.settings.sizes)
        if request.size is not None and not 16 <= request.size <= largest:
            raise SafetyError(
                f"size must be one of {self.settings.sizes}, or any value between 16 and {largest}."
            )

    # -- generation ------------------------------------------------------
    async def generate(self, request: AvatarRequest) -> AvatarResult:
        attrs, seed, seed_key, extra, negative_extra, plain = self._resolve(request)
        composed = await self._compose(attrs, seed, extra, negative_extra, plain)

        steps = self.settings.fast_steps if request.fast else self.settings.steps

        digest = compute_hash(
            {
                "kind": "image",
                "prompt": composed.prompt,
                "negative_prompt": composed.negative_prompt,
                "seed": seed,
                "model": self.backend.model_id,
                "backend": self.backend.name,
                "steps": steps,
                "guidance": self.settings.guidance,
                "width": self.settings.width,
                "height": self.settings.height,
                "sizes": self.settings.sizes,
            }
        )

        if has_variants(self.settings.outputs_dir, digest, self.settings.sizes):
            existing = self.db.get_avatar(digest)
            if existing:
                if seed_key and not existing.get("seed_key"):
                    existing["seed_key"] = seed_key
                    self.db.upsert_avatar({**existing, "id": digest})
                return self._to_result(existing, cached=True)
            logger.warning("Files for %s exist but the index row is missing; reindexing", digest)

        started = time.perf_counter()
        image = await self.backend.generate(
            RenderSpec(
                prompt=composed.subject,
                prompt_2=composed.style,
                negative_prompt=composed.negative_subject,
                negative_prompt_2=composed.negative_style,
                width=self.settings.width,
                height=self.settings.height,
                steps=steps,
                guidance=self.settings.guidance,
                seed=seed,
            )
        )
        duration_ms = int((time.perf_counter() - started) * 1000)

        persona_dict = composed.persona.model_dump() if composed.persona else None
        metadata = build_metadata(
            model=self.backend.model_id,
            backend=self.backend.name,
            seed=seed,
            prompt=composed.prompt,
            negative_prompt=composed.negative_prompt,
            attributes=attrs.to_dict(),
            persona=persona_dict,
            version=__version__,
        )
        write_variants(
            image,
            avatar_dir(self.settings.outputs_dir, digest),
            self.settings.sizes,
            metadata,
            webp_quality=self.settings.webp_quality,
        )

        record = {
            "id": digest,
            "seed": seed,
            "seed_key": seed_key,
            "attributes": attrs.to_dict(),
            "persona": persona_dict,
            "prompt": composed.prompt,
            "negative_prompt": composed.negative_prompt,
            "model": self.backend.model_id,
            "backend": self.backend.name,
            "composer": composed.source,
            "sizes": self.settings.sizes,
            "duration_ms": duration_ms,
            "created_at": datetime.now(UTC).isoformat(),
        }
        self.db.upsert_avatar(record)
        self.db.record_combo(_combo_key(attrs))
        logger.info(
            "Generated %s in %dms (%s, %s, seed=%d)",
            digest[:12],
            duration_ms,
            self.backend.name,
            composed.source,
            seed,
        )
        return self._to_result(record, cached=False)

    # -- batch -----------------------------------------------------------
    def plan_batch(self, request: BatchRequest) -> list[AvatarRequest]:
        """Expand a batch request into concrete, de-duplicated single requests.

        Random sampling clusters on whatever the vocabulary weights favour, so
        50 random avatars come out looking like 12 people. ``diversity="even"``
        walks a shuffled cross product of sex x age x ancestry instead, and
        combinations already present in the database are re-rolled, so a second
        batch does not repeat the first.
        """
        batch_seed, _ = normalise_seed(request.seed)
        base = request.overrides.model_dump(exclude_none=True)
        base.pop("seed", None)

        if request.diversity == "even":
            pins = self.sampler.strata(request.n, batch_seed, request.overrides.pinned_attributes())
        else:
            pins = [request.overrides.pinned_attributes() for _ in range(request.n)]

        out: list[AvatarRequest] = []
        for index, pin in enumerate(pins):
            seed = (batch_seed + index * 0x9E3779B1) % (1 << 63)
            for attempt in range(6):
                candidate = (seed + attempt * 0x85EBCA6B) % (1 << 63)
                attrs = self.sampler.sample(
                    candidate,
                    {**pin},
                    exact_age=base.get("age"),
                    min_age=0 if self.settings.allow_minors else self.settings.min_age,
                    max_age=self.settings.max_age,
                )
                if not self.db.seen_combo(_combo_key(attrs)):
                    seed = candidate
                    break
            out.append(AvatarRequest(**{**base, **pin, "seed": seed}))
        return out

    # -- helpers ---------------------------------------------------------
    def _to_result(self, record: dict[str, Any], *, cached: bool) -> AvatarResult:
        avatar_id = record["id"]
        sizes = record["sizes"]
        urls = {str(size): f"/v1/avatars/{avatar_id}/image?size={size}" for size in sizes}
        urls["default"] = f"/v1/avatars/{avatar_id}/image"
        persona = record.get("persona")
        return AvatarResult(
            id=avatar_id,
            hash=avatar_id,
            seed=record["seed"],
            seed_key=record.get("seed_key"),
            attributes=record["attributes"],
            persona=Persona(**persona) if persona else None,
            prompt=record["prompt"],
            negative_prompt=record["negative_prompt"],
            model=record["model"],
            backend=record["backend"],
            composer=record["composer"],
            sizes=sizes,
            urls=urls,
            created_at=_parse_dt(record["created_at"]),
            cached=cached,
            duration_ms=record.get("duration_ms"),
        )

    def get(self, avatar_id: str) -> AvatarResult | None:
        record = self.db.get_avatar(avatar_id)
        return self._to_result(record, cached=True) if record else None

    def recent(self, limit: int = 60, offset: int = 0) -> list[AvatarResult]:
        return [self._to_result(r, cached=True) for r in self.db.list_avatars(limit, offset)]

    def delete(self, avatar_id: str) -> bool:
        """Remove one avatar and its files."""
        if not self.db.delete_avatar(avatar_id):
            return False
        shutil.rmtree(avatar_dir(self.settings.outputs_dir, avatar_id), ignore_errors=True)
        return True

    def clear_all(self) -> int:
        """Remove every avatar and its files. Returns how many rows went.

        Files are removed per avatar rather than by wiping ``outputs_dir``
        wholesale: the directory is user-configurable, and an unconditional
        rmtree of a path someone pointed at the wrong place is not a mistake
        worth making recoverable-by-backup only.
        """
        ids = self.db.all_avatar_ids()
        removed = self.db.delete_all_avatars()
        for avatar_id in ids:
            shutil.rmtree(avatar_dir(self.settings.outputs_dir, avatar_id), ignore_errors=True)
        logger.info("Cleared %d avatars", removed)
        return removed


def _combo_key(attrs: Attributes) -> str:
    """Identity of an attribute set for de-duplication purposes.

    Only the axes a person would notice at avatar size. Two faces that differ
    only in background colour are still "the same combination".
    """
    subset = {
        axis: attrs.values.get(axis, "")
        for axis in (
            "sex",
            "ethnicity",
            "skin_tone",
            "profession",
            "hair",
            "facial_hair",
            "glasses",
        )
    }
    subset["age_range"] = attrs.age_range
    return json.dumps(subset, sort_keys=True)


def _parse_dt(value: Any) -> datetime:
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return datetime.now(UTC)


__all__ = ["AvatarService", "SafetyError"]
