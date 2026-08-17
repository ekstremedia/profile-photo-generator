"""Content-addressed image storage.

The path of an avatar is derived from everything that went into making it, so
an identical request never renders twice, and a changed default (a new
scheduler, a different step count) produces a different path rather than
silently serving a stale face.

    data/outputs/1f/9c/1f9c4e...d2/1024.webp

Two levels of fan-out keep directory listings usable once there are a few
hundred thousand avatars.
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

from ppg import PIPELINE_VERSION

HASH_BYTES = 16  # 32 hex characters - short enough for a URL, no collisions in practice


def compute_hash(payload: dict[str, Any]) -> str:
    """Stable hash of a request's full identity.

    ``PIPELINE_VERSION`` is folded in so a change to the realism cues or the
    sampler invalidates the whole store instead of mixing old and new output.
    """
    canonical = json.dumps(
        {**payload, "_pipeline": PIPELINE_VERSION},
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=str,
    )
    return hashlib.blake2b(canonical.encode("utf-8"), digest_size=HASH_BYTES).hexdigest()


# Avatar ids are blake2b hex digests and nothing else. Validated here, at the
# point where an id becomes a filesystem path, rather than trusting every
# caller to have checked: `avatar_dir(out, "../../../etc")` used to produce
# "/../../../etc", and one of this module's callers hands that straight to
# shutil.rmtree.
_DIGEST_RE = re.compile(r"^[0-9a-f]{8,64}$")


class InvalidDigest(ValueError):
    """The supplied avatar id is not a hex digest."""


def validate_digest(digest: str) -> str:
    if not _DIGEST_RE.fullmatch(digest):
        raise InvalidDigest(f"Not a valid avatar id: {digest!r}")
    return digest


def avatar_dir(outputs_dir: Path, digest: str) -> Path:
    validate_digest(digest)
    return outputs_dir / digest[:2] / digest[2:4] / digest


def variant_path(outputs_dir: Path, digest: str, size: int, fmt: str = "webp") -> Path:
    return avatar_dir(outputs_dir, digest) / f"{size}.{fmt}"


def has_variants(outputs_dir: Path, digest: str, sizes: list[int]) -> bool:
    """True when every expected file is already on disk.

    Checked before trusting a database hit: the database and the filesystem can
    disagree if someone clears ``data/outputs`` without clearing the database,
    and serving a 404 for a row that claims to exist is a confusing failure.
    """
    try:
        directory = avatar_dir(outputs_dir, digest)
    except InvalidDigest:
        return False
    if not directory.is_dir():
        return False
    return all((directory / f"{size}.{fmt}").is_file() for size in sizes for fmt in ("webp", "png"))


def pick_size(requested: int | None, available: list[int]) -> int:
    """Resolve a requested size to one that exists.

    Falls back to the smallest size that is at least as large as the request,
    so ``?size=200`` gets the 256px file rather than a 404.
    """
    if not available:
        raise ValueError("No output sizes are configured.")
    ordered = sorted(available)
    if requested is None:
        return ordered[-1]
    if requested in ordered:
        return requested
    for size in ordered:
        if size >= requested:
            return size
    return ordered[-1]
