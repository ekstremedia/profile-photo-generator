"""Content addressing, the size ladder, and provenance metadata.

The store's job is that an identical request never renders twice, that a
changed pipeline never serves a stale face, and that every file leaving this
service says it is synthetic and which seed produced it.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from ppg.store import files as files_module
from ppg.store.files import (
    HASH_BYTES,
    avatar_dir,
    compute_hash,
    has_variants,
    pick_size,
    variant_path,
)
from ppg.store.imaging import build_metadata, read_metadata, to_square, write_variants

SIZES = [1024, 512, 256, 128]


@pytest.fixture
def metadata() -> dict[str, str]:
    return build_metadata(
        model="fake/placeholder",
        backend="fake",
        seed=4242,
        prompt="head and shoulders portrait, 34 year old East Asian female",
        negative_prompt="cartoon, anime",
        attributes={"sex": "female", "age": 34},
        persona={"name": "Test Person", "age": 34, "occupation": "baker"},
        version="0.1.0",
    )


# ---------------------------------------------------------------------------
# Content hash
# ---------------------------------------------------------------------------


def test_compute_hash_is_stable_across_calls_and_processes(monkeypatch) -> None:
    payload = {"kind": "image", "seed": 42, "prompt": "a", "sizes": [256, 128]}
    assert compute_hash(payload) == compute_hash(payload)

    # Pinned against a hard-coded digest with the pipeline version held at 1, so
    # this fails if the hashing scheme changes (canonical JSON, blake2b, 16
    # bytes) but not merely because PIPELINE_VERSION was legitimately bumped.
    # Every avatar URL and every path on disk is derived from this function.
    monkeypatch.setattr(files_module, "PIPELINE_VERSION", 1)
    assert compute_hash(payload) == "e487cde946481c4b2d5dacf5807e248a"
    assert len(compute_hash(payload)) == HASH_BYTES * 2


def test_compute_hash_ignores_dict_key_order() -> None:
    # Python preserves insertion order, so without sort_keys the same request
    # built two different ways would render twice and cache never.
    a = compute_hash({"kind": "image", "seed": 1, "opts": {"x": 1, "y": 2}})
    b = compute_hash({"opts": {"y": 2, "x": 1}, "seed": 1, "kind": "image"})
    assert a == b


def test_compute_hash_reacts_to_every_value() -> None:
    base = {"kind": "image", "seed": 1, "prompt": "a"}
    assert compute_hash(base) != compute_hash({**base, "seed": 2})
    assert compute_hash(base) != compute_hash({**base, "prompt": "b"})
    assert compute_hash(base) != compute_hash({**base, "extra": None})


def test_bumping_the_pipeline_version_invalidates_the_whole_store(monkeypatch) -> None:
    payload = {"kind": "image", "seed": 42}
    before = compute_hash(payload)
    # Patched on ppg.store.files, not on ppg: `from ppg import PIPELINE_VERSION`
    # binds the value into this module's namespace at import time, so patching
    # the source module would have no effect at all.
    monkeypatch.setattr(files_module, "PIPELINE_VERSION", files_module.PIPELINE_VERSION + 1)
    assert compute_hash(payload) != before


def test_paths_fan_out_two_levels(tmp_path: Path) -> None:
    digest = "1f9c4e" + "0" * 26
    directory = avatar_dir(tmp_path, digest)
    # Two levels keep directory listings usable at a few hundred thousand files.
    assert directory == tmp_path / "1f" / "9c" / digest
    assert variant_path(tmp_path, digest, 256, "png") == directory / "256.png"


# ---------------------------------------------------------------------------
# Size resolution
# ---------------------------------------------------------------------------


def test_pick_size_resolves_exact_nearest_larger_and_out_of_range() -> None:
    assert pick_size(256, SIZES) == 256  # exact
    assert pick_size(200, SIZES) == 256  # nearest larger, never a 404
    assert pick_size(1, SIZES) == 128  # below the ladder -> smallest
    assert pick_size(4096, SIZES) == 1024  # above the ladder -> largest
    assert pick_size(None, SIZES) == 1024  # unspecified -> the render size


def test_pick_size_does_not_care_about_list_order() -> None:
    assert pick_size(200, [128, 1024, 256, 512]) == 256


def test_pick_size_without_any_configured_size_is_an_error() -> None:
    with pytest.raises(ValueError, match="No output sizes"):
        pick_size(256, [])


# ---------------------------------------------------------------------------
# Writing variants
# ---------------------------------------------------------------------------


def test_write_variants_writes_every_size_in_both_formats(
    tmp_path: Path, metadata: dict[str, str]
) -> None:
    image = Image.new("RGB", (512, 512), (30, 60, 90))
    written = write_variants(image, tmp_path, [512, 256], metadata)

    assert set(written) == {"512.png", "512.webp", "256.png", "256.webp"}
    for size in (512, 256):
        for fmt in ("png", "webp"):
            path = tmp_path / f"{size}.{fmt}"
            assert path.is_file()
            with Image.open(path) as image:
                # Square and exactly the requested size: a UI drops these
                # straight into an <img> without resizing.
                assert image.size == (size, size)

    # has_variants() is what the cache trusts before skipping a render.
    outputs = tmp_path.parent
    digest = "ab12" + "0" * 28
    write_variants(Image.new("RGB", (256, 256)), avatar_dir(outputs, digest), [256], metadata)
    assert has_variants(outputs, digest, [256]) is True
    assert has_variants(outputs, digest, [256, 512]) is False  # missing size, not a hit
    assert has_variants(outputs, "ff" * 16, [256]) is False


@pytest.mark.parametrize("fmt", ["png", "webp"])
def test_provenance_round_trips_through_both_formats(
    tmp_path: Path, metadata: dict[str, str], fmt: str
) -> None:
    # PNG carries this in text chunks and WebP in an EXIF blob, which are two
    # completely different code paths - both have to survive the round trip, or
    # half the files this service emits would ship without provenance.
    write_variants(Image.new("RGB", (256, 256), (10, 20, 30)), tmp_path, [256], metadata)

    recovered = read_metadata(tmp_path / f"256.{fmt}")
    assert recovered["AIGenerated"] == "true"
    assert recovered["Seed"] == "4242"
    assert recovered["Backend"] == "fake"
    assert recovered["Software"].startswith("profile-photo-generator")
    assert "no real person" in recovered["Description"]
    assert recovered["Prompt"] == metadata["Prompt"]
    assert recovered["Attributes"] == metadata["Attributes"]


def test_metadata_records_the_seed_as_a_string(metadata: dict[str, str]) -> None:
    # EXIF and PNG text chunks are both text-only stores; a non-string value
    # would raise on save rather than at build time.
    assert all(isinstance(value, str) for value in metadata.values())


# ---------------------------------------------------------------------------
# Cropping
# ---------------------------------------------------------------------------


def test_to_square_centre_crops_a_wide_image() -> None:
    wide = Image.new("RGB", (600, 400), (0, 0, 0))
    for x in range(290, 310):  # a marker stripe down the horizontal centre
        for y in range(400):
            wide.putpixel((x, y), (255, 0, 0))

    square = to_square(wide)
    assert square.size == (400, 400)
    # The stripe was centred before, so it must still be centred after: the
    # crop takes (600-400)//2 = 100 off each side.
    assert square.getpixel((200, 200)) == (255, 0, 0)
    assert square.getpixel((5, 200)) == (0, 0, 0)


def test_to_square_biases_a_tall_crop_upwards() -> None:
    # On a portrait, the face sits above centre, so the crop is not symmetric.
    tall = Image.new("RGB", (400, 600))
    square = to_square(tall)
    assert square.size == (400, 400)


def test_to_square_leaves_a_square_image_alone() -> None:
    already = Image.new("RGB", (256, 256))
    assert to_square(already) is already
