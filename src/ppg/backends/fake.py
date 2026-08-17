"""Placeholder backend - no model, no GPU, no download.

Two real uses:

* The test suite and CI run against this, so the API, sampler, cache and store
  are all provable on a free GitHub runner in seconds.
* ``PPG_BACKEND=fake docker compose up`` lets someone try the API, the CLI and
  the gallery before deciding whether to spend 7GB on weights.

Output is a deterministic abstract portrait-shaped card derived from the seed,
not a face. It is obviously not a photograph, which is the point.
"""

from __future__ import annotations

import colorsys
import hashlib

from PIL import Image, ImageDraw

from ppg.backends.base import RenderSpec
from ppg.config import Settings


class FakeBackend:
    name = "fake"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._loaded = False

    @property
    def model_id(self) -> str:
        return "fake/placeholder"

    @property
    def loaded(self) -> bool:
        return self._loaded

    async def load(self) -> None:
        self._loaded = True

    async def unload(self) -> None:
        self._loaded = False

    async def generate(self, spec: RenderSpec) -> Image.Image:
        return self._render(spec)

    # -- rendering ------------------------------------------------------
    @staticmethod
    def _palette(seed: int, prompt: str) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        digest = hashlib.blake2b(f"{seed}:{prompt}".encode(), digest_size=8).digest()
        hue = digest[0] / 255.0
        alt_hue = (hue + 0.42) % 1.0
        top = colorsys.hsv_to_rgb(hue, 0.35, 0.85)
        bottom = colorsys.hsv_to_rgb(alt_hue, 0.45, 0.45)
        to_255 = lambda c: tuple(int(v * 255) for v in c)  # noqa: E731
        return to_255(top), to_255(bottom)  # type: ignore[return-value]

    def _render(self, spec: RenderSpec) -> Image.Image:
        width, height = spec.width, spec.height
        top, bottom = self._palette(spec.seed, spec.prompt)

        image = Image.new("RGB", (width, height), top)
        draw = ImageDraw.Draw(image)

        # Vertical gradient.
        for y in range(height):
            t = y / max(height - 1, 1)
            colour = tuple(int(top[i] * (1 - t) + bottom[i] * t) for i in range(3))
            draw.line([(0, y), (width, y)], fill=colour)

        # A head-and-shoulders silhouette, so the aspect and framing of the
        # placeholder match what a real avatar looks like in a UI.
        fg = tuple(min(255, c + 60) for c in bottom)
        head_r = width * 0.18
        cx, cy = width / 2, height * 0.38
        draw.ellipse([cx - head_r, cy - head_r, cx + head_r, cy + head_r], fill=fg)
        shoulder_w = width * 0.62
        draw.ellipse(
            [cx - shoulder_w / 2, height * 0.62, cx + shoulder_w / 2, height * 1.35],
            fill=fg,
        )

        # Short seed fingerprint, so two placeholders are visibly different.
        tag = f"{spec.seed:016x}"[:8]
        draw.text((12, height - 24), f"fake:{tag}", fill=(255, 255, 255))
        return image
