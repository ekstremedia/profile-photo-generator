"""Image backend interface.

Swapping how pixels get made should never touch the API, the sampler or the
store. Everything above this line deals in :class:`RenderSpec` in and a PIL
image out.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from PIL import Image

from ppg.config import Settings


@dataclass(frozen=True)
class RenderSpec:
    """One image request.

    A single prompt: on SDXL both text encoders receive the same text, because
    the pooled conditioning comes from the second encoder alone and giving it
    anything other than the full prompt measurably costs adherence. See
    ``prompt/templates.py``.
    """

    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: int


@runtime_checkable
class ImageBackend(Protocol):
    name: str

    @property
    def model_id(self) -> str: ...

    @property
    def loaded(self) -> bool: ...

    async def load(self) -> None: ...

    async def generate(self, spec: RenderSpec) -> Image.Image: ...

    async def unload(self) -> None: ...


class BackendError(RuntimeError):
    """Generation failed. Surfaces as a failed job, not a crashed worker."""


def build_backend(settings: Settings) -> ImageBackend:
    """Instantiate the configured backend without importing the others.

    The imports are deliberately local: `diffusers_sdxl` pulls in torch, which
    is a multi-second import and is not installed at all in the test image.
    """
    if settings.backend == "fake":
        from ppg.backends.fake import FakeBackend

        return FakeBackend(settings)
    if settings.backend == "openai_images":
        from ppg.backends.openai_images import OpenAIImagesBackend

        return OpenAIImagesBackend(settings)

    from ppg.backends.diffusers_sdxl import DiffusersSDXLBackend

    return DiffusersSDXLBackend(settings)
