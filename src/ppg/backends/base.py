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

    ``prompt`` and ``prompt_2`` are the two halves described in
    ``prompt/templates.py``: the subject and the photographic style. SDXL feeds
    them to its two text encoders, each with its own 77-token budget. Backends
    with a single text input should join them with ", ".
    """

    prompt: str
    negative_prompt: str
    width: int
    height: int
    steps: int
    guidance: float
    seed: int
    prompt_2: str = ""
    negative_prompt_2: str = ""

    @property
    def full_prompt(self) -> str:
        return f"{self.prompt}, {self.prompt_2}" if self.prompt_2 else self.prompt

    @property
    def full_negative_prompt(self) -> str:
        if self.negative_prompt_2:
            return f"{self.negative_prompt}, {self.negative_prompt_2}"
        return self.negative_prompt


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
