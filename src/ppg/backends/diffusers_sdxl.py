"""SDXL via the diffusers library. The default backend.

Runs in-process rather than talking to a separate ComfyUI/A1111 service: one
container, one dependency tree, and no workflow JSON to keep in sync with the
API. The trade-off is less flexibility, which is why the backend interface
exists.

Three details in here are load-bearing and easy to get wrong:

1. **The VAE.** SDXL's stock VAE overflows in float16 and every image comes
   out pure black. ``madebyollin/sdxl-vae-fp16-fix`` replaces it. That repo has
   no ``.fp16.safetensors`` variant, so it must be loaded with no ``variant``
   argument and cast afterwards.
2. **The scheduler.** DPM++ 2M SDE with Karras sigmas is what RealVisXL was
   tuned against. The default Euler scheduler gives noticeably plasticky skin.
3. **Guidance.** RealVisXL V5 wants CFG around 4-5, not the SD 1.5-era 7-8.
   High CFG is the main cause of the over-saturated, over-contrasted "AI face".
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from typing import Any, cast

from PIL import Image

from ppg.backends.base import BackendError, RenderSpec
from ppg.config import Settings

logger = logging.getLogger(__name__)


class DiffusersSDXLBackend:
    name = "diffusers"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._pipe: Any = None
        self._device = settings.resolve_device()
        self._lock = asyncio.Lock()

    @property
    def model_id(self) -> str:
        return self.settings.model_id

    @property
    def loaded(self) -> bool:
        return self._pipe is not None

    @property
    def device(self) -> str:
        return self._device

    # -- loading --------------------------------------------------------
    async def load(self) -> None:
        if self._pipe is not None:
            return
        async with self._lock:
            if self._pipe is not None:
                return
            started = time.perf_counter()
            self._pipe = await asyncio.to_thread(self._load_sync)
            logger.info(
                "Loaded %s on %s in %.1fs",
                self.settings.model_id,
                self._device,
                time.perf_counter() - started,
            )

    def _load_sync(self) -> Any:
        # Reduces allocator fragmentation, which is what actually causes the
        # "tried to allocate 1GB, 1GB free" failures on a card that is also
        # driving a desktop. Must be set before the CUDA context is created.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

        try:
            import torch
            from diffusers import (
                AutoencoderKL,
                DPMSolverMultistepScheduler,
                StableDiffusionXLPipeline,
            )
        except ImportError as exc:  # pragma: no cover - depends on install extras
            raise BackendError(
                "The diffusers backend needs the GPU extras: "
                "pip install '.[gpu]' (see docs/MODELS.md), or run with "
                "PPG_BACKEND=fake to try the API without a model."
            ) from exc

        dtype = torch.float16 if self._device == "cuda" else torch.float32

        # See module docstring: no `variant` for this repo, it has no fp16 files.
        vae = AutoencoderKL.from_pretrained(self.settings.vae_id, torch_dtype=dtype)

        common: dict[str, Any] = {
            "vae": vae,
            "torch_dtype": dtype,
            "use_safetensors": True,
            # We write our own provenance metadata; the built-in invisible
            # watermarker would add a dependency for no benefit here.
            "add_watermarker": False,
        }

        try:
            pipe = StableDiffusionXLPipeline.from_pretrained(
                self.settings.model_id,
                variant="fp16" if dtype == torch.float16 else None,
                **common,
            )
        except Exception as exc:
            logger.warning(
                "from_pretrained(variant=fp16) failed for %s (%s); retrying without a variant",
                self.settings.model_id,
                exc,
            )
            pipe = StableDiffusionXLPipeline.from_pretrained(self.settings.model_id, **common)

        pipe.scheduler = DPMSolverMultistepScheduler.from_config(
            pipe.scheduler.config,
            algorithm_type="sde-dpmsolver++",
            use_karras_sigmas=True,
        )

        if self.settings.low_vram and self._device == "cuda":
            # Streams submodules to the GPU on demand. Roughly 2x slower but
            # fits comfortably on an 8GB card.
            pipe.enable_model_cpu_offload()
        else:
            pipe.to(self._device)

        # VAE decode of a 1024px latent wants a ~1GB contiguous block, which is
        # exactly where a 12GB card with a desktop session on it runs out.
        # Slicing handles batches, tiling handles resolution - we need tiling.
        # (`pipe.enable_vae_slicing()` is deprecated in diffusers 0.40.)
        for target, method in ((pipe.vae, "enable_slicing"), (pipe.vae, "enable_tiling")):
            if hasattr(target, method):
                getattr(target, method)()

        pipe.set_progress_bar_config(disable=True)

        if self.settings.compile and self._device == "cuda":
            logger.info("torch.compile enabled - the first image will take several minutes")
            pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

        return pipe

    async def unload(self) -> None:
        async with self._lock:
            if self._pipe is None:
                return
            self._pipe = None
            try:
                import gc

                import torch

                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:  # pragma: no cover
                pass

    # -- generation -----------------------------------------------------
    async def generate(self, spec: RenderSpec) -> Image.Image:
        await self.load()
        async with self._lock:
            return await asyncio.to_thread(self._generate_sync, spec)

    def _generate_sync(self, spec: RenderSpec) -> Image.Image:
        import torch

        # Offloaded pipelines expect a CPU generator; resident ones want the
        # device generator. Getting this wrong silently breaks reproducibility.
        gen_device = "cpu" if self.settings.low_vram else self._device
        generator = torch.Generator(device=gen_device).manual_seed(spec.seed % (2**63))

        try:
            with torch.inference_mode():
                output = self._pipe(
                    prompt=spec.prompt,
                    # The second text encoder gets its own 77 tokens. Without
                    # this the realism cues fall off the end of encoder 1 and
                    # the output drifts back towards retouched stock photos.
                    prompt_2=spec.prompt_2 or None,
                    negative_prompt=spec.negative_prompt,
                    negative_prompt_2=spec.negative_prompt_2 or None,
                    width=spec.width,
                    height=spec.height,
                    num_inference_steps=spec.steps,
                    guidance_scale=spec.guidance,
                    generator=generator,
                )
        except torch.cuda.OutOfMemoryError as exc:  # pragma: no cover - hardware dependent
            torch.cuda.empty_cache()
            raise BackendError(
                "CUDA out of memory. Set PPG_LOW_VRAM=true, or reduce PPG_WIDTH/PPG_HEIGHT "
                "to 768. See docs/TROUBLESHOOTING.md."
            ) from exc
        except Exception as exc:
            raise BackendError(f"Image generation failed: {exc}") from exc

        image = output.images[0]
        self._assert_not_black(image)
        return image

    @staticmethod
    def _assert_not_black(image: Image.Image) -> None:
        """Catch the fp16 VAE failure explicitly rather than serving black PNGs."""
        # Converted to a single band first, so getextrema() is a plain
        # (min, max) pair - the multi-band return shape cannot occur here.
        _lowest, brightest = cast("tuple[int, int]", image.convert("L").getextrema())
        if brightest < 5:
            raise BackendError(
                "The model produced an all-black image. This is almost always the "
                "float16 VAE overflow: check that PPG_VAE_ID points at "
                "madebyollin/sdxl-vae-fp16-fix and that it downloaded correctly."
            )
