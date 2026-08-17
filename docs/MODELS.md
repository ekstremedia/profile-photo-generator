# Models

The default is `SG161222/RealVisXL_V5.0` with `madebyollin/sdxl-vae-fp16-fix`
as the VAE. That combination is what the defaults in `.env.example` are tuned
for, and it is what the measurements below come from.

## What the diffusers backend can load

`backends/diffusers_sdxl.py` builds a `StableDiffusionXLPipeline`. That is a
hard constraint, not a preference: **`PPG_MODEL_ID` must be an SDXL-architecture
checkpoint in diffusers layout.** An SD 1.5 checkpoint, a FLUX checkpoint or a
Z-Image checkpoint will fail to load, because they need different pipeline
classes. Options for those are at the bottom of this page.

The scheduler is also fixed in code: DPM++ 2M SDE with Karras sigmas
(`DPMSolverMultistepScheduler`, `algorithm_type="sde-dpmsolver++"`,
`use_karras_sigmas=True`). RealVisXL was tuned against it, and the stock Euler
scheduler gives noticeably plasticky skin.

## Installing torch

`pip install -e ".[gpu]"` takes torch from PyPI, which is the right build for
most recent NVIDIA cards on Linux — the Linux wheel bundles its own CUDA
runtime, cuDNN and cuBLAS, which is also why the Dockerfile uses a plain
`python:3.12-slim` base rather than an `nvidia/cuda` image.

If that build does not match your driver, install torch first from the wheel
index that does, then the rest:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e ".[gpu]"
```

Check the result before anything else:

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
```

`False` there means every generation will run on the CPU at minutes per image.

## Candidates

Disk figures are fp16 diffusers weights. Only the first row was measured on the
reference machine (RTX 4070 SUPER 12 GB, torch 2.13.0+cu130, diffusers 0.39.0).

| Model | Arch | VRAM (fp16) | Disk | Speed | Licence |
| --- | --- | --- | --- | --- | --- |
| `SG161222/RealVisXL_V5.0` (default) | SDXL | ~7 GB resident | 6.94 GB | ~8 s at 1024², 30 steps; 3.3 s load | CreativeML Open RAIL++-M |
| `RunDiffusion/Juggernaut-XL-v9` | SDXL | ~7 GB | ~7 GB | comparable; same arch and step count | CreativeML Open RAIL++-M |
| `stabilityai/stable-diffusion-xl-base-1.0` | SDXL | ~7 GB | ~7 GB | comparable | CreativeML Open RAIL++-M |
| `Tongyi-MAI/Z-Image-Turbo` (6B) | Z-Image | 14–16 GB in BF16 | ~12 GB | few-step model, so fast where it fits | Apache-2.0 |
| SD 1.5-class realistic checkpoints | SD 1.5 | ~4 GB | ~2 GB | fastest; 512² native | CreativeML Open RAIL-M |

Notes on each:

**RealVisXL V5.0** is the default because it is tuned for photorealistic faces
and behaves well at low guidance. Its repository ships fp16 *and* fp32 copies
of everything — see the download trap below.

**Juggernaut XL** is the obvious alternative: same architecture, same VRAM,
same scheduler, a slightly different aesthetic (a little more contrast and
polish, which for avatars can read as more "produced"). Worth trying if
RealVisXL's output feels too flat for you.

**SDXL base 1.0** is the reference point rather than a recommendation. It is
less flattering on faces than either fine-tune, but it is the model everything
else is derived from and it is the safest thing to test a pipeline change
against.

**Z-Image-Turbo** is attractive because it is Apache-2.0, which removes the
use-based restrictions that come with the RAIL++ weights, and because it is a
few-step model. The catch is memory: at BF16 it wants roughly 14–16 GB, so a
12 GB card needs an fp8 quantisation or CPU offload, and neither is wired up
here. It is also not an SDXL pipeline, so the `diffusers` backend cannot load
it — run it in Ollama or another server and point the `openai_images` backend
at it. Follow the model card for step count and guidance; turbo models want far
fewer steps and much lower CFG than the defaults in this project.

**SD 1.5-class checkpoints** are the answer for a 4–6 GB card. They render at
512² natively and are several times faster, at a visible cost in skin detail
and hand/ear anatomy. They need a `StableDiffusionPipeline`, which means
writing a small backend adapter (see [CONTRIBUTING.md](../CONTRIBUTING.md)) —
about thirty lines, mostly the same as the SDXL one.

## Swapping the checkpoint

```bash
# .env
PPG_MODEL_ID=RunDiffusion/Juggernaut-XL-v9
PPG_VAE_ID=madebyollin/sdxl-vae-fp16-fix
```

Then fetch the weights and check the result:

```bash
python scripts/download-models.py --model RunDiffusion/Juggernaut-XL-v9
ppg doctor
ppg warmup

# in Docker
docker compose run --rm api python scripts/download-models.py --model RunDiffusion/Juggernaut-XL-v9
docker compose restart api
```

`ppg warmup` loads the pipeline and renders one throwaway image, reporting both
timings. It is the fastest way to find out that a swap did not work.

### What to check when you swap

1. **VAE compatibility.** Every SDXL fine-tune shares the SDXL VAE, so
   `madebyollin/sdxl-vae-fp16-fix` stays correct. Do not point `PPG_VAE_ID` at
   a model's bundled VAE unless you know it is fp16-safe — that is how you get
   all-black images. An SD 1.5 model needs an SD 1.5 VAE instead.
2. **Guidance range.** RealVisXL and most modern SDXL fine-tunes want CFG
   4–5 (`PPG_GUIDANCE=4.5`). SD 1.5-era models want 7–8. Turbo and distilled
   models want ~1. Carrying the wrong number across is the single most common
   reason a swapped model looks worse.
3. **Scheduler.** Hard-coded to DPM++ 2M SDE Karras. If your model's card
   insists on something else, that is a code change in
   `backends/diffusers_sdxl.py`, not a configuration change.
4. **Step count.** `PPG_STEPS=30` and `PPG_FAST_STEPS=15` suit a 25–40 step
   model. A turbo model at 30 steps is wasted time and often worse output.
5. **The cache.** The model id is part of the content hash, so swapping models
   does not serve you stale images — but it also means your existing avatars
   will not be regenerated in the new style unless you delete them.

## The fp32 duplicate download trap

An unfiltered clone of `SG161222/RealVisXL_V5.0` pulls **34.6 GB**, because the
repository contains fp32 copies of the UNet and text encoders alongside the
fp16 ones, plus single-file checkpoints. Only 6.94 GB of that is used.

`scripts/download-models.py` therefore fetches with explicit include patterns:

```python
[
    "model_index.json",
    "scheduler/*",
    "tokenizer*/*",
    "*/config.json",
    "*.fp16.safetensors",
]
```

Two consequences:

- **`--model` does not do this.** Repositories passed with
  `--model <repo>` are fetched with `["*"]`, i.e. everything. For a repo with
  fp32 duplicates, pass explicit patterns to `hf download` yourself instead, or
  accept the extra tens of gigabytes.
- **Some repos have no `.fp16.safetensors` files at all.**
  `madebyollin/sdxl-vae-fp16-fix` is one of them, which is why the backend
  loads the VAE with no `variant` argument and casts afterwards. For the main
  checkpoint the backend tries `variant="fp16"` first and retries without it,
  logging a warning; that fallback works, but it downloads fp32 weights and
  costs both disk and load time.

Check what you actually have:

```bash
du -sh data/hf-cache/hub/*
ppg doctor      # the "weights" row reports the cached size
```

## The openai_images backend

Point the service at any OpenAI-compatible `/v1/images/generations` endpoint —
a hosted provider, a separate GPU box, or Ollama once its image generation
supports Linux:

```bash
PPG_BACKEND=openai_images
PPG_OPENAI_BASE_URL=http://host.docker.internal:11434/v1
PPG_OPENAI_MODEL=x/z-image-turbo
PPG_OPENAI_API_KEY=            # only if the remote endpoint needs one
```

Everything else — sampling, prompt composition, caching, the size ladder,
provenance metadata, the API — is unchanged.

Two things do degrade, because the OpenAI images schema has neither field:

- **No negative prompt.** It is folded into the positive prompt as an
  "Avoid: ..." clause, which models respect much less reliably.
- **No seed.** One is sent as a non-standard field; compliant servers ignore
  it, and the adapter retries without it if the server returns 400 or 422. Your
  attributes and persona stay deterministic — the pixels may not.

`PPG_STEPS`, `PPG_GUIDANCE` and `PPG_LOW_VRAM` have no effect on this backend;
the remote server decides.

## The fake backend

```bash
PPG_BACKEND=fake
```

No model, no GPU, no download. It renders a deterministic gradient card with a
head-and-shoulders silhouette and a short seed fingerprint — obviously not a
photograph, which is the point. Use it for CI, for client development, and for
trying the API before spending 7 GB.

## Licences in one line

The code is MIT. The default RealVisXL and Juggernaut weights are CreativeML
Open RAIL++-M, which carries use-based restrictions MIT does not have. The
fp16-fix VAE is MIT. If you need permissive weights, Z-Image-Turbo is
Apache-2.0. Details and alternatives are in [NOTICE.md](../NOTICE.md).
