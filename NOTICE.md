# Notices and third-party licences

Short version: **the code is MIT, the default model weights are not.** Nothing
here is unusual for a project that uses Stable Diffusion, but the difference is
worth two minutes of your time if you are shipping something commercial.

## This project's code

MIT, Copyright (c) 2026 ekstremedia. See [LICENSE](LICENSE). That covers
everything in `src/`, `scripts/`, `static/`, `docs/` and the Docker files.

Model weights are **not** distributed with this repository. They are downloaded
from Hugging Face at setup time, under their own licences.

## Default model weights

| Artefact | Licence | Notes |
| --- | --- | --- |
| `SG161222/RealVisXL_V5.0` | CreativeML Open RAIL++-M | the default checkpoint; an SDXL fine-tune |
| `madebyollin/sdxl-vae-fp16-fix` | MIT | the fp16-safe VAE; required |
| `RunDiffusion/Juggernaut-XL-v9` | CreativeML Open RAIL++-M | listed alternative |
| `stabilityai/stable-diffusion-xl-base-1.0` | CreativeML Open RAIL++-M | the base model both are derived from |

### What CreativeML Open RAIL++-M actually means

It is an open licence with a use-based restriction appendix. In practice:

- You may use, modify, redistribute and commercialise the model and its
  outputs.
- The licence does **not** claim ownership of images you generate.
- It attaches a list of prohibited uses — generating content that exploits
  minors, disinformation, harassment, discrimination, and similar — and
  requires you to pass the same restrictions on to anyone you redistribute the
  model or a derivative to.
- The restrictions travel with the *model*, not with this repository. Using
  this MIT-licensed code with different weights is not affected by them.

That is the whole practical difference from MIT: an obligation to not do a
listed set of things, and to pass that obligation on. For most people building
an avatar service it changes nothing. If your legal review requires
"no use restrictions on any component", it matters, and the alternatives below
exist.

Read the actual text on the model's Hugging Face page rather than trusting this
summary. Licences on Hugging Face can change between revisions.

## Permissively licensed alternatives

| Model | Licence | How it plugs in |
| --- | --- | --- |
| `Tongyi-MAI/Z-Image-Turbo` (6B) | Apache-2.0 | not an SDXL pipeline: run it in Ollama or another server and use `PPG_BACKEND=openai_images` |
| `black-forest-labs/FLUX.1-schnell` | Apache-2.0 | likewise, through `openai_images` or a custom backend |
| `PPG_BACKEND=fake` | MIT (this repo) | no model at all; placeholder cards for CI and demos |

Both permissive options need more VRAM than SDXL: Z-Image-Turbo wants roughly
14–16 GB at BF16, so a 12 GB card needs an fp8 quantisation or offload.
[docs/MODELS.md](docs/MODELS.md) has the details and the configuration.

The `openai_images` backend also lets you use a hosted provider, in which case
that provider's terms apply to the images and this section does not.

## Generated images

Images produced by this project depict people who do not exist. Every output
file carries provenance metadata — `AIGenerated: true`, the model, the backend,
the seed, the prompt — in PNG text chunks and EXIF. Keeping that metadata
intact is a good idea and costs nothing; see [SECURITY.md](SECURITY.md).

Whether you can use a generated image for a given purpose depends on the model
licence above, on the RAIL++ use restrictions where they apply, and on the law
where you are — several jurisdictions now require synthetic media to be
disclosed. This document is not legal advice.

## Python dependencies

Runtime, from `pyproject.toml`:

| Package | Licence |
| --- | --- |
| fastapi, pydantic, pydantic-settings, typer, rich, pyyaml, python-multipart | MIT |
| uvicorn, httpx, starlette | BSD-3-Clause |
| pillow | MIT-CMU |

GPU extras:

| Package | Licence |
| --- | --- |
| diffusers, transformers, accelerate | Apache-2.0 |
| torch, torchvision | BSD-3-Clause |
| safetensors, sentencepiece | Apache-2.0 |

Development only: pytest (MIT), ruff (MIT), mypy (MIT).

Package metadata is the authority; this table is a convenience and may drift as
versions change. `pip-licenses` will print the current state of your
environment.

## Vocabulary and name data

`src/ppg/attributes/vocab.yaml` and `src/ppg/prompt/names.yaml` are part of
this repository and MIT-licensed. The names in `names.yaml` are ordinary common
given names and surnames, combined at random; they are not intended to refer to
any real person, and any resemblance is coincidence.
