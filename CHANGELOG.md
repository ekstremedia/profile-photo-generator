# Changelog

All notable changes to this project are documented here.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [Unreleased]

### Fixed

- **Requested attributes were being silently discarded.** Prompts are now
  assembled to a word budget in an explicit priority order — framing, identity
  and age cues, `prompt_extra`, pinned attributes, skin tone, realism cues —
  with generated filler added only if room remains. Previously `prompt_extra`
  was appended last, so a request for a red scarf sat behind forty words of
  invented detail and fell past CLIP's 77-token cutoff without any warning that
  it had been dropped.
- **Ages at the extremes did not render.** A bare "90 year old" produced a
  well-preserved sixty, because SDXL's training data skews heavily towards
  attractive thirty-somethings. Each age band now carries explicit appearance
  cues ("very elderly, deeply wrinkled skin, age spots, sparse white hair") and
  a matching push away from youth in the negative prompt.
- **Options describing an absence had no effect.** "No glasses" in a prompt is
  as likely to summon glasses as to prevent them. Vocabulary options can now
  carry a `negative` field, used by `glasses: none` and the clean-shaven
  facial-hair values.
- **Reverted the two-text-encoder prompt split.** Routing the subject to
  encoder 1 and photographic style to encoder 2 appeared to double the token
  budget, but SDXL's pooled conditioning comes from the second encoder alone,
  so it measurably lost requested details — at a fixed seed, a red scarf and
  wire glasses both vanished. Both encoders now receive the same full prompt.

## [0.1.0] - 2026-08-17

Initial release.

### Added

- **Seeded attribute sampler** over `src/ppg/attributes/vocab.yaml`: thirteen
  axes (sex, age range, ethnicity, skin tone, profession, hair, facial hair,
  glasses, expression, clothing, background, lighting, camera) with per-option
  weights, age-aware rules and skin-tone affinities. The same seed and pins
  always produce the same person.
- **Prompt composition** in two interchangeable forms. `TemplateComposer` needs
  nothing but Python and produces complete prompts; `LLMComposer` uses a local
  Ollama model for more varied wording and a richer persona. `PPG_COMPOSER=auto`
  picks the LLM when Ollama is reachable and falls back silently when it is not,
  and `PPG_OLLAMA_MODEL=auto` selects a model the user already has.
- **SDXL rendering** through `diffusers`, defaulting to
  `SG161222/RealVisXL_V5.0` with the `madebyollin/sdxl-vae-fp16-fix` VAE, DPM++
  2M SDE Karras scheduling, and guidance 4.5. About 8 seconds per 1024×1024
  image at 30 steps on an RTX 4070 SUPER.
- **`openai_images` backend** for any OpenAI-compatible
  `/v1/images/generations` endpoint, including Ollama once its image generation
  reaches Linux.
- **`fake` backend** needing no GPU and no weights, for CI and for trying the
  API before downloading 7 GB.
- **HTTP API**: `POST /v1/avatars`, `POST /v1/avatars/batch`,
  `GET /v1/avatars/by-seed/{key}`, avatar listing, metadata, image serving and
  deletion, job status and results, `GET /v1/options`, `/healthz`, `/readyz`,
  `/metrics`, and an OpenAI-compatible `POST /v1/images/generations` shim.
  Optional bearer authentication through `PPG_API_KEY`.
- **Deterministic avatars**: `by-seed/{key}` hashes any string to a seed with
  blake2b, so `by-seed/{md5(email)}` is a stable per-user avatar that is
  generated once and cached forever.
- **Two caches**: a prompt cache that pins the wording a seed first produced,
  and a content-addressed image cache keyed on prompt, seed, model, sampler
  settings and `PIPELINE_VERSION`.
- **Single-worker job queue** with position, ETA from a rolling mean, and
  bounded depth (`PPG_MAX_QUEUE`). One GPU, one worker, on purpose.
- **Storage**: content-addressed files at four sizes (1024/512/256/128) in PNG
  and WebP, provenance metadata in PNG text chunks and EXIF, and a SQLite index
  in WAL mode.
- **CLI** (`ppg`): `doctor`, `warmup`, `generate`, `batch` (with contact sheet),
  `options`, `clear`, `serve`, `version`. `generate` also works against a remote
  server through `--url` / `PPG_URL`.
- **Deletion**: `DELETE /v1/avatars/{id}` for one, and `DELETE /v1/avatars`
  with `?confirm=true` for all. Cached prompts survive a clear, so a
  regenerated `by-seed` avatar comes back as the same face.
- **Safety rules**: free-text filtering for sexual content, age descriptors and
  real-person phrasings; age clamped to `PPG_MIN_AGE`–`PPG_MAX_AGE`; an opt-in
  minor mode that forces plain framing and drops styling overrides. Refusals
  surface as HTTP 422 before a request is queued.
- **Gallery UI** served from `static/`, driven by `/v1/options`. Generate one or
  a batch, inspect a face's persona, attributes and prompt, delete a single
  avatar from its card, or clear the whole gallery behind a two-step button.
- **Docker setup**: `docker/Dockerfile`, `docker/entrypoint.sh`, `compose.yaml`
  and `scripts/setup-host.sh`. `./data` is bind-mounted to `/data` so weights,
  images and the database stay on the host, `host.docker.internal` is wired for
  a host Ollama, and `cpu`, `ollama` and `dev` profiles cover the machines that
  do not match the default.
- **`scripts/download-models.py`**, which fetches the weights with include
  patterns — 6.94 GB instead of the 34.6 GB an unfiltered clone of the
  RealVisXL repository pulls, because it ships fp32 duplicates.
- **Documentation**: architecture, API, models, prompting, Laravel integration
  and troubleshooting, plus contributing, security and licence notices.

### Known limitations

- The `diffusers` backend loads SDXL checkpoints only; SD 1.5, FLUX and Z-Image
  need a new adapter or the `openai_images` backend.
- Jobs are in-memory and do not survive a restart. The avatars they produce do.
- The safety filter is a narrow text filter, not a classifier. See
  [SECURITY.md](SECURITY.md).
- Prompts are built to a word budget with a fixed priority order, so CLIP's
  77-token truncation removes generated filler rather than anything the caller
  asked for. A very long `prompt_extra` can still overflow on its own.

[Unreleased]: https://github.com/ekstremedia/profile-photo-generator/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/ekstremedia/profile-photo-generator/releases/tag/v0.1.0
