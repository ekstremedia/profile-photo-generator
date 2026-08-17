# Contributing

Thanks for looking. This project is small, opinionated and easy to work on: one
Python package, no build step, and a test suite that runs on any machine
because it does not need a GPU.

## Development setup

```bash
git clone https://github.com/ekstremedia/profile-photo-generator
cd profile-photo-generator
python -m venv .venv && source .venv/bin/activate

pip install -e ".[dev]"          # everything except torch/diffusers
pip install -e ".[gpu,dev]"      # add this if you have an NVIDIA card

cp .env.example .env
```

You do not need the model weights to work on most of the code. Start with:

```bash
PPG_BACKEND=fake ppg serve
```

That gives you the full API, the CLI, the sampler, the caches, the store and
the gallery, with placeholder images instead of faces.

## Running the tests

```bash
PPG_BACKEND=fake pytest
```

`IMAGE_BACKEND=fake` is accepted as an alias, because it reads better in CI
invocations. The suite is expected to pass with no GPU, no weights and no
Ollama running — if a change makes that untrue, the change needs rethinking.
Anything requiring real weights belongs behind an explicit opt-in, not in the
default run.

## Linting and types

```bash
ruff check .
ruff format .
mypy
```

Ruff is configured in `pyproject.toml`: line length 100, targeting Python 3.11,
with `E, F, I, UP, B, C4, SIM, RUF` enabled. `B008` is ignored because
FastAPI's `Depends()` in a default argument is idiomatic.

mypy runs against the `ppg` package with `mypy_path = src`. It is configured
for Python 3.12 even though the floor is 3.11 — numpy's stubs use 3.12-only
syntax that mypy refuses to parse under an older target. Keep the code itself
3.11-compatible.

## Project layout

```
src/ppg/
  __init__.py            version, PIPELINE_VERSION, the CUDA allocator setting
  config.py              pydantic-settings; every PPG_* variable lives here
  schemas.py             the public request/response contract - keep changes additive
  safety.py              input rules; raises SafetyError, which becomes a 422
  service.py             the generation pipeline, end to end
  cli.py                 doctor, warmup, generate, batch, options, serve, version
  api/                   FastAPI app factory, routers, the bearer-token dependency
  attributes/            vocab.yaml and the seeded sampler
  prompt/                template and LLM composers, Ollama client, names.yaml
  backends/              ImageBackend implementations
  pipeline/worker.py     single-consumer job queue
  store/                 content-addressed files, image writing, SQLite
scripts/                 download-models.py, setup-host.sh
static/                  the gallery UI
docs/                    the documentation you are reading
```

[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) explains how these fit together
and why the non-obvious decisions were made. Read it before a structural
change.

## The best first contribution: a vocabulary entry

`src/ppg/attributes/vocab.yaml` decides what kind of people get generated.
Adding a profession, a hairstyle or a lighting setup is one line of YAML and no
Python:

```yaml
    - { value: bicycle_mechanic, prompt: bicycle mechanic }
```

Then look at the result:

```bash
ppg options profession
ppg generate --profession bicycle_mechanic --seed demo
ppg batch -n 24 --diversity even --seed check   # writes ./out/contact-sheet.jpg
```

House style, and what a review will look for:

- Breadth over polish. Trades, care work, service work and manual work are
  deliberately over-represented relative to founders and designers.
- Visual phrasing. `prompt` text goes to an image model: "ship's engineer"
  gives it something to draw, "logistics coordination specialist" does not.
- Weight the long tail down rather than out (`weight: 0.3`), so it appears
  occasionally instead of never or constantly.
- No implied correlations. Ancestry, profession, class and setting are sampled
  independently on purpose.
- Rebalancing existing weights is a real contribution, not a lesser one. If a
  batch of 50 comes out with twelve people in blazers, the fix is a weight.

Additions to `src/ppg/prompt/names.yaml` from people who know a region better
than that file does are equally welcome.

Full details of the option format, the `rules` section and
`skin_tone_affinity` are in [docs/PROMPTING.md](docs/PROMPTING.md).

## Adding a backend adapter

A backend turns a `RenderSpec` into a PIL image and nothing else. See
`backends/base.py`:

```python
class ImageBackend(Protocol):
    name: str
    @property
    def model_id(self) -> str: ...
    @property
    def loaded(self) -> bool: ...
    async def load(self) -> None: ...
    async def generate(self, spec: RenderSpec) -> Image.Image: ...
    async def unload(self) -> None: ...
```

To add one:

1. Write `src/ppg/backends/<name>.py` implementing the protocol. Raise
   `BackendError` for anything the caller could act on; it is reported as a
   failed job rather than crashing the worker.
2. Add the literal to `BackendName` in `config.py` and a branch to
   `build_backend()` in `backends/base.py`. Keep the import inside the branch —
   importing torch takes seconds and is not installed in the test image.
3. Add any settings to `Settings` and document them in `.env.example`.
4. Add a row to the table in [docs/MODELS.md](docs/MODELS.md).

`name` and `model_id` go into the content hash, so images from different
backends never collide in the cache. Pick a stable `name`; changing it later
invalidates every image that backend produced.

`backends/fake.py` is 90 lines and is the easiest one to copy.

## Changing behaviour that affects existing images

If a change would make already-cached images stale — a different sampler, a
different realism cue block, different defaults — bump `PIPELINE_VERSION` in
`src/ppg/__init__.py`. It is folded into every cache key, so bumping it
invalidates the whole store at once instead of mixing old and new output.

Adding a field to a response model is fine and needs no bump. Removing or
renaming one is a breaking change to the contract that the CLI, the gallery and
every Laravel client depend on; discuss it in an issue first.

## Commit style

[Conventional Commits](https://www.conventionalcommits.org/):

```
feat(vocab): add bicycle mechanic and locksmith professions
fix(backends): retry SDXL load without a variant when fp16 files are absent
docs(laravel): show the proxy route needed when PPG_API_KEY is set
chore(deps): raise the diffusers floor to 0.31
```

Types in use: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`.
Scopes follow the module names above (`vocab`, `api`, `backends`, `prompt`,
`store`, `cli`, `docker`). Keep the subject line under 72 characters and in the
imperative mood.

## Pull requests

- One logical change per PR.
- `ruff check`, `mypy` and `PPG_BACKEND=fake pytest` all clean.
- If it changes output, include a contact sheet or a before/after image.
- If it changes configuration, update `.env.example` in the same PR.
- If it changes the HTTP surface, update [docs/API.md](docs/API.md).

Bug reports should include the output of `ppg doctor`. Nine times out of ten it
contains the answer already.

By contributing you agree that your contribution is licensed under the MIT
licence, and that you will follow the
[Code of Conduct](CODE_OF_CONDUCT.md).
