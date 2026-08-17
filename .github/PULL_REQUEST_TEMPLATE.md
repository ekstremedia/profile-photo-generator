## What this changes

<!-- One or two sentences. Link an issue if there is one. -->

## Why

<!-- What was wrong, or what this makes possible. -->

## Checks

- [ ] Ran `pytest` with the fake backend (`PPG_BACKEND=fake pytest`) - no GPU or
      model weights required
- [ ] `ruff check .` and `ruff format .` are clean
- [ ] Documented anything a user would notice (README, `docs/`, `.env.example`)

## Notes

Additions to `src/ppg/attributes/vocab.yaml` and `src/ppg/prompt/names.yaml` are
very welcome and do not need an issue first - a wider vocabulary is the whole
point. If you changed sampling or the prompt text, say whether previously
generated avatars would change, so `PIPELINE_VERSION` can be bumped.
