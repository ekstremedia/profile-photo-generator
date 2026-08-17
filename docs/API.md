# HTTP API

Base URL in these examples is `http://localhost:8000`. A running instance
serves interactive OpenAPI documentation at
[`/docs`](http://localhost:8000/docs) and the raw schema at
[`/openapi.json`](http://localhost:8000/openapi.json). The gallery UI is
mounted at `/`.

## Endpoints

| Method | Path | Auth | Purpose |
| --- | --- | --- | --- |
| POST | `/v1/avatars` | yes | generate one avatar |
| POST | `/v1/avatars/batch` | yes | queue many |
| GET | `/v1/avatars/by-seed/{key}` | yes | deterministic avatar image for any string |
| GET | `/v1/avatars` | yes | recent avatars |
| GET | `/v1/avatars/{id}` | yes | metadata for one avatar |
| GET | `/v1/avatars/{id}/image` | yes | the image file |
| DELETE | `/v1/avatars/{id}` | yes | delete row and files |
| DELETE | `/v1/avatars` | yes | delete every avatar, requires `?confirm=true` |
| GET | `/v1/jobs/{id}` | yes | job status |
| GET | `/v1/jobs/{id}/results` | yes | results of a finished job |
| POST | `/v1/images/generations` | yes | OpenAI-compatible shim |
| GET | `/v1/options` | no | every attribute value this instance accepts |
| GET | `/healthz` | no | liveness |
| GET | `/readyz` | no | readiness, including model load state |
| GET | `/metrics` | no | Prometheus exposition |

## Authentication

Off by default, so `docker compose up` works with no configuration. Set
`PPG_API_KEY` and clients then send a bearer token:

```bash
curl -H "Authorization: Bearer $PPG_API_KEY" http://localhost:8000/v1/avatars
```

A wrong or missing token gives `401` with `WWW-Authenticate: Bearer`.

The check is attached to the avatars and OpenAI-compat routers only.
`/v1/options`, `/healthz`, `/readyz` and `/metrics` stay open even when a key
is set — they expose no images and are useful to monitoring. If that is not
acceptable in your deployment, block them at the reverse proxy.

CORS origins come from `PPG_CORS_ORIGINS` (comma-separated, `*` by default);
credentials are not allowed.

## POST /v1/avatars

Every field is optional. An empty body gives a fully random face. Unknown
fields are rejected (`extra="forbid"`), which catches typos rather than
silently ignoring them.

| Field | Type | Notes |
| --- | --- | --- |
| `sex`, `age_range`, `ethnicity`, `skin_tone`, `profession`, `hair`, `facial_hair`, `glasses`, `expression`, `clothing`, `background`, `lighting`, `camera` | string | pin an axis; anything unset is sampled from the seed. Values come from `/v1/options`, but unknown values pass straight through to the prompt |
| `age` | int | exact age; overrides `age_range`, clamped to `PPG_MIN_AGE`–`PPG_MAX_AGE` |
| `seed` | int or string | a string is hashed to a seed with blake2b and echoed back as `seed_key`. Omit for a random face |
| `size` | int | validated against the configured ladder; the response returns URLs for every size, so this is a client-side convenience |
| `fast` | bool | uses `PPG_FAST_STEPS` (15) instead of `PPG_STEPS` (30) |
| `prompt_extra` | string | appended to the generated prompt; passes through the safety filter, max 500 chars |
| `negative_extra` | string | appended to the negative prompt; length-checked only, since blocking words there would be backwards |

Query parameter `wait` (seconds, 0–600) controls blocking. The default is
`PPG_DEFAULT_WAIT` (60). `wait=0` returns a job immediately.

```bash
curl -s -X POST 'http://localhost:8000/v1/avatars' \
  -H 'content-type: application/json' \
  -d '{"sex":"female","age":41,"profession":"marine_biologist","seed":"demo-1"}'
```

```json
{
  "id": "cfbc85922772aad0efd5ee528c288cbd",
  "hash": "cfbc85922772aad0efd5ee528c288cbd",
  "seed": 2743158823318417408,
  "seed_key": "demo-1",
  "attributes": {
    "sex": "female",
    "ethnicity": "southern_european",
    "skin_tone": "fitzpatrick_iii",
    "profession": "marine_biologist",
    "hair": "shoulder_length_wavy",
    "facial_hair": "clean_shaven",
    "glasses": "none",
    "expression": "slight_smile",
    "clothing": "profession_appropriate",
    "background": "blurred_greenery",
    "lighting": "soft_window",
    "camera": "85mm",
    "age": 41,
    "age_range": "35-44"
  },
  "persona": {
    "name": "Chiara Ferrero",
    "age": 41,
    "occupation": "marine biologist",
    "city": "Genoa",
    "country": null,
    "bio": "Marine biologist in Genoa."
  },
  "prompt": "head and shoulders portrait, centred, looking at the camera, 41 year old Southern European female, working as a marine biologist, light olive skin, ...",
  "negative_prompt": "cartoon, anime, illustration, painting, drawing, sketch, 3d render, cgi, ...",
  "model": "SG161222/RealVisXL_V5.0",
  "backend": "diffusers",
  "composer": "template",
  "sizes": [1024, 512, 256, 128],
  "urls": {
    "1024": "/v1/avatars/cfbc85922772aad0efd5ee528c288cbd/image?size=1024",
    "512": "/v1/avatars/cfbc85922772aad0efd5ee528c288cbd/image?size=512",
    "256": "/v1/avatars/cfbc85922772aad0efd5ee528c288cbd/image?size=256",
    "128": "/v1/avatars/cfbc85922772aad0efd5ee528c288cbd/image?size=128",
    "default": "/v1/avatars/cfbc85922772aad0efd5ee528c288cbd/image"
  },
  "created_at": "2026-08-17T21:26:41.882431+00:00",
  "cached": false,
  "duration_ms": 8214
}
```

Two response headers come with a `200`:

- `X-PPG-Cache: hit | miss` — whether the image was already on disk.
- `X-PPG-Composer: llm | template` — which composer wrote the prompt. Useful
  for spotting that Ollama quietly stopped being reachable.

If the wait expires first, the status becomes `202` and the body is a `JobInfo`
instead:

```json
{
  "id": "3f9a1c2e4b7d8a05",
  "status": "running",
  "kind": "single",
  "total": 1,
  "completed": 0,
  "position": null,
  "eta_seconds": 8.4,
  "avatar_ids": [],
  "error": null,
  "created_at": "2026-08-17T21:31:02.104553+00:00",
  "finished_at": null
}
```

## POST /v1/avatars/batch

```bash
curl -s -X POST 'http://localhost:8000/v1/avatars/batch' \
  -H 'content-type: application/json' \
  -d '{"n":50,"diversity":"even","seed":"seed-users-2026"}'
```

Returns `202` with a `JobInfo`. Fields: `n` (1–500), `diversity`
(`even` | `random`), `seed`, and `overrides` (a full `AvatarRequest` applied to
every avatar in the batch). The `wait` query parameter (0–3600) blocks if you
want it to.

`diversity: "even"` walks a shuffled cross product of sex × age bucket ×
ancestry and re-rolls combinations already present in the database, so a batch
of 50 looks like 50 different people and a second batch does not repeat the
first. `"random"` samples each avatar independently, which clusters on whatever
the vocabulary weights favour.

Poll it:

```bash
curl -s http://localhost:8000/v1/jobs/3f9a1c2e4b7d8a05
curl -s http://localhost:8000/v1/jobs/3f9a1c2e4b7d8a05/results | jq '.[].id'
```

`eta_seconds` is derived from a rolling mean of the last 20 non-cached
generations. `position` is the number of queued jobs ahead of this one, `0`
meaning next up, and `null` once the job is running or finished.

Jobs are in-memory. They do not survive a restart; the avatars they produced
do, and stay available through `/v1/avatars`.

## GET /v1/avatars/by-seed/{key}

Returns the image file directly, generating it on first request.

```bash
curl -o ada.webp \
  "http://localhost:8000/v1/avatars/by-seed/$(printf 'ada@example.com' | md5sum | cut -d' ' -f1)?size=256"
```

Query parameters: `size` (nearest available size that is at least as large is
used) and `format` (`webp`, the default, or `png`).

The key is a path parameter that accepts slashes, so `by-seed/users/42` works.
The first call blocks for up to `PPG_DEFAULT_WAIT` seconds; if generation has
not finished by then you get `503` with `Retry-After: 10` rather than a
half-baked image. Every later call is a static file read.

One edge case: a purely numeric key (`by-seed/12345`) is treated as an integer
seed rather than a hashed key, so it is not indexed under `seed_key`. The face
is still identical every time — the content hash makes the second request a
cache hit — but the lookup path is slightly longer. Hex digests from `md5` or
`sha1` are effectively never all digits, so this rarely matters in practice.

## GET /v1/avatars

```bash
curl -s 'http://localhost:8000/v1/avatars?limit=12&offset=0' | jq '.[].id'
```

Newest first. `limit` 1–200 (default 60), `offset` ≥ 0. Returns full
`AvatarResult` objects with `cached: true`.

## GET /v1/avatars/{id} and /v1/avatars/{id}/image

`/v1/avatars/{id}` returns the metadata record. `/v1/avatars/{id}/image`
returns the file:

```bash
curl -sI 'http://localhost:8000/v1/avatars/cfbc85922772aad0efd5ee528c288cbd/image?size=512'
```

```
HTTP/1.1 200 OK
content-type: image/webp
cache-control: public, max-age=31536000, immutable
etag: "cfbc85922772aad0efd5ee528c288cbd-512-webp"
x-ppg-size: 512
```

`size` resolves to the smallest available size that is at least as large as the
request, so `?size=200` serves the 256 px file rather than a 404. Omitting
`size` serves the largest. `format` is `webp` or `png`.

The files are content-addressed and never change, hence the immutable
cache header.

## DELETE /v1/avatars/{id}

`204` on success, `404` if the id is unknown. Removes the database row and the
whole directory of variants. The avatar will be regenerated identically if the
same request arrives again.

## DELETE /v1/avatars

Deletes every avatar and every image file.

```bash
curl -s -X DELETE 'http://localhost:8000/v1/avatars?confirm=true'
```

```json
{ "deleted": 51 }
```

Without `?confirm=true` this returns `400` and does nothing, so a mistyped
`curl -X DELETE .../v1/avatars` cannot empty the library by accident.

Model weights are untouched. Cached *prompts* are also kept on purpose: after a
clear, `GET /v1/avatars/by-seed/{key}` regenerates the identical face rather
than a new one, so clearing the gallery does not silently re-face every user in
an application that relies on it.

The CLI equivalent is `ppg clear` (`--yes` to skip the prompt), and the gallery
UI exposes it as a two-step "Clear all" button.

## GET /v1/options

```bash
curl -s http://localhost:8000/v1/options | jq '{sizes, backend, model, composer, axes: (.axes|keys)}'
```

```json
{
  "sizes": [1024, 512, 256, 128],
  "backend": "diffusers",
  "model": "SG161222/RealVisXL_V5.0",
  "composer": "auto",
  "axes": ["sex","age_range","ethnicity","skin_tone","profession","hair","facial_hair","glasses","expression","clothing","background","lighting","camera"]
}
```

Each axis is a list of `{value, weight, label}`. This reflects the running
instance's own `vocab.yaml`, which the operator may have edited, so it is the
right thing for a client-side validator to fetch once and cache rather than
hard-coding a list.

## Health and metrics

```bash
curl -s http://localhost:8000/healthz
# {"status":"ok","version":"0.1.0"}

curl -s http://localhost:8000/readyz
```

```json
{
  "status": "ready",
  "version": "0.1.0",
  "backend": "diffusers",
  "device": "cuda",
  "model_loaded": true,
  "ollama_reachable": true,
  "queue_depth": 0
}
```

`status` is `loading` until the checkpoint is resident, `error` if the load
failed, `ready` afterwards. `ollama_reachable` is `null` when
`PPG_COMPOSER=template`. `/healthz` is 200 for the whole process lifetime; gate
traffic on `/readyz`.

`/metrics` is a hand-rolled Prometheus exposition with four numbers:
`ppg_avatars_total`, `ppg_queue_depth`, `ppg_model_loaded` and
`ppg_generation_seconds` (rolling mean).

## POST /v1/images/generations (OpenAI-compatible)

For tooling that already speaks the OpenAI images API — `openai-php` in a
Laravel app, the `openai` Python package, and so on.

```bash
curl -s -X POST http://localhost:8000/v1/images/generations \
  -H 'content-type: application/json' \
  -d '{"prompt":"wearing a knitted sweater, soft window light","n":1,"size":"512x512","user":"ada@example.com"}' \
  | jq '.data[0] | {revised_prompt, bytes: (.b64_json|length)}'
```

Accepted fields: `prompt` (1–500 chars, required), `model`, `n` (1–10), `size`
(`WxH`, the width is matched to the nearest configured size), `response_format`
(`b64_json` default, or `url`), `user`, `quality`, `style`.

**The mapping is lossy, deliberately.** This service generates from
*attributes*, not from free text:

- `prompt` is appended as extra styling detail on top of a randomly sampled
  person. It does not describe the person from scratch, and it still goes
  through the safety filter.
- `prompt` (or `user`, when given) is hashed into the seed, so the same input
  keeps returning the same face.
- `model`, `quality` and `style` are accepted and ignored — the model is
  whatever the instance is configured with.
- `response_format: "url"` returns a relative path on this service, not a
  signed remote URL.
- `b64_json` is re-encoded from the stored PNG master, because OpenAI clients
  expect a PNG rather than the WebP served over HTTP.

If you want real control, use `POST /v1/avatars`. That is the native API and it
is better.

## Status codes

| Code | When |
| --- | --- |
| 200 | avatar generated or served |
| 202 | the wait expired, or a batch was queued; body is a `JobInfo` |
| 204 | avatar deleted |
| 400 | empty `by-seed` key, or an unparseable `size` on the OpenAI shim |
| 401 | `PPG_API_KEY` is set and the bearer token was missing or wrong |
| 404 | unknown avatar id, unknown job id, or a missing variant file |
| 422 | refused by the safety filter, or a schema validation failure |
| 500 | generation failed for a reason the backend could not classify |
| 503 | queue full (`PPG_MAX_QUEUE`), `by-seed` still rendering (`Retry-After: 10`), or the app has not finished starting |
| 504 | the OpenAI shim timed out waiting for a large `n` |

A `422` from the safety filter has a plain-text explanation:

```bash
curl -s -X POST http://localhost:8000/v1/avatars \
  -H 'content-type: application/json' \
  -d '{"prompt_extra":"lookalike of a famous actor"}'
```

```json
{
  "detail": "prompt_extra rejected: this service only generates people who do not exist. It will not imitate a real or identifiable person. Blocked term(s): famous, lookalike."
}
```

Age is a separate refusal: `{"age": 14}` with the default settings gives
`422` explaining that the minimum is 18 and that `PPG_ALLOW_MINORS` exists.
Setting an age above `PPG_MAX_AGE` clamps silently rather than failing.

Note that a failed generation *inside a batch* does not fail the request: the
job collects the error strings in `error` and returns whichever avatars did
succeed.
