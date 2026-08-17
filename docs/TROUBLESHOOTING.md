# Troubleshooting

## Run `ppg doctor` first

```bash
ppg doctor                                 # local install
docker compose exec api ppg doctor         # in Docker
ppg doctor --url http://localhost:8000     # also probe a running server
```

It checks Python, the backend, torch and CUDA, diffusers, free disk, the model
cache and Ollama, and prints a table with a verdict per row:

```
                              ppg doctor
  check      result   detail
  python     pass     3.13.7
  ppg        pass     0.1.0
  backend    pass     diffusers (device=auto)
  cuda       pass     NVIDIA GeForce RTX 4070 SUPER, 12.9 GB VRAM
  torch      pass     2.13.0+cu130
  diffusers  pass     0.39.0
  disk       pass     412.6 GB free at /www/profile_photo_generator/data
  weights    pass     SG161222/RealVisXL_V5.0 cached (7.3 GB)
  ollama     pass     using llama3.2:3b (4 model(s) installed)
```

It exits non-zero when something failed, and it is what the bug report template
asks for. Almost every problem below has a corresponding row.

---

## All-black images

**Symptom.** Generation "succeeds" but every image is pure black, or the API
returns a 500 with:

> The model produced an all-black image. This is almost always the float16 VAE
> overflow: check that PPG_VAE_ID points at madebyollin/sdxl-vae-fp16-fix and
> that it downloaded correctly.

**Cause.** SDXL's stock VAE overflows in float16. The decode produces NaNs and
the image comes out black. This is the single most common "it doesn't work"
report for any SDXL project, and it is not a bug in your card or your driver.

**Fix.** Keep `PPG_VAE_ID` pointed at the fp16-safe replacement and make sure
it is actually present:

```bash
grep PPG_VAE_ID .env
# PPG_VAE_ID=madebyollin/sdxl-vae-fp16-fix

ls data/hf-cache/hub/models--madebyollin--sdxl-vae-fp16-fix/snapshots/*/
# config.json  diffusion_pytorch_model.safetensors

python scripts/download-models.py    # re-run if it is missing; it skips what is there
```

That repository has **no `.fp16.safetensors` variant**, so it is loaded with no
`variant` argument and cast afterwards. If you have edited the backend and
added `variant="fp16"` to the VAE load, it will fail to find the files.

The backend checks every render for this and raises rather than serving a black
PNG, so a black image never reaches the cache. If you do have black files on
disk from an older build, delete `data/outputs` and let them regenerate.

Running on CPU uses float32, where the stock VAE is fine — which is why the
problem appears only on GPU.

---

## CUDA out of memory when Ollama shares the GPU

**Symptom.** Generation fails with:

> CUDA out of memory. Set PPG_LOW_VRAM=true, or reduce PPG_WIDTH/PPG_HEIGHT to
> 768.

...on a card that has plenty of memory when nothing else is running. It often
works the first time and fails after you have used the chat model.

**Cause.** Ollama keeps a model resident for five minutes after a request by
default. An 8B model parked on 5.4 GB of a 12 GB card leaves too little for
SDXL's ~7 GB, and the diffusion load or the VAE decode fails. Because the
prompt composer *is* an Ollama call, this happens immediately before every
generation — the worst possible timing.

**Fix, already the default:**

```dotenv
PPG_OLLAMA_KEEP_ALIVE=0
```

`0` unloads the prompt model as soon as it has composed. That costs about a
second of model reload per request and buys back the VRAM that actually
matters. Only raise it (`5m`) if Ollama runs on a different machine or a
different card.

Check what is resident:

```bash
ollama ps          # models currently loaded and their size
nvidia-smi         # everything else holding VRAM, including your desktop session
```

Related settings:

- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set automatically in
  `ppg/__init__.py` before anything can create a CUDA context. It stops
  allocator fragmentation, which is what produces "tried to allocate 26 MiB,
  174 MiB free" on a card that is also driving a desktop. Do not unset it.
- `PPG_LOW_VRAM=true` enables `enable_model_cpu_offload()`, streaming
  submodules to the GPU on demand. Roughly twice as slow, but it fits
  comfortably on an 8 GB card. `ppg doctor` suggests it below 7.5 GB.
- `PPG_WIDTH=768` / `PPG_HEIGHT=768` is the next lever. Below that SDXL quality
  degrades quickly.

A last resort that always works: run Ollama on the CPU (`OLLAMA_NUM_GPU=0`) or
set `PPG_COMPOSER=template` and skip the LLM entirely. Template prompts are
good; that is the whole point of the fallback.

---

## Ollama unreachable from inside Docker

**Symptom.** The container logs:

> No Ollama model available at http://host.docker.internal:11434 - using
> template prompts. This is fine; Ollama only adds wording variety.

...or `X-PPG-Composer: template` on responses where you expected `llm`, or
`ollama_reachable: false` from `/readyz`.

This is not fatal. Prompts fall back to templates and generation continues.

**Two separate causes.**

1. **The address.** `127.0.0.1` inside a container is the container. Use
   `host.docker.internal`, which `compose.yaml` already sets:

   ```dotenv
   PPG_OLLAMA_BASE_URL=http://host.docker.internal:11434
   ```

   On Linux this needs `extra_hosts: ["host.docker.internal:host-gateway"]` in
   the compose service, which is also already there.

2. **The host binding.** Ollama listens on `127.0.0.1:11434` by default, which
   refuses connections from the Docker bridge. Bind it to all interfaces:

   ```bash
   sudo systemctl edit ollama
   # [Service]
   # Environment="OLLAMA_HOST=0.0.0.0:11434"
   sudo systemctl restart ollama
   ```

   `scripts/setup-host.sh` checks this for you. Note the security implication:
   `0.0.0.0` exposes port 11434 to your whole network. Firewall it, or bind to
   the Docker bridge address only. See [SECURITY.md](../SECURITY.md).

Verify from inside the container:

```bash
docker compose exec api python -c \
  "import httpx; print(httpx.get('http://host.docker.internal:11434/api/tags').status_code)"
```

If you would rather not touch the host daemon at all, run Ollama in a container
instead:

```bash
echo 'PPG_COMPOSE_OLLAMA_URL=http://ollama:11434' >> .env
docker compose --profile ollama up -d
```

That profile also pulls `llama3.2:3b` once and deliberately gives the Ollama
container no GPU reservation, so it cannot compete with SDXL for VRAM.

---

## `could not select device driver "nvidia"`

**Symptom.** `docker compose up` fails with something like:

> Error response from daemon: could not select device driver "nvidia" with
> capabilities: [[gpu]]

...or the container starts but its banner says:

> [ppg] no nvidia device nodes - generation will fall back to CPU (minutes per image).

**Cause.** The NVIDIA Container Toolkit is not installed or not configured, so
Docker cannot pass the GPU through. The host driver being fine is not enough.

**Fix.** `scripts/setup-host.sh` does this for you on Debian-like systems, one
prompt per step (`--yes` to accept them all). By hand:

```bash
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
  | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Then confirm:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

`setup-host.sh` verifies the result by running `nvidia-smi` inside a small CUDA
container, because the toolkit being installed and Docker knowing about it are
two different things.

---

## Disk full

**Symptom.** `download-models.py` refuses to start:

> Not enough free space: need ~8.74 GB including working room, have 3.10 GB.
> Tip: `docker builder prune` often reclaims a great deal.

...or a build fails with `no space left on device`, or SQLite starts throwing
`disk I/O error`.

**Fix, in order of how much they usually reclaim:**

```bash
docker builder prune -af      # build cache; frequently tens of gigabytes
docker image prune -af        # dangling and unused images
du -sh data/*                 # outputs/, hf-cache/ and ppg.db
```

An unfiltered clone of the RealVisXL repository pulls 34.6 GB because it ships
fp32 duplicates; `scripts/download-models.py` uses include patterns to fetch
6.94 GB instead. If `data/hf-cache` is far larger than about 7.3 GB, something
fetched the whole repository — see [MODELS.md](MODELS.md#the-fp32-duplicate-download-trap).

Generated avatars are cheap individually (four sizes in two formats) but they
add up over tens of thousands. They are safe to delete: anything still
referenced will be regenerated identically on the next request.

---

## Weights re-download on every restart

**Symptom.** A 7 GB download starts again after a container rebuild, or after
running `ppg` from a different directory.

**Cause.** `HF_HOME` is not set, so `huggingface_hub` falls back to
`~/.cache/huggingface` — a different path in a fresh container, and a different
path per user on the host.

**Fix.** The application handles this itself: `Settings.ensure_dirs()` exports
`HF_HOME` from `PPG_DATA_DIR/hf-cache` if nothing else set it, and `PPG_DATA_DIR`
is resolved to an absolute path so the working directory cannot change it. What
you need to make sure of:

- In Docker, the cache must be on the mount, not the container filesystem.
  `compose.yaml` bind-mounts `./data:/data` and sets `HF_HOME=/data/hf-cache`
  in `environment:` — which takes precedence over `.env`, where `HF_HOME` is
  empty and would otherwise send 7 GB to `~/.cache/huggingface` inside a
  throwaway container. Do not delete `./data` on the host.
- If you set `HF_HOME` yourself, set it to the same path everywhere — the shell,
  `.env`, and the container.
- `ppg doctor` prints the resolved cache and its size in the `weights` row.
  Check that it points where you expect.

The container does not download weights unless you ask it to. Either run the
script once:

```bash
docker compose run --rm api python scripts/download-models.py
```

or set `PPG_AUTO_DOWNLOAD=1`, which makes the entrypoint fetch them on start.
A missing checkpoint is not fatal — the server starts and `/readyz` stays
`loading` — because a crash loop that re-attempts a 7 GB download would bury
the real cause.

---

## Generation takes minutes, not seconds

Almost always CPU fallback. Check:

```bash
curl -s http://localhost:8000/readyz | jq .device      # "cpu" or "cuda"
ppg doctor                                             # the cuda row
```

Causes, in order of likelihood:

1. **torch has no CUDA support.** A plain `pip install torch` may install the
   CPU build. Install from the CUDA wheel index that matches your driver.
2. **The container cannot see the GPU** — see the nvidia-container-toolkit
   section above.
3. **`PPG_DEVICE=cpu`** is set explicitly somewhere.
4. **`PPG_LOW_VRAM=true` on a card that does not need it.** Offload is roughly
   twice as slow; turn it off above 10 GB.
5. **`PPG_COMPILE=true` on a first run.** `torch.compile` adds minutes to the
   first image and about 15% to later ones. It is off by default for that
   reason.

`ppg warmup` times a load and a single render, and warns when a CUDA render
takes more than 10 seconds:

```bash
ppg warmup
#   loaded in 3.3s
#   rendered 1024x1024 in 8.1s
```

If the device says `cuda` and it is still slow, check `nvidia-smi` for another
process (Ollama, a game, a browser with hardware acceleration) competing for
the card.

---

## `Token indices sequence length is longer than ... (77)`

**Symptom.** A warning from transformers on every generation:

> The following part of your input was truncated because CLIP can only handle
> sequences up to 77 tokens

**Usually harmless, occasionally a real symptom.** Prompts are built to a word
budget (`PROMPT_WORD_BUDGET` in `prompt/templates.py`) that normally keeps them
under the limit, so this warning should be rare. When it does appear it is
because a long `prompt_extra`, or a long free-form `profession`, pushed the
*required* part of the prompt over on its own — and required parts are never
trimmed, precisely so that what you asked for is not what gets thrown away.

What gets discarded is always the tail. The ordering is: framing, identity and
age cues, your `prompt_extra`, your pinned attributes, skin tone, realism cues,
then filler. So a warning means the filler is gone and possibly the last of the
realism cues; it does not mean your request was dropped.

If you see it constantly, shorten `prompt_extra`. If you want a bigger budget,
raise `PROMPT_WORD_BUDGET`, accept the truncation, and bump `PIPELINE_VERSION`.

---

## Other things that come up

**`503 Queue is full (128 images pending, limit 128)`** — the queue is bounded
by `PPG_MAX_QUEUE`. Raise it, or slow the client down. There is one worker;
queueing more does not make it faster.

**`404 No job <id>. Jobs are in-memory and do not survive a restart`** — exactly
what it says. The avatars that job produced are still in `/v1/avatars`.

**`503 Service is still starting up.`** — the request arrived before the
lifespan finished. Gate on `/readyz`, not `/healthz`.

**`/readyz` says `"status": "error"`** — the background model load failed. The
reason is in the server log; it is usually missing weights, a bad
`PPG_MODEL_ID`, or missing GPU extras (`pip install -e ".[gpu]"`).

**`422` on a request you think is reasonable** — the safety filter matched a
word in `prompt_extra`. The `detail` field names the blocked terms. Age cannot
be set through free text at all; use the `age` parameter.

**Images exist on disk but the API 404s** — the SQLite index and
`data/outputs` have diverged. The service handles the common direction (files
missing, row present) by regenerating. For the other direction, the file is
orphaned and harmless; delete it or leave it.

**Nothing works and the error is confusing** — try
`PPG_BACKEND=fake ppg serve`. If that works, the problem is the GPU, the
weights or the driver, and not the API, the sampler or your client.
