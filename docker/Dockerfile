# syntax=docker/dockerfile:1.7
#
# Profile Photo Generator - CUDA-capable image, built on plain python:3.14-slim.
#
# Why not an `nvidia/cuda:*` base image? The PyTorch wheel published on PyPI for
# Linux already ships the CUDA runtime, cuDNN and cuBLAS it needs, bundled as
# `nvidia-*` pip packages. Layering that on top of a CUDA base image means two
# copies of the same libraries and roughly 3GB of dead weight; the container
# only ever loads the ones inside the wheel. The host still needs the NVIDIA
# driver plus `nvidia-container-toolkit` - see scripts/setup-host.sh.
#
# Python 3.12 rather than 3.13: the project runs fine on 3.13, but a few of the
# GPU dependencies still lag a release behind on 3.13 wheels, and a build that
# falls back to compiling from source here would take an hour.

# ---------------------------------------------------------------------------
# Stage 1 - build the virtualenv
# ---------------------------------------------------------------------------
FROM python:3.14-slim AS builder

# uv resolves and installs this dependency tree in a fraction of the time pip
# takes, which matters when torch alone is a 900MB download.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /usr/local/bin/uv

ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never

WORKDIR /src

# A self-contained venv is the unit copied into the runtime stage: nothing from
# this stage's system Python or uv cache follows it.
RUN uv venv /opt/venv
ENV VIRTUAL_ENV=/opt/venv \
    PATH="/opt/venv/bin:$PATH"

# Dependencies are installed against a stub package so that editing src/ does
# not invalidate the layer holding several GB of torch and CUDA wheels. The
# metadata hatchling needs (version, readme, license) all lives in pyproject.
COPY pyproject.toml README.md LICENSE ./
RUN mkdir -p src/ppg \
    && touch src/ppg/__init__.py \
    && uv pip install --no-cache ".[gpu]"

# --no-cache above: uv's wheel cache for this tree is ~3GB and would sit in the
# build cache indefinitely. Slower rebuilds, but this project is aimed at
# workstations whose disk is already mostly model weights.

# The real code. --no-deps because the resolution above is authoritative;
# --reinstall because the stub was installed under the same version number.
COPY src ./src
RUN uv pip install --no-cache --no-deps --reinstall .

# ---------------------------------------------------------------------------
# Stage 2 - runtime
# ---------------------------------------------------------------------------
FROM python:3.14-slim AS runtime

# libgomp1: torch links against OpenMP for its CPU kernels, which are used for
#   image pre/post-processing even when generation runs on the GPU.
# ca-certificates: huggingface_hub downloads over TLS when weights are missing.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgomp1 \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# uid 1000 matches the default first user on a Debian/Ubuntu workstation, so
# files written into the bind-mounted ./data stay editable on the host without
# a chown dance.
RUN groupadd --gid 1000 ppg \
    && useradd --uid 1000 --gid 1000 --create-home --shell /bin/bash ppg

# Left owned by root: the application only ever reads it, and a non-root process
# that cannot rewrite its own dependencies is one less thing to worry about.
COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
COPY --chown=1000:1000 src /app/src
COPY --chown=1000:1000 static /app/static
COPY --chown=1000:1000 scripts /app/scripts
# The licence travels with the image, not only with the repository.
COPY --chown=1000:1000 LICENSE README.md /app/
COPY --chmod=0755 docker/entrypoint.sh /usr/local/bin/entrypoint.sh

ENV PATH="/opt/venv/bin:$PATH" \
    VIRTUAL_ENV=/opt/venv \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# /app/src ahead of site-packages so the `dev` compose profile can bind-mount a
# live checkout over it and have the change take effect. Both copies come from
# the same COPY above, so this changes nothing for the other profiles.
ENV PYTHONPATH=/app/src

# Weights and outputs both live under the single bind-mounted /data, so a
# rebuild never re-downloads the 7GB checkpoint.
ENV HF_HOME=/data/hf-cache \
    PPG_DATA_DIR=/data

# The gallery assets are not part of the wheel; app.py checks this variable
# before falling back to guessing a location.
ENV PPG_STATIC_DIR=/app/static

# Cuts allocator fragmentation, which is what actually produces "tried to
# allocate 1GB, 1GB free" on a card that is also driving a desktop. Set here as
# well as in the backend because it must precede CUDA context creation.
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Created here as well as mounted, so `docker run` without a volume still works.
RUN mkdir -p /data && chown 1000:1000 /data /app

USER 1000:1000
EXPOSE 8000

# No VOLUME declaration on purpose: it would make a plain `docker run` create an
# anonymous volume and quietly re-download 7GB of weights into it. compose.yaml
# bind-mounts ./data instead.

# /healthz is liveness only - it answers as soon as uvicorn is up. Readiness
# (model resident) is /readyz, deliberately not used here: a container that is
# still loading a 7GB checkpoint is healthy, just not ready. The long start
# period covers an optional first-run weight download.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD python -c "import os,urllib.request;urllib.request.urlopen('http://127.0.0.1:'+os.environ.get('PPG_PORT','8000')+'/healthz',timeout=4)" || exit 1

ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
