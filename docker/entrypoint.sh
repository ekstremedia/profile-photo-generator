#!/usr/bin/env bash
#
# Container entrypoint. Reports the configuration it is about to run with,
# optionally fetches the model weights, then hands PID 1 to uvicorn.
#
# The banner exists because nearly every "it does not work" report for this
# project comes down to one of three things being different from what the
# operator assumed: the backend, the device, or where the data directory
# actually points. Printing them costs one line of log and answers all three.

set -euo pipefail

PPG_HOST="${PPG_HOST:-0.0.0.0}"
PPG_PORT="${PPG_PORT:-8000}"
PPG_BACKEND="${PPG_BACKEND:-diffusers}"
PPG_DEVICE="${PPG_DEVICE:-auto}"
PPG_DATA_DIR="${PPG_DATA_DIR:-/data}"
PPG_MODEL_ID="${PPG_MODEL_ID:-SG161222/RealVisXL_V5.0}"
HF_HOME="${HF_HOME:-${PPG_DATA_DIR}/hf-cache}"
export PPG_HOST PPG_PORT PPG_BACKEND PPG_DEVICE PPG_DATA_DIR PPG_MODEL_ID HF_HOME

log() {
    printf '[ppg] %s\n' "$*"
}

# Whether the NVIDIA devices were actually passed through. Checking the device
# nodes rather than importing torch keeps startup instant, and it distinguishes
# the two failure modes that look identical from the outside: a missing
# nvidia-container-toolkit on the host, versus a container that simply has no
# GPU reservation.
gpu_visible() {
    [ -e /dev/nvidiactl ] || [ -e /dev/dxg ]
}

# Mirrors the check in `ppg doctor`: huggingface_hub stores a repo as
# <cache>/hub/models--<owner>--<name>, but older layouts omit the hub/ level.
weights_present() {
    local slug="models--${PPG_MODEL_ID//\//--}"
    [ -d "${HF_HOME}/hub/${slug}" ] || [ -d "${HF_HOME}/${slug}" ]
}

log "backend=${PPG_BACKEND} device=${PPG_DEVICE} data=${PPG_DATA_DIR} hf_home=${HF_HOME}"
if [ "${PPG_BACKEND}" = "diffusers" ] && [ "${PPG_DEVICE}" != "cpu" ]; then
    if gpu_visible; then
        log "nvidia devices visible in the container"
    else
        log "no nvidia device nodes - generation will fall back to CPU (minutes per image)."
        log "  install nvidia-container-toolkit on the host: scripts/setup-host.sh"
    fi
fi

# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------
if [ "${PPG_BACKEND}" = "diffusers" ] && ! weights_present; then
    if [ "${PPG_AUTO_DOWNLOAD:-0}" = "1" ]; then
        log "weights for ${PPG_MODEL_ID} not in ${HF_HOME} - downloading (~7GB, once)"
        # Not fatal: exiting here under `restart: unless-stopped` would produce a
        # crash loop that re-attempts a 7GB download and buries the real cause.
        # The server starts regardless and reports the backend error on /readyz.
        if ! python /app/scripts/download-models.py; then
            log "download failed - starting anyway, see /readyz for the backend error"
        fi
    else
        log "weights for ${PPG_MODEL_ID} not found in ${HF_HOME}."
        log "  The server will start, but /readyz stays 'loading' until they arrive."
        log "  Fetch them with:  docker compose run --rm api python scripts/download-models.py"
        log "  Or set PPG_AUTO_DOWNLOAD=1 to have this container do it on start."
    fi
fi

# ---------------------------------------------------------------------------
# Optional advisory self-check
# ---------------------------------------------------------------------------
# Deliberately non-fatal: `ppg doctor` exits non-zero for things that are only
# warnings in a container (no Ollama on the host, for instance), and refusing to
# start over those would be worse than starting degraded.
if [ "${PPG_DOCTOR_ON_START:-0}" = "1" ]; then
    log "running 'ppg doctor' (advisory, failures do not stop startup)"
    ppg doctor || log "doctor reported problems - see the table above"
fi

# An explicit command (the dev profile's uvicorn --reload, a one-off `ppg
# batch`, a shell) replaces the server but still gets the banner and the
# weight check above.
if [ "$#" -gt 0 ]; then
    exec "$@"
fi

log "serving on http://${PPG_HOST}:${PPG_PORT}"
# One worker on purpose: the model occupies most of a consumer GPU's VRAM and a
# second worker would load a second copy of it. Concurrency is handled inside
# the process by ppg's job queue.
exec uvicorn ppg.api.app:app --host "$PPG_HOST" --port "$PPG_PORT"
