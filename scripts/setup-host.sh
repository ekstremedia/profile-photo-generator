#!/usr/bin/env bash
#
# One-off host preparation for running Profile Photo Generator in Docker.
#
# Three separate things, each asked about individually and each safe to re-run:
#
#   1. reclaim disk from the Docker build cache
#   2. install nvidia-container-toolkit, without which a container cannot see
#      the GPU no matter what compose.yaml says
#   3. expose a host Ollama to containers (optional, and it has a security cost)
#
# Nothing is changed without a y/n prompt. --yes answers y to everything, which
# is only reasonable once you have read what each step does.
#
# Usage: scripts/setup-host.sh [--yes]

set -euo pipefail

ASSUME_YES=0

NVIDIA_GPGKEY_URL="https://nvidia.github.io/libnvidia-container/gpgkey"
NVIDIA_LIST_URL="https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list"
NVIDIA_KEYRING="/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg"
NVIDIA_LIST="/etc/apt/sources.list.d/nvidia-container-toolkit.list"
# Any small CUDA image works; this one is a documented, long-lived tag and the
# base variant is about 250MB rather than several GB.
CUDA_PROBE_IMAGE="nvidia/cuda:12.4.1-base-ubuntu22.04"
OLLAMA_DROPIN_DIR="/etc/systemd/system/ollama.service.d"
OLLAMA_DROPIN="${OLLAMA_DROPIN_DIR}/override.conf"

# ---------------------------------------------------------------------------
# Output and prompting
# ---------------------------------------------------------------------------

heading() { printf '\n=== %s ===\n\n' "$*"; }
info() { printf '  %s\n' "$*"; }
blank() { printf '\n'; }
warn() { printf '  WARNING: %s\n' "$*" >&2; }

# Prompts on /dev/tty rather than stdin so the script still behaves when its
# stdin is a pipe. A missing tty is treated as "no", never as "yes".
confirm() {
    if [ "$ASSUME_YES" -eq 1 ]; then
        info "[--yes] $1"
        return 0
    fi
    if [ ! -e /dev/tty ]; then
        info "no terminal available, skipping: $1"
        return 1
    fi
    local reply=""
    printf '  %s [y/N] ' "$1" > /dev/tty
    read -r reply < /dev/tty || reply=""
    case "$reply" in
        y | Y | yes | YES | Yes) return 0 ;;
        *)
            info "skipped."
            return 1
            ;;
    esac
}

# Echoes what it is about to do, so the transcript of a run doubles as the list
# of commands to repeat by hand elsewhere.
run() {
    printf '  $ %s\n' "$*"
    "$@"
}

usage() {
    printf 'Usage: %s [--yes]\n\n' "$0"
    printf '  --yes, -y   answer yes to every prompt\n'
    printf '  --help, -h  this message\n'
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        -y | --yes) ASSUME_YES=1 ;;
        -h | --help)
            usage
            exit 0
            ;;
        *)
            printf 'Unknown option: %s\n\n' "$1" >&2
            usage >&2
            exit 2
            ;;
    esac
    shift
done

# An array rather than a string, so the empty (already root) case expands to
# nothing at all instead of an empty argument.
SUDO=()
if [ "$(id -u)" -ne 0 ]; then
    if ! command -v sudo > /dev/null 2>&1; then
        warn "not root and sudo is not installed; the mutating steps will fail."
    fi
    SUDO=(sudo)
fi

OS_ID=""
OS_PRETTY=""
OS_LIKE=""
if [ -r /etc/os-release ]; then
    # shellcheck disable=SC1091  # path is fixed and only read for ID fields
    . /etc/os-release
    OS_ID="${ID:-}"
    OS_LIKE="${ID_LIKE:-}"
    OS_PRETTY="${PRETTY_NAME:-$OS_ID}"
fi

is_debian_like() {
    case " ${OS_ID} ${OS_LIKE} " in
        *" debian "* | *" ubuntu "*) return 0 ;;
        *) return 1 ;;
    esac
}

heading "Profile Photo Generator - host setup"
info "host: ${OS_PRETTY:-unknown}"
info "This script asks before every change. Answer n to skip a step."

if ! command -v docker > /dev/null 2>&1; then
    warn "docker is not installed. Install it first:"
    info "  https://docs.docker.com/engine/install/debian/"
    exit 1
fi
if ! docker info > /dev/null 2>&1; then
    warn "cannot talk to the Docker daemon. Either it is not running, or this"
    info "  user is not in the 'docker' group (log out and back in after adding it)."
fi

# ---------------------------------------------------------------------------
# 1. Disk
# ---------------------------------------------------------------------------

heading "1/3  Docker disk usage"

docker system df || warn "could not read docker disk usage."
blank
info "The line to watch is BUILD CACHE. On a machine that has built a few image"
info "variants it is routinely tens of GB, and it is entirely disposable: the"
info "only cost of clearing it is that the next build starts from scratch."
info "Worth knowing before this project asks for another ~10GB for the image"
info "and ~7GB for the model weights."
blank

if confirm "Run 'docker builder prune -f' now?"; then
    run docker builder prune -f
fi

# ---------------------------------------------------------------------------
# 2. GPU access for containers
# ---------------------------------------------------------------------------

heading "2/3  GPU access for containers"

if ! command -v nvidia-smi > /dev/null 2>&1; then
    warn "no nvidia-smi on this host, so the NVIDIA driver itself is missing."
    info "  The container toolkit is useless without it. On Debian, enable the"
    info "  non-free-firmware component and install 'nvidia-driver', then reboot."
    blank
fi

toolkit_present=0
runtime_registered=0
if command -v nvidia-ctk > /dev/null 2>&1; then
    toolkit_present=1
fi
# The toolkit being installed is not the same as Docker knowing about it: the
# runtime has to be written into /etc/docker/daemon.json and the daemon
# restarted, which is what `nvidia-ctk runtime configure` does.
if docker info 2> /dev/null | grep -qi nvidia; then
    runtime_registered=1
fi

install_toolkit() {
    if ! command -v curl > /dev/null 2>&1; then
        warn "curl is not installed; cannot fetch the repository definition."
        info "  apt-get install -y curl gnupg   then re-run this script."
        return 1
    fi

    run "${SUDO[@]}" install -m 0755 -d /usr/share/keyrings

    # --yes so that re-running overwrites the existing keyring rather than
    # failing on "file exists", which is what makes this step idempotent.
    printf '  $ curl -fsSL %s | gpg --dearmor -o %s\n' "$NVIDIA_GPGKEY_URL" "$NVIDIA_KEYRING"
    curl -fsSL "$NVIDIA_GPGKEY_URL" | "${SUDO[@]}" gpg --dearmor --yes -o "$NVIDIA_KEYRING"
    run "${SUDO[@]}" chmod 0644 "$NVIDIA_KEYRING"

    # The list file NVIDIA publishes carries no signed-by option, and apt on
    # Debian 12+ rejects a repository whose key is not pinned to it. The sed is
    # the vendor's own documented fix, not a workaround.
    printf '  $ curl -fsSL %s | sed (add signed-by) | tee %s\n' "$NVIDIA_LIST_URL" "$NVIDIA_LIST"
    curl -fsSL "$NVIDIA_LIST_URL" \
        | sed "s#deb https://#deb [signed-by=${NVIDIA_KEYRING}] https://#g" \
        | "${SUDO[@]}" tee "$NVIDIA_LIST" > /dev/null

    run "${SUDO[@]}" apt-get update
    run "${SUDO[@]}" apt-get install -y nvidia-container-toolkit
    run "${SUDO[@]}" nvidia-ctk runtime configure --runtime=docker
    run "${SUDO[@]}" systemctl restart docker
}

if [ "$toolkit_present" -eq 1 ] && [ "$runtime_registered" -eq 1 ]; then
    info "nvidia-container-toolkit is installed and registered with Docker."
elif ! is_debian_like; then
    warn "this script only knows the Debian/Ubuntu installation route."
    info "  Detected: ${OS_PRETTY:-unknown}. Follow the vendor instructions:"
    info "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    info "  Then run: nvidia-ctk runtime configure --runtime=docker && systemctl restart docker"
    info "  and re-run this script to verify."
else
    if [ "$toolkit_present" -eq 1 ]; then
        info "nvidia-container-toolkit is installed but Docker has no nvidia runtime."
    else
        info "nvidia-container-toolkit is not installed. Without it, 'docker run --gpus all'"
        info "fails and the api container silently falls back to CPU generation."
    fi
    blank
    info "About to:"
    info "  - write NVIDIA's signing key to ${NVIDIA_KEYRING}"
    info "  - write ${NVIDIA_LIST}"
    info "  - apt-get update && apt-get install -y nvidia-container-toolkit"
    info "  - nvidia-ctk runtime configure --runtime=docker"
    info "  - systemctl restart docker   (this stops every running container)"
    blank
    if confirm "Install nvidia-container-toolkit?"; then
        install_toolkit
    fi
fi

blank
info "Verification runs a throwaway CUDA container and asks it for nvidia-smi."
info "It pulls ${CUDA_PROBE_IMAGE} (~250MB) the first time."
if confirm "Run the GPU verification container?"; then
    if run docker run --rm --gpus all "$CUDA_PROBE_IMAGE" nvidia-smi; then
        blank
        info "The GPU is visible from inside containers."
    else
        blank
        warn "the probe container could not see the GPU."
        info "  Check in this order:"
        info "    nvidia-smi                     driver present on the host?"
        info "    docker info | grep -i nvidia   runtime registered?"
        info "    sudo journalctl -u docker -n50 daemon complaints?"
    fi
fi

# ---------------------------------------------------------------------------
# 3. Host Ollama
# ---------------------------------------------------------------------------

heading "3/3  Reaching a host Ollama from containers (optional)"

info "Ollama only improves prompt wording here. With no Ollama at all the"
info "project still works, using templates instead."
blank

if ! command -v systemctl > /dev/null 2>&1; then
    info "no systemd on this host - skipping."
elif ! systemctl cat ollama.service > /dev/null 2>&1; then
    info "no ollama.service on this host - skipping."
    info "  To get one in a container instead:"
    info "    echo 'PPG_COMPOSE_OLLAMA_URL=http://ollama:11434' >> .env"
    info "    docker compose --profile ollama up -d"
elif grep -qs 'OLLAMA_HOST=0\.0\.0\.0' "$OLLAMA_DROPIN"; then
    info "already configured: ${OLLAMA_DROPIN} sets OLLAMA_HOST=0.0.0.0."
else
    info "Ollama binds 127.0.0.1 by default, which a container cannot reach even"
    info "through host.docker.internal. A systemd drop-in at"
    info "${OLLAMA_DROPIN} changes that:"
    blank
    info "    [Service]"
    info '    Environment="OLLAMA_HOST=0.0.0.0:11434"'
    blank
    warn "this exposes Ollama on EVERY interface, not only the Docker bridge."
    info "  Ollama has no authentication. Anyone who can reach port 11434 can"
    info "  run models, read the model list, and use your GPU. Before doing this:"
    info "    - firewall 11434 to the docker0/br-* subnets only, e.g."
    info "        sudo ufw allow in on docker0 to any port 11434"
    info "        sudo ufw deny 11434"
    info "    - or skip this entirely and use the containerised Ollama:"
    info "        docker compose --profile ollama up -d"
    blank

    if confirm "Write the drop-in and restart Ollama?"; then
        run "${SUDO[@]}" mkdir -p "$OLLAMA_DROPIN_DIR"
        printf '  $ tee %s\n' "$OLLAMA_DROPIN"
        printf '%s\n' \
            '[Service]' \
            'Environment="OLLAMA_HOST=0.0.0.0:11434"' \
            | "${SUDO[@]}" tee "$OLLAMA_DROPIN" > /dev/null
        run "${SUDO[@]}" systemctl daemon-reload
        run "${SUDO[@]}" systemctl restart ollama
        if command -v ss > /dev/null 2>&1; then
            blank
            info "Listening sockets on 11434 (expect 0.0.0.0 or *):"
            ss -ltn 'sport = :11434' || true
        fi
    fi
fi

# ---------------------------------------------------------------------------

heading "Next"

info "cp .env.example .env                  # if you have not already"
info "docker compose build                  # ~10GB image, mostly torch + CUDA"
info "docker compose up -d"
info "docker compose logs -f api            # the model load takes ~20s"
info "curl -s localhost:8000/readyz         # model_loaded:true when ready"
blank
info "If the weights are not in ./data/hf-cache yet (~7GB):"
info "  docker compose run --rm api python scripts/download-models.py"
blank
info "No GPU:      docker compose --profile cpu up api-cpu"
info "No Ollama:   docker compose --profile ollama up -d"
info "Development: docker compose --profile dev up api-dev"
blank
