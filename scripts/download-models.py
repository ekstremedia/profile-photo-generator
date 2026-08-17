#!/usr/bin/env python3
"""Download the image model weights.

Run this once before the first generation:

    python scripts/download-models.py

The point of this script rather than a bare `hf download` is the include
patterns. The RealVisXL repository ships fp16 *and* fp32 copies of everything,
so an unfiltered clone pulls ~34.6 GB when only ~7 GB is used. It also pulls
the fp16-fix VAE, which is not optional: SDXL's stock VAE produces black
images in float16, and that is the single most common "it doesn't work"
report for any SDXL project.

Safe to re-run - anything already in the cache is skipped.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# (repo_id, allow_patterns, approximate download size in GB, why we need it)
DEFAULT_MODELS: list[tuple[str, list[str], float, str]] = [
    (
        "SG161222/RealVisXL_V5.0",
        [
            "model_index.json",
            "scheduler/*",
            "tokenizer*/*",
            "*/config.json",
            "*.fp16.safetensors",
        ],
        6.94,
        "SDXL checkpoint tuned for photorealistic faces",
    ),
    (
        "madebyollin/sdxl-vae-fp16-fix",
        ["config.json", "diffusion_pytorch_model.safetensors"],
        0.34,
        "fp16-safe VAE (without it, every image comes out black)",
    ),
]


def human_gb(value: float) -> str:
    return f"{value:.2f} GB"


def free_space_gb(path: Path) -> float:
    while not path.exists():
        path = path.parent
    return shutil.disk_usage(path).free / 1e9


def resolve_cache_dir(explicit: str | None) -> Path:
    if explicit:
        return Path(explicit).expanduser().resolve()
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]).expanduser().resolve()
    return (REPO_ROOT / "data" / "hf-cache").resolve()


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--cache-dir",
        help="Where to store weights. Defaults to $HF_HOME, else ./data/hf-cache",
    )
    # The service loads whatever PPG_MODEL_ID / PPG_VAE_ID say, so the
    # downloader has to be able to follow. Without these, an operator who
    # changed the checkpoint got a cache full of the default weights and a
    # 7GB surprise download on the first request.
    parser.add_argument(
        "--model-id",
        default=os.environ.get("PPG_MODEL_ID"),
        help="Checkpoint repo id. Defaults to $PPG_MODEL_ID, then the project default.",
    )
    parser.add_argument(
        "--vae-id",
        default=os.environ.get("PPG_VAE_ID"),
        help="VAE repo id. Defaults to $PPG_VAE_ID, then the project default.",
    )
    parser.add_argument(
        "--model",
        action="append",
        help="Extra repo id to fetch in full (e.g. a LoRA). Repeatable.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Show what would be downloaded.")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print(
            "huggingface_hub is not installed.\n\n"
            "  pip install 'huggingface_hub[hf_transfer]'\n"
            "  # Debian/Ubuntu block system-wide pip, so use one of:\n"
            "  uv tool install 'huggingface_hub[cli]'\n"
            "  pipx install 'huggingface_hub[cli]'\n",
            file=sys.stderr,
        )
        return 1

    cache_dir = resolve_cache_dir(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    # Rust-backed parallel downloader, roughly 2-3x faster on a fast link.
    # Only enable it if actually installed - hf_hub errors out otherwise.
    try:
        import hf_transfer  # noqa: F401
    except ImportError:
        pass
    else:
        os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    checkpoint, vae = DEFAULT_MODELS
    if args.model_id and args.model_id != checkpoint[0]:
        # Patterns are kept: a different SDXL checkpoint has the same layout,
        # and the fp32 duplicates are just as much of a waste there.
        checkpoint = (args.model_id, checkpoint[1], 0.0, "checkpoint (configured)")
    if args.vae_id and args.vae_id != vae[0]:
        vae = (args.vae_id, vae[1], 0.0, "VAE (configured)")

    targets = [checkpoint, vae]
    for extra in args.model or []:
        targets.append((extra, ["*"], 0.0, "user requested"))

    total = sum(size for _, _, size, _ in targets)
    available = free_space_gb(cache_dir)

    print(f"Cache directory : {cache_dir}")
    print(f"Free disk space : {human_gb(available)}")
    print(f"To download     : ~{human_gb(total)}\n")

    for repo, patterns, size, why in targets:
        print(f"  {repo}")
        print(f"    ~{human_gb(size)} - {why}")
        print(f"    patterns: {' '.join(patterns)}")
    print()

    if available < total * 1.2:
        print(
            f"Not enough free space: need ~{human_gb(total * 1.2)} including working "
            f"room, have {human_gb(available)}.\n"
            "Tip: `docker builder prune` often reclaims a great deal.",
            file=sys.stderr,
        )
        return 1

    if args.dry_run:
        print("Dry run - nothing downloaded.")
        return 0

    for repo, patterns, _size, _why in targets:
        print(f"--> {repo}")
        try:
            path = snapshot_download(
                repo_id=repo,
                allow_patterns=patterns,
                cache_dir=str(cache_dir / "hub"),
            )
        except Exception as exc:
            print(f"\nFailed to download {repo}: {exc}", file=sys.stderr)
            if "hf_transfer" in str(exc):
                print("Retry with HF_HUB_ENABLE_HF_TRANSFER=0", file=sys.stderr)
            return 1
        print(f"    ok -> {path}\n")

    print("All weights present.\n")
    print("Next:")
    print("  ppg doctor      # verify GPU, disk, Ollama and model")
    print("  ppg warmup      # load the pipeline and time one image")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
