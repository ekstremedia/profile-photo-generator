"""Command line interface.

Two modes, chosen automatically:

* **local** - loads the model in this process. Good for one-offs, batches and
  `ppg doctor`.
* **remote** - talks to a running server when ``--url`` or ``PPG_URL`` is set.
  The same commands then drive a container or another machine.

``ppg doctor`` is the command to reach for first when something is wrong; it
checks every external thing this project depends on and says which one is
unhappy.
"""

from __future__ import annotations

import asyncio
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import httpx
import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn, TimeElapsedColumn
from rich.table import Table

from ppg import __version__
from ppg.config import Settings, get_settings
from ppg.schemas import AvatarRequest, BatchRequest

app = typer.Typer(
    name="ppg",
    help="Generate photorealistic synthetic profile photos locally.",
    no_args_is_help=True,
    add_completion=False,
)
console = Console()
err = Console(stderr=True)

PASS, WARN, FAIL = "[green]pass[/green]", "[yellow]warn[/yellow]", "[red]fail[/red]"


# ---------------------------------------------------------------------------
# Shared plumbing
# ---------------------------------------------------------------------------


def _settings() -> Settings:
    """Load settings and create the data directories.

    ``ensure_dirs`` also exports HF_HOME. Skipping it means huggingface_hub
    quietly falls back to ~/.cache/huggingface and re-downloads 7GB of weights
    that are already sitting in ./data/hf-cache.
    """
    settings = get_settings()
    settings.ensure_dirs()
    return settings


async def _make_service(settings: Settings):
    from ppg.backends.base import build_backend
    from ppg.prompt.composer import build_composer_auto
    from ppg.service import AvatarService
    from ppg.store.db import Database

    settings.ensure_dirs()
    db = Database(settings.db_path)
    backend = build_backend(settings)
    composer, _model = await build_composer_auto(settings)
    return AvatarService(settings, db, backend, composer), db, backend


def _copy_out(
    settings: Settings, avatar_id: str, sizes: list[int], out_dir: Path, stem: str
) -> list[Path]:
    from ppg.store.files import variant_path

    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for size in sizes:
        for fmt in ("png", "webp"):
            source = variant_path(settings.outputs_dir, avatar_id, size, fmt)
            if source.is_file():
                target = out_dir / f"{stem}-{size}.{fmt}"
                shutil.copy2(source, target)
                written.append(target)
    return written


def _print_result(result: Any, out_files: list[Path]) -> None:
    attrs = result["attributes"] if isinstance(result, dict) else result.attributes
    persona = (
        result["persona"]
        if isinstance(result, dict)
        else (result.persona.model_dump() if result.persona else None)
    )
    prompt = result["prompt"] if isinstance(result, dict) else result.prompt
    avatar_id = result["id"] if isinstance(result, dict) else result.id

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_row("[bold]id[/bold]", avatar_id)
    if persona:
        table.add_row(
            "[bold]person[/bold]", f"{persona['name']}, {persona['age']}, {persona['occupation']}"
        )
        if persona.get("city"):
            table.add_row("[bold]city[/bold]", persona["city"])
    interesting = ("sex", "age", "ethnicity", "skin_tone", "profession", "expression")
    table.add_row(
        "[bold]traits[/bold]",
        ", ".join(f"{k}={attrs[k]}" for k in interesting if k in attrs),
    )
    console.print(table)
    console.print(Panel(prompt, title="prompt", border_style="dim", expand=False))
    for path in out_files:
        console.print(f"  [green]->[/green] {path}")


# ---------------------------------------------------------------------------
# doctor
# ---------------------------------------------------------------------------


@app.command()
def doctor(url: str | None = typer.Option(None, help="Also probe a running server.")) -> None:
    """Check every external dependency and report what is wrong.

    Run this first when something does not work. It is also what the bug
    report template asks for.
    """
    settings = _settings()
    table = Table(title="ppg doctor", header_style="bold")
    table.add_column("check")
    table.add_column("result")
    table.add_column("detail", overflow="fold")

    failures = 0

    table.add_row("python", PASS, sys.version.split()[0])
    table.add_row("ppg", PASS, __version__)
    table.add_row("backend", PASS, f"{settings.backend} (device={settings.device})")

    # --- GPU -----------------------------------------------------------
    if settings.backend != "diffusers":
        table.add_row("torch", WARN, f"not required for the {settings.backend} backend")
    else:
        try:
            import torch

            if torch.cuda.is_available():
                name = torch.cuda.get_device_name(0)
                vram = torch.cuda.get_device_properties(0).total_memory / 1e9
                verdict = PASS if vram >= 7.5 else WARN
                note = f"{name}, {vram:.1f} GB VRAM"
                if vram < 7.5:
                    note += " - set PPG_LOW_VRAM=true"
                table.add_row("cuda", verdict, note)
            else:
                table.add_row("cuda", WARN, "no CUDA device - will run on CPU, minutes per image")
            table.add_row("torch", PASS, torch.__version__)
        except ImportError:
            failures += 1
            table.add_row("torch", FAIL, "not installed - pip install '.[gpu]'")

        try:
            import diffusers

            table.add_row("diffusers", PASS, diffusers.__version__)
        except ImportError:
            failures += 1
            table.add_row("diffusers", FAIL, "not installed - pip install '.[gpu]'")

    # --- disk ----------------------------------------------------------
    free_gb = shutil.disk_usage(settings.data_dir).free / 1e9
    table.add_row(
        "disk",
        PASS if free_gb > 10 else (WARN if free_gb > 3 else FAIL),
        f"{free_gb:.1f} GB free at {settings.data_dir}",
    )
    if free_gb <= 3:
        failures += 1

    # --- weights -------------------------------------------------------
    cache = settings.model_cache_dir
    slug = "models--" + settings.model_id.replace("/", "--")
    present = (cache / "hub" / slug).is_dir() or (cache / slug).is_dir()
    if settings.backend != "diffusers":
        table.add_row("weights", WARN, "not needed for this backend")
    elif present:
        size = sum(f.stat().st_size for f in (cache).rglob("*") if f.is_file()) / 1e9
        table.add_row("weights", PASS, f"{settings.model_id} cached ({size:.1f} GB)")
    else:
        failures += 1
        table.add_row(
            "weights", FAIL, f"{settings.model_id} missing - run scripts/download-models.py"
        )

    # --- ollama --------------------------------------------------------
    if settings.composer == "template":
        table.add_row("ollama", PASS, "disabled (PPG_COMPOSER=template)")
    else:
        from ppg.prompt.composer import resolve_ollama_model
        from ppg.prompt.ollama_client import OllamaClient

        client = OllamaClient(settings.ollama_base_url, settings.ollama_model, timeout=5.0)

        async def _probe() -> tuple[bool, str | None, list[str]]:
            if not await client.reachable():
                return False, None, []
            names = await client.available_models()
            chosen = await resolve_ollama_model(client, settings.ollama_model)
            return True, chosen, names

        reachable, chosen, names = asyncio.run(_probe())
        if not reachable:
            verdict = FAIL if settings.composer == "llm" else WARN
            if settings.composer == "llm":
                failures += 1
            table.add_row(
                "ollama",
                verdict,
                f"unreachable at {settings.ollama_base_url} - "
                + (
                    "PPG_COMPOSER=llm requires it"
                    if settings.composer == "llm"
                    else "optional, prompts fall back to templates"
                ),
            )
        elif chosen is None:
            table.add_row("ollama", WARN, "reachable but no models pulled - using templates")
        else:
            table.add_row("ollama", PASS, f"using {chosen} ({len(names)} model(s) installed)")

    # --- server --------------------------------------------------------
    target = url
    if target:
        try:
            response = httpx.get(f"{target.rstrip('/')}/readyz", timeout=10.0)
            payload = response.json()
            table.add_row(
                "server",
                PASS if payload.get("model_loaded") else WARN,
                f"{target} status={payload.get('status')} queue={payload.get('queue_depth')}",
            )
        except (httpx.HTTPError, ValueError) as exc:
            failures += 1
            table.add_row("server", FAIL, f"{target}: {exc}")

    console.print(table)
    if failures:
        err.print(f"\n[red]{failures} check(s) failed.[/red] See docs/TROUBLESHOOTING.md")
        raise typer.Exit(code=1)
    console.print("\n[green]All checks passed.[/green]")


# ---------------------------------------------------------------------------
# warmup
# ---------------------------------------------------------------------------


@app.command()
def warmup() -> None:
    """Load the model and render one throwaway image, reporting timings.

    Useful as a post-deploy smoke test: it proves the weights, the VAE and the
    GPU all work together before any real request arrives.
    """
    import time

    settings = _settings()

    async def _run() -> None:
        from ppg.backends.base import RenderSpec, build_backend

        backend = build_backend(settings)
        with console.status(f"Loading {backend.model_id} ..."):
            started = time.perf_counter()
            await backend.load()
            load_s = time.perf_counter() - started
        console.print(f"  loaded in [bold]{load_s:.1f}s[/bold]")

        with console.status("Rendering a test image ..."):
            started = time.perf_counter()
            image = await backend.generate(
                RenderSpec(
                    prompt="head and shoulders portrait of a person, natural window light, 85mm",
                    negative_prompt="cartoon, 3d render",
                    width=settings.width,
                    height=settings.height,
                    steps=settings.steps,
                    guidance=settings.guidance,
                    seed=1234,
                )
            )
            render_s = time.perf_counter() - started

        console.print(f"  rendered {image.size[0]}x{image.size[1]} in [bold]{render_s:.1f}s[/bold]")
        if render_s > 10 and settings.resolve_device() == "cuda":
            console.print(
                "[yellow]That is slower than expected for a GPU. Check whether "
                "PPG_LOW_VRAM is on, or whether the model fell back to float32.[/yellow]"
            )
        await backend.unload()

    asyncio.run(_run())


# ---------------------------------------------------------------------------
# generate
# ---------------------------------------------------------------------------


@app.command()
def generate(
    sex: str | None = typer.Option(None),
    age: int | None = typer.Option(None),
    ethnicity: str | None = typer.Option(None),
    skin_tone: str | None = typer.Option(None),
    profession: str | None = typer.Option(None),
    expression: str | None = typer.Option(None),
    glasses: str | None = typer.Option(None),
    seed: str | None = typer.Option(None, help="Integer, or any string (e.g. an email)."),
    fast: bool = typer.Option(False, help="Fewer steps. Quicker, slightly softer."),
    extra: str | None = typer.Option(None, "--extra", help="Extra prompt detail."),
    out: Path = typer.Option(Path("./out"), help="Where to write the image files."),
    url: str | None = typer.Option(None, envvar="PPG_URL", help="Use a running server instead."),
    show_json: bool = typer.Option(False, "--json", help="Print the full result as JSON."),
) -> None:
    """Generate one avatar. Everything not given is randomised from the seed."""
    payload: dict[str, Any] = {
        k: v
        for k, v in {
            "sex": sex,
            "age": age,
            "ethnicity": ethnicity,
            "skin_tone": skin_tone,
            "profession": profession,
            "expression": expression,
            "glasses": glasses,
            "seed": seed,
            "fast": fast,
            "prompt_extra": extra,
        }.items()
        if v is not None
    }

    if url:
        result = _remote_generate(url, payload)
        if show_json:
            console.print_json(json.dumps(result))
            return
        console.print(f"[dim]Generated on {url}[/dim]")
        _print_result(result, [])
        console.print(f"  image: {url.rstrip('/')}{result['urls']['default']}")
        return

    settings = _settings()

    async def _run() -> None:
        service, db, backend = await _make_service(settings)
        try:
            request = AvatarRequest(**payload)
            service.precheck(request)
            with console.status("Generating ..."):
                result = await service.generate(request)
            files = _copy_out(settings, result.id, result.sizes, out, result.id[:12])
            if show_json:
                console.print_json(result.model_dump_json())
            else:
                _print_result(result, files)
                if result.cached:
                    console.print("[dim]  (served from cache)[/dim]")
        finally:
            await backend.unload()
            db.close()

    asyncio.run(_run())


def _remote_generate(url: str, payload: dict[str, Any]) -> dict[str, Any]:
    try:
        response = httpx.post(f"{url.rstrip('/')}/v1/avatars", json=payload, timeout=300.0)
        response.raise_for_status()
        return response.json()
    except httpx.HTTPStatusError as exc:
        err.print(f"[red]{exc.response.status_code}: {exc.response.text[:300]}[/red]")
        raise typer.Exit(code=1) from exc
    except httpx.HTTPError as exc:
        err.print(f"[red]Could not reach {url}: {exc}[/red]")
        raise typer.Exit(code=1) from exc


# ---------------------------------------------------------------------------
# batch
# ---------------------------------------------------------------------------


@app.command()
def batch(
    n: int = typer.Option(10, "-n", "--count", min=1, max=500),
    diversity: str = typer.Option("even", help="even | random"),
    seed: str | None = typer.Option(None),
    out: Path = typer.Option(Path("./out"), help="Where to write the image files."),
    contact_sheet: bool = typer.Option(True, help="Also write a contact sheet montage."),
) -> None:
    """Generate many avatars at once, spread evenly across the main axes."""
    settings = _settings()

    async def _run() -> None:
        service, db, backend = await _make_service(settings)
        try:
            request = BatchRequest(n=n, diversity=diversity, seed=seed)  # type: ignore[arg-type]
            requests = service.plan_batch(request)
            paths: list[Path] = []
            ids: list[str] = []
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("{task.completed}/{task.total}"),
                TimeElapsedColumn(),
                console=console,
            ) as progress:
                task = progress.add_task("generating", total=len(requests))
                for index, single in enumerate(requests):
                    result = await service.generate(single)
                    ids.append(result.id)
                    paths += _copy_out(
                        settings, result.id, [max(result.sizes)], out, f"{index:03d}"
                    )
                    progress.advance(task)

            console.print(f"[green]{len(ids)} avatars[/green] -> {out}")
            if contact_sheet and ids:
                sheet = _contact_sheet(settings, ids, out / "contact-sheet.jpg")
                console.print(f"  contact sheet -> {sheet}")
        finally:
            await backend.unload()
            db.close()

    asyncio.run(_run())


def _contact_sheet(settings: Settings, ids: list[str], target: Path) -> Path:
    """Tile the batch into one image. The fastest way to judge whether a batch
    is actually varied or whether it produced the same face twelve times."""
    from PIL import Image

    from ppg.store.files import variant_path

    thumb = 256
    columns = min(6, len(ids))
    rows = (len(ids) + columns - 1) // columns
    sheet = Image.new("RGB", (columns * thumb, rows * thumb), (18, 18, 20))
    for index, avatar_id in enumerate(ids):
        path = variant_path(settings.outputs_dir, avatar_id, thumb, "webp")
        if not path.is_file():
            continue
        with Image.open(path) as tile:
            sheet.paste(
                tile.convert("RGB"), ((index % columns) * thumb, (index // columns) * thumb)
            )
    target.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(target, "JPEG", quality=88)
    return target


# ---------------------------------------------------------------------------
# misc
# ---------------------------------------------------------------------------


@app.command()
def options(axis: str | None = typer.Argument(None, help="Show one axis only.")) -> None:
    """List every attribute value this instance accepts."""
    from ppg.attributes.sampler import get_vocabulary

    vocab = get_vocabulary()
    axes = {axis: vocab.axes[axis]} if axis and axis in vocab.axes else vocab.axes
    if axis and axis not in vocab.axes:
        err.print(f"[red]Unknown axis {axis!r}.[/red] Known: {', '.join(vocab.axes)}")
        raise typer.Exit(code=1)

    for name, opts in axes.items():
        table = Table(title=name, header_style="bold", show_lines=False)
        table.add_column("value")
        table.add_column("weight", justify="right")
        table.add_column("prompt", overflow="fold")
        for opt in opts:
            table.add_row(opt.value, f"{opt.weight:g}", opt.phrase or "[dim](omitted)[/dim]")
        console.print(table)


@app.command()
def clear(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete every generated avatar and its image files.

    Model weights are untouched - this only clears `data/outputs` and the
    index. Cached prompts are kept, so regenerating a `by-seed` avatar
    afterwards returns the identical face.
    """
    settings = _settings()
    from ppg.store.db import Database

    db = Database(settings.db_path)
    try:
        total = db.count_avatars()
        if not total:
            console.print("Nothing to delete.")
            return
        if not yes and not typer.confirm(f"Delete all {total} avatars and their image files?"):
            console.print("Cancelled.")
            raise typer.Exit(code=1)

        from ppg.backends.fake import FakeBackend
        from ppg.prompt.composer import TemplateComposer
        from ppg.service import AvatarService

        # Deleting needs the store, not a model, so the cheapest backend will
        # do - loading 7GB of weights to remove files would be absurd.
        service = AvatarService(settings, db, FakeBackend(settings), TemplateComposer())
        console.print(f"[green]Deleted {service.clear_all()} avatars.[/green]")
    finally:
        db.close()


@app.command()
def serve(
    host: str | None = typer.Option(None),
    port: int | None = typer.Option(None),
    reload: bool = typer.Option(False, help="Auto-reload on code changes."),
) -> None:
    """Run the HTTP server without Docker."""
    import uvicorn

    settings = _settings()
    uvicorn.run(
        "ppg.api.app:app",
        host=host or settings.host,
        port=port or settings.port,
        reload=reload,
        log_level="info",
    )


@app.command()
def version() -> None:
    """Print the version."""
    console.print(__version__)


if __name__ == "__main__":
    app()
