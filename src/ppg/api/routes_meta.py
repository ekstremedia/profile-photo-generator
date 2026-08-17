"""Options, health and metrics.

Note on authentication: unlike ``/v1/avatars``, nothing in this module
requires ``PPG_API_KEY``. That is deliberate. ``/healthz``, ``/readyz`` and
``/metrics`` need to be reachable by probes and scrapers that have no
credentials, and ``/v1/options`` only returns the contents of ``vocab.yaml`` -
a file that ships in the public repository - while the gallery UI, which has
nowhere to put a bearer token, needs it to render its form. Generation, which
is the part that costs GPU time, stays behind the key.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import PlainTextResponse

from ppg import __version__
from ppg.api.deps import AppState, get_state
from ppg.attributes.sampler import get_vocabulary
from ppg.prompt.ollama_client import OllamaClient
from ppg.schemas import AxisOption, HealthResponse, OptionsResponse

router = APIRouter(tags=["meta"])


@router.get("/v1/options", response_model=OptionsResponse, summary="Valid attribute values")
async def options(state: AppState = Depends(get_state)) -> OptionsResponse:
    """Everything this instance accepts.

    The gallery builds its form from this, and it is the right thing for a
    client (a Laravel validator, say) to fetch once and cache - it reflects the
    running instance's own `vocab.yaml`, which the operator may have edited.
    """
    vocab = get_vocabulary()
    axes = {
        axis: [AxisOption(value=opt.value, weight=opt.weight, label=opt.display) for opt in opts]
        for axis, opts in vocab.axes.items()
    }
    return OptionsResponse(
        axes=axes,
        sizes=state.settings.sizes,
        formats=["webp", "png"],
        age_bounds={"min": state.settings.min_age, "max": state.settings.max_age},
        backend=state.backend.name,
        model=state.backend.model_id,
        composer=state.settings.composer,
    )


@router.get("/healthz", summary="Liveness")
async def healthz() -> dict[str, str]:
    """Always 200 while the process is up. Use /readyz for traffic decisions."""
    return {"status": "ok", "version": __version__}


@router.get("/readyz", response_model=HealthResponse, summary="Readiness")
async def readyz(state: AppState = Depends(get_state)) -> HealthResponse:
    """False until the model is loaded.

    Loading a 7GB checkpoint takes a while, and a request that arrives first
    would simply block. Orchestrators should gate on this.
    """
    ollama_reachable: bool | None = None
    if state.settings.composer != "template":
        client = OllamaClient(state.settings.ollama_base_url, state.ollama_model or "", timeout=3.0)
        ollama_reachable = await client.reachable()

    return HealthResponse(
        status="ready" if state.backend.loaded else ("error" if state.warm_error else "loading"),
        version=__version__,
        backend=state.backend.name,
        device=state.settings.resolve_device(),
        model_loaded=state.backend.loaded,
        ollama_reachable=ollama_reachable,
        queue_depth=state.queue.depth,
    )


@router.get("/metrics", summary="Prometheus metrics", response_class=PlainTextResponse)
async def metrics(state: AppState = Depends(get_state)) -> str:
    """Minimal Prometheus exposition.

    Deliberately hand-rolled rather than pulling in a client library - there
    are four numbers worth reporting and the format is one line each.
    """
    average = state.queue.average_seconds
    lines = [
        "# HELP ppg_avatars_total Avatars stored in the index.",
        "# TYPE ppg_avatars_total counter",
        f"ppg_avatars_total {state.db.count_avatars()}",
        "# HELP ppg_queue_depth Images waiting to be rendered.",
        "# TYPE ppg_queue_depth gauge",
        f"ppg_queue_depth {state.queue.depth}",
        "# HELP ppg_model_loaded Whether the image model is resident.",
        "# TYPE ppg_model_loaded gauge",
        f"ppg_model_loaded {int(state.backend.loaded)}",
        "# HELP ppg_generation_seconds Rolling mean generation time.",
        "# TYPE ppg_generation_seconds gauge",
        f"ppg_generation_seconds {average:.3f}" if average else "ppg_generation_seconds 0",
    ]
    return "\n".join(lines) + "\n"
