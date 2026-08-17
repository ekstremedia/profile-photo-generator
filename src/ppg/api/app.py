"""FastAPI application factory."""

from __future__ import annotations

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from ppg import __version__
from ppg.api.deps import AppState
from ppg.api.routes_avatars import router as avatars_router
from ppg.api.routes_meta import router as meta_router
from ppg.api.routes_openai import router as openai_router
from ppg.backends.base import build_backend
from ppg.config import Settings, get_settings
from ppg.pipeline.worker import JobQueue
from ppg.prompt.composer import build_composer_auto
from ppg.safety import SafetyError
from ppg.service import AvatarService
from ppg.store.db import Database

logger = logging.getLogger(__name__)


def _static_dir() -> Path | None:
    """Locate the gallery assets in both a source checkout and a container."""
    candidates = []
    if override := os.environ.get("PPG_STATIC_DIR"):
        candidates.append(Path(override))
    candidates += [
        Path(__file__).resolve().parents[3] / "static",  # source checkout
        Path("/app/static"),  # container layout
        Path.cwd() / "static",
    ]
    return next((path for path in candidates if path.is_dir()), None)


DESCRIPTION = """
Generate photorealistic synthetic profile photos locally.

* `POST /v1/avatars` - make one, optionally steered by attributes
* `GET /v1/avatars/by-seed/{key}` - deterministic: the same key always returns
  the same face, so `by-seed/{md5(email)}` works as a drop-in avatar service
* `GET /v1/options` - every attribute value this instance accepts

Every face is synthetic and depicts no real person. Output files carry
metadata saying so.
"""


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings: Settings = app.state.settings
    settings.ensure_dirs()

    db = Database(settings.db_path)
    backend = build_backend(settings)
    composer, ollama_model = await build_composer_auto(settings)
    service = AvatarService(settings, db, backend, composer)
    queue = JobQueue(service, max_size=settings.max_queue)

    state = AppState(
        settings=settings,
        db=db,
        backend=backend,
        service=service,
        queue=queue,
        ollama_model=ollama_model,
    )
    app.state.ppg = state
    await queue.start()

    # Load the model in the background: startup should not block for the
    # seconds a 7GB checkpoint takes, but /readyz must report false until it is
    # done so an orchestrator does not route traffic too early.
    async def warm() -> None:
        try:
            await backend.load()
            logger.info("Backend %s ready (%s)", backend.name, backend.model_id)
        except Exception as exc:
            state.warm_error = str(exc)
            logger.error("Backend failed to load: %s", exc)

    warm_task = asyncio.create_task(warm(), name="ppg-warmup")

    try:
        yield
    finally:
        warm_task.cancel()
        await queue.stop()
        await backend.unload()
        db.close()


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()

    app = FastAPI(
        title="Profile Photo Generator",
        description=DESCRIPTION,
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url=None,
        openapi_url="/openapi.json",
    )
    app.state.settings = settings

    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=False,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.exception_handler(SafetyError)
    async def _safety_handler(_request: Request, exc: SafetyError) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(exc)})

    app.include_router(meta_router)
    app.include_router(avatars_router)
    app.include_router(openai_router)

    # Mounted last so it cannot shadow the API routes.
    if static_dir := _static_dir():
        app.mount("/", StaticFiles(directory=str(static_dir), html=True), name="gallery")

    return app


app = create_app()
