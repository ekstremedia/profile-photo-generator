"""Shared application state and request dependencies."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from ppg.backends.base import ImageBackend
from ppg.config import Settings
from ppg.pipeline.worker import JobQueue
from ppg.service import AvatarService
from ppg.store.db import Database

_bearer = HTTPBearer(auto_error=False)


@dataclass
class AppState:
    settings: Settings
    db: Database
    backend: ImageBackend
    service: AvatarService
    queue: JobQueue
    ollama_model: str | None = None
    warm_error: str | None = None


def get_state(request: Request) -> AppState:
    state: AppState | None = getattr(request.app.state, "ppg", None)
    if state is None:  # pragma: no cover - only if the lifespan did not run
        raise HTTPException(status_code=503, detail="Service is still starting up.")
    return state


def get_settings_dep(state: AppState = Depends(get_state)) -> Settings:
    return state.settings


def get_service(state: AppState = Depends(get_state)) -> AvatarService:
    return state.service


def get_queue(state: AppState = Depends(get_state)) -> JobQueue:
    return state.queue


def require_api_key(
    state: AppState = Depends(get_state),
    credentials: HTTPAuthorizationCredentials | None = Depends(_bearer),
) -> None:
    """Bearer-token check, active only when PPG_API_KEY is set.

    Off by default so `docker compose up` works with no configuration. The
    README and SECURITY.md both say to set the key before exposing the port.
    """
    expected = state.settings.api_key
    if not expected:
        return
    if credentials is None or credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing bearer token.",
            headers={"WWW-Authenticate": "Bearer"},
        )
