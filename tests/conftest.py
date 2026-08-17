"""Shared fixtures.

Every test in this suite runs with **no GPU, no model weights and no network**:

* ``PPG_BACKEND=fake`` swaps Stable Diffusion XL for a deterministic placeholder
  card drawn with Pillow, so nothing is downloaded and nothing needs CUDA.
* ``PPG_COMPOSER=template`` keeps prompt composition in pure Python, so no
  request is ever made to Ollama.
* ``data_dir`` is a per-test ``tmp_path``, so the developer's real ``./data``
  (database, outputs, model cache) is never read or written.

If any of those three slip, the suite would try to pull 7GB of weights on a CI
runner, so the ``settings`` fixture asserts them rather than trusting them.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterator
from contextlib import ExitStack

import pytest
from fastapi.testclient import TestClient

from ppg.api.app import create_app
from ppg.config import Settings, get_settings


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> Iterator[None]:
    """``get_settings`` is ``lru_cache``d, so a cached Settings from one test
    would otherwise leak its ``tmp_path`` into the next one."""
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def make_settings(tmp_path, monkeypatch) -> Callable[..., Settings]:
    """Factory for isolated Settings objects.

    Configuration goes through environment variables because several fields
    (``backend``, ``sizes_raw``) declare a ``validation_alias`` and therefore
    silently ignore constructor keywords - ``Settings(backend="fake")`` alone
    would leave the backend at its ``diffusers`` default.
    """
    # Drop anything the developer happens to have exported, so a local
    # PPG_MODEL_ID or PPG_ALLOW_MINORS cannot change what the tests mean.
    for name in list(os.environ):
        if name.startswith("PPG_") or name == "IMAGE_BACKEND":
            monkeypatch.delenv(name, raising=False)

    monkeypatch.setenv("PPG_BACKEND", "fake")
    monkeypatch.setenv("PPG_COMPOSER", "template")
    # A two-rung size ladder: small enough to keep the suite fast, but still
    # exercises `?size=`, pick_size() and the multi-size write path.
    monkeypatch.setenv("PPG_SIZES", "256,128")
    # HF_HOME is read without the PPG_ prefix and ensure_dirs() creates it, so
    # it has to be redirected or the real huggingface cache dir gets touched.
    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf-cache"))

    def _make(**overrides) -> Settings:
        # _env_file=None ignores any .env sitting in the checkout; the
        # environment set above is the whole configuration.
        return Settings(_env_file=None, data_dir=tmp_path, width=256, height=256, **overrides)

    return _make


@pytest.fixture
def settings(make_settings, tmp_path) -> Settings:
    settings = make_settings()
    # Fail loudly here rather than 20 seconds into a torch import.
    assert settings.backend == "fake", "tests must never touch the real image backend"
    assert settings.composer == "template", "tests must never talk to Ollama"
    assert settings.data_dir == tmp_path.resolve()
    return settings


@pytest.fixture
def make_client() -> Iterator[Callable[[Settings], TestClient]]:
    """Factory for TestClients, each entered as a context manager.

    The context manager is what runs the FastAPI lifespan; without it there is
    no ``app.state.ppg`` and every request answers 503. The ExitStack also
    guarantees the queue worker and database are shut down after the test.
    """
    with ExitStack() as stack:

        def _make(settings: Settings) -> TestClient:
            return stack.enter_context(TestClient(create_app(settings)))

        yield _make


@pytest.fixture
def client(settings, make_client) -> TestClient:
    return make_client(settings)
