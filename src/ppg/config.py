"""Application settings.

Everything is configurable through environment variables prefixed with ``PPG_``
(or a ``.env`` file). See ``.env.example`` for the annotated list.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

BackendName = Literal["diffusers", "fake", "openai_images"]
ComposerMode = Literal["auto", "llm", "template"]
DeviceName = Literal["auto", "cuda", "cpu"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_prefix="PPG_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
        # Without this, a field carrying a `validation_alias` can only be set
        # through its alias, so `Settings(backend="fake")` is silently dropped
        # by `extra="ignore"` and you get the default instead - which for the
        # backend means loading a 7GB model when you asked for the fake one.
        populate_by_name=True,
    )

    # --- Server ---------------------------------------------------------
    host: str = "0.0.0.0"
    port: int = 8000
    api_key: str | None = None
    cors_origins_raw: str = Field(default="*", validation_alias="PPG_CORS_ORIGINS")

    # --- Image backend --------------------------------------------------
    backend: BackendName = Field(
        default="diffusers",
        # IMAGE_BACKEND is accepted as an alias because it reads better in
        # test/CI invocations: `IMAGE_BACKEND=fake pytest`.
        validation_alias=AliasChoices("PPG_BACKEND", "IMAGE_BACKEND"),
    )
    device: DeviceName = "auto"
    model_id: str = "SG161222/RealVisXL_V5.0"
    vae_id: str = "madebyollin/sdxl-vae-fp16-fix"

    steps: int = 30
    fast_steps: int = 15
    guidance: float = 4.5
    width: int = 1024
    height: int = 1024

    low_vram: bool = False
    compile: bool = False

    openai_base_url: str = "http://127.0.0.1:11434/v1"
    openai_api_key: str | None = None
    openai_model: str = "x/z-image-turbo"

    # --- Prompt composer ------------------------------------------------
    composer: ComposerMode = "auto"
    ollama_base_url: str = "http://127.0.0.1:11434"
    # "auto" uses whatever the user already has installed, preferring a small
    # model, and needs no Ollama at all if none is running.
    ollama_model: str = "auto"
    ollama_timeout: float = 90.0
    # How long Ollama should keep the prompt model in VRAM after a request.
    # "0" unloads immediately, which is what you want when the image model
    # shares the same GPU. Use "5m" if Ollama runs elsewhere or you have VRAM
    # to spare - it saves a second of model load per request.
    ollama_keep_alive: str = "0"

    # --- Storage --------------------------------------------------------
    data_dir: Path = Path("./data")
    sizes_raw: str = Field(default="1024,512,256,128", validation_alias="PPG_SIZES")
    webp_quality: int = 90
    # Read without the PPG_ prefix because huggingface_hub owns this variable.
    hf_home: Path | None = Field(default=None, validation_alias="HF_HOME")

    # --- Safety ---------------------------------------------------------
    min_age: int = 18
    max_age: int = 90
    allow_minors: bool = False

    # --- Queue ----------------------------------------------------------
    max_queue: int = 128
    default_wait: float = 60.0

    # -------------------------------------------------------------------
    # Derived values
    # -------------------------------------------------------------------
    @field_validator("hf_home", "api_key", "openai_api_key", mode="before")
    @classmethod
    def _empty_is_none(cls, v: object) -> object:
        """Treat an empty env var as unset.

        `.env` files habitually contain `HF_HOME=` as a placeholder. Without
        this, pydantic coerces that empty string to `Path(".")` and the model
        cache silently lands in the current working directory - which then
        re-downloads 7GB on the next run from a different directory.
        """
        return None if isinstance(v, str) and not v.strip() else v

    @field_validator("data_dir")
    @classmethod
    def _expand(cls, v: Path) -> Path:
        # Absolute, so `ppg` behaves the same from any working directory and
        # huggingface_hub cannot resolve HF_HOME against the wrong cwd.
        return v.expanduser().resolve()

    @property
    def sizes(self) -> list[int]:
        """Output sizes, largest first. The first entry is the render size."""
        parsed = [int(s.strip()) for s in self.sizes_raw.split(",") if s.strip()]
        return sorted(set(parsed), reverse=True)

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.cors_origins_raw.split(",") if o.strip()]

    @property
    def outputs_dir(self) -> Path:
        return self.data_dir / "outputs"

    @property
    def db_path(self) -> Path:
        return self.data_dir / "ppg.db"

    def resolve_device(self) -> str:
        """Turn ``auto`` into a concrete device, without importing torch eagerly."""
        if self.device != "auto":
            return self.device
        try:
            import torch
        except ImportError:
            return "cpu"
        return "cuda" if torch.cuda.is_available() else "cpu"

    @property
    def model_cache_dir(self) -> Path:
        return (self.hf_home or (self.data_dir / "hf-cache")).expanduser()

    def ensure_dirs(self) -> None:
        """Create the data directories and point huggingface_hub at ours.

        Without this, weights land in ``~/.cache/huggingface`` and a container
        rebuild silently re-downloads 7GB. Setting the variable here rather
        than relying on the shell means it also works when the value came from
        a ``.env`` file, which pydantic-settings reads without exporting.
        """
        self.outputs_dir.mkdir(parents=True, exist_ok=True)
        cache = self.model_cache_dir
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("HF_HOME", str(cache))


@lru_cache
def get_settings() -> Settings:
    return Settings()
