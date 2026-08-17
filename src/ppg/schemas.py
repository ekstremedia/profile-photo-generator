"""Public request/response models.

These are the contract. The gallery UI, the CLI and any Laravel client all
speak exactly this, so keep changes additive.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

# The attribute axes a caller may pin. Anything left unset is drawn from the
# vocabulary using the request's seed. Order matters: it is the order used when
# building the prompt and when rendering the gallery form.
ATTRIBUTE_AXES: tuple[str, ...] = (
    "sex",
    "age_range",
    "ethnicity",
    "skin_tone",
    "profession",
    "hair",
    "facial_hair",
    "glasses",
    "expression",
    "clothing",
    "background",
    "lighting",
    "camera",
)

JobStatus = Literal["queued", "running", "done", "failed"]


class AvatarRequest(BaseModel):
    """Everything is optional. An empty body gives a fully random face."""

    model_config = ConfigDict(extra="forbid")

    sex: str | None = None
    age: int | None = Field(default=None, description="Exact age. Overrides age_range.")
    age_range: str | None = None
    ethnicity: str | None = None
    skin_tone: str | None = Field(
        default=None, description="Fitzpatrick I-VI, or a plain descriptor like 'deep'."
    )
    profession: str | None = None
    hair: str | None = None
    facial_hair: str | None = None
    glasses: str | None = None
    expression: str | None = None
    clothing: str | None = None
    background: str | None = None
    lighting: str | None = None
    camera: str | None = None

    seed: int | str | None = Field(
        default=None,
        description="Integer seed, or any string (e.g. an email) hashed to one. "
        "The same value always produces the same face.",
    )
    size: int | None = Field(
        default=None, description="Served size; must be one of /v1/options.sizes"
    )
    fast: bool = Field(default=False, description="Fewer sampling steps. Quicker, slightly softer.")
    prompt_extra: str | None = Field(default=None, description="Appended to the generated prompt.")
    negative_extra: str | None = Field(default=None, description="Appended to the negative prompt.")

    def pinned_attributes(self) -> dict[str, str]:
        """Caller-specified axes only, normalised to lowercase strings."""
        out: dict[str, str] = {}
        for axis in ATTRIBUTE_AXES:
            value = getattr(self, axis, None)
            if value is not None and str(value).strip():
                out[axis] = str(value).strip()
        return out


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    n: int = Field(default=10, ge=1, le=500)
    diversity: Literal["even", "random"] = "even"
    seed: int | str | None = None
    overrides: AvatarRequest = Field(default_factory=AvatarRequest)


class Persona(BaseModel):
    """Plausible identity for the generated face. Entirely fictional."""

    name: str
    age: int
    occupation: str
    city: str | None = None
    country: str | None = None
    bio: str | None = None


class ComposedPrompt(BaseModel):
    """What the prompt composer returns. Validated before it reaches the model."""

    prompt: str = Field(min_length=10)
    negative_extra: str = ""
    persona: Persona


class AvatarResult(BaseModel):
    id: str
    hash: str
    seed: int
    seed_key: str | None = None
    attributes: dict[str, Any]
    persona: Persona | None = None
    prompt: str
    negative_prompt: str
    model: str
    backend: str
    composer: Literal["llm", "template"]
    sizes: list[int]
    urls: dict[str, str]
    created_at: datetime
    cached: bool = False
    duration_ms: int | None = None


class JobInfo(BaseModel):
    id: str
    status: JobStatus
    kind: Literal["single", "batch"] = "single"
    total: int = 1
    completed: int = 0
    position: int | None = Field(default=None, description="Place in the queue, 0 = next up.")
    eta_seconds: float | None = None
    avatar_ids: list[str] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime
    finished_at: datetime | None = None


class AxisOption(BaseModel):
    value: str
    weight: float = 1.0
    label: str | None = None


class OptionsResponse(BaseModel):
    """Drives the gallery form and any client-side validation."""

    axes: dict[str, list[AxisOption]]
    sizes: list[int]
    formats: list[str]
    age_bounds: dict[str, int]
    backend: str
    model: str
    composer: str


class HealthResponse(BaseModel):
    status: str
    version: str
    backend: str
    device: str
    model_loaded: bool
    ollama_reachable: bool | None = None
    queue_depth: int = 0
