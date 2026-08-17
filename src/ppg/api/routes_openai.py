"""OpenAI-compatible image endpoint.

Exists so existing tooling works unmodified - `openai-php` in a Laravel app,
the `openai` Python package, anything that already speaks
`POST /v1/images/generations`.

The mapping is necessarily lossy, and it is worth being explicit about how:
this service generates from *attributes*, not from free text. The `prompt`
field is therefore appended as extra styling detail on top of a randomly
sampled person, and is also hashed into the seed so the same prompt keeps
returning the same face. If you want real control, use `POST /v1/avatars` -
that is the native API and it is better.
"""

from __future__ import annotations

import asyncio
import base64
import io
import time
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException
from PIL import Image
from pydantic import BaseModel, Field

from ppg.api.deps import AppState, get_state, require_api_key
from ppg.pipeline.worker import QueueFull
from ppg.schemas import AvatarRequest
from ppg.store.files import pick_size, variant_path

router = APIRouter(prefix="/v1", tags=["openai-compat"], dependencies=[Depends(require_api_key)])


class ImageGenerationRequest(BaseModel):
    prompt: str = Field(min_length=1, max_length=500)
    model: str | None = None
    n: int = Field(default=1, ge=1, le=10)
    size: str = "1024x1024"
    response_format: Literal["b64_json", "url"] = "b64_json"
    user: str | None = Field(
        default=None,
        description="Used as the deterministic seed key when present.",
    )
    quality: str | None = None
    style: str | None = None


class ImageData(BaseModel):
    b64_json: str | None = None
    url: str | None = None
    revised_prompt: str | None = None


class ImageGenerationResponse(BaseModel):
    created: int
    data: list[ImageData]


def _parse_size(raw: str, available: list[int]) -> int:
    try:
        width = int(raw.lower().split("x")[0])
    except (ValueError, IndexError) as exc:
        raise HTTPException(status_code=400, detail=f"Unparseable size {raw!r}.") from exc
    return pick_size(width, available)


@router.post(
    "/images/generations",
    response_model=ImageGenerationResponse,
    summary="OpenAI-compatible image generation",
)
async def images_generations(
    request: ImageGenerationRequest,
    state: AppState = Depends(get_state),
) -> ImageGenerationResponse:
    size = _parse_size(request.size, state.settings.sizes)

    avatar_requests = []
    for index in range(request.n):
        seed_key = f"{request.user}:{index}" if request.user else f"{request.prompt}:{index}"
        avatar_request = AvatarRequest(prompt_extra=request.prompt, seed=seed_key)
        state.service.precheck(avatar_request)
        avatar_requests.append(avatar_request)

    try:
        job = state.queue.submit(avatar_requests, "batch" if request.n > 1 else "single")
    except QueueFull as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    timeout = max(state.settings.default_wait, 30.0 * request.n)
    if not await state.queue.wait(job, timeout):
        raise HTTPException(
            status_code=504,
            detail="Generation did not finish in time. Use POST /v1/avatars with wait=0 "
            "and poll /v1/jobs/{id} for long batches.",
        )
    if not job.results:
        raise HTTPException(status_code=500, detail=job.error or "Generation failed.")

    data: list[ImageData] = []
    for result in job.results:
        if request.response_format == "url":
            data.append(
                ImageData(
                    url=f"/v1/avatars/{result.id}/image?size={size}", revised_prompt=result.prompt
                )
            )
            continue
        encoded = await asyncio.to_thread(_encode_png, state, result.id, result.sizes, size)
        data.append(ImageData(b64_json=encoded, revised_prompt=result.prompt))

    return ImageGenerationResponse(created=int(time.time()), data=data)


def _encode_png(state: AppState, avatar_id: str, sizes: list[int], requested: int) -> str:
    """Read the stored file and return it base64-encoded.

    Re-encodes from the PNG master so the response is a real PNG, which is what
    OpenAI clients expect, rather than the WebP we serve over HTTP.
    """
    resolved = pick_size(requested, sizes)
    path = variant_path(state.settings.outputs_dir, avatar_id, resolved, "png")
    if not path.is_file():
        raise HTTPException(status_code=500, detail=f"Missing image file for {avatar_id}.")
    with Image.open(path) as image:
        buffer = io.BytesIO()
        image.save(buffer, "PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")
