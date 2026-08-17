"""Avatar creation and retrieval."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Query, Response, status
from fastapi.responses import FileResponse

from ppg.api.deps import AppState, get_state, require_api_key
from ppg.pipeline.worker import Job, QueueFull
from ppg.safety import SafetyError
from ppg.schemas import AvatarRequest, AvatarResult, BatchRequest, JobInfo
from ppg.store.files import pick_size, variant_path

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/v1", tags=["avatars"], dependencies=[Depends(require_api_key)])

# Content-addressed files never change, so they can be cached forever.
IMMUTABLE = "public, max-age=31536000, immutable"


def _submit(state: AppState, requests: list[AvatarRequest], kind: str) -> Job:
    try:
        return state.queue.submit(requests, kind)  # type: ignore[arg-type]
    except QueueFull as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=str(exc)
        ) from exc


def _first_result(job: Job) -> AvatarResult:
    if job.results:
        return job.results[0]
    detail = job.error or "Generation failed for an unknown reason."
    raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=detail)


# ---------------------------------------------------------------------------
# Creation
# ---------------------------------------------------------------------------


@router.post(
    "/avatars",
    response_model=None,
    status_code=status.HTTP_200_OK,
    summary="Generate an avatar",
)
async def create_avatar(
    request: AvatarRequest,
    response: Response,
    wait: float | None = Query(
        default=None,
        ge=0,
        le=600,
        description="Seconds to block for the result. 0 returns a job immediately.",
    ),
    state: AppState = Depends(get_state),
) -> AvatarResult | JobInfo:
    """Create one avatar.

    Blocks for the result by default, because generation takes a few seconds
    and a blocking call is far simpler to consume. Pass `wait=0` for a job id
    instead, or a shorter timeout if you would rather poll.
    """
    state.service.precheck(request)
    job = _submit(state, [request], "single")
    timeout = state.settings.default_wait if wait is None else wait

    if await state.queue.wait(job, timeout):
        result = _first_result(job)
        response.headers["X-PPG-Cache"] = "hit" if result.cached else "miss"
        response.headers["X-PPG-Composer"] = result.composer
        return result

    response.status_code = status.HTTP_202_ACCEPTED
    return state.queue.info(job)


@router.post(
    "/avatars/batch",
    response_model=JobInfo,
    status_code=status.HTTP_202_ACCEPTED,
    summary="Generate many avatars",
)
async def create_batch(
    request: BatchRequest,
    wait: float = Query(
        default=0, ge=0, le=3600, description="Seconds to block. 0 returns at once."
    ),
    state: AppState = Depends(get_state),
) -> JobInfo:
    """Queue `n` avatars.

    With `diversity="even"` the batch is spread across sex, age and ancestry
    rather than sampled independently, and combinations already in the database
    are re-rolled - so a batch of 50 looks like 50 different people, and a
    second batch does not repeat the first.
    """
    state.service.precheck(request.overrides)
    requests = state.service.plan_batch(request)
    job = _submit(state, requests, "batch")
    if wait:
        await state.queue.wait(job, wait)
    return state.queue.info(job)


# ---------------------------------------------------------------------------
# Deterministic lookup
#
# Declared before /avatars/{avatar_id} so the literal path segment wins.
# ---------------------------------------------------------------------------


@router.get(
    "/avatars/by-seed/{key:path}",
    response_class=FileResponse,
    summary="Deterministic avatar for any key",
)
async def avatar_by_seed(
    key: str,
    size: int | None = Query(default=None),
    format: str = Query(default="webp", pattern="^(webp|png)$"),
    state: AppState = Depends(get_state),
) -> FileResponse:
    """Return the avatar for `key`, generating it on first request.

    The key is hashed to a seed, so the same key always yields the same face -
    forever, across restarts and machines. That makes this usable directly in
    a template:

        <img src="{{ config('ppg.url') }}/v1/avatars/by-seed/{{ md5($user->email) }}?size=256">

    First call for a given key blocks while the image renders; every later call
    is a static file read.
    """
    if not key.strip():
        raise HTTPException(status_code=400, detail="A non-empty key is required.")

    record = state.db.find_by_seed_key(key)
    if record is None:
        request = AvatarRequest(seed=key)
        state.service.precheck(request)
        job = _submit(state, [request], "single")
        if not await state.queue.wait(job, state.settings.default_wait):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Still generating. Retry shortly.",
                headers={"Retry-After": "10"},
            )
        result = _first_result(job)
        avatar_id, sizes = result.id, result.sizes
    else:
        avatar_id, sizes = record["id"], record["sizes"]

    return _serve(state, avatar_id, sizes, size, format)


# ---------------------------------------------------------------------------
# Retrieval
# ---------------------------------------------------------------------------


@router.get("/avatars", response_model=list[AvatarResult], summary="Recent avatars")
async def list_avatars(
    limit: int = Query(default=60, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    state: AppState = Depends(get_state),
) -> list[AvatarResult]:
    return state.service.recent(limit=limit, offset=offset)


@router.get("/avatars/{avatar_id}", response_model=AvatarResult, summary="Avatar metadata")
async def get_avatar(avatar_id: str, state: AppState = Depends(get_state)) -> AvatarResult:
    result = state.service.get(avatar_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"No avatar with id {avatar_id}.")
    return result


@router.get(
    "/avatars/{avatar_id}/image",
    response_class=FileResponse,
    summary="Avatar image",
)
async def get_avatar_image(
    avatar_id: str,
    size: int | None = Query(default=None, description="Nearest available size is used."),
    format: str = Query(default="webp", pattern="^(webp|png)$"),
    state: AppState = Depends(get_state),
) -> FileResponse:
    record = state.db.get_avatar(avatar_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No avatar with id {avatar_id}.")
    return _serve(state, avatar_id, record["sizes"], size, format)


@router.delete("/avatars/{avatar_id}", status_code=204, summary="Delete an avatar")
async def delete_avatar(avatar_id: str, state: AppState = Depends(get_state)) -> Response:
    import shutil

    from ppg.store.files import avatar_dir

    if not state.db.delete_avatar(avatar_id):
        raise HTTPException(status_code=404, detail=f"No avatar with id {avatar_id}.")
    shutil.rmtree(avatar_dir(state.settings.outputs_dir, avatar_id), ignore_errors=True)
    return Response(status_code=204)


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------


@router.get("/jobs/{job_id}", response_model=JobInfo, summary="Job status")
async def get_job(job_id: str, state: AppState = Depends(get_state)) -> JobInfo:
    job = state.queue.get(job_id)
    if job is None:
        raise HTTPException(
            status_code=404,
            detail=f"No job {job_id}. Jobs are in-memory and do not survive a restart; "
            "generated avatars do.",
        )
    return state.queue.info(job)


@router.get("/jobs/{job_id}/results", response_model=list[AvatarResult], summary="Job results")
async def get_job_results(job_id: str, state: AppState = Depends(get_state)) -> list[AvatarResult]:
    job = state.queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"No job {job_id}.")
    return job.results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serve(
    state: AppState,
    avatar_id: str,
    sizes: list[int],
    requested: int | None,
    fmt: str,
) -> FileResponse:
    try:
        resolved = pick_size(requested, sizes)
    except ValueError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    path = variant_path(state.settings.outputs_dir, avatar_id, resolved, fmt)
    if not path.is_file():
        raise HTTPException(
            status_code=404,
            detail=f"Avatar {avatar_id} has no {resolved}px {fmt} file on disk.",
        )
    return FileResponse(
        path,
        media_type="image/webp" if fmt == "webp" else "image/png",
        headers={
            "Cache-Control": IMMUTABLE,
            "ETag": f'"{avatar_id}-{resolved}-{fmt}"',
            "X-PPG-Size": str(resolved),
        },
    )


__all__ = ["SafetyError", "router"]
