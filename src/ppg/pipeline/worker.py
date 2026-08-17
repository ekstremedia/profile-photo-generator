"""Single-consumer job queue.

There is exactly one GPU, and a diffusion model saturates it, so running two
generations at once makes both slower and risks an out-of-memory error. The
queue therefore has one worker on purpose - it is a serialisation point, not a
scalability bottleneck waiting to be widened.

Jobs live in memory. They are ephemeral progress trackers; the avatars they
produce are in SQLite and on disk, so a restart loses the progress bar, not
any work.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Literal

from ppg.safety import SafetyError
from ppg.schemas import AvatarRequest, AvatarResult, JobInfo

logger = logging.getLogger(__name__)

# How many recent generations feed the ETA estimate.
_TIMING_WINDOW = 20


@dataclass
class Job:
    id: str
    kind: Literal["single", "batch"]
    requests: list[AvatarRequest]
    status: Literal["queued", "running", "done", "failed"] = "queued"
    results: list[AvatarResult] = field(default_factory=list)
    error: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    finished_at: datetime | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def total(self) -> int:
        return len(self.requests)

    @property
    def completed(self) -> int:
        return len(self.results)


class QueueFull(RuntimeError):
    """The queue is at capacity. Surfaces as HTTP 503."""


class JobQueue:
    def __init__(self, service, max_size: int = 128) -> None:
        self.service = service
        self.max_size = max_size
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._jobs: dict[str, Job] = {}
        self._order: deque[str] = deque()
        self._timings: deque[float] = deque(maxlen=_TIMING_WINDOW)
        self._task: asyncio.Task | None = None
        self._current: str | None = None

    # -- lifecycle -------------------------------------------------------
    async def start(self) -> None:
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run(), name="ppg-worker")

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await self._task
        self._task = None

    # -- submission ------------------------------------------------------
    def submit(self, requests: list[AvatarRequest], kind: Literal["single", "batch"]) -> Job:
        if self.depth + len(requests) > self.max_size:
            raise QueueFull(
                f"Queue is full ({self.depth} images pending, limit {self.max_size}). "
                "Retry shortly or raise PPG_MAX_QUEUE."
            )
        job = Job(id=uuid.uuid4().hex[:16], kind=kind, requests=requests)
        self._jobs[job.id] = job
        self._order.append(job.id)
        self._queue.put_nowait(job.id)
        return job

    async def wait(self, job: Job, timeout: float) -> bool:
        """Block until the job finishes. Returns False on timeout."""
        if timeout <= 0:
            return job.done.is_set()
        try:
            await asyncio.wait_for(job.done.wait(), timeout=timeout)
            return True
        except TimeoutError:
            return False

    # -- inspection ------------------------------------------------------
    def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    @property
    def depth(self) -> int:
        """Images still to render, including the one in flight."""
        return sum(
            job.total - job.completed
            for job in self._jobs.values()
            if job.status in ("queued", "running")
        )

    @property
    def average_seconds(self) -> float | None:
        if not self._timings:
            return None
        return sum(self._timings) / len(self._timings)

    def position(self, job: Job) -> int | None:
        """How many jobs are ahead of this one. 0 means it is next up."""
        if job.status != "queued":
            return None
        pending = [jid for jid in self._order if self._jobs[jid].status == "queued"]
        try:
            return pending.index(job.id)
        except ValueError:
            return None

    def info(self, job: Job) -> JobInfo:
        eta: float | None = None
        average = self.average_seconds
        if average is not None and job.status in ("queued", "running"):
            ahead = sum(
                other.total - other.completed
                for other in self._jobs.values()
                if other.status == "queued" and other.created_at < job.created_at
            )
            running = sum(
                other.total - other.completed
                for other in self._jobs.values()
                if other.status == "running" and other.id != job.id
            )
            eta = round(average * (ahead + running + job.total - job.completed), 1)

        return JobInfo(
            id=job.id,
            status=job.status,
            kind=job.kind,
            total=job.total,
            completed=job.completed,
            position=self.position(job),
            eta_seconds=eta,
            avatar_ids=[r.id for r in job.results],
            error=job.error,
            created_at=job.created_at,
            finished_at=job.finished_at,
        )

    def prune(self, keep: int = 500) -> None:
        """Drop the oldest finished jobs so a long-running server does not grow."""
        while len(self._order) > keep:
            job_id = self._order.popleft()
            job = self._jobs.get(job_id)
            if job and job.status in ("done", "failed"):
                self._jobs.pop(job_id, None)
            elif job:
                self._order.append(job_id)
                return

    # -- the worker ------------------------------------------------------
    async def _run(self) -> None:
        logger.info("Worker started")
        while True:
            job_id = await self._queue.get()
            job = self._jobs.get(job_id)
            if job is None:
                continue
            self._current = job_id
            job.status = "running"
            try:
                await self._process(job)
                job.status = "failed" if job.error and not job.results else "done"
            except asyncio.CancelledError:
                job.status = "failed"
                job.error = "Server shutting down."
                job.done.set()
                raise
            except Exception as exc:
                logger.exception("Job %s failed", job.id)
                job.status = "failed"
                job.error = str(exc)
            finally:
                self._current = None
                job.finished_at = datetime.now(UTC)
                job.done.set()
                self._queue.task_done()
                self.prune()

    async def _process(self, job: Job) -> None:
        errors: list[str] = []
        for request in job.requests:
            loop = asyncio.get_running_loop()
            started = loop.time()
            try:
                result = await self.service.generate(request)
            except SafetyError as exc:
                # A refused request is a client error, not a worker failure.
                errors.append(str(exc))
                continue
            except Exception as exc:
                logger.exception("Generation failed within job %s", job.id)
                errors.append(str(exc))
                continue
            if not result.cached:
                self._timings.append(loop.time() - started)
            job.results.append(result)
        if errors:
            job.error = "; ".join(errors[:5])
