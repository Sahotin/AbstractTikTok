"""Persistent single-process batch runner with bounded workers and rate limiting."""

from __future__ import annotations

import asyncio
import logging
import time
from collections import deque
from typing import Optional
from uuid import UUID

from analysis.domain import AnalysisJob, AnalysisJobItem, AnalysisJobItemStatus, AnalysisJobStatus, NormalizedComment
from analysis.preprocessing import utc_now
from analysis.repositories import AnalysisRepository

from .comment_analysis_service import CommentAnalysisService


logger = logging.getLogger("analysis.job_runner")


class AsyncRateLimiter:
    """Sliding-window request limiter, independent from worker concurrency."""

    def __init__(self, max_requests: Optional[int], *, window_seconds: float = 60.0):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._timestamps: deque[float] = deque()
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        if self.max_requests is None:
            return
        while True:
            async with self._lock:
                now = time.monotonic()
                while self._timestamps and now - self._timestamps[0] >= self.window_seconds:
                    self._timestamps.popleft()
                if len(self._timestamps) < self.max_requests:
                    self._timestamps.append(now)
                    return
                wait_seconds = self.window_seconds - (now - self._timestamps[0])
            await asyncio.sleep(max(wait_seconds, 0.001))


class AnalysisJobRunner:
    def __init__(
        self,
        repository: AnalysisRepository,
        analysis_service: CommentAnalysisService,
        *,
        rate_limit_window_seconds: float = 60.0,
    ):
        self.repository = repository
        self.analysis_service = analysis_service
        self.rate_limit_window_seconds = rate_limit_window_seconds

    async def run(self, job_id: UUID | str) -> AnalysisJob:
        job_id_string = str(job_id)
        started = time.perf_counter()
        job = await self._load_and_validate_job(job_id_string)
        if job.status in {AnalysisJobStatus.COMPLETED, AnalysisJobStatus.CANCELLED}:
            return job
        try:
            await self.repository.reset_processing_job_items(job_id_string)
            job = await self.repository.update_analysis_job_status(
                job_id_string,
                AnalysisJobStatus.RUNNING,
                started_at=utc_now(),
            )
            if job is None:
                raise KeyError(f"Analysis job not found: {job_id}")
            limiter = AsyncRateLimiter(
                job.max_requests_per_minute,
                window_seconds=self.rate_limit_window_seconds,
            )
            queue: asyncio.Queue[tuple[AnalysisJobItem, NormalizedComment] | None] = asyncio.Queue(
                maxsize=max(job.batch_size, job.max_concurrency)
            )
            workers = [
                asyncio.create_task(self._worker(job_id_string, queue, limiter))
                for _ in range(job.max_concurrency)
            ]
            try:
                while True:
                    current = await self.repository.get_analysis_job(job_id_string)
                    if current is None:
                        raise KeyError(f"Analysis job not found: {job_id}")
                    if current.status == AnalysisJobStatus.CANCELLED:
                        break
                    items = await self.repository.list_job_items(
                        job_id_string,
                        statuses=[AnalysisJobItemStatus.PENDING],
                        limit=job.batch_size,
                    )
                    if not items:
                        break
                    comments = await self.repository.get_comments_by_ids(
                        [str(item.comment_id) for item in items]
                    )
                    comments_by_id = {str(comment.id): comment for comment in comments}
                    for item in items:
                        comment = comments_by_id.get(str(item.comment_id))
                        if comment is None:
                            await self._fail_missing_comment(item)
                            continue
                        await queue.put((item, comment))
                    await self._wait_for_batch(queue, workers)
            finally:
                for worker in workers:
                    worker.cancel()
                await asyncio.gather(*workers, return_exceptions=True)

            final_job = await self.repository.get_analysis_job(job_id_string)
            if final_job is None:
                raise KeyError(f"Analysis job not found: {job_id}")
            if final_job.status != AnalysisJobStatus.CANCELLED:
                final_job = await self.repository.update_analysis_job_status(
                    job_id_string,
                    AnalysisJobStatus.COMPLETED,
                    finished_at=utc_now(),
                )
            if final_job is None:
                raise RuntimeError("Job final status was not persisted")
            self._log_job(final_job, started)
            return final_job
        except Exception as exc:
            failed_job = await self.repository.update_analysis_job_status(
                job_id_string,
                AnalysisJobStatus.FAILED,
                finished_at=utc_now(),
                error_message=f"{type(exc).__name__}: {str(exc)[:500]}",
            )
            if failed_job is not None:
                self._log_job(failed_job, started)
            raise

    async def _worker(
        self,
        job_id: str,
        queue: asyncio.Queue[tuple[AnalysisJobItem, NormalizedComment] | None],
        limiter: AsyncRateLimiter,
    ) -> None:
        while True:
            payload = await queue.get()
            try:
                if payload is None:
                    return
                item, comment = payload
                current = await self.repository.get_analysis_job(job_id)
                if current is None or current.status == AnalysisJobStatus.CANCELLED:
                    continue
                cached = await self.analysis_service.get_cached_analysis(comment)
                if cached is None:
                    await limiter.acquire()
                    current = await self.repository.get_analysis_job(job_id)
                    if current is None or current.status == AnalysisJobStatus.CANCELLED:
                        continue
                claimed = await self.repository.claim_job_item(str(item.id), utc_now())
                if not claimed:
                    continue
                if cached is not None:
                    await self.repository.finish_job_item(
                        str(item.id),
                        status=AnalysisJobItemStatus.SKIPPED,
                        finished_at=utc_now(),
                    )
                    continue
                try:
                    cached_after_wait = await self.analysis_service.get_cached_analysis(comment)
                    if cached_after_wait is not None:
                        terminal_status = AnalysisJobItemStatus.SKIPPED
                    else:
                        await self.analysis_service.analyze_comment(comment)
                        terminal_status = AnalysisJobItemStatus.COMPLETED
                    await self.repository.finish_job_item(
                        str(item.id),
                        status=terminal_status,
                        finished_at=utc_now(),
                    )
                except Exception as exc:
                    await self.repository.finish_job_item(
                        str(item.id),
                        status=AnalysisJobItemStatus.FAILED,
                        finished_at=utc_now(),
                        error_type=type(exc).__name__,
                        error_message=str(exc)[:500],
                    )
            finally:
                queue.task_done()

    @staticmethod
    async def _wait_for_batch(
        queue: asyncio.Queue,
        workers: list[asyncio.Task],
    ) -> None:
        """Finish a batch or surface a crashed worker instead of hanging."""

        join_task = asyncio.create_task(queue.join())
        done, _pending = await asyncio.wait(
            [join_task, *workers],
            return_when=asyncio.FIRST_COMPLETED,
        )
        if join_task in done:
            await join_task
            return
        join_task.cancel()
        await asyncio.gather(join_task, return_exceptions=True)
        for worker in done:
            error = worker.exception()
            if error is not None:
                raise error
        raise RuntimeError("Analysis worker stopped before its queue batch completed")

    async def _load_and_validate_job(self, job_id: str) -> AnalysisJob:
        job = await self.repository.get_analysis_job(job_id)
        if job is None:
            raise KeyError(f"Analysis job not found: {job_id}")
        expected = (
            self.analysis_service.provider.provider_name,
            self.analysis_service.provider.model,
            self.analysis_service.prompt_version,
            self.analysis_service.analysis_version,
        )
        actual = (job.provider, job.model, job.prompt_version, job.analysis_version)
        if expected != actual:
            raise ValueError("Current LLM configuration does not match the persisted job version")
        return job

    async def _fail_missing_comment(self, item: AnalysisJobItem) -> None:
        if await self.repository.claim_job_item(str(item.id), utc_now()):
            await self.repository.finish_job_item(
                str(item.id),
                status=AnalysisJobItemStatus.FAILED,
                finished_at=utc_now(),
                error_type="MissingComment",
                error_message="Normalized comment no longer exists",
            )

    @staticmethod
    def _log_job(job: AnalysisJob, started: float) -> None:
        logger.info(
            "analysis_job job_id=%s status=%s total=%s completed=%s failed=%s skipped=%s elapsed_ms=%.2f estimated_tokens=%s estimated_cost=%s",
            job.id,
            job.status.value,
            job.total_comments,
            job.completed_count,
            job.failed_count,
            job.skipped_count,
            (time.perf_counter() - started) * 1000,
            job.estimated_input_tokens + job.estimated_output_tokens,
            job.estimated_cost,
        )
