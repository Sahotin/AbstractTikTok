from __future__ import annotations

import asyncio
import time

import pytest

from analysis.domain import AnalysisJobItemStatus, AnalysisJobStatus
from analysis.llm import FakeLLMProvider, LLMConfig, LLMProviderError
from analysis.services import (
    AnalysisJobRunner,
    AnalysisJobService,
    BatchAnalysisConfig,
    CommentAnalysisService,
)


def _runtime(context, provider, *, max_retries: int = 0, window_seconds: float = 0.05):
    llm_config = LLMConfig(
        provider="fake",
        model=provider.model,
        timeout=1,
        max_retries=max_retries,
        retry_base_delay=0,
    )
    analysis = CommentAnalysisService(context.repository, provider, llm_config, max_batch_size=50)
    jobs = AnalysisJobService(context.repository, context.service, analysis)
    runner = AnalysisJobRunner(context.repository, analysis, rate_limit_window_seconds=window_seconds)
    return analysis, jobs, runner


async def _create(jobs, comments, *, concurrency=3, batch_size=10, rpm=None):
    return await jobs.create_job(
        limit=len(comments),
        config=BatchAnalysisConfig(
            batch_size=batch_size,
            max_concurrency=concurrency,
            max_requests_per_minute=rpm,
        ),
        comment_ids=[comment.id for comment in comments],
    )


@pytest.mark.asyncio
async def test_worker_analyzes_ten_comments(phase1b_context, job_comments) -> None:
    provider = FakeLLMProvider()
    _analysis, jobs, runner = _runtime(phase1b_context, provider)
    job = await _create(jobs, job_comments[:10])

    result = await runner.run(job.id)

    assert result.status == AnalysisJobStatus.COMPLETED
    assert result.completed_count == 10
    assert result.failed_count == 0
    assert provider.call_count == 10
    assert await phase1b_context.repository.count_comment_analyses() == 10


@pytest.mark.asyncio
async def test_worker_pool_never_exceeds_max_concurrency(phase1b_context, job_comments) -> None:
    provider = FakeLLMProvider(delays=[0.2] * 10)
    _analysis, jobs, runner = _runtime(phase1b_context, provider)
    job = await _create(jobs, job_comments[:10], concurrency=3, batch_size=10)

    await runner.run(job.id)

    assert 1 < provider.max_observed_concurrency <= 3


@pytest.mark.asyncio
async def test_cache_is_checked_before_provider(phase1b_context, job_comments) -> None:
    warm_provider = FakeLLMProvider()
    warm_analysis, _jobs, _runner = _runtime(phase1b_context, warm_provider)
    for comment in job_comments[:5]:
        await warm_analysis.analyze_comment(comment)

    provider = FakeLLMProvider()
    _analysis, jobs, runner = _runtime(phase1b_context, provider)
    job = await _create(jobs, job_comments[:10])
    result = await runner.run(job.id)

    assert result.completed_count == 5
    assert result.skipped_count == 5
    assert provider.call_count == 5


@pytest.mark.asyncio
async def test_two_comment_failures_do_not_fail_job(phase1b_context, job_comments) -> None:
    provider = FakeLLMProvider(failure_call_indices={2, 7})
    _analysis, jobs, runner = _runtime(phase1b_context, provider)
    job = await _create(jobs, job_comments[:10], concurrency=1)

    result = await runner.run(job.id)

    assert result.status == AnalysisJobStatus.COMPLETED
    assert result.completed_count == 8
    assert result.failed_count == 2
    assert result.pending_count == 0


@pytest.mark.asyncio
async def test_permanent_provider_rejection_fails_fast(phase1b_context, job_comments) -> None:
    provider = FakeLLMProvider(failure=LLMProviderError("Provider returned HTTP 402"), failures_before_success=20)
    _analysis, jobs, runner = _runtime(phase1b_context, provider)
    job = await _create(jobs, job_comments[:10], concurrency=2, batch_size=10)

    with pytest.raises(LLMProviderError, match="HTTP 402"):
        await runner.run(job.id)

    stored = await phase1b_context.repository.get_analysis_job(str(job.id))
    assert stored is not None
    assert stored.status == AnalysisJobStatus.FAILED
    assert stored.failed_count == 10
    assert stored.pending_count == 0
    assert provider.call_count <= 2


@pytest.mark.asyncio
async def test_interrupted_job_resumes_only_remaining_items(phase1b_context, job_comments) -> None:
    first_provider = FakeLLMProvider()
    first_analysis, jobs, _runner = _runtime(phase1b_context, first_provider)
    job = await _create(jobs, job_comments[:10])
    items = await phase1b_context.repository.list_job_items(str(job.id), limit=10)
    comments_by_id = {str(comment.id): comment for comment in job_comments[:10]}
    for item in items[:6]:
        await phase1b_context.repository.claim_job_item(str(item.id), job.created_at)
        await first_analysis.analyze_comment(comments_by_id[str(item.comment_id)])
        await phase1b_context.repository.finish_job_item(
            str(item.id),
            status=AnalysisJobItemStatus.COMPLETED,
            finished_at=job.created_at,
        )
    # Simulate a process crash after one worker claimed an item but before it
    # produced or persisted an analysis.
    await phase1b_context.repository.claim_job_item(str(items[6].id), job.created_at)
    await phase1b_context.repository.update_analysis_job_status(str(job.id), AnalysisJobStatus.INTERRUPTED)

    resume_provider = FakeLLMProvider()
    _analysis, _jobs, resume_runner = _runtime(phase1b_context, resume_provider)
    result = await resume_runner.run(job.id)

    assert result.completed_count == 10
    assert resume_provider.call_count == 4


@pytest.mark.asyncio
async def test_cancellation_stops_starting_new_requests(phase1b_context, job_comments) -> None:
    provider = FakeLLMProvider(delays=[0.05] * 20)
    _analysis, jobs, runner = _runtime(phase1b_context, provider)
    job = await _create(jobs, job_comments[:20], concurrency=2, batch_size=20)
    running = asyncio.create_task(runner.run(job.id))
    while provider.call_count < 2:
        await asyncio.sleep(0.001)
    cancelled = await jobs.cancel_job(job.id)
    calls_at_cancel = provider.call_count
    result = await running

    assert cancelled.status == AnalysisJobStatus.CANCELLED
    assert result.status == AnalysisJobStatus.CANCELLED
    assert provider.call_count == calls_at_cancel
    assert result.pending_count > 0


@pytest.mark.asyncio
async def test_rate_limit_is_independent_from_concurrency(phase1b_context, job_comments) -> None:
    provider = FakeLLMProvider()
    _analysis, jobs, runner = _runtime(phase1b_context, provider, window_seconds=0.05)
    job = await _create(jobs, job_comments[:4], concurrency=4, batch_size=4, rpm=2)

    started = time.perf_counter()
    result = await runner.run(job.id)
    elapsed = time.perf_counter() - started

    assert result.completed_count == 4
    assert elapsed >= 0.045
    assert provider.call_count == 4


@pytest.mark.asyncio
async def test_reexecuting_completed_job_is_idempotent(phase1b_context, job_comments) -> None:
    provider = FakeLLMProvider()
    _analysis, jobs, runner = _runtime(phase1b_context, provider)
    job = await _create(jobs, job_comments[:10])

    first = await runner.run(job.id)
    second = await runner.run(job.id)
    duplicate_job = await _create(jobs, job_comments[:10])
    duplicate_result = await runner.run(duplicate_job.id)

    assert first == second
    assert duplicate_result.skipped_count == 10
    assert provider.call_count == 10
    assert await phase1b_context.repository.count_comment_analyses() == 10


@pytest.mark.asyncio
async def test_same_cache_key_concurrency_calls_provider_once(phase1b_context, job_comments) -> None:
    provider = FakeLLMProvider(delays=[0.03])
    analysis, _jobs, _runner = _runtime(phase1b_context, provider)

    results = await asyncio.gather(
        *(analysis.analyze_comment(job_comments[0]) for _ in range(5))
    )

    assert provider.call_count == 1
    assert len({result.id for result in results}) == 1
    assert await phase1b_context.repository.count_comment_analyses() == 1
