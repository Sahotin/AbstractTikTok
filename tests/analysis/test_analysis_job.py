from __future__ import annotations

import pytest
import httpx

from analysis.domain import AnalysisJobItemStatus, AnalysisJobStatus
from analysis.llm import FakeLLMProvider, LLMConfig
from analysis.services import (
    AnalysisJobRunner,
    AnalysisJobService,
    BatchAnalysisConfig,
    CommentAnalysisService,
)
from api.main import app
from api.routers.analysis import (
    get_analysis_job_manager,
    get_analysis_job_service,
    get_analysis_repository,
)


def _llm_config() -> LLMConfig:
    return LLMConfig(
        provider="fake",
        model="fake-model-v1",
        timeout=1,
        max_retries=0,
        retry_base_delay=0,
    )


def _services(context, provider=None):
    provider = provider or FakeLLMProvider()
    analysis = CommentAnalysisService(context.repository, provider, _llm_config(), max_batch_size=50)
    jobs = AnalysisJobService(context.repository, context.service, analysis)
    runner = AnalysisJobRunner(context.repository, analysis, rate_limit_window_seconds=0.05)
    return provider, analysis, jobs, runner


@pytest.mark.asyncio
async def test_job_creation_persists_ten_pending_items(phase1b_context, job_comments) -> None:
    _provider, _analysis, jobs, _runner = _services(phase1b_context)

    job = await jobs.create_job(
        limit=10,
        config=BatchAnalysisConfig(batch_size=4, max_concurrency=3),
        comment_ids=[comment.id for comment in job_comments[:10]],
    )
    items = await phase1b_context.repository.list_job_items(str(job.id), limit=20)

    assert job.status == AnalysisJobStatus.PENDING
    assert job.total_comments == 10
    assert job.pending_count == 10
    assert len(items) == 10
    assert all(item.status == AnalysisJobItemStatus.PENDING for item in items)


@pytest.mark.asyncio
async def test_cost_estimate_is_null_without_prices(phase1b_context, job_comments) -> None:
    _provider, _analysis, jobs, _runner = _services(phase1b_context)

    job = await jobs.create_job(
        limit=10,
        config=BatchAnalysisConfig(batch_size=5, max_concurrency=2),
        comment_ids=[comment.id for comment in job_comments[:10]],
    )

    assert job.estimated_input_tokens > 0
    assert job.estimated_output_tokens == 1500
    assert job.estimated_cost is None


@pytest.mark.asyncio
async def test_retry_failed_creates_child_job(phase1b_context, job_comments) -> None:
    failing_provider = FakeLLMProvider(failure_call_indices={0, 1})
    _provider, _analysis, jobs, runner = _services(phase1b_context, failing_provider)
    parent = await jobs.create_job(
        limit=10,
        config=BatchAnalysisConfig(batch_size=10, max_concurrency=1),
        comment_ids=[comment.id for comment in job_comments[:10]],
    )
    parent = await runner.run(parent.id)
    assert parent.failed_count == 2

    healthy_provider = FakeLLMProvider()
    _provider, _analysis, retry_jobs, retry_runner = _services(phase1b_context, healthy_provider)
    child = await retry_jobs.retry_failed_job(parent.id)
    child = await retry_runner.run(child.id)

    assert child.parent_job_id == parent.id
    assert child.total_comments == 2
    assert child.completed_count == 2
    assert child.failed_count == 0
    assert healthy_provider.call_count == 2


@pytest.mark.asyncio
async def test_job_api_creates_returns_and_cancels_without_waiting(phase1b_context, job_comments) -> None:
    _provider, _analysis, jobs, _runner = _services(phase1b_context)

    class RecordingManager:
        def __init__(self):
            self.started = []

        def start(self, job_id):
            self.started.append(str(job_id))
            return True

    manager = RecordingManager()

    async def override_job_service():
        yield jobs

    async def override_repository():
        yield phase1b_context.repository

    app.dependency_overrides[get_analysis_job_service] = override_job_service
    app.dependency_overrides[get_analysis_repository] = override_repository
    app.dependency_overrides[get_analysis_job_manager] = lambda: manager
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(
                "/api/analysis/jobs",
                json={
                    "platform": "douyin",
                    "limit": 10,
                    "batch_size": 4,
                    "max_concurrency": 2,
                },
            )
            assert response.status_code == 202
            job_id = response.json()["id"]
            status_response = await client.get(f"/api/analysis/jobs/{job_id}")
            cancel_response = await client.post(f"/api/analysis/jobs/{job_id}/cancel")
    finally:
        app.dependency_overrides.clear()

    assert response.json()["status"] == "pending"
    assert response.json()["total_comments"] == 10
    assert manager.started == [job_id]
    assert status_response.status_code == 200
    assert cancel_response.json()["status"] == "cancelled"
