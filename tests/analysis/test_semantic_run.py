from __future__ import annotations

import httpx
import pytest

from analysis.domain import SemanticRunStatus, SemanticScopeType
from analysis.embeddings import EmbeddingConfig, FakeEmbeddingProvider
from analysis.services import EmbeddingService, SemanticRunService
from api.main import app
from api.routers.analysis import get_semantic_repository, get_semantic_run_manager, get_semantic_run_service


def _service(phase1b_context, semantic_repository, **config_updates) -> SemanticRunService:
    values = {
        "provider": "fake",
        "model": "semantic-fake",
        "embedding_version": "semantic-v1",
        "batch_size": 10,
        "max_concurrency": 2,
    }
    values.update(config_updates)
    config = EmbeddingConfig(**values)
    provider = FakeEmbeddingProvider(dimension=8, model=config.model)
    return SemanticRunService(
        phase1b_context.repository,
        semantic_repository,
        EmbeddingService(semantic_repository, provider, config),
        config,
    )


@pytest.mark.asyncio
async def test_semantic_run_create_and_status_transition(phase1b_context, semantic_repository) -> None:
    service = _service(phase1b_context, semantic_repository)
    run = await service.create_run(platform="douyin", run_id=phase1b_context.run_id, limit=2)

    assert run.status == SemanticRunStatus.PENDING
    assert run.scope_type == SemanticScopeType.RUN
    assert run.embedding_config["provider"] == "fake"
    assert run.embedding_version == "semantic-v1"
    assert "api_key" not in run.embedding_config

    completed, report = await service.execute_run(run.id)
    assert completed.status == SemanticRunStatus.COMPLETED
    assert completed.started_at is not None and completed.finished_at is not None
    assert completed.embedded_comments == report.embedded_comments == 2
    assert completed.topic_count == completed.opinion_count == 0


@pytest.mark.asyncio
async def test_semantic_run_content_scope(phase1b_context, semantic_repository) -> None:
    content = (await phase1b_context.service.list_contents(limit=1)).items[0]
    run = await _service(phase1b_context, semantic_repository).create_run(
        platform="douyin",
        content_id=content.id,
        limit=1,
    )
    assert run.scope_type == SemanticScopeType.CONTENT
    assert run.content_id == content.id


@pytest.mark.asyncio
async def test_semantic_run_rejects_configuration_drift(phase1b_context, semantic_repository) -> None:
    original = _service(phase1b_context, semantic_repository)
    run = await original.create_run(platform="douyin", limit=1)
    changed = _service(phase1b_context, semantic_repository, embedding_version="semantic-v2")
    with pytest.raises(ValueError, match="configuration"):
        await changed.execute_run(run.id)
    stored = await semantic_repository.get_semantic_run(str(run.id))
    assert stored is not None and stored.status == SemanticRunStatus.FAILED


class _ManagerStub:
    def __init__(self):
        self.started = []

    def start(self, run_id):
        self.started.append(run_id)
        return True


@pytest.mark.asyncio
async def test_semantic_run_post_and_get_api(phase1b_context, semantic_repository) -> None:
    service = _service(phase1b_context, semantic_repository)
    manager = _ManagerStub()

    async def override_service():
        yield service

    app.dependency_overrides[get_semantic_run_service] = override_service
    app.dependency_overrides[get_semantic_repository] = lambda: semantic_repository
    app.dependency_overrides[get_semantic_run_manager] = lambda: manager
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            created = await client.post("/api/analysis/semantic-runs", json={"platform": "douyin", "limit": 1})
            fetched = await client.get(f"/api/analysis/semantic-runs/{created.json()['id']}")
    finally:
        app.dependency_overrides.clear()

    assert created.status_code == 202
    assert created.json()["status"] == "pending"
    assert len(manager.started) == 1
    assert fetched.status_code == 200
    assert fetched.json()["id"] == created.json()["id"]
