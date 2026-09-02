from __future__ import annotations

import httpx
import pytest

from analysis.domain import SemanticSearchQuery
from analysis.embeddings import EmbeddingConfig, FakeEmbeddingProvider
from analysis.services import EmbeddingService, SemanticRunService, SemanticSearchService
from api.main import app
from api.routers.analysis import get_semantic_search_service


async def _search_context(phase1b_context, semantic_repository):
    comments = (await phase1b_context.service.list_comments(limit=10)).items
    vectors = {
        comments[0].text: [1.0, 0.0],
        comments[1].text: [0.0, 1.0],
        "价格太贵": [1.0, 0.0],
    }
    config = EmbeddingConfig(provider="fake", model="search-v1", embedding_version="search-v1")
    provider = FakeEmbeddingProvider(vectors, dimension=2, model="search-v1")
    run_service = SemanticRunService(
        phase1b_context.repository,
        semantic_repository,
        EmbeddingService(semantic_repository, provider, config),
        config,
    )
    run = await run_service.create_run(platform="douyin", limit=len(comments))
    run, _ = await run_service.execute_run(run.id)
    return comments, run, SemanticSearchService(semantic_repository, provider)


@pytest.mark.asyncio
async def test_semantic_search_ranks_fixed_snapshot(phase1b_context, semantic_repository) -> None:
    comments, run, service = await _search_context(phase1b_context, semantic_repository)
    result = await service.search(SemanticSearchQuery(
        semantic_run_id=run.id,
        query="价格太贵",
        top_k=2,
    ))

    assert result.searched_comments == 2
    assert result.hits[0].comment_id == comments[0].id
    assert result.hits[0].similarity == pytest.approx(1.0)
    assert result.embedding_version == "search-v1"


@pytest.mark.asyncio
async def test_semantic_search_rejects_provider_mismatch(phase1b_context, semantic_repository) -> None:
    _comments, run, _service = await _search_context(phase1b_context, semantic_repository)
    mismatch = SemanticSearchService(
        semantic_repository, FakeEmbeddingProvider(dimension=2, model="other-model")
    )
    with pytest.raises(ValueError, match="does not match"):
        await mismatch.search(SemanticSearchQuery(semantic_run_id=run.id, query="test"))


@pytest.mark.asyncio
async def test_semantic_search_api(phase1b_context, semantic_repository) -> None:
    _comments, run, service = await _search_context(phase1b_context, semantic_repository)

    async def override_search_service():
        yield service

    app.dependency_overrides[get_semantic_search_service] = override_search_service
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.post("/api/analysis/semantic-search", json={
                "semantic_run_id": str(run.id),
                "query": "价格太贵",
                "top_k": 1,
            })
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert len(response.json()["hits"]) == 1
