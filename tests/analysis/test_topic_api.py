from __future__ import annotations

import httpx
import pytest

from analysis.domain import TopicClusterConfig
from analysis.services import TopicService
from api.main import app
from api.routers.analysis import get_semantic_repository

from .test_topic_service import _prepared_run


@pytest.mark.asyncio
async def test_topic_list_and_detail_are_bounded(phase1b_context, semantic_repository) -> None:
    run, _ = await _prepared_run(phase1b_context, semantic_repository)
    await TopicService(semantic_repository).cluster_run(
        run.id, TopicClusterConfig(distance_threshold=0.2, min_cluster_size=2)
    )
    app.dependency_overrides[get_semantic_repository] = lambda: semantic_repository
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            listing = await client.get("/api/analysis/topics", params={
                "semantic_run_id": str(run.id), "limit": 1,
            })
            topic_id = listing.json()["items"][0]["id"]
            detail = await client.get(
                f"/api/analysis/topics/{topic_id}", params={"representative_limit": 2}
            )
    finally:
        app.dependency_overrides.clear()

    assert listing.status_code == 200
    assert listing.json()["total"] == 2
    assert len(listing.json()["items"]) == 1
    assert detail.status_code == 200
    payload = detail.json()
    assert payload["algorithm"] == "agglomerative"
    assert payload["analysis_coverage"] == 1.0
    assert len(payload["representative_comments"]) <= 2
    assert "centroid" not in payload
    assert "members" not in payload


@pytest.mark.asyncio
async def test_topic_api_filters_and_missing_detail(phase1b_context, semantic_repository) -> None:
    run, _ = await _prepared_run(phase1b_context, semantic_repository)
    await TopicService(semantic_repository).cluster_run(
        run.id, TopicClusterConfig(distance_threshold=0.2, min_cluster_size=2)
    )
    app.dependency_overrides[get_semantic_repository] = lambda: semantic_repository
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            filtered = await client.get("/api/analysis/topics", params={"min_size": 999})
            missing = await client.get("/api/analysis/topics/00000000-0000-0000-0000-000000000000")
    finally:
        app.dependency_overrides.clear()
    assert filtered.status_code == 200 and filtered.json()["total"] == 0
    assert missing.status_code == 404
