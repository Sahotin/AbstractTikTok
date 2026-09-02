from __future__ import annotations

from uuid import uuid4

import httpx
import pytest

from api.main import app
from api.routers.analysis import get_analysis_query_service


@pytest.fixture
def analysis_app(phase1b_context):
    async def override_query_service():
        yield phase1b_context.service

    app.dependency_overrides[get_analysis_query_service] = override_query_service
    try:
        yield app
    finally:
        app.dependency_overrides.clear()


@pytest.mark.asyncio
async def test_contents_endpoint_returns_bounded_page(analysis_app) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=analysis_app), base_url="http://test") as client:
        response = await client.get("/api/analysis/contents", params={"limit": 1})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert len(payload["items"]) == 1
    assert "raw_payload" not in payload["items"][0]


@pytest.mark.asyncio
async def test_comments_endpoint_supports_depth_filter(analysis_app) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=analysis_app), base_url="http://test") as client:
        response = await client.get("/api/analysis/comments", params={"depth": 1, "limit": 50})

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["depth"] == 1


@pytest.mark.asyncio
async def test_authors_endpoint_returns_authors(analysis_app) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=analysis_app), base_url="http://test") as client:
        response = await client.get("/api/analysis/authors", params={"platform": "douyin"})

    assert response.status_code == 200
    assert response.json()["total"] == 3


@pytest.mark.asyncio
async def test_statistics_endpoint_returns_sql_aggregates(analysis_app) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=analysis_app), base_url="http://test") as client:
        response = await client.get("/api/analysis/statistics")

    assert response.status_code == 200
    payload = response.json()
    assert payload["contents"]["content_count"] == 1
    assert payload["comments"]["comment_count"] == 2
    assert payload["comments"]["root_comment_count"] == 1
    assert payload["comments"]["reply_comment_count"] == 1
    assert payload["authors"]["author_count"] == 3


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("path", "params"),
    [
        ("/api/analysis/comments", {"limit": 0}),
        ("/api/analysis/comments", {"limit": 501}),
        ("/api/analysis/comments", {"depth": -1}),
        ("/api/analysis/contents", {"run_id": "not-a-uuid"}),
    ],
)
async def test_api_rejects_invalid_query_parameters(analysis_app, path, params) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=analysis_app), base_url="http://test") as client:
        response = await client.get(path, params=params)

    assert response.status_code == 422


@pytest.mark.asyncio
async def test_missing_resource_returns_404(analysis_app) -> None:
    async with httpx.AsyncClient(transport=httpx.ASGITransport(app=analysis_app), base_url="http://test") as client:
        response = await client.get(f"/api/analysis/contents/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Content not found"
