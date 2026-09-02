from __future__ import annotations

from datetime import datetime, timezone

import httpx
import pytest

from analysis.domain import TrendQuery
from analysis.repositories.sqlalchemy import TrendBucketAggregate
from analysis.services import TrendAnalysisService
from api.main import app
from api.routers.analysis import get_trend_service


class _TrendRepository:
    def __init__(self, rows):
        self.rows = rows
        self.calls = []

    async def get_trend_buckets(self, **kwargs):
        self.calls.append(kwargs)
        return self.rows


def _row(day: int, total: int, *, analyzed: int, negative=0, medium=0, high=0):
    return TrendBucketAggregate(
        bucket_start=f"2026-06-{day:02d}T00:00:00",
        total_comments=total,
        analyzed_comments=analyzed,
        positive_count=max(analyzed - negative, 0),
        neutral_count=0,
        negative_count=negative,
        low_risk_count=max(analyzed - medium - high, 0),
        medium_risk_count=medium,
        high_risk_count=high,
        mean_sentiment_score=0.0 if analyzed else None,
    )


@pytest.mark.asyncio
async def test_trend_scores_and_detects_bounded_inflections() -> None:
    repository = _TrendRepository([
        _row(1, 10, analyzed=10),
        _row(2, 30, analyzed=30, negative=20, high=10),
        _row(3, 10, analyzed=10),
    ])
    result = await TrendAnalysisService(repository).get_trend(
        TrendQuery(bucket="day", min_bucket_comments=3)
    )

    assert result.total_comments == 50
    assert result.analyzed_comments == 50
    assert result.data_quality == "sufficient"
    assert result.points[1].risk_score == 45.0
    assert [item.kind for item in result.inflections] == ["risk_spike", "recovery"]
    assert result.peak_risk_at == datetime(2026, 6, 2, tzinfo=timezone.utc)
    assert result.overall_risk_score == 27.0


@pytest.mark.asyncio
async def test_trend_does_not_compare_sparse_nonconsecutive_buckets() -> None:
    repository = _TrendRepository([
        _row(1, 3, analyzed=0),
        _row(3, 30, analyzed=0),
    ])
    result = await TrendAnalysisService(repository).get_trend(
        TrendQuery(bucket="day", min_bucket_comments=3)
    )
    assert result.inflections == []


@pytest.mark.asyncio
async def test_trend_reports_partial_and_no_analysis_data() -> None:
    partial = await TrendAnalysisService(_TrendRepository([
        _row(1, 10, analyzed=2, negative=1, high=1),
    ])).get_trend(TrendQuery())
    empty_analysis = await TrendAnalysisService(_TrendRepository([
        _row(1, 5, analyzed=0),
    ])).get_trend(TrendQuery())

    assert partial.data_quality == "partial"
    assert partial.analysis_coverage == 0.2
    assert empty_analysis.data_quality == "no_analyses"
    assert empty_analysis.peak_risk_at is None


@pytest.mark.asyncio
async def test_trend_api_is_read_only_and_validates_range() -> None:
    service = TrendAnalysisService(_TrendRepository([
        _row(1, 10, analyzed=10, negative=2, high=1),
    ]))

    async def override_trend_service():
        yield service

    app.dependency_overrides[get_trend_service] = override_trend_service
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://test"
        ) as client:
            response = await client.get("/api/analysis/trends", params={"bucket": "day"})
            invalid = await client.get(
                "/api/analysis/trends",
                params={
                    "start_at": "2026-06-02T00:00:00Z",
                    "end_at": "2026-06-01T00:00:00Z",
                },
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["points"][0]["negative_count"] == 2
    assert invalid.status_code == 422
