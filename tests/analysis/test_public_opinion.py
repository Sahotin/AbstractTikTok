from __future__ import annotations

from datetime import datetime, timedelta, timezone
from uuid import NAMESPACE_URL, uuid4, uuid5

import httpx
import pytest

from analysis.domain import (
    CollectionRun,
    CollectionRunStatus,
    CommentAnalysis,
    NormalizedComment,
    NormalizedContent,
    Platform,
    PublicOpinionQuery,
    TrendQuery,
)
from analysis.preprocessing import compute_text_hash, utc_now
from analysis.services import PublicOpinionService, TrendAnalysisService
from api.main import app
from api.routers.analysis import get_public_opinion_service


def _analysis(comment_id, index: int, *, sentiment: str, emotion: str, stance: str, risk: str, topics, keywords, reasons):
    identity = f"public-opinion|{comment_id}|{index}"
    return CommentAnalysis(
        id=uuid5(NAMESPACE_URL, identity),
        comment_id=comment_id,
        sentiment=sentiment,
        sentiment_score={"positive": 0.8, "negative": -0.7, "neutral": 0.0}[sentiment],
        emotion=emotion,
        topics=topics,
        stance=stance,
        risk_level=risk,
        risk_reasons=reasons,
        keywords=keywords,
        summary=f"summary {index}",
        model="fake-model-v1",
        provider="fake",
        prompt_version="comment_analysis_v1",
        analysis_version="schema-v1",
        input_hash=compute_text_hash(identity),
        created_at=utc_now(),
    )


async def _seed_primary_analyses(context):
    run_id = uuid4()
    content_id = uuid4()
    now = utc_now()
    await context.repository.upsert_collection_run(
        CollectionRun(
            id=run_id,
            platform=Platform.DOUYIN,
            crawler_type="test",
            status=CollectionRunStatus.COMPLETED,
            started_at=now,
            finished_at=now,
        )
    )
    await context.repository.upsert_contents([
        NormalizedContent(
            id=content_id,
            run_id=run_id,
            platform=Platform.DOUYIN,
            native_content_id="opinion-primary-content",
            content_type="video",
            raw_payload={},
            collected_at=now,
        )
    ])
    comments = [
        NormalizedComment(
            id=uuid4(),
            run_id=run_id,
            platform=Platform.DOUYIN,
            native_comment_id=f"opinion-primary-{index}",
            content_id=content_id,
            native_content_id="opinion-primary-content",
            text=f"public opinion test {index}",
            published_at=datetime(2026, 6, 1, tzinfo=timezone.utc) + timedelta(days=index % 3),
            text_hash=compute_text_hash(f"public opinion test {index}"),
            raw_payload={},
            collected_at=now,
        )
        for index in range(10)
    ]
    await context.repository.upsert_comments(comments)
    values = [
        ("positive", "joy", "support", "low", ["price", "quality"], ["fast", "good"], []),
        ("positive", "joy", "support", "high", ["price"], ["fast"], ["safety"]),
        ("positive", "surprise", "support", "low", ["service"], ["fast"], []),
        ("positive", "joy", "neutral", "medium", ["shipping"], ["delivery"], ["delay"]),
        ("positive", "joy", "support", "low", ["value"], ["good"], []),
        ("negative", "anger", "oppose", "high", ["quality"], ["bad"], ["quality issue"]),
        ("negative", "anger", "oppose", "medium", ["service"], ["slow"], ["delay"]),
        ("negative", "sadness", "oppose", "low", ["refund"], ["bad"], []),
        ("neutral", "neutral", "neutral", "low", ["details"], ["question"], []),
        ("neutral", "neutral", "unclear", "low", ["details"], ["question"], []),
    ]
    for index, (sentiment, emotion, stance, risk, topics, keywords, reasons) in enumerate(values):
        await context.repository.save_analysis(
            _analysis(
                comments[index].id,
                index,
                sentiment=sentiment,
                emotion=emotion,
                stance=stance,
                risk=risk,
                topics=topics,
                keywords=keywords,
                reasons=reasons,
            )
        )
    return comments


async def _seed_partial_scope(context):
    run_id = uuid4()
    content_id = uuid4()
    now = utc_now()
    await context.repository.upsert_collection_run(
        CollectionRun(
            id=run_id,
            platform=Platform.DOUYIN,
            crawler_type="test",
            status=CollectionRunStatus.COMPLETED,
            started_at=now,
            finished_at=now,
        )
    )
    await context.repository.upsert_contents([
        NormalizedContent(
            id=content_id,
            run_id=run_id,
            platform=Platform.DOUYIN,
            native_content_id="opinion-partial-content",
            content_type="video",
            raw_payload={},
            collected_at=now,
        )
    ])
    comments = [
        NormalizedComment(
            id=uuid4(),
            run_id=run_id,
            platform=Platform.DOUYIN,
            native_comment_id=f"opinion-partial-{index}",
            content_id=content_id,
            native_content_id="opinion-partial-content",
            text=f"partial comment {index}",
            published_at=datetime(2026, 7, 1, tzinfo=timezone.utc),
            text_hash=compute_text_hash(f"partial comment {index}"),
            raw_payload={},
            collected_at=now,
        )
        for index in range(10)
    ]
    await context.repository.upsert_comments(comments)
    for index, comment in enumerate(comments[:8]):
        await context.repository.save_analysis(
            _analysis(
                comment.id,
                100 + index,
                sentiment="positive",
                emotion="joy",
                stance="support",
                risk="low",
                topics=["partial"],
                keywords=["partial"],
                reasons=[],
            )
        )
    return run_id, content_id


@pytest.mark.asyncio
async def test_public_opinion_aggregates_structured_results(phase1b_context) -> None:
    comments = await _seed_primary_analyses(phase1b_context)
    service = PublicOpinionService(phase1b_context.repository)

    summary = await service.get_public_opinion(PublicOpinionQuery(content_id=comments[0].content_id, top_k=20))

    assert summary.total_comments == 10
    assert summary.analyzed_comments == 10
    assert summary.analysis_coverage == 1
    assert summary.sentiment_distribution["positive"].count == 5
    assert summary.sentiment_distribution["positive"].percentage == 50.0
    assert summary.sentiment_distribution["negative"].count == 3
    assert summary.sentiment_distribution["neutral"].count == 2
    assert sum(item.count for item in summary.sentiment_distribution.values()) == 10
    assert summary.emotion_distribution["joy"].count == 4
    assert summary.stance_target is None
    assert summary.stance_distribution == {}
    assert summary.risk_distribution["high"].count == 2
    assert summary.high_risk_count == 2
    assert summary.ai_analysis_coverage == 1
    assert summary.collection_coverage == "unknown"
    assert summary.risk_explanation.risk_level == "low"
    assert summary.risk_explanation.risk_score == 25
    assert {item.key for item in summary.risk_explanation.components} >= {
        "negative_ratio", "high_risk_ratio", "risk_topic_concentration"
    }
    assert summary.risk_explanation.representative_evidence
    assert summary.conclusion.startswith("基于当前采集的 10 条评论")
    assert summary.sentiment_score_statistics.minimum == -0.7
    assert summary.sentiment_score_statistics.maximum == 0.8
    assert summary.sentiment_score_statistics.mean == pytest.approx(0.19, abs=0.0001)
    assert [(item.value, item.count) for item in summary.topics[:2]] == [("details", 2), ("price", 2)]
    assert summary.topic_sentiment_distribution["price"] == {"positive": 2}
    assert summary.topic_risk_distribution["quality"] == {"high": 1, "low": 1}
    assert [(item.value, item.count) for item in summary.risk_reason_frequency] == [
        ("delay", 2),
        ("quality issue", 1),
        ("safety", 1),
    ]
    assert [(item.date.isoformat(), item.count) for item in summary.comment_volume_by_date] == [
        ("2026-06-01", 4),
        ("2026-06-02", 3),
        ("2026-06-03", 3),
    ]


@pytest.mark.asyncio
async def test_trend_uses_latest_analysis_sql_buckets(phase1b_context) -> None:
    comments = await _seed_primary_analyses(phase1b_context)
    summary = await TrendAnalysisService(phase1b_context.repository).get_trend(
        TrendQuery(content_id=comments[0].content_id, bucket="day", min_bucket_comments=1)
    )

    assert [point.total_comments for point in summary.points] == [4, 3, 3]
    assert [point.analyzed_comments for point in summary.points] == [4, 3, 3]
    assert summary.points[0].negative_count == 1
    assert summary.points[0].medium_risk_count == 2
    assert summary.analysis_coverage == 1
    assert summary.data_quality == "sufficient"


@pytest.mark.asyncio
async def test_high_risk_comment_query_returns_latest_analysis(phase1b_context) -> None:
    comments = await _seed_primary_analyses(phase1b_context)
    rows = await phase1b_context.repository.list_high_risk_comments(
        content_id=str(comments[0].content_id), limit=10
    )

    assert len(rows) == 2
    assert all(row.risk_level == "high" for row in rows)
    assert {row.summary for row in rows} == {"summary 1", "summary 5"}


@pytest.mark.asyncio
async def test_public_opinion_scope_coverage_and_empty_handling(phase1b_context) -> None:
    service = PublicOpinionService(phase1b_context.repository)
    empty = await service.get_public_opinion(PublicOpinionQuery(platform=Platform.DOUYIN))
    assert empty.total_comments == 2
    assert empty.analyzed_comments == 0
    assert empty.analysis_status == "no_analyses"
    assert empty.sentiment_distribution == {}
    assert empty.analysis_coverage == 0

    await _seed_primary_analyses(phase1b_context)
    partial_run_id, partial_content_id = await _seed_partial_scope(phase1b_context)
    all_scope = await service.get_public_opinion(PublicOpinionQuery())
    run_scope = await service.get_public_opinion(PublicOpinionQuery(run_id=partial_run_id))
    content_scope = await service.get_public_opinion(PublicOpinionQuery(content_id=partial_content_id))

    assert all_scope.total_comments == 22
    assert run_scope.total_comments == content_scope.total_comments == 10
    assert run_scope.analyzed_comments == content_scope.analyzed_comments == 8
    assert run_scope.analysis_coverage == content_scope.analysis_coverage == 0.8
    assert run_scope.analysis_status == "partial"


@pytest.mark.asyncio
async def test_public_opinion_api_returns_aggregate_only(phase1b_context) -> None:
    comments = await _seed_primary_analyses(phase1b_context)
    service = PublicOpinionService(phase1b_context.repository)

    async def override_public_opinion_service():
        yield service

    app.dependency_overrides[get_public_opinion_service] = override_public_opinion_service
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get(
                "/api/analysis/public-opinion",
                params={"content_id": str(comments[0].content_id), "top_k": 5},
            )
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["analyzed_comments"] == 10
    assert len(payload["topics"]) == 5
    assert "comment_text" not in payload
    assert "raw_payload" not in payload
