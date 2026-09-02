from __future__ import annotations

import importlib
from uuid import uuid4

import httpx
import pytest
from typer.testing import CliRunner

from analysis.cli import app
from analysis.evaluation import (
    CommentQualityEvaluation,
    CommentQualityEvaluator,
    deterministic_stratified_sample,
    add_cost_range,
    write_comment_quality_report,
)
from analysis.llm import FakeLLMProvider, LLMConfig
from analysis.services import CommentAnalysisService
from api.main import app as api_app


@pytest.mark.asyncio
async def test_stratified_sample_is_deterministic_and_bounded(phase1b_context) -> None:
    source = (await phase1b_context.service.list_comments(limit=1)).items[0]
    comments = [
        source.model_copy(update={
            "id": uuid4(),
            "native_comment_id": f"quality-{index}",
            "text": text,
            "text_hash": f"{index:064x}",
            "depth": index % 2,
        })
        for index, text in enumerate((["支持", "123", "[笑哭]", "这个观点很有启发"] * 10))
    ]

    first = deterministic_stratified_sample(comments, limit=20, seed="stable")
    second = deterministic_stratified_sample(list(reversed(comments)), limit=20, seed="stable")

    assert [item.comment_id for item in first] == [item.comment_id for item in second]
    assert len(first) == 20
    assert any(item.quality_eligible for item in first)
    assert any(not item.quality_eligible for item in first)


@pytest.mark.asyncio
async def test_evaluator_reports_structure_and_label_consistency(phase1b_context) -> None:
    source = (await phase1b_context.service.list_comments(limit=1)).items[0]
    duplicate = source.model_copy(update={"id": uuid4(), "native_comment_id": "quality-duplicate"})
    await phase1b_context.repository.upsert_comments([duplicate])
    sample = deterministic_stratified_sample([source, duplicate], limit=2, seed="duplicates")
    provider = FakeLLMProvider()
    service = CommentAnalysisService(
        phase1b_context.repository,
        provider,
        LLMConfig(provider="fake", model=provider.model, max_retries=0),
    )
    analyses = {
        str(comment.id): await service.analyze_comment(comment)
        for comment in (source, duplicate)
    }

    result = CommentQualityEvaluator().evaluate(sample, analyses)

    assert result.structure_success_rate == 1.0
    assert result.duplicate_group_count == 1
    assert result.duplicate_consistency_rate == 1.0
    assert result.duplicate_consistency_by_dimension == {
        "sentiment": 1.0,
        "emotion": 1.0,
        "stance": 1.0,
        "risk_level": 1.0,
    }
    assert result.consistency_violation_count == 0
    assert provider.call_count == 1

    labeled = [item.model_copy(update={"gold_sentiment": "positive"}) for item in sample]
    supervised = CommentQualityEvaluator().evaluate(labeled, analyses).supervised_metrics
    assert supervised["sentiment"]["accuracy"] == 1.0
    assert supervised["sentiment"]["macro_f1"] == 1.0


def test_comment_quality_refuses_production_database() -> None:
    result = CliRunner().invoke(
        app,
        ["comment-quality", "--database", "database/analysis.db", "--limit", "1"],
    )

    assert result.exit_code == 1
    assert "refuses database/analysis.db" in result.output


def test_cost_range_is_explicit_when_cache_split_is_unavailable() -> None:
    result = add_cost_range(
        {"prompt_tokens": 1000, "completion_tokens": 500},
        cache_hit_input_price_per_1m=0.01,
        cache_miss_input_price_per_1m=0.4,
        output_price_per_1m=1.2,
        pricing_label="test",
    )

    assert result["cost_usd_min"] == 0.00061
    assert result["cost_usd_max"] == 0.001
    assert "range" in result["pricing_assumption"]


@pytest.mark.asyncio
async def test_latest_quality_api_returns_machine_readable_artifact(tmp_path, monkeypatch) -> None:
    annotation = tmp_path / "annotations.jsonl"
    annotation.write_text("", encoding="utf-8")
    evaluation = CommentQualityEvaluation(
        sample_size=10,
        analyzed_count=10,
        missing_count=0,
        structure_success_rate=1.0,
        semantic_eligible_count=8,
        duplicate_group_count=1,
        duplicate_consistency_rate=1.0,
        consistency_violation_count=0,
        consistency_violations={},
        sentiment_distribution={"neutral": 10},
        stance_distribution={"unclear": 10},
        risk_distribution={"low": 10},
        provider_usage={"request_count": 8, "total_tokens": 1000},
    )
    write_comment_quality_report(
        evaluation,
        output_dir=tmp_path,
        stamp="20260901_120000",
        database=tmp_path / "validation.db",
        seed="test-seed",
        job=None,
        annotation_path=annotation,
        elapsed_seconds=1.25,
    )
    router_module = importlib.import_module("api.routers.analysis")
    monkeypatch.setattr(router_module, "DEFAULT_QUALITY_REPORT_DIR", tmp_path)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=api_app), base_url="http://test"
    ) as client:
        response = await client.get("/api/analysis/quality/latest")

    assert response.status_code == 200
    assert response.json()["evaluation"]["structure_success_rate"] == 1.0
    assert response.json()["database_name"] == "validation.db"
