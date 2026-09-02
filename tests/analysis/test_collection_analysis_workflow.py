from __future__ import annotations

import asyncio
import importlib
import shutil
from pathlib import Path

import pytest
import httpx

from analysis.domain import Platform
from analysis.llm import FakeLLMProvider, LLMConfig
from api.main import app
from api.routers.analysis import get_analysis_repository, get_semantic_repository
from api.schemas.analysis import CollectionAnalysisWorkflowRequest
from api.schemas.crawler import CrawlerStartRequest, PlatformEnum, SaveDataOptionEnum
from api.services.analysis_workflow_manager import AnalysisWorkflowManager
from api.services.crawler_manager import CrawlerManager


FIXTURES = Path(__file__).parent.parent / "fixtures"
workflow_module = importlib.import_module("api.services.analysis_workflow_manager")
data_router_module = importlib.import_module("api.routers.data")


@pytest.mark.asyncio
async def test_collection_workflow_ingests_then_exposes_dashboard_context(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    source_dir = data_root / "dy" / "json"
    source_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "douyin_contents.json", source_dir / "detail_contents.json")
    shutil.copy(FIXTURES / "douyin_comments.jsonl", source_dir / "detail_comments.jsonl")
    monkeypatch.setattr(workflow_module, "DATA_ROOT", data_root)
    monkeypatch.setattr(workflow_module, "ANALYSIS_DATABASE", tmp_path / "analysis.db")

    manager = AnalysisWorkflowManager()
    workflow = manager.start(CollectionAnalysisWorkflowRequest(
        platform=Platform.DOUYIN,
        source_files=["dy/json/detail_contents.json", "dy/json/detail_comments.jsonl"],
        analysis_items=["trend"],
        limit=2,
    ))

    for _ in range(1000):
        await asyncio.sleep(0.01)
        workflow = manager.get(workflow.id)
        if workflow and workflow.status in {"completed", "failed"}:
            break

    assert workflow is not None
    assert workflow.status == "completed", f"{workflow.stage}: {workflow.error_message or workflow.message}"
    assert workflow.run_id is not None
    assert workflow.comment_count == 2
    assert workflow.progress_percent == 100
    assert f"workflow_id={workflow.id}" in (workflow.dashboard_url or "")


@pytest.mark.asyncio
async def test_collection_workflow_fails_when_every_llm_item_fails(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    source_dir = data_root / "dy" / "json"
    source_dir.mkdir(parents=True)
    shutil.copy(FIXTURES / "douyin_contents.json", source_dir / "detail_contents.json")
    shutil.copy(FIXTURES / "douyin_comments.jsonl", source_dir / "detail_comments.jsonl")
    monkeypatch.setattr(workflow_module, "DATA_ROOT", data_root)
    monkeypatch.setattr(workflow_module, "ANALYSIS_DATABASE", tmp_path / "analysis.db")
    config = LLMConfig(provider="fake", model="fake-model", timeout=1, max_retries=0, retry_base_delay=0)
    monkeypatch.setattr(workflow_module.LLMConfig, "from_environment", classmethod(lambda _cls: config))
    monkeypatch.setattr(workflow_module, "create_provider", lambda _config: FakeLLMProvider(failure_call_indices={0, 1}))

    manager = AnalysisWorkflowManager()
    workflow = manager.start(CollectionAnalysisWorkflowRequest(
        platform=Platform.DOUYIN,
        source_files=["dy/json/detail_contents.json", "dy/json/detail_comments.jsonl"],
        analysis_items=["sentiment"],
        limit=2,
        max_concurrency=1,
    ))
    for _ in range(1000):
        await asyncio.sleep(0.01)
        workflow = manager.get(workflow.id)
        if workflow and workflow.status in {"completed", "failed"}:
            break

    assert workflow is not None
    assert workflow.status == "failed"
    assert workflow.analyzed_count == 0
    assert workflow.failed_count == 2
    assert "未产出有效结果" in (workflow.error_message or "")


def test_crawler_analysis_context_supports_latest_bilibili_excel(tmp_path: Path) -> None:
    manager = CrawlerManager()
    manager._data_root = tmp_path / "data"
    manager.current_config = CrawlerStartRequest(
        platform=PlatformEnum.BILIBILI,
        save_option=SaveDataOptionEnum.EXCEL,
    )
    output = manager._data_root / "bilibili" / "bili_detail_20260901_190000.xlsx"
    output.parent.mkdir(parents=True)
    output.write_bytes(b"workbook-placeholder")

    context = manager.get_analysis_context()

    assert context["available"] is True
    assert context["analysis_platform"] == "bilibili"
    assert context["files"][0]["path"] == "bilibili/bili_detail_20260901_190000.xlsx"


def test_crawler_analysis_context_groups_latest_douyin_json_outputs(tmp_path: Path) -> None:
    manager = CrawlerManager()
    manager._data_root = tmp_path / "data"
    output_dir = manager._data_root / "douyin" / "json"
    output_dir.mkdir(parents=True)
    (output_dir / "search_contents_2026-09-01.json").write_text("[]", encoding="utf-8")
    (output_dir / "search_comments_2026-09-01.json").write_text("[]", encoding="utf-8")

    context = manager.get_analysis_context()

    assert context["available"] is True
    assert context["analysis_platform"] == "douyin"
    assert {item["name"] for item in context["files"]} == {
        "search_contents_2026-09-01.json",
        "search_comments_2026-09-01.json",
    }


def test_historical_data_files_expose_analysis_group_metadata(tmp_path: Path, monkeypatch) -> None:
    data_root = tmp_path / "data"
    output_dir = data_root / "douyin" / "json"
    output_dir.mkdir(parents=True)
    contents = output_dir / "search_contents_2026-09-01.json"
    comments = output_dir / "search_comments_2026-09-01.jsonl"
    contents.write_text("[]", encoding="utf-8")
    comments.write_text('{"comment_id": "1"}\n{"comment_id": "2"}\n', encoding="utf-8")
    monkeypatch.setattr(data_router_module, "DATA_DIR", data_root)

    content_info = data_router_module.get_file_info(contents)
    comment_info = data_router_module.get_file_info(comments)

    assert content_info["analysis_supported"] is True
    assert content_info["analysis_platform"] == "douyin"
    assert content_info["crawler_type"] == "search"
    assert content_info["analysis_group_id"] == comment_info["analysis_group_id"]
    assert comment_info["record_count"] == 2


@pytest.mark.asyncio
async def test_result_scopes_are_durable_and_open_without_new_analysis(phase1b_context, semantic_repository) -> None:
    async def override_analysis_repository():
        yield phase1b_context.repository

    async def override_semantic_repository():
        yield semantic_repository

    app.dependency_overrides[get_analysis_repository] = override_analysis_repository
    app.dependency_overrides[get_semantic_repository] = override_semantic_repository
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.get("/api/analysis/result-scopes")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    payload = response.json()
    assert payload["total"] == 1
    assert payload["items"][0]["run_id"] == phase1b_context.run_id
    assert payload["items"][0]["dashboard_url"].startswith("/dashboard?platform=douyin&run_id=")
    assert payload["items"][0]["analysis_coverage"] == 0
