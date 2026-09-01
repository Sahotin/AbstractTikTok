from __future__ import annotations

import asyncio
import importlib
import shutil
from pathlib import Path

import pytest

from analysis.domain import Platform
from api.schemas.analysis import CollectionAnalysisWorkflowRequest
from api.schemas.crawler import CrawlerStartRequest, PlatformEnum, SaveDataOptionEnum
from api.services.analysis_workflow_manager import AnalysisWorkflowManager
from api.services.crawler_manager import CrawlerManager


FIXTURES = Path(__file__).parent.parent / "fixtures"
workflow_module = importlib.import_module("api.services.analysis_workflow_manager")


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
