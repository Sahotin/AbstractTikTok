from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import pytest_asyncio

from analysis.repositories import AnalysisRepository
from analysis.repositories import SemanticRepository
from analysis.services import AnalysisQueryService, IngestionService
from database.semantic_models import (
    CommentEmbeddingModel,
    SemanticAnalysisRunModel,
    SemanticRunItemModel,
    TopicClusterModel,
)


FIXTURES = Path(__file__).parent.parent / "fixtures"


@dataclass(frozen=True)
class Phase1BContext:
    repository: AnalysisRepository
    service: AnalysisQueryService
    run_id: str


@pytest_asyncio.fixture
async def phase1b_context(tmp_path: Path):
    repository = AnalysisRepository(tmp_path / "analysis.db")
    ingestion = IngestionService(repository, batch_size=1)
    report = await ingestion.ingest(
        [FIXTURES / "douyin_contents.json", FIXTURES / "douyin_comments.jsonl"],
        platform="douyin",
    )
    context = Phase1BContext(
        repository=repository,
        service=AnalysisQueryService(repository),
        run_id=str(report.run_id),
    )
    try:
        yield context
    finally:
        await repository.close()


@pytest_asyncio.fixture
async def job_comments(phase1b_context):
    source = (await phase1b_context.service.list_comments(limit=1)).items[0]
    comments = [
        source.model_copy(
            update={
                "id": uuid4(),
                "native_comment_id": f"job-comment-{index:03d}",
                "text": f"{source.text} 测试样本 {index:03d}",
            }
        )
        for index in range(50)
    ]
    await phase1b_context.repository.upsert_comments(comments)
    return comments


@pytest_asyncio.fixture
async def semantic_repository(phase1b_context):
    """Create Phase 3B tables explicitly in an isolated test database."""

    tables = [
        CommentEmbeddingModel.__table__,
        SemanticAnalysisRunModel.__table__,
        TopicClusterModel.__table__,
        SemanticRunItemModel.__table__,
    ]
    async with phase1b_context.repository.engine.begin() as connection:
        await connection.run_sync(
            lambda sync_connection: CommentEmbeddingModel.metadata.create_all(
                sync_connection,
                tables=tables,
            )
        )
    repository = SemanticRepository(phase1b_context.repository.database_path)
    try:
        yield repository
    finally:
        await repository.close()
