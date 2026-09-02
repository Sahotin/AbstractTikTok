from __future__ import annotations

import sqlite3
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import pytest
from alembic import command
from alembic.config import Config

from analysis.domain import EmbeddingRecord
from analysis.preprocessing import compute_text_hash, utc_now


def _record(comment_id, *, provider="fake", model="model-a", version="v1") -> EmbeddingRecord:
    input_hash = compute_text_hash("same text")
    return EmbeddingRecord(
        id=uuid5(NAMESPACE_URL, f"{comment_id}:{provider}:{model}:{version}"),
        comment_id=comment_id,
        input_hash=input_hash,
        provider=provider,
        model=model,
        embedding_version=version,
        dimension=3,
        vector=[1.0, 0.0, 0.0],
        created_at=utc_now(),
    )


@pytest.mark.asyncio
async def test_embedding_round_trip_and_content_hash_lookup(semantic_repository, job_comments) -> None:
    first = _record(job_comments[0].id)
    await semantic_repository.save_embedding(first)

    exact = await semantic_repository.get_embedding(
        comment_id=str(first.comment_id),
        input_hash=first.input_hash,
        provider=first.provider,
        model=first.model,
        embedding_version=first.embedding_version,
    )
    reused = await semantic_repository.get_embedding_by_content_hash(
        input_hash=first.input_hash,
        provider=first.provider,
        model=first.model,
        embedding_version=first.embedding_version,
    )

    assert exact is not None and exact.vector == pytest.approx(first.vector)
    assert reused is not None and reused.comment_id == first.comment_id


@pytest.mark.asyncio
async def test_cache_identity_includes_provider_model_and_version(semantic_repository, job_comments) -> None:
    records = [
        _record(job_comments[index].id, provider=provider, model=model, version=version)
        for index, (provider, model, version) in enumerate([
            ("fake", "model-a", "v1"),
            ("other", "model-a", "v1"),
            ("fake", "model-b", "v1"),
            ("fake", "model-a", "v2"),
        ])
    ]
    await semantic_repository.save_embeddings(records)

    assert await semantic_repository.count_embeddings() == 4
    for record in records:
        assert await semantic_repository.get_embedding_by_content_hash(
            input_hash=record.input_hash,
            provider=record.provider,
            model=record.model,
            embedding_version=record.embedding_version,
        ) is not None


@pytest.mark.asyncio
async def test_unique_cache_write_is_idempotent(semantic_repository, job_comments) -> None:
    record = _record(job_comments[0].id)
    await semantic_repository.save_embeddings([record, record])
    assert await semantic_repository.count_embeddings() == 1


def test_alembic_builds_fresh_analysis_database(tmp_path) -> None:
    database_path = tmp_path / "fresh.db"
    project_root = Path(__file__).parents[2]
    config = Config(str(project_root / "alembic.ini"))
    config.set_main_option("script_location", str(project_root / "migrations"))
    config.set_main_option("sqlalchemy.url", f"sqlite:///{database_path.as_posix()}")

    command.upgrade(config, "head")

    connection = sqlite3.connect(database_path)
    try:
        tables = {
            row[0]
            for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
        }
        assert {
            "collection_runs",
            "normalized_comments",
            "comment_embeddings",
            "semantic_analysis_runs",
        } <= tables
        assert connection.execute("SELECT version_num FROM alembic_version").fetchone()[0] == "0004_phase4a3_semantic_accounting"
        assert connection.execute("PRAGMA integrity_check").fetchone()[0] == "ok"
        assert connection.execute("PRAGMA foreign_key_check").fetchall() == []
    finally:
        connection.close()
