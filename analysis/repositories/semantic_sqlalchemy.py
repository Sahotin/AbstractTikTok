"""Async SQLite repository dedicated to embeddings and semantic-run state."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, Sequence
from uuid import UUID

from sqlalchemy import bindparam, case, delete, event, func, select, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine
from sqlalchemy.exc import OperationalError

from analysis.domain import (
    AnalysisSnapshot,
    CommentAnalysis,
    EmbeddingRecord,
    NormalizedComment,
    SemanticAnalysisRun,
    SemanticRunItem,
    SemanticRunStatus,
    TopicCluster,
    TopicMembership,
    TopicRepresentativeComment,
    TopicStageStatus,
)
from analysis.embeddings.codec import decode_vector, encode_vector
from database.analysis_models import CommentAnalysisModel, NormalizedCommentModel
from database.semantic_models import (
    CommentEmbeddingModel,
    SemanticAnalysisRunModel,
    SemanticRunItemModel,
    TopicClusterModel,
)


@dataclass(frozen=True)
class TopicSnapshotRow:
    item: SemanticRunItem
    comment: NormalizedComment
    embedding: EmbeddingRecord | None
    analysis: CommentAnalysis | None


@dataclass(frozen=True)
class TopicPage:
    items: list[TopicCluster]
    total: int
    limit: int
    offset: int


def _utc(value: datetime | None) -> datetime | None:
    if value is None:
        return None
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _embedding_row(record: EmbeddingRecord) -> dict[str, Any]:
    return {
        "id": str(record.id),
        "comment_id": str(record.comment_id),
        "input_hash": record.input_hash,
        "provider": record.provider,
        "model": record.model,
        "embedding_version": record.embedding_version,
        "dimension": record.dimension,
        "dtype": record.dtype,
        "normalized": record.normalized,
        "vector_blob": encode_vector(record.vector),
        "created_at": record.created_at,
    }


def _embedding_domain(row: CommentEmbeddingModel) -> EmbeddingRecord:
    return EmbeddingRecord(
        id=row.id,
        comment_id=row.comment_id,
        input_hash=row.input_hash,
        provider=row.provider,
        model=row.model,
        embedding_version=row.embedding_version,
        dimension=row.dimension,
        dtype=row.dtype,
        normalized=row.normalized,
        vector=decode_vector(row.vector_blob, dimension=row.dimension, dtype=row.dtype),
        created_at=_utc(row.created_at),
    )


def _run_row(run: SemanticAnalysisRun) -> dict[str, Any]:
    values = run.model_dump(mode="python")
    for key, value in tuple(values.items()):
        if hasattr(value, "value"):
            values[key] = value.value
    for key in ("id", "run_id", "content_id"):
        if values.get(key) is not None:
            values[key] = str(values[key])
    return values


def _run_domain(row: SemanticAnalysisRunModel) -> SemanticAnalysisRun:
    values = {column.name: getattr(row, column.name) for column in row.__table__.columns}
    for key in (
        "created_at", "started_at", "finished_at", "topic_started_at", "topic_finished_at"
    ):
        values[key] = _utc(values[key])
    return SemanticAnalysisRun.model_validate(values)


def _orm_domain(row: Any, domain_type):
    values = {column.name: getattr(row, column.name) for column in row.__table__.columns}
    for key in ("created_at", "published_at", "collected_at"):
        if key in values:
            values[key] = _utc(values[key])
    return domain_type.model_validate(values)


def _item_domain(row: SemanticRunItemModel) -> SemanticRunItem:
    return _orm_domain(row, SemanticRunItem)


def _topic_domain(row: TopicClusterModel) -> TopicCluster:
    values = {column.name: getattr(row, column.name) for column in row.__table__.columns}
    values["centroid"] = decode_vector(
        row.centroid_blob, dimension=row.dimension, dtype="float32-le"
    )
    values.pop("centroid_blob")
    values["created_at"] = _utc(values["created_at"])
    return TopicCluster.model_validate(values)


class SemanticRepository:
    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path).expanduser().resolve()
        self.engine: AsyncEngine = create_async_engine(
            f"sqlite+aiosqlite:///{self.database_path.as_posix()}",
            future=True,
        )
        self._session_factory = async_sessionmaker(self.engine, expire_on_commit=False)
        self._write_lock = asyncio.Lock()

        @event.listens_for(self.engine.sync_engine, "connect")
        def _configure_sqlite(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    async def close(self) -> None:
        await self.engine.dispose()

    async def save_embedding(self, record: EmbeddingRecord) -> EmbeddingRecord:
        await self.save_embeddings([record])
        stored = await self.get_embedding(
            comment_id=str(record.comment_id),
            input_hash=record.input_hash,
            provider=record.provider,
            model=record.model,
            embedding_version=record.embedding_version,
        )
        if stored is None:
            raise RuntimeError("Embedding was not persisted")
        return stored

    async def save_embeddings(self, records: Sequence[EmbeddingRecord]) -> None:
        if not records:
            return
        async with self._write_lock:
            async with self._session_factory.begin() as session:
                # 80 rows keep the statement below SQLite's legacy 999-variable
                # limit while retaining one transaction for the whole write.
                for offset in range(0, len(records), 80):
                    statement = sqlite_insert(CommentEmbeddingModel).values(
                        [_embedding_row(record) for record in records[offset : offset + 80]]
                    )
                    statement = statement.on_conflict_do_nothing(
                        index_elements=["comment_id", "input_hash", "provider", "model", "embedding_version"]
                    )
                    await session.execute(statement)

    async def get_embedding(
        self,
        *,
        comment_id: str,
        input_hash: str,
        provider: str,
        model: str,
        embedding_version: str,
    ) -> EmbeddingRecord | None:
        statement = select(CommentEmbeddingModel).where(
            CommentEmbeddingModel.comment_id == comment_id,
            CommentEmbeddingModel.input_hash == input_hash,
            CommentEmbeddingModel.provider == provider,
            CommentEmbeddingModel.model == model,
            CommentEmbeddingModel.embedding_version == embedding_version,
        )
        async with self._session_factory() as session:
            row = (await session.execute(statement)).scalar_one_or_none()
        return _embedding_domain(row) if row else None

    async def get_embedding_by_content_hash(
        self,
        *,
        input_hash: str,
        provider: str,
        model: str,
        embedding_version: str,
    ) -> EmbeddingRecord | None:
        statement = (
            select(CommentEmbeddingModel)
            .where(
                CommentEmbeddingModel.input_hash == input_hash,
                CommentEmbeddingModel.provider == provider,
                CommentEmbeddingModel.model == model,
                CommentEmbeddingModel.embedding_version == embedding_version,
            )
            .order_by(CommentEmbeddingModel.created_at, CommentEmbeddingModel.id)
            .limit(1)
        )
        async with self._session_factory() as session:
            row = (await session.execute(statement)).scalar_one_or_none()
        return _embedding_domain(row) if row else None

    async def get_embeddings_for_comments(
        self,
        comment_ids: Sequence[str],
        *,
        provider: str,
        model: str,
        embedding_version: str,
    ) -> list[EmbeddingRecord]:
        if not comment_ids:
            return []
        rows = []
        async with self._session_factory() as session:
            for offset in range(0, len(comment_ids), 900):
                statement = select(CommentEmbeddingModel).where(
                    CommentEmbeddingModel.comment_id.in_(comment_ids[offset : offset + 900]),
                    CommentEmbeddingModel.provider == provider,
                    CommentEmbeddingModel.model == model,
                    CommentEmbeddingModel.embedding_version == embedding_version,
                )
                rows.extend((await session.execute(statement)).scalars().all())
        return [_embedding_domain(row) for row in rows]

    async def get_embeddings_by_content_hashes(
        self,
        input_hashes: Sequence[str],
        *,
        provider: str,
        model: str,
        embedding_version: str,
    ) -> dict[str, EmbeddingRecord]:
        if not input_hashes:
            return {}
        rows = []
        async with self._session_factory() as session:
            for offset in range(0, len(input_hashes), 900):
                statement = (
                    select(CommentEmbeddingModel)
                    .where(
                        CommentEmbeddingModel.input_hash.in_(input_hashes[offset : offset + 900]),
                        CommentEmbeddingModel.provider == provider,
                        CommentEmbeddingModel.model == model,
                        CommentEmbeddingModel.embedding_version == embedding_version,
                    )
                    .order_by(CommentEmbeddingModel.created_at, CommentEmbeddingModel.id)
                )
                rows.extend((await session.execute(statement)).scalars().all())
        result: dict[str, EmbeddingRecord] = {}
        for row in rows:
            result.setdefault(row.input_hash, _embedding_domain(row))
        return result

    async def get_embeddings_by_ids(self, embedding_ids: Sequence[str]) -> dict[str, EmbeddingRecord]:
        if not embedding_ids:
            return {}
        result: dict[str, EmbeddingRecord] = {}
        async with self._session_factory() as session:
            for offset in range(0, len(embedding_ids), 900):
                rows = (
                    await session.execute(
                        select(CommentEmbeddingModel).where(
                            CommentEmbeddingModel.id.in_(embedding_ids[offset : offset + 900])
                        )
                    )
                ).scalars().all()
                result.update({row.id: _embedding_domain(row) for row in rows})
        return result

    async def get_latest_analyses_for_comments(
        self, comment_ids: Sequence[str]
    ) -> dict[str, CommentAnalysis]:
        if not comment_ids:
            return {}
        result: dict[str, CommentAnalysis] = {}
        async with self._session_factory() as session:
            for offset in range(0, len(comment_ids), 900):
                rows = (
                    await session.execute(
                        select(CommentAnalysisModel)
                        .where(CommentAnalysisModel.comment_id.in_(comment_ids[offset : offset + 900]))
                        .order_by(
                            CommentAnalysisModel.comment_id,
                            CommentAnalysisModel.created_at.desc(),
                            CommentAnalysisModel.id.desc(),
                        )
                    )
                ).scalars().all()
                for row in rows:
                    result.setdefault(row.comment_id, _orm_domain(row, CommentAnalysis))
        return result

    async def count_embeddings(self) -> int:
        from sqlalchemy import func

        async with self._session_factory() as session:
            return int(await session.scalar(select(func.count()).select_from(CommentEmbeddingModel)) or 0)

    async def create_semantic_run(self, run: SemanticAnalysisRun) -> SemanticAnalysisRun:
        statement = sqlite_insert(SemanticAnalysisRunModel).values(_run_row(run)).on_conflict_do_nothing(
            index_elements=["id"]
        )
        async with self._write_lock:
            async with self._session_factory.begin() as session:
                await session.execute(statement)
        stored = await self.get_semantic_run(str(run.id))
        if stored is None:
            raise RuntimeError("Semantic run was not persisted")
        return stored

    async def replace_semantic_run_items(
        self,
        run_id: str,
        items: Sequence[SemanticRunItem],
        *,
        analysis_snapshot_hash: str,
        analyzed_comments: int,
    ) -> None:
        rows = []
        for item in items:
            row = item.model_dump(mode="python")
            for key, value in tuple(row.items()):
                if hasattr(value, "value"):
                    row[key] = value.value
                elif isinstance(value, UUID):
                    row[key] = str(value)
            rows.append(row)
        async with self._write_lock:
            async with self._session_factory.begin() as session:
                # Replacing items invalidates every persisted membership and
                # analysis-derived topic metadata.  Leaving topic clusters in
                # place would make the Dashboard show an orphaned historical
                # topic result with no representatives.
                await session.execute(
                    delete(TopicClusterModel).where(TopicClusterModel.semantic_run_id == run_id)
                )
                await session.execute(
                    delete(SemanticRunItemModel).where(SemanticRunItemModel.semantic_run_id == run_id)
                )
                for offset in range(0, len(rows), 80):
                    await session.execute(
                        sqlite_insert(SemanticRunItemModel).values(rows[offset : offset + 80])
                    )
                await session.execute(
                    update(SemanticAnalysisRunModel)
                    .where(SemanticAnalysisRunModel.id == run_id)
                    .values(
                        analysis_snapshot_hash=analysis_snapshot_hash,
                        analyzed_comments=analyzed_comments,
                        topic_status=TopicStageStatus.NOT_STARTED.value,
                        topic_count=0,
                        clustered_comments=0,
                        clustering_input_comments=0,
                        clustering_failed_comments=0,
                        noise_comments=0,
                        topic_started_at=None,
                        topic_finished_at=None,
                        topic_error_message=None,
                    )
                )

    async def list_semantic_run_items(self, run_id: str) -> list[SemanticRunItem]:
        async with self._session_factory() as session:
            rows = (
                await session.execute(
                    select(SemanticRunItemModel)
                    .where(SemanticRunItemModel.semantic_run_id == run_id)
                    .order_by(SemanticRunItemModel.comment_id)
                )
            ).scalars().all()
        return [_item_domain(row) for row in rows]

    async def get_analysis_snapshot(self, run_id: str) -> AnalysisSnapshot | None:
        """Read the one versioned analysis identity referenced by a run.

        SemanticRunItem.analysis_id is the durable boundary between an
        embedding/topic result and its source LLM analysis.  Returning one
        identity here lets API consumers display and audit that boundary
        without loading embeddings or comment text.
        """

        statement = (
            select(
                CommentAnalysisModel.provider,
                CommentAnalysisModel.model,
                CommentAnalysisModel.prompt_version,
                CommentAnalysisModel.analysis_version,
            )
            .select_from(SemanticRunItemModel)
            .join(CommentAnalysisModel, SemanticRunItemModel.analysis_id == CommentAnalysisModel.id)
            .where(SemanticRunItemModel.semantic_run_id == run_id)
            .distinct()
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        if not rows:
            return None
        if len(rows) != 1:
            raise ValueError("Semantic run references more than one analysis snapshot")
        provider, model, prompt_version, analysis_version = rows[0]
        return AnalysisSnapshot(
            provider=str(provider),
            model=str(model),
            prompt_version=str(prompt_version),
            analysis_version=str(analysis_version),
        )

    async def get_topic_snapshot(self, run_id: str) -> list[TopicSnapshotRow]:
        statement = (
            select(
                SemanticRunItemModel,
                NormalizedCommentModel,
                CommentEmbeddingModel,
                CommentAnalysisModel,
            )
            .join(NormalizedCommentModel, SemanticRunItemModel.comment_id == NormalizedCommentModel.id)
            .outerjoin(CommentEmbeddingModel, SemanticRunItemModel.embedding_id == CommentEmbeddingModel.id)
            .outerjoin(CommentAnalysisModel, SemanticRunItemModel.analysis_id == CommentAnalysisModel.id)
            .where(SemanticRunItemModel.semantic_run_id == run_id)
            .order_by(SemanticRunItemModel.comment_id)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        return [
            TopicSnapshotRow(
                item=_item_domain(item),
                comment=_orm_domain(comment, NormalizedComment),
                embedding=_embedding_domain(embedding) if embedding is not None else None,
                analysis=_orm_domain(analysis, CommentAnalysis) if analysis is not None else None,
            )
            for item, comment, embedding, analysis in rows
        ]

    async def replace_topic_results(
        self,
        run_id: str,
        clusters: Sequence[TopicCluster],
        memberships: Sequence[TopicMembership],
        *,
        run_updates: dict[str, Any],
    ) -> None:
        cluster_rows = []
        for cluster in clusters:
            row = cluster.model_dump(mode="python", exclude={"centroid"})
            row["id"] = str(cluster.id)
            row["semantic_run_id"] = str(cluster.semantic_run_id)
            row["medoid_comment_id"] = str(cluster.medoid_comment_id)
            for key in ("dominant_sentiment", "dominant_stance", "dominant_risk"):
                if row.get(key) is not None:
                    row[key] = row[key].value if hasattr(row[key], "value") else row[key]
            row["top_topics"] = [item.model_dump(mode="json") for item in cluster.top_topics]
            row["top_keywords"] = [item.model_dump(mode="json") for item in cluster.top_keywords]
            row["centroid_blob"] = encode_vector(cluster.centroid)
            cluster_rows.append(row)

        membership_rows = [
            {
                "p_run_id": run_id,
                "p_comment_id": str(item.comment_id),
                "p_cluster_id": str(item.topic_cluster_id) if item.topic_cluster_id else None,
                "p_similarity": item.similarity,
                "p_rank": item.representative_rank,
                "p_score": item.representative_score,
                "p_status": item.status.value,
                "p_error": item.error_message,
            }
            for item in memberships
        ]
        table = SemanticRunItemModel.__table__
        membership_statement = (
            update(table)
            .where(
                table.c.semantic_run_id == bindparam("p_run_id"),
                table.c.comment_id == bindparam("p_comment_id"),
            )
            .values(
                status=bindparam("p_status"),
                topic_cluster_id=bindparam("p_cluster_id"),
                topic_similarity=bindparam("p_similarity"),
                topic_representative_rank=bindparam("p_rank"),
                topic_representative_score=bindparam("p_score"),
                error_message=bindparam("p_error"),
            )
        )
        async with self._write_lock:
            async with self._session_factory.begin() as session:
                await session.execute(
                    update(SemanticRunItemModel)
                    .where(SemanticRunItemModel.semantic_run_id == run_id)
                    .values(
                        status=case(
                            (SemanticRunItemModel.embedding_id.is_not(None), "embedded"),
                            else_="missing_embedding",
                        ),
                        error_message=case(
                            (SemanticRunItemModel.embedding_id.is_not(None), None),
                            else_="missing_embedding",
                        ),
                        topic_cluster_id=None,
                        topic_similarity=None,
                        topic_representative_rank=None,
                        topic_representative_score=None,
                    )
                )
                await session.execute(
                    delete(TopicClusterModel).where(TopicClusterModel.semantic_run_id == run_id)
                )
                for offset in range(0, len(cluster_rows), 40):
                    await session.execute(
                        sqlite_insert(TopicClusterModel).values(cluster_rows[offset : offset + 40])
                    )
                for offset in range(0, len(membership_rows), 500):
                    await session.execute(
                        membership_statement, membership_rows[offset : offset + 500]
                    )
                await session.execute(
                    update(SemanticAnalysisRunModel)
                    .where(SemanticAnalysisRunModel.id == run_id)
                    .values(**run_updates)
                )

    async def fail_topic_run(self, run_id: str, *, error_message: str, finished_at: datetime) -> None:
        """Atomically remove stale topic output and persist a failed topic stage."""

        async with self._write_lock:
            async with self._session_factory.begin() as session:
                await session.execute(
                    update(SemanticRunItemModel)
                    .where(SemanticRunItemModel.semantic_run_id == run_id)
                    .values(
                        status=case(
                            (SemanticRunItemModel.embedding_id.is_not(None), "embedded"),
                            else_="missing_embedding",
                        ),
                        error_message=case(
                            (SemanticRunItemModel.embedding_id.is_not(None), None),
                            else_="missing_embedding",
                        ),
                        topic_cluster_id=None,
                        topic_similarity=None,
                        topic_representative_rank=None,
                        topic_representative_score=None,
                    )
                )
                await session.execute(
                    delete(TopicClusterModel).where(TopicClusterModel.semantic_run_id == run_id)
                )
                await session.execute(
                    update(SemanticAnalysisRunModel)
                    .where(SemanticAnalysisRunModel.id == run_id)
                    .values(
                        topic_status="failed",
                        topic_finished_at=finished_at,
                        topic_error_message=error_message,
                        clustered_comments=0,
                        noise_comments=0,
                        topic_count=0,
                    )
                )

    async def list_topic_clusters(
        self,
        *,
        semantic_run_id: str | None = None,
        platform: str | None = None,
        run_id: str | None = None,
        content_id: str | None = None,
        min_size: int = 1,
        limit: int = 50,
        offset: int = 0,
    ) -> TopicPage:
        filters = [TopicClusterModel.size >= min_size]
        if semantic_run_id:
            filters.append(TopicClusterModel.semantic_run_id == semantic_run_id)
        if platform:
            filters.append(SemanticAnalysisRunModel.platform == platform)
        if run_id:
            filters.append(SemanticAnalysisRunModel.run_id == run_id)
        if content_id:
            filters.append(SemanticAnalysisRunModel.content_id == content_id)
        base = (
            select(TopicClusterModel)
            .join(SemanticAnalysisRunModel, TopicClusterModel.semantic_run_id == SemanticAnalysisRunModel.id)
            .where(*filters)
        )
        count_statement = (
            select(func.count())
            .select_from(TopicClusterModel)
            .join(SemanticAnalysisRunModel, TopicClusterModel.semantic_run_id == SemanticAnalysisRunModel.id)
            .where(*filters)
        )
        async with self._session_factory() as session:
            total = int(await session.scalar(count_statement) or 0)
            rows = (
                await session.execute(
                    base.order_by(TopicClusterModel.size.desc(), TopicClusterModel.medoid_comment_id)
                    .limit(limit)
                    .offset(offset)
                )
            ).scalars().all()
        return TopicPage([_topic_domain(row) for row in rows], total, limit, offset)

    async def get_topic_cluster(self, topic_id: str) -> TopicCluster | None:
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(TopicClusterModel).where(TopicClusterModel.id == topic_id)
                )
            ).scalar_one_or_none()
        return _topic_domain(row) if row else None

    async def get_topic_representatives(
        self, topic_id: str, *, limit: int = 3
    ) -> list[TopicRepresentativeComment]:
        statement = (
            select(SemanticRunItemModel, NormalizedCommentModel, CommentAnalysisModel)
            .join(NormalizedCommentModel, SemanticRunItemModel.comment_id == NormalizedCommentModel.id)
            .outerjoin(CommentAnalysisModel, SemanticRunItemModel.analysis_id == CommentAnalysisModel.id)
            .where(
                SemanticRunItemModel.topic_cluster_id == topic_id,
                SemanticRunItemModel.topic_representative_rank.is_not(None),
                SemanticRunItemModel.topic_representative_rank <= limit,
            )
            .order_by(SemanticRunItemModel.topic_representative_rank, SemanticRunItemModel.comment_id)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        return [
            TopicRepresentativeComment(
                comment_id=item.comment_id,
                text=comment.text,
                like_count=comment.like_count,
                similarity=float(item.topic_similarity or 0.0),
                rank=item.topic_representative_rank,
                score=float(item.topic_representative_score or 0.0),
                sentiment=analysis.sentiment if analysis is not None else None,
                risk_level=analysis.risk_level if analysis is not None else None,
            )
            for item, comment, analysis in rows
        ]

    async def get_topic_analysis_distributions(self, topic_id: str) -> dict[str, dict[str, int]]:
        """Aggregate the fixed analysis snapshot for one existing cluster."""

        output: dict[str, dict[str, int]] = {"sentiment": {}, "risk": {}}
        for key, column in (
            ("sentiment", CommentAnalysisModel.sentiment),
            ("risk", CommentAnalysisModel.risk_level),
        ):
            statement = (
                select(column, func.count())
                .select_from(SemanticRunItemModel)
                .join(CommentAnalysisModel, SemanticRunItemModel.analysis_id == CommentAnalysisModel.id)
                .where(SemanticRunItemModel.topic_cluster_id == topic_id)
                .group_by(column)
                .order_by(column)
            )
            async with self._session_factory() as session:
                rows = (await session.execute(statement)).all()
            output[key] = {str(value): int(count) for value, count in rows}
        return output

    async def get_semantic_run(self, run_id: str) -> SemanticAnalysisRun | None:
        async with self._session_factory() as session:
            row = (
                await session.execute(
                    select(SemanticAnalysisRunModel).where(SemanticAnalysisRunModel.id == run_id)
                )
            ).scalar_one_or_none()
        return _run_domain(row) if row else None

    async def latest_semantic_run_for_collection(self, collection_run_id: str) -> SemanticAnalysisRun | None:
        statement = (
            select(SemanticAnalysisRunModel)
            .where(SemanticAnalysisRunModel.run_id == collection_run_id)
            .order_by(SemanticAnalysisRunModel.created_at.desc(), SemanticAnalysisRunModel.id.desc())
            .limit(1)
        )
        try:
            async with self._session_factory() as session:
                row = (await session.execute(statement)).scalar_one_or_none()
        except OperationalError:
            return None
        return _run_domain(row) if row else None

    async def update_semantic_run(
        self,
        run_id: str,
        *,
        status: SemanticRunStatus | None = None,
        **updates: Any,
    ) -> SemanticAnalysisRun | None:
        values = dict(updates)
        if status is not None:
            values["status"] = status.value
        allowed = {column.name for column in SemanticAnalysisRunModel.__table__.columns} - {"id"}
        unknown = set(values) - allowed
        if unknown:
            raise ValueError(f"Unsupported semantic run fields: {sorted(unknown)}")
        async with self._write_lock:
            async with self._session_factory.begin() as session:
                await session.execute(
                    update(SemanticAnalysisRunModel)
                    .where(SemanticAnalysisRunModel.id == run_id)
                    .values(**values)
                )
        return await self.get_semantic_run(run_id)
