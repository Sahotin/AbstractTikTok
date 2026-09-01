"""Async SQLite repository for normalized data ingestion and bounded reads."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Optional, Sequence, Type, TypeVar

from sqlalchemy import case, event, func, select, true, update
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.ext.asyncio import AsyncEngine, async_sessionmaker, create_async_engine

from analysis.domain import (
    AnalysisJob,
    AnalysisJobItem,
    AnalysisJobItemStatus,
    AnalysisJobStatus,
    CommentAnalysis,
    CollectionRun,
    NormalizedAuthor,
    NormalizedComment,
    NormalizedContent,
)
from database.analysis_models import (
    AnalysisJobItemModel,
    AnalysisJobModel,
    CommentAnalysisModel,
    CollectionRunModel,
    NormalizedAuthorModel,
    NormalizedCommentModel,
    NormalizedContentModel,
)


@dataclass(frozen=True)
class UpsertResult:
    """Counts from one idempotent batch write."""

    inserted: int
    duplicates: int


@dataclass(frozen=True)
class ImportDatabaseStats:
    """Final database facts for one stable collection run."""

    content_count: int
    comment_count: int
    first_level_comment_count: int
    second_level_comment_count: int
    author_count: int
    video_count: int
    earliest_comment_at: datetime | None
    latest_comment_at: datetime | None
    total_database_contents: int
    total_database_comments: int
    total_database_authors: int


@dataclass(frozen=True)
class ContentQueryStatistics:
    content_count: int


@dataclass(frozen=True)
class CommentQueryStatistics:
    comment_count: int
    root_comment_count: int
    reply_comment_count: int
    earliest_comment_time: datetime | None
    latest_comment_time: datetime | None


@dataclass(frozen=True)
class AuthorQueryStatistics:
    author_count: int


@dataclass(frozen=True)
class AggregateValue:
    value: str
    count: int


@dataclass(frozen=True)
class SentimentScoreAggregate:
    minimum: float | None
    maximum: float | None
    mean: float | None


@dataclass(frozen=True)
class TopicDimensionAggregate:
    topic: str
    dimension: str
    count: int


@dataclass(frozen=True)
class CommentVolumeAggregate:
    date: str
    count: int


@dataclass(frozen=True)
class TrendBucketAggregate:
    bucket_start: str
    total_comments: int
    analyzed_comments: int
    positive_count: int
    neutral_count: int
    negative_count: int
    low_risk_count: int
    medium_risk_count: int
    high_risk_count: int
    mean_sentiment_score: float | None


@dataclass(frozen=True)
class HighRiskCommentAggregate:
    comment: NormalizedComment
    sentiment: str
    sentiment_score: float
    risk_level: str
    risk_reasons: list[str]
    summary: str


DomainEntity = TypeVar(
    "DomainEntity",
    NormalizedContent,
    NormalizedComment,
    NormalizedAuthor,
    CommentAnalysis,
    AnalysisJob,
    AnalysisJobItem,
)


def _domain_row(item: Any) -> dict[str, Any]:
    row = item.model_dump(mode="python")
    for key, value in tuple(row.items()):
        if hasattr(value, "value"):
            row[key] = value.value
        elif key == "id" or key.endswith("_id") and value is not None and value.__class__.__name__ == "UUID":
            row[key] = str(value)
    # UUID values can also appear in optional foreign-key fields.
    for key in ("id", "run_id", "content_id", "author_id"):
        if row.get(key) is not None:
            row[key] = str(row[key])
    return row


def _as_utc(value: datetime | None) -> datetime | None:
    """Restore UTC tzinfo lost by SQLite's timezone-naive datetime storage."""

    if value is None:
        return None
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _orm_to_domain(instance: Any, domain_type: Type[DomainEntity]) -> DomainEntity:
    """Map one ORM row to a validated domain model without exposing the ORM."""

    data = {
        column.name: getattr(instance, column.name)
        for column in instance.__table__.columns
    }
    for field_name in ("started_at", "finished_at", "published_at", "collected_at", "created_at"):
        if field_name in data:
            data[field_name] = _as_utc(data[field_name])
    return domain_type.model_validate(data)


class AnalysisRepository:
    """Own all persistence and query access for normalized entities."""

    def __init__(self, database_path: Path | str):
        self.database_path = Path(database_path).expanduser().resolve()
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self.engine: AsyncEngine = create_async_engine(
            f"sqlite+aiosqlite:///{self.database_path.as_posix()}",
            future=True,
        )
        self._session_factory = async_sessionmaker(self.engine, expire_on_commit=False)
        self._write_lock = asyncio.Lock()

        @event.listens_for(self.engine.sync_engine, "connect")
        def _enable_foreign_keys(dbapi_connection, _connection_record) -> None:
            cursor = dbapi_connection.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.execute("PRAGMA busy_timeout=5000")
            cursor.close()

    async def initialize(self) -> None:
        """Create the normalized data tables owned by this repository."""

        tables = [
            CollectionRunModel.__table__,
            NormalizedAuthorModel.__table__,
            NormalizedContentModel.__table__,
            NormalizedCommentModel.__table__,
            CommentAnalysisModel.__table__,
            AnalysisJobModel.__table__,
            AnalysisJobItemModel.__table__,
        ]
        async with self.engine.begin() as connection:
            await connection.run_sync(lambda sync_connection: CollectionRunModel.metadata.create_all(sync_connection, tables=tables))

    async def close(self) -> None:
        await self.engine.dispose()

    async def upsert_collection_run(self, run: CollectionRun) -> None:
        row = _domain_row(run)
        statement = sqlite_insert(CollectionRunModel).values(row)
        update_values = {
            column.name: statement.excluded[column.name]
            for column in CollectionRunModel.__table__.columns
            if column.name != "id"
        }
        statement = statement.on_conflict_do_update(index_elements=["id"], set_=update_values)
        async with self._session_factory.begin() as session:
            await session.execute(statement)

    async def upsert_authors(self, authors: Sequence[NormalizedAuthor]) -> UpsertResult:
        return await self._upsert_platform_entities(
            NormalizedAuthorModel,
            authors,
            native_field="native_author_id",
            preserve_existing_nullables=True,
        )

    async def upsert_contents(self, contents: Sequence[NormalizedContent]) -> UpsertResult:
        return await self._upsert_platform_entities(
            NormalizedContentModel,
            contents,
            native_field="native_content_id",
        )

    async def upsert_comments(self, comments: Sequence[NormalizedComment]) -> UpsertResult:
        return await self._upsert_platform_entities(
            NormalizedCommentModel,
            comments,
            native_field="native_comment_id",
        )

    async def _upsert_platform_entities(
        self,
        orm_model: Type[Any],
        items: Sequence[Any],
        native_field: str,
        preserve_existing_nullables: bool = False,
    ) -> UpsertResult:
        if not items:
            return UpsertResult(inserted=0, duplicates=0)

        # Last occurrence wins inside a batch while all repeated rows are still
        # included in the duplicate count.
        unique_rows: dict[tuple[str, str], dict[str, Any]] = {}
        for item in items:
            row = _domain_row(item)
            unique_rows[(row["platform"], row[native_field])] = row

        inserted = 0
        async with self._session_factory.begin() as session:
            for platform in sorted({key[0] for key in unique_rows}):
                platform_rows = [row for (row_platform, _), row in unique_rows.items() if row_platform == platform]
                native_ids = [row[native_field] for row in platform_rows]
                native_column = getattr(orm_model, native_field)
                existing_result = await session.execute(
                    select(native_column).where(
                        orm_model.platform == platform,
                        native_column.in_(native_ids),
                    )
                )
                existing_ids = set(existing_result.scalars().all())
                inserted += len(native_ids) - len(existing_ids)

                statement = sqlite_insert(orm_model).values(platform_rows)
                update_values: dict[str, Any] = {}
                for column in orm_model.__table__.columns:
                    if column.name in {"id", "platform", native_field}:
                        continue
                    excluded_value = statement.excluded[column.name]
                    if preserve_existing_nullables and column.nullable:
                        update_values[column.name] = func.coalesce(excluded_value, getattr(orm_model, column.name))
                    else:
                        update_values[column.name] = excluded_value
                statement = statement.on_conflict_do_update(
                    index_elements=["platform", native_field],
                    set_=update_values,
                )
                await session.execute(statement)

        return UpsertResult(inserted=inserted, duplicates=len(items) - inserted)

    async def existing_content_ids(self, platform: str, native_ids: Iterable[str]) -> set[str]:
        unique_ids = set(native_ids)
        if not unique_ids:
            return set()
        async with self._session_factory() as session:
            result = await session.execute(
                select(NormalizedContentModel.native_content_id).where(
                    NormalizedContentModel.platform == platform,
                    NormalizedContentModel.native_content_id.in_(unique_ids),
                )
            )
            return set(result.scalars().all())

    async def get_content(self, content_id: str) -> NormalizedContent | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(NormalizedContentModel).where(NormalizedContentModel.id == content_id)
            )
            row = result.scalar_one_or_none()
        return _orm_to_domain(row, NormalizedContent) if row else None

    async def list_contents(
        self,
        *,
        platform: Optional[str] = None,
        native_content_id: Optional[str] = None,
        run_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[NormalizedContent], int]:
        filters = []
        if platform is not None:
            filters.append(NormalizedContentModel.platform == platform)
        if native_content_id is not None:
            filters.append(NormalizedContentModel.native_content_id == native_content_id)
        if run_id is not None:
            filters.append(NormalizedContentModel.run_id == run_id)

        async with self._session_factory() as session:
            total = await session.scalar(
                select(func.count()).select_from(NormalizedContentModel).where(*filters)
            ) or 0
            result = await session.execute(
                select(NormalizedContentModel)
                .where(*filters)
                .order_by(
                    NormalizedContentModel.published_at.is_(None),
                    NormalizedContentModel.published_at.desc(),
                    NormalizedContentModel.id,
                )
                .limit(limit)
                .offset(offset)
            )
            rows = result.scalars().all()
        return [_orm_to_domain(row, NormalizedContent) for row in rows], int(total)

    async def get_comment(self, comment_id: str) -> NormalizedComment | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(NormalizedCommentModel).where(NormalizedCommentModel.id == comment_id)
            )
            row = result.scalar_one_or_none()
        return _orm_to_domain(row, NormalizedComment) if row else None

    async def get_comments_by_ids(self, comment_ids: Sequence[str]) -> list[NormalizedComment]:
        unique_ids = list(dict.fromkeys(comment_ids))
        if not unique_ids:
            return []
        async with self._session_factory() as session:
            result = await session.execute(
                select(NormalizedCommentModel).where(NormalizedCommentModel.id.in_(unique_ids))
            )
            rows = result.scalars().all()
        by_id = {row.id: _orm_to_domain(row, NormalizedComment) for row in rows}
        return [by_id[comment_id] for comment_id in unique_ids if comment_id in by_id]

    async def list_comments(
        self,
        *,
        platform: Optional[str] = None,
        content_id: Optional[str] = None,
        native_content_id: Optional[str] = None,
        run_id: Optional[str] = None,
        parent_comment_id: Optional[str] = None,
        depth: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[NormalizedComment], int]:
        filters = []
        if platform is not None:
            filters.append(NormalizedCommentModel.platform == platform)
        if content_id is not None:
            filters.append(NormalizedCommentModel.content_id == content_id)
        if native_content_id is not None:
            filters.append(NormalizedCommentModel.native_content_id == native_content_id)
        if run_id is not None:
            filters.append(NormalizedCommentModel.run_id == run_id)
        if parent_comment_id is not None:
            filters.append(NormalizedCommentModel.parent_comment_id == parent_comment_id)
        if depth is not None:
            filters.append(NormalizedCommentModel.depth == depth)

        async with self._session_factory() as session:
            total = await session.scalar(
                select(func.count()).select_from(NormalizedCommentModel).where(*filters)
            ) or 0
            result = await session.execute(
                select(NormalizedCommentModel)
                .where(*filters)
                .order_by(
                    NormalizedCommentModel.published_at.is_(None),
                    NormalizedCommentModel.published_at,
                    NormalizedCommentModel.id,
                )
                .limit(limit)
                .offset(offset)
            )
            rows = result.scalars().all()
        return [_orm_to_domain(row, NormalizedComment) for row in rows], int(total)

    async def get_author(self, author_id: str) -> NormalizedAuthor | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(NormalizedAuthorModel).where(NormalizedAuthorModel.id == author_id)
            )
            row = result.scalar_one_or_none()
        return _orm_to_domain(row, NormalizedAuthor) if row else None

    async def list_authors(
        self,
        *,
        platform: Optional[str] = None,
        native_author_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[NormalizedAuthor], int]:
        filters = []
        if platform is not None:
            filters.append(NormalizedAuthorModel.platform == platform)
        if native_author_id is not None:
            filters.append(NormalizedAuthorModel.native_author_id == native_author_id)

        async with self._session_factory() as session:
            total = await session.scalar(
                select(func.count()).select_from(NormalizedAuthorModel).where(*filters)
            ) or 0
            result = await session.execute(
                select(NormalizedAuthorModel)
                .where(*filters)
                .order_by(NormalizedAuthorModel.nickname.is_(None), NormalizedAuthorModel.nickname, NormalizedAuthorModel.id)
                .limit(limit)
                .offset(offset)
            )
            rows = result.scalars().all()
        return [_orm_to_domain(row, NormalizedAuthor) for row in rows], int(total)

    async def get_content_statistics(
        self,
        *,
        platform: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> ContentQueryStatistics:
        filters = []
        if platform is not None:
            filters.append(NormalizedContentModel.platform == platform)
        if run_id is not None:
            filters.append(NormalizedContentModel.run_id == run_id)
        async with self._session_factory() as session:
            count = await session.scalar(
                select(func.count()).select_from(NormalizedContentModel).where(*filters)
            ) or 0
        return ContentQueryStatistics(content_count=int(count))

    async def get_comment_statistics(
        self,
        *,
        platform: Optional[str] = None,
        content_id: Optional[str] = None,
        native_content_id: Optional[str] = None,
        run_id: Optional[str] = None,
    ) -> CommentQueryStatistics:
        filters = []
        if platform is not None:
            filters.append(NormalizedCommentModel.platform == platform)
        if content_id is not None:
            filters.append(NormalizedCommentModel.content_id == content_id)
        if native_content_id is not None:
            filters.append(NormalizedCommentModel.native_content_id == native_content_id)
        if run_id is not None:
            filters.append(NormalizedCommentModel.run_id == run_id)
        statement = select(
            func.count(),
            func.sum(case((NormalizedCommentModel.depth == 0, 1), else_=0)),
            func.sum(case((NormalizedCommentModel.depth > 0, 1), else_=0)),
            func.min(NormalizedCommentModel.published_at),
            func.max(NormalizedCommentModel.published_at),
        ).where(*filters)
        async with self._session_factory() as session:
            result = await session.execute(statement)
            count, roots, replies, earliest, latest = result.one()
        return CommentQueryStatistics(
            comment_count=int(count or 0),
            root_comment_count=int(roots or 0),
            reply_comment_count=int(replies or 0),
            earliest_comment_time=_as_utc(earliest),
            latest_comment_time=_as_utc(latest),
        )

    async def get_author_statistics(self, *, platform: Optional[str] = None) -> AuthorQueryStatistics:
        filters = []
        if platform is not None:
            filters.append(NormalizedAuthorModel.platform == platform)
        async with self._session_factory() as session:
            count = await session.scalar(
                select(func.count()).select_from(NormalizedAuthorModel).where(*filters)
            ) or 0
        return AuthorQueryStatistics(author_count=int(count))

    @staticmethod
    def _comment_scope_filters(
        *,
        platform: Optional[str] = None,
        run_id: Optional[str] = None,
        content_id: Optional[str] = None,
    ) -> list[Any]:
        filters = []
        if platform is not None:
            filters.append(NormalizedCommentModel.platform == platform)
        if run_id is not None:
            filters.append(NormalizedCommentModel.run_id == run_id)
        if content_id is not None:
            filters.append(NormalizedCommentModel.content_id == content_id)
        return filters

    def _latest_analysis_cte(
        self,
        *,
        platform: Optional[str] = None,
        run_id: Optional[str] = None,
        content_id: Optional[str] = None,
    ) -> Any:
        """Return one latest versioned analysis per scoped comment in SQL."""

        filters = self._comment_scope_filters(platform=platform, run_id=run_id, content_id=content_id)
        ranked = (
            select(
                CommentAnalysisModel.id.label("id"),
                CommentAnalysisModel.comment_id.label("comment_id"),
                CommentAnalysisModel.sentiment.label("sentiment"),
                CommentAnalysisModel.sentiment_score.label("sentiment_score"),
                CommentAnalysisModel.emotion.label("emotion"),
                CommentAnalysisModel.topics.label("topics"),
                CommentAnalysisModel.stance.label("stance"),
                CommentAnalysisModel.risk_level.label("risk_level"),
                CommentAnalysisModel.risk_reasons.label("risk_reasons"),
                CommentAnalysisModel.keywords.label("keywords"),
                CommentAnalysisModel.summary.label("summary"),
                func.row_number().over(
                    partition_by=CommentAnalysisModel.comment_id,
                    order_by=(CommentAnalysisModel.created_at.desc(), CommentAnalysisModel.id.desc()),
                ).label("row_number"),
            )
            .select_from(CommentAnalysisModel)
            .join(NormalizedCommentModel, CommentAnalysisModel.comment_id == NormalizedCommentModel.id)
            .where(*filters)
            .cte("ranked_comment_analyses")
        )
        return select(ranked).where(ranked.c.row_number == 1).cte("latest_comment_analyses")

    async def get_public_opinion_counts(
        self,
        *,
        platform: Optional[str] = None,
        run_id: Optional[str] = None,
        content_id: Optional[str] = None,
    ) -> tuple[int, int]:
        filters = self._comment_scope_filters(platform=platform, run_id=run_id, content_id=content_id)
        latest = self._latest_analysis_cte(platform=platform, run_id=run_id, content_id=content_id)
        async with self._session_factory() as session:
            total = await session.scalar(select(func.count()).select_from(NormalizedCommentModel).where(*filters)) or 0
            analyzed = await session.scalar(select(func.count()).select_from(latest)) or 0
        return int(total), int(analyzed)

    async def get_analysis_distribution(
        self,
        field: str,
        *,
        platform: Optional[str] = None,
        run_id: Optional[str] = None,
        content_id: Optional[str] = None,
    ) -> list[AggregateValue]:
        allowed_fields = {"sentiment", "emotion", "stance", "risk_level"}
        if field not in allowed_fields:
            raise ValueError(f"Unsupported analysis distribution field: {field}")
        latest = self._latest_analysis_cte(platform=platform, run_id=run_id, content_id=content_id)
        column = latest.c[field]
        statement = (
            select(column, func.count().label("count"))
            .select_from(latest)
            .group_by(column)
            .order_by(column)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        return [AggregateValue(value=str(value), count=int(count)) for value, count in rows]

    async def get_sentiment_score_statistics(
        self,
        *,
        platform: Optional[str] = None,
        run_id: Optional[str] = None,
        content_id: Optional[str] = None,
    ) -> SentimentScoreAggregate:
        latest = self._latest_analysis_cte(platform=platform, run_id=run_id, content_id=content_id)
        statement = select(
            func.min(latest.c.sentiment_score),
            func.max(latest.c.sentiment_score),
            func.avg(latest.c.sentiment_score),
        )
        async with self._session_factory() as session:
            minimum, maximum, mean = (await session.execute(statement)).one()
        return SentimentScoreAggregate(
            minimum=float(minimum) if minimum is not None else None,
            maximum=float(maximum) if maximum is not None else None,
            mean=float(mean) if mean is not None else None,
        )

    async def get_json_frequency(
        self,
        field: str,
        *,
        platform: Optional[str] = None,
        run_id: Optional[str] = None,
        content_id: Optional[str] = None,
        top_k: int = 20,
    ) -> list[AggregateValue]:
        allowed_fields = {"topics", "keywords", "risk_reasons"}
        if field not in allowed_fields:
            raise ValueError(f"Unsupported JSON frequency field: {field}")
        latest = self._latest_analysis_cte(platform=platform, run_id=run_id, content_id=content_id)
        item = func.json_each(latest.c[field]).table_valued("key", "value").alias("item")
        value = item.c.value
        statement = (
            select(value.label("value"), func.count().label("count"))
            .select_from(latest.join(item, true()))
            .where(func.trim(value) != "")
            .group_by(value)
            .order_by(func.count().desc(), value.asc())
            .limit(top_k)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        return [AggregateValue(value=str(value), count=int(count)) for value, count in rows]

    async def get_topic_dimension_distribution(
        self,
        dimension: str,
        *,
        platform: Optional[str] = None,
        run_id: Optional[str] = None,
        content_id: Optional[str] = None,
        top_k: int = 20,
    ) -> list[TopicDimensionAggregate]:
        allowed_dimensions = {"sentiment", "risk_level"}
        if dimension not in allowed_dimensions:
            raise ValueError(f"Unsupported topic dimension: {dimension}")
        latest = self._latest_analysis_cte(platform=platform, run_id=run_id, content_id=content_id)
        ranked_topic_item = func.json_each(latest.c.topics).table_valued("key", "value").alias("ranked_topic_item")
        ranked_topic = ranked_topic_item.c.value
        top_topics = (
            select(ranked_topic.label("topic"), func.count().label("count"))
            .select_from(latest.join(ranked_topic_item, true()))
            .where(ranked_topic != "")
            .group_by(ranked_topic)
            .order_by(func.count().desc(), ranked_topic.asc())
            .limit(top_k)
            .cte("top_topics")
        )
        topic_item = func.json_each(latest.c.topics).table_valued("key", "value").alias("topic_item")
        topic = topic_item.c.value
        statement = (
            select(topic.label("topic"), latest.c[dimension].label("dimension"), func.count().label("count"))
            .select_from(
                latest.join(topic_item, true()).join(top_topics, topic == top_topics.c.topic)
            )
            .where(topic != "")
            .group_by(topic, latest.c[dimension])
            .order_by(topic.asc(), latest.c[dimension].asc())
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        return [
            TopicDimensionAggregate(topic=str(topic_value), dimension=str(value), count=int(count))
            for topic_value, value, count in rows
        ]

    async def get_comment_volume_by_date(
        self,
        *,
        platform: Optional[str] = None,
        run_id: Optional[str] = None,
        content_id: Optional[str] = None,
    ) -> list[CommentVolumeAggregate]:
        filters = self._comment_scope_filters(platform=platform, run_id=run_id, content_id=content_id)
        comment_date = func.date(NormalizedCommentModel.published_at)
        statement = (
            select(comment_date.label("date"), func.count().label("count"))
            .select_from(NormalizedCommentModel)
            .where(*filters, NormalizedCommentModel.published_at.is_not(None))
            .group_by(comment_date)
            .order_by(comment_date.asc())
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        return [CommentVolumeAggregate(date=str(value), count=int(count)) for value, count in rows]

    async def get_trend_buckets(
        self,
        *,
        bucket: str,
        platform: Optional[str] = None,
        run_id: Optional[str] = None,
        content_id: Optional[str] = None,
        start_at: datetime | None = None,
        end_at: datetime | None = None,
    ) -> list[TrendBucketAggregate]:
        """Aggregate one latest analysis per comment into bounded SQL time buckets."""

        if bucket not in {"hour", "day"}:
            raise ValueError(f"Unsupported trend bucket: {bucket}")
        filters = self._comment_scope_filters(
            platform=platform, run_id=run_id, content_id=content_id
        )
        filters.append(NormalizedCommentModel.published_at.is_not(None))
        if start_at is not None:
            filters.append(NormalizedCommentModel.published_at >= start_at)
        if end_at is not None:
            filters.append(NormalizedCommentModel.published_at < end_at)
        latest = self._latest_analysis_cte(
            platform=platform, run_id=run_id, content_id=content_id
        )
        pattern = "%Y-%m-%dT%H:00:00" if bucket == "hour" else "%Y-%m-%dT00:00:00"
        bucket_start = func.strftime(pattern, NormalizedCommentModel.published_at)

        def count_when(condition):
            return func.sum(case((condition, 1), else_=0))

        statement = (
            select(
                bucket_start.label("bucket_start"),
                func.count(NormalizedCommentModel.id).label("total_comments"),
                count_when(latest.c.comment_id.is_not(None)).label("analyzed_comments"),
                count_when(latest.c.sentiment == "positive").label("positive_count"),
                count_when(latest.c.sentiment == "neutral").label("neutral_count"),
                count_when(latest.c.sentiment == "negative").label("negative_count"),
                count_when(latest.c.risk_level == "low").label("low_risk_count"),
                count_when(latest.c.risk_level == "medium").label("medium_risk_count"),
                count_when(latest.c.risk_level == "high").label("high_risk_count"),
                func.avg(latest.c.sentiment_score).label("mean_sentiment_score"),
            )
            .select_from(
                NormalizedCommentModel.__table__.outerjoin(
                    latest, latest.c.comment_id == NormalizedCommentModel.id
                )
            )
            .where(*filters)
            .group_by(bucket_start)
            .order_by(bucket_start.asc())
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        return [
            TrendBucketAggregate(
                bucket_start=str(row.bucket_start),
                total_comments=int(row.total_comments or 0),
                analyzed_comments=int(row.analyzed_comments or 0),
                positive_count=int(row.positive_count or 0),
                neutral_count=int(row.neutral_count or 0),
                negative_count=int(row.negative_count or 0),
                low_risk_count=int(row.low_risk_count or 0),
                medium_risk_count=int(row.medium_risk_count or 0),
                high_risk_count=int(row.high_risk_count or 0),
                mean_sentiment_score=(
                    float(row.mean_sentiment_score)
                    if row.mean_sentiment_score is not None
                    else None
                ),
            )
            for row in rows
        ]

    async def list_high_risk_comments(
        self,
        *,
        platform: Optional[str] = None,
        run_id: Optional[str] = None,
        content_id: Optional[str] = None,
        limit: int = 20,
    ) -> list[HighRiskCommentAggregate]:
        if not 1 <= limit <= 100:
            raise ValueError("limit must be between 1 and 100")
        latest = self._latest_analysis_cte(
            platform=platform, run_id=run_id, content_id=content_id
        )
        statement = (
            select(
                NormalizedCommentModel,
                latest.c.sentiment,
                latest.c.sentiment_score,
                latest.c.risk_level,
                latest.c.risk_reasons,
                latest.c.summary,
            )
            .join(latest, latest.c.comment_id == NormalizedCommentModel.id)
            .where(latest.c.risk_level == "high")
            .order_by(
                NormalizedCommentModel.like_count.is_(None),
                NormalizedCommentModel.like_count.desc(),
                NormalizedCommentModel.published_at.desc(),
                NormalizedCommentModel.id,
            )
            .limit(limit)
        )
        async with self._session_factory() as session:
            rows = (await session.execute(statement)).all()
        return [
            HighRiskCommentAggregate(
                comment=_orm_to_domain(row[0], NormalizedComment),
                sentiment=str(row.sentiment),
                sentiment_score=float(row.sentiment_score),
                risk_level=str(row.risk_level),
                risk_reasons=list(row.risk_reasons or []),
                summary=str(row.summary),
            )
            for row in rows
        ]

    async def get_cached_analysis(
        self,
        *,
        comment_id: str,
        input_hash: str,
        provider: str,
        model: str,
        prompt_version: str,
        analysis_version: str,
    ) -> CommentAnalysis | None:
        """Return an exact versioned cache hit without fuzzy matching."""

        statement = select(CommentAnalysisModel).where(
            CommentAnalysisModel.comment_id == comment_id,
            CommentAnalysisModel.input_hash == input_hash,
            CommentAnalysisModel.provider == provider,
            CommentAnalysisModel.model == model,
            CommentAnalysisModel.prompt_version == prompt_version,
            CommentAnalysisModel.analysis_version == analysis_version,
        )
        async with self._session_factory() as session:
            result = await session.execute(statement)
            row = result.scalar_one_or_none()
        return _orm_to_domain(row, CommentAnalysis) if row else None

    async def get_cached_analysis_by_content_hash(
        self,
        *,
        input_hash: str,
        provider: str,
        model: str,
        prompt_version: str,
        analysis_version: str,
    ) -> CommentAnalysis | None:
        """Return a reusable result for identical normalized text and versions."""

        statement = (
            select(CommentAnalysisModel)
            .where(
                CommentAnalysisModel.input_hash == input_hash,
                CommentAnalysisModel.provider == provider,
                CommentAnalysisModel.model == model,
                CommentAnalysisModel.prompt_version == prompt_version,
                CommentAnalysisModel.analysis_version == analysis_version,
            )
            .order_by(CommentAnalysisModel.created_at, CommentAnalysisModel.id)
            .limit(1)
        )
        async with self._session_factory() as session:
            result = await session.execute(statement)
            row = result.scalar_one_or_none()
        return _orm_to_domain(row, CommentAnalysis) if row else None

    async def save_analysis(self, analysis: CommentAnalysis) -> CommentAnalysis:
        """Persist once by cache key and return the winning stored row."""

        row = _domain_row(analysis)
        statement = sqlite_insert(CommentAnalysisModel).values(row).on_conflict_do_nothing(
            index_elements=[
                "comment_id",
                "input_hash",
                "provider",
                "model",
                "prompt_version",
                "analysis_version",
            ]
        )
        async with self._write_lock:
            async with self._session_factory.begin() as session:
                await session.execute(statement)
        stored = await self.get_cached_analysis(
            comment_id=str(analysis.comment_id),
            input_hash=analysis.input_hash,
            provider=analysis.provider,
            model=analysis.model,
            prompt_version=analysis.prompt_version,
            analysis_version=analysis.analysis_version,
        )
        if stored is None:
            raise RuntimeError("Comment analysis was not persisted")
        return stored

    async def count_comment_analyses(self, *, comment_id: Optional[str] = None) -> int:
        statement = select(func.count()).select_from(CommentAnalysisModel)
        if comment_id is not None:
            statement = statement.where(CommentAnalysisModel.comment_id == comment_id)
        async with self._session_factory() as session:
            count = await session.scalar(statement) or 0
        return int(count)

    async def create_analysis_job(
        self,
        job: AnalysisJob,
        items: Sequence[AnalysisJobItem],
    ) -> AnalysisJob:
        job_row = _domain_row(job)
        item_rows = [_domain_row(item) for item in items]
        async with self._write_lock:
            async with self._session_factory.begin() as session:
                await session.execute(sqlite_insert(AnalysisJobModel).values(job_row))
                if item_rows:
                    await session.execute(sqlite_insert(AnalysisJobItemModel).values(item_rows))
        stored = await self.get_analysis_job(str(job.id))
        if stored is None:
            raise RuntimeError("Analysis job was not persisted")
        return stored

    async def get_analysis_job(self, job_id: str) -> AnalysisJob | None:
        async with self._session_factory() as session:
            result = await session.execute(
                select(AnalysisJobModel).where(AnalysisJobModel.id == job_id)
            )
            row = result.scalar_one_or_none()
        return _orm_to_domain(row, AnalysisJob) if row else None

    async def list_job_items(
        self,
        job_id: str,
        *,
        statuses: Optional[Sequence[AnalysisJobItemStatus | str]] = None,
        limit: int = 100,
    ) -> list[AnalysisJobItem]:
        statement = select(AnalysisJobItemModel).where(AnalysisJobItemModel.job_id == job_id)
        if statuses:
            status_values = [status.value if hasattr(status, "value") else str(status) for status in statuses]
            statement = statement.where(AnalysisJobItemModel.status.in_(status_values))
        statement = statement.order_by(AnalysisJobItemModel.id).limit(limit)
        async with self._session_factory() as session:
            result = await session.execute(statement)
            rows = result.scalars().all()
        return [_orm_to_domain(row, AnalysisJobItem) for row in rows]

    async def claim_job_item(self, item_id: str, started_at: datetime) -> bool:
        async with self._write_lock:
            async with self._session_factory.begin() as session:
                result = await session.execute(
                    update(AnalysisJobItemModel)
                    .where(
                        AnalysisJobItemModel.id == item_id,
                        AnalysisJobItemModel.status == AnalysisJobItemStatus.PENDING.value,
                    )
                    .values(
                        status=AnalysisJobItemStatus.PROCESSING.value,
                        attempt_count=AnalysisJobItemModel.attempt_count + 1,
                        started_at=started_at,
                        finished_at=None,
                        error_type=None,
                        error_message=None,
                    )
                )
                if result.rowcount != 1:
                    return False
                item_job_id = await session.scalar(
                    select(AnalysisJobItemModel.job_id).where(AnalysisJobItemModel.id == item_id)
                )
                await session.execute(
                    update(AnalysisJobModel)
                    .where(AnalysisJobModel.id == item_job_id)
                    .values(
                        pending_count=AnalysisJobModel.pending_count - 1,
                        processing_count=AnalysisJobModel.processing_count + 1,
                    )
                )
        return True

    async def finish_job_item(
        self,
        item_id: str,
        *,
        status: AnalysisJobItemStatus,
        finished_at: datetime,
        error_type: Optional[str] = None,
        error_message: Optional[str] = None,
    ) -> bool:
        if status not in {
            AnalysisJobItemStatus.COMPLETED,
            AnalysisJobItemStatus.FAILED,
            AnalysisJobItemStatus.SKIPPED,
        }:
            raise ValueError(f"Invalid terminal item status: {status}")
        count_column = {
            AnalysisJobItemStatus.COMPLETED: AnalysisJobModel.completed_count,
            AnalysisJobItemStatus.FAILED: AnalysisJobModel.failed_count,
            AnalysisJobItemStatus.SKIPPED: AnalysisJobModel.skipped_count,
        }[status]
        async with self._write_lock:
            async with self._session_factory.begin() as session:
                item_job_id = await session.scalar(
                    select(AnalysisJobItemModel.job_id).where(AnalysisJobItemModel.id == item_id)
                )
                result = await session.execute(
                    update(AnalysisJobItemModel)
                    .where(
                        AnalysisJobItemModel.id == item_id,
                        AnalysisJobItemModel.status == AnalysisJobItemStatus.PROCESSING.value,
                    )
                    .values(
                        status=status.value,
                        finished_at=finished_at,
                        error_type=error_type,
                        error_message=error_message,
                    )
                )
                if result.rowcount != 1:
                    return False
                await session.execute(
                    update(AnalysisJobModel)
                    .where(AnalysisJobModel.id == item_job_id)
                    .values(
                        processing_count=AnalysisJobModel.processing_count - 1,
                        **{count_column.key: count_column + 1},
                    )
                )
        return True

    async def reset_processing_job_items(self, job_id: str) -> int:
        async with self._write_lock:
            async with self._session_factory.begin() as session:
                processing_count = await session.scalar(
                    select(func.count()).select_from(AnalysisJobItemModel).where(
                        AnalysisJobItemModel.job_id == job_id,
                        AnalysisJobItemModel.status == AnalysisJobItemStatus.PROCESSING.value,
                    )
                ) or 0
                if processing_count:
                    await session.execute(
                        update(AnalysisJobItemModel)
                        .where(
                            AnalysisJobItemModel.job_id == job_id,
                            AnalysisJobItemModel.status == AnalysisJobItemStatus.PROCESSING.value,
                        )
                        .values(status=AnalysisJobItemStatus.PENDING.value, started_at=None)
                    )
                    await session.execute(
                        update(AnalysisJobModel)
                        .where(AnalysisJobModel.id == job_id)
                        .values(
                            pending_count=AnalysisJobModel.pending_count + int(processing_count),
                            processing_count=AnalysisJobModel.processing_count - int(processing_count),
                        )
                    )
        return int(processing_count)

    async def update_analysis_job_status(
        self,
        job_id: str,
        status: AnalysisJobStatus,
        *,
        started_at: Optional[datetime] = None,
        finished_at: Optional[datetime] = None,
        error_message: Optional[str] = None,
    ) -> AnalysisJob | None:
        values: dict[str, Any] = {"status": status.value, "error_message": error_message}
        if started_at is not None:
            values["started_at"] = func.coalesce(AnalysisJobModel.started_at, started_at)
        if finished_at is not None:
            values["finished_at"] = finished_at
        async with self._write_lock:
            async with self._session_factory.begin() as session:
                await session.execute(
                    update(AnalysisJobModel).where(AnalysisJobModel.id == job_id).values(**values)
                )
        return await self.get_analysis_job(job_id)

    async def cancel_analysis_job(self, job_id: str, finished_at: datetime) -> AnalysisJob | None:
        async with self._write_lock:
            async with self._session_factory.begin() as session:
                await session.execute(
                    update(AnalysisJobModel)
                    .where(
                        AnalysisJobModel.id == job_id,
                        AnalysisJobModel.status.in_([
                            AnalysisJobStatus.PENDING.value,
                            AnalysisJobStatus.RUNNING.value,
                            AnalysisJobStatus.INTERRUPTED.value,
                        ]),
                    )
                    .values(status=AnalysisJobStatus.CANCELLED.value, finished_at=finished_at)
                )
        return await self.get_analysis_job(job_id)

    async def failed_job_comment_ids(self, job_id: str) -> list[str]:
        async with self._session_factory() as session:
            result = await session.execute(
                select(AnalysisJobItemModel.comment_id)
                .where(
                    AnalysisJobItemModel.job_id == job_id,
                    AnalysisJobItemModel.status == AnalysisJobItemStatus.FAILED.value,
                )
                .order_by(AnalysisJobItemModel.id)
            )
            return list(result.scalars().all())

    async def stats_for_run(self, run_id: str) -> ImportDatabaseStats:
        """Return acceptance statistics after all batches are committed."""

        async with self._session_factory() as session:
            content_count = await session.scalar(
                select(func.count()).select_from(NormalizedContentModel).where(NormalizedContentModel.run_id == run_id)
            ) or 0
            comment_count = await session.scalar(
                select(func.count()).select_from(NormalizedCommentModel).where(NormalizedCommentModel.run_id == run_id)
            ) or 0
            first_level_count = await session.scalar(
                select(func.count()).select_from(NormalizedCommentModel).where(
                    NormalizedCommentModel.run_id == run_id,
                    NormalizedCommentModel.depth == 0,
                )
            ) or 0
            second_level_count = await session.scalar(
                select(func.count()).select_from(NormalizedCommentModel).where(
                    NormalizedCommentModel.run_id == run_id,
                    NormalizedCommentModel.depth > 0,
                )
            ) or 0
            time_result = await session.execute(
                select(
                    func.min(NormalizedCommentModel.published_at),
                    func.max(NormalizedCommentModel.published_at),
                ).where(NormalizedCommentModel.run_id == run_id)
            )
            earliest, latest = time_result.one()

            content_author_result = await session.execute(
                select(NormalizedContentModel.author_id).where(
                    NormalizedContentModel.run_id == run_id,
                    NormalizedContentModel.author_id.is_not(None),
                )
            )
            comment_author_result = await session.execute(
                select(NormalizedCommentModel.author_id).where(
                    NormalizedCommentModel.run_id == run_id,
                    NormalizedCommentModel.author_id.is_not(None),
                )
            )
            author_ids = set(content_author_result.scalars()).union(comment_author_result.scalars())

            total_contents = await session.scalar(select(func.count()).select_from(NormalizedContentModel)) or 0
            total_comments = await session.scalar(select(func.count()).select_from(NormalizedCommentModel)) or 0
            total_authors = await session.scalar(select(func.count()).select_from(NormalizedAuthorModel)) or 0

        return ImportDatabaseStats(
            content_count=int(content_count),
            comment_count=int(comment_count),
            first_level_comment_count=int(first_level_count),
            second_level_comment_count=int(second_level_count),
            author_count=len(author_ids),
            video_count=int(content_count),
            earliest_comment_at=_as_utc(earliest),
            latest_comment_at=_as_utc(latest),
            total_database_contents=int(total_contents),
            total_database_comments=int(total_comments),
            total_database_authors=int(total_authors),
        )
