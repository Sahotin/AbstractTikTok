"""Stable read boundary for normalized data consumers and future tools."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Generic, Optional, TypeVar
from uuid import UUID

from analysis.domain import NormalizedAuthor, NormalizedComment, NormalizedContent, Platform
from analysis.repositories import (
    AnalysisRepository,
    AuthorQueryStatistics,
    CommentQueryStatistics,
    ContentQueryStatistics,
)


Entity = TypeVar("Entity")


@dataclass(frozen=True)
class QueryPage(Generic[Entity]):
    """One bounded page plus its SQL-computed total."""

    items: list[Entity]
    total: int
    limit: int
    offset: int


def _value(value: UUID | Platform | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Platform):
        return value.value
    return str(value)


def _validate_page(limit: int, offset: int) -> None:
    if not 1 <= limit <= 1000:
        raise ValueError("limit must be between 1 and 1000")
    if offset < 0:
        raise ValueError("offset must be non-negative")


class AnalysisQueryService:
    """Read normalized entities without exposing SQLAlchemy to callers."""

    def __init__(self, repository: AnalysisRepository):
        self.repository = repository

    async def get_content(self, content_id: UUID | str) -> NormalizedContent | None:
        return await self.repository.get_content(str(content_id))

    async def list_contents(
        self,
        *,
        platform: Platform | str | None = None,
        native_content_id: Optional[str] = None,
        run_id: UUID | str | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> QueryPage[NormalizedContent]:
        _validate_page(limit, offset)
        items, total = await self.repository.list_contents(
            platform=_value(platform),
            native_content_id=native_content_id,
            run_id=_value(run_id),
            limit=limit,
            offset=offset,
        )
        return QueryPage(items=items, total=total, limit=limit, offset=offset)

    async def get_comment(self, comment_id: UUID | str) -> NormalizedComment | None:
        return await self.repository.get_comment(str(comment_id))

    async def list_comments(
        self,
        *,
        platform: Platform | str | None = None,
        content_id: UUID | str | None = None,
        native_content_id: Optional[str] = None,
        run_id: UUID | str | None = None,
        parent_comment_id: Optional[str] = None,
        depth: Optional[int] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> QueryPage[NormalizedComment]:
        _validate_page(limit, offset)
        if depth is not None and depth < 0:
            raise ValueError("depth must be non-negative")
        items, total = await self.repository.list_comments(
            platform=_value(platform),
            content_id=_value(content_id),
            native_content_id=native_content_id,
            run_id=_value(run_id),
            parent_comment_id=parent_comment_id,
            depth=depth,
            limit=limit,
            offset=offset,
        )
        return QueryPage(items=items, total=total, limit=limit, offset=offset)

    async def get_author(self, author_id: UUID | str) -> NormalizedAuthor | None:
        return await self.repository.get_author(str(author_id))

    async def list_authors(
        self,
        *,
        platform: Platform | str | None = None,
        native_author_id: Optional[str] = None,
        limit: int = 50,
        offset: int = 0,
    ) -> QueryPage[NormalizedAuthor]:
        _validate_page(limit, offset)
        items, total = await self.repository.list_authors(
            platform=_value(platform),
            native_author_id=native_author_id,
            limit=limit,
            offset=offset,
        )
        return QueryPage(items=items, total=total, limit=limit, offset=offset)

    async def get_content_statistics(
        self,
        *,
        platform: Platform | str | None = None,
        run_id: UUID | str | None = None,
    ) -> ContentQueryStatistics:
        return await self.repository.get_content_statistics(
            platform=_value(platform),
            run_id=_value(run_id),
        )

    async def get_comment_statistics(
        self,
        *,
        platform: Platform | str | None = None,
        content_id: UUID | str | None = None,
        native_content_id: Optional[str] = None,
        run_id: UUID | str | None = None,
    ) -> CommentQueryStatistics:
        return await self.repository.get_comment_statistics(
            platform=_value(platform),
            content_id=_value(content_id),
            native_content_id=native_content_id,
            run_id=_value(run_id),
        )

    async def get_author_statistics(
        self,
        *,
        platform: Platform | str | None = None,
    ) -> AuthorQueryStatistics:
        return await self.repository.get_author_statistics(platform=_value(platform))
