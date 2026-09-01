"""Validated domain models shared by ingestion and future AI modules."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class Platform(str, Enum):
    """Platforms understood by the normalized data layer."""

    DOUYIN = "douyin"
    XHS = "xhs"
    KUAISHOU = "kuaishou"
    BILIBILI = "bilibili"
    WEIBO = "weibo"
    TIEBA = "tieba"
    ZHIHU = "zhihu"


class CollectionRunStatus(str, Enum):
    """Lifecycle states for an ingestion run."""

    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class DomainModel(BaseModel):
    """Strict base model so schema drift is caught at ingestion time."""

    model_config = ConfigDict(extra="forbid")


class CollectionRun(DomainModel):
    """One stable, reproducible import of one or more crawler output files."""

    id: UUID
    platform: Platform
    crawler_type: str
    query: Optional[str] = None
    specified_ids: list[str] = Field(default_factory=list)
    status: CollectionRunStatus
    started_at: datetime
    finished_at: Optional[datetime] = None
    config_snapshot: dict[str, Any] = Field(default_factory=dict)
    output_location: list[str] = Field(default_factory=list)
    content_count: int = Field(default=0, ge=0)
    comment_count: int = Field(default=0, ge=0)
    error_message: Optional[str] = None


class NormalizedAuthor(DomainModel):
    """Cross-platform author fields plus platform-specific attributes."""

    id: UUID
    platform: Platform
    native_author_id: str
    nickname: Optional[str] = None
    profile_url: Optional[str] = None
    avatar_url: Optional[str] = None
    description: Optional[str] = None
    gender: Optional[str] = None
    ip_location: Optional[str] = None
    following_count: Optional[int] = Field(default=None, ge=0)
    follower_count: Optional[int] = Field(default=None, ge=0)
    content_count: Optional[int] = Field(default=None, ge=0)
    verified: Optional[bool] = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    raw_payload: dict[str, Any] = Field(default_factory=dict)
    collected_at: datetime


class NormalizedContent(DomainModel):
    """Cross-platform content/video representation."""

    id: UUID
    run_id: UUID
    platform: Platform
    native_content_id: str
    content_type: str
    title: Optional[str] = None
    body: Optional[str] = None
    url: Optional[str] = None
    author_id: Optional[UUID] = None
    native_author_id: Optional[str] = None
    published_at: Optional[datetime] = None
    like_count: Optional[int] = Field(default=None, ge=0)
    comment_count: Optional[int] = Field(default=None, ge=0)
    share_count: Optional[int] = Field(default=None, ge=0)
    favorite_count: Optional[int] = Field(default=None, ge=0)
    view_count: Optional[int] = Field(default=None, ge=0)
    source_keyword: Optional[str] = None
    media_urls: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    raw_payload: dict[str, Any]
    collected_at: datetime


class NormalizedComment(DomainModel):
    """Cross-platform comment representation with reply-tree metadata."""

    id: UUID
    run_id: UUID
    platform: Platform
    native_comment_id: str
    content_id: UUID
    native_content_id: str
    author_id: Optional[UUID] = None
    native_author_id: Optional[str] = None
    parent_comment_id: Optional[str] = None
    root_comment_id: Optional[str] = None
    depth: int = Field(default=0, ge=0)
    text: str
    published_at: Optional[datetime] = None
    like_count: Optional[int] = Field(default=None, ge=0)
    reply_count: Optional[int] = Field(default=None, ge=0)
    ip_location: Optional[str] = None
    media_urls: list[str] = Field(default_factory=list)
    text_hash: str
    language: Optional[str] = None
    raw_payload: dict[str, Any]
    collected_at: datetime

