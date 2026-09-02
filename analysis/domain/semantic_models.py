"""Versioned embedding records and basic semantic-analysis run state."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional
from uuid import UUID

from pydantic import Field

from .models import DomainModel, Platform
from .topic_models import TopicStageStatus


class SemanticRunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class SemanticScopeType(str, Enum):
    PLATFORM = "platform"
    RUN = "run"
    CONTENT = "content"


class EmbeddingRecord(DomainModel):
    id: UUID
    comment_id: UUID
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=255)
    embedding_version: str = Field(min_length=1, max_length=64)
    dimension: int = Field(gt=0)
    dtype: str = "float32-le"
    normalized: bool = True
    vector: list[float]
    created_at: datetime


class SemanticAnalysisRun(DomainModel):
    id: UUID
    status: SemanticRunStatus
    scope_type: SemanticScopeType
    platform: Platform
    run_id: Optional[UUID] = None
    content_id: Optional[UUID] = None
    requested_limit: int = Field(ge=1)
    scope_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    input_snapshot_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_snapshot_hash: Optional[str] = Field(default=None, pattern=r"^[0-9a-f]{64}$")
    embedding_provider: str
    embedding_model: str
    embedding_version: str
    embedding_dimension: Optional[int] = Field(default=None, gt=0)
    embedding_config: dict[str, Any] = Field(default_factory=dict)
    algorithm: str = "none"
    algorithm_version: str = "phase3b1"
    algorithm_config: dict[str, Any] = Field(default_factory=dict)
    config_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    total_comments: int = Field(default=0, ge=0)
    eligible_comments: int = Field(default=0, ge=0)
    excluded_comments: int = Field(default=0, ge=0)
    embedded_comments: int = Field(default=0, ge=0)
    embedding_failed_comments: int = Field(default=0, ge=0)
    failed_comments: int = Field(default=0, ge=0)
    analyzed_comments: int = Field(default=0, ge=0)
    noise_comments: int = Field(default=0, ge=0)
    topic_count: int = Field(default=0, ge=0)
    opinion_count: int = Field(default=0, ge=0)
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None
    topic_status: TopicStageStatus = TopicStageStatus.NOT_STARTED
    clustering_input_comments: int = Field(default=0, ge=0)
    clustered_comments: int = Field(default=0, ge=0)
    clustering_failed_comments: int = Field(default=0, ge=0)
    topic_started_at: Optional[datetime] = None
    topic_finished_at: Optional[datetime] = None
    topic_error_message: Optional[str] = None
