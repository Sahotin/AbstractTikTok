"""Persistent batch-analysis job and per-comment execution state."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional
from uuid import UUID

from pydantic import Field

from .models import DomainModel, Platform


class AnalysisJobStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    INTERRUPTED = "interrupted"


class AnalysisJobItemStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class AnalysisJob(DomainModel):
    id: UUID
    parent_job_id: Optional[UUID] = None
    status: AnalysisJobStatus
    platform: Platform
    run_id: Optional[UUID] = None
    content_id: Optional[UUID] = None
    requested_limit: int = Field(ge=1)
    total_comments: int = Field(default=0, ge=0)
    pending_count: int = Field(default=0, ge=0)
    processing_count: int = Field(default=0, ge=0)
    completed_count: int = Field(default=0, ge=0)
    failed_count: int = Field(default=0, ge=0)
    skipped_count: int = Field(default=0, ge=0)
    provider: str
    model: str
    prompt_version: str
    analysis_version: str
    max_concurrency: int = Field(ge=1)
    batch_size: int = Field(ge=1)
    max_requests_per_minute: Optional[int] = Field(default=None, ge=1)
    estimated_input_tokens: int = Field(default=0, ge=0)
    estimated_output_tokens: int = Field(default=0, ge=0)
    input_price_per_1m_tokens: Optional[float] = Field(default=None, ge=0)
    output_price_per_1m_tokens: Optional[float] = Field(default=None, ge=0)
    estimated_cost: Optional[float] = Field(default=None, ge=0)
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None


class AnalysisJobItem(DomainModel):
    id: UUID
    job_id: UUID
    comment_id: UUID
    status: AnalysisJobItemStatus
    attempt_count: int = Field(default=0, ge=0)
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
