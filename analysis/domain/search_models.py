"""Semantic comment-search contracts over one immutable embedding run."""

from __future__ import annotations

from datetime import datetime
from typing import Optional
from uuid import UUID

from pydantic import Field

from .analysis_models import RiskLevel, Sentiment
from .models import DomainModel


class SemanticSearchQuery(DomainModel):
    semantic_run_id: UUID
    query: str = Field(min_length=1, max_length=500)
    top_k: int = Field(default=10, ge=1, le=50)
    min_similarity: float = Field(default=0.0, ge=-1.0, le=1.0)
    risk_level: Optional[RiskLevel] = None
    sentiment: Optional[Sentiment] = None
    content_id: Optional[UUID] = None
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    exclude_low_information: bool = True
    deduplicate_text: bool = True


class SemanticSearchHit(DomainModel):
    comment_id: UUID
    text: str
    similarity: float = Field(ge=-1.0, le=1.0)
    like_count: Optional[int] = Field(default=None, ge=0)
    published_at: Optional[datetime] = None
    content_id: UUID
    topic_cluster_id: Optional[UUID] = None
    sentiment: Optional[Sentiment] = None
    risk_level: Optional[RiskLevel] = None
    summary: Optional[str] = None


class SemanticSearchResult(DomainModel):
    semantic_run_id: UUID
    query: str
    embedding_provider: str
    embedding_model: str
    embedding_version: str
    searched_comments: int = Field(ge=0)
    excluded_comments: int = Field(ge=0)
    returned_comments: int = Field(default=0, ge=0)
    hits: list[SemanticSearchHit] = Field(default_factory=list)
