"""Deterministic, explainable aggregate models for analyzed comments."""

from __future__ import annotations

from datetime import date
from typing import Optional
from uuid import UUID

from pydantic import Field

from .models import DomainModel, Platform


class PublicOpinionQuery(DomainModel):
    """The same AND-combined scope semantics used by normalized-data queries."""

    platform: Optional[Platform] = None
    run_id: Optional[UUID] = None
    content_id: Optional[UUID] = None
    top_k: int = Field(default=20, ge=1, le=100)


class PublicOpinionScope(DomainModel):
    platform: Optional[Platform] = None
    run_id: Optional[UUID] = None
    content_id: Optional[UUID] = None


class DistributionValue(DomainModel):
    count: int = Field(ge=0)
    percentage: float = Field(ge=0, le=100)


class SentimentScoreStatistics(DomainModel):
    """SQL aggregate score statistics. Median is intentionally deferred."""

    minimum: Optional[float] = None
    maximum: Optional[float] = None
    mean: Optional[float] = None


class FrequencyItem(DomainModel):
    value: str
    count: int = Field(ge=0)
    percentage: float = Field(ge=0, le=100)


class CommentVolumeByDate(DomainModel):
    date: date
    count: int = Field(ge=0)


class PublicOpinionSummary(DomainModel):
    """A repeatable aggregate over the latest analysis of each scoped comment."""

    scope: PublicOpinionScope
    total_comments: int = Field(ge=0)
    analyzed_comments: int = Field(ge=0)
    unanalyzed_comments: int = Field(ge=0)
    analysis_coverage: float = Field(ge=0, le=1)
    analysis_status: str
    sentiment_distribution: dict[str, DistributionValue] = Field(default_factory=dict)
    sentiment_score_statistics: SentimentScoreStatistics
    emotion_distribution: dict[str, DistributionValue] = Field(default_factory=dict)
    stance_distribution: dict[str, DistributionValue] = Field(default_factory=dict)
    risk_distribution: dict[str, DistributionValue] = Field(default_factory=dict)
    high_risk_count: int = Field(ge=0)
    risk_reason_frequency: list[FrequencyItem] = Field(default_factory=list)
    topics: list[FrequencyItem] = Field(default_factory=list)
    keywords: list[FrequencyItem] = Field(default_factory=list)
    topic_sentiment_distribution: dict[str, dict[str, int]] = Field(default_factory=dict)
    topic_risk_distribution: dict[str, dict[str, int]] = Field(default_factory=dict)
    comment_volume_by_date: list[CommentVolumeByDate] = Field(default_factory=list)
