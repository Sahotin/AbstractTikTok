"""Deterministic, explainable aggregate models for analyzed comments."""

from __future__ import annotations

from datetime import date
from typing import Literal, Optional
from uuid import UUID

from pydantic import Field

from .models import DomainModel, Platform


class PublicOpinionQuery(DomainModel):
    """The same AND-combined scope semantics used by normalized-data queries."""

    platform: Optional[Platform] = None
    run_id: Optional[UUID] = None
    content_id: Optional[UUID] = None
    # A completed semantic run is an immutable mapping from eligible comments
    # to the exact CommentAnalysis rows used by that run.  Aggregates must use
    # it instead of independently selecting each comment's newest analysis.
    semantic_run_id: Optional[UUID] = None
    eligible_comments: Optional[int] = Field(default=None, ge=0)
    analysis_snapshot: Optional["AnalysisSnapshot"] = None
    top_k: int = Field(default=20, ge=1, le=100)


class AnalysisSnapshot(DomainModel):
    """The versioned LLM analysis identity frozen by a semantic run."""

    provider: str = Field(min_length=1, max_length=64)
    model: str = Field(min_length=1, max_length=255)
    prompt_version: str = Field(min_length=1, max_length=64)
    analysis_version: str = Field(min_length=1, max_length=64)


class PublicOpinionScope(DomainModel):
    platform: Optional[Platform] = None
    run_id: Optional[UUID] = None
    content_id: Optional[UUID] = None
    semantic_run_id: Optional[UUID] = None


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


class RiskComponent(DomainModel):
    key: str
    label: str
    value: float = Field(ge=0)
    display_value: str
    weight: float = Field(ge=0.0, le=1.0)
    contribution: float = Field(ge=0.0, le=100.0)


class RiskEvidence(DomainModel):
    comment_id: UUID
    text: str
    sentiment: str
    risk_level: str
    risk_reasons: list[str] = Field(default_factory=list)
    like_count: Optional[int] = Field(default=None, ge=0)


class RiskExplanation(DomainModel):
    risk_level: Literal["unknown", "low", "medium", "high"]
    risk_score: int = Field(ge=0, le=100)
    components: list[RiskComponent] = Field(default_factory=list)
    triggered_rules: list[str] = Field(default_factory=list)
    representative_evidence: list[RiskEvidence] = Field(default_factory=list)


class PublicOpinionSummary(DomainModel):
    """A repeatable aggregate over the latest analysis of each scoped comment."""

    scope: PublicOpinionScope
    total_comments: int = Field(ge=0)
    eligible_comments: int = Field(ge=0)
    excluded_comments: int = Field(ge=0)
    analyzed_comments: int = Field(ge=0)
    unanalyzed_eligible_comments: int = Field(ge=0)
    unanalyzed_comments: int = Field(ge=0)
    analysis_coverage: float = Field(ge=0, le=1)
    ai_analysis_coverage: float = Field(ge=0, le=1)
    analysis_status: str
    analysis_snapshot: Optional[AnalysisSnapshot] = None
    collection_coverage: Literal["unknown"] = "unknown"
    stance_target: Optional[str] = None
    conclusion: str
    conclusion_points: list[str] = Field(default_factory=list)
    sentiment_distribution: dict[str, DistributionValue] = Field(default_factory=dict)
    sentiment_score_statistics: SentimentScoreStatistics
    emotion_distribution: dict[str, DistributionValue] = Field(default_factory=dict)
    stance_distribution: dict[str, DistributionValue] = Field(default_factory=dict)
    risk_distribution: dict[str, DistributionValue] = Field(default_factory=dict)
    high_risk_count: int = Field(ge=0)
    risk_explanation: RiskExplanation
    risk_reason_frequency: list[FrequencyItem] = Field(default_factory=list)
    topics: list[FrequencyItem] = Field(default_factory=list)
    keywords: list[FrequencyItem] = Field(default_factory=list)
    topic_sentiment_distribution: dict[str, dict[str, int]] = Field(default_factory=dict)
    topic_risk_distribution: dict[str, dict[str, int]] = Field(default_factory=dict)
    comment_volume_by_date: list[CommentVolumeByDate] = Field(default_factory=list)
