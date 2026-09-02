"""Deterministic time-series contracts for public-opinion trend analysis."""

from __future__ import annotations

from datetime import datetime
from typing import Literal, Optional
from uuid import UUID

from pydantic import Field, model_validator

from .models import DomainModel, Platform
from .public_opinion import AnalysisSnapshot, PublicOpinionScope


class TrendQuery(DomainModel):
    platform: Optional[Platform] = None
    run_id: Optional[UUID] = None
    content_id: Optional[UUID] = None
    semantic_run_id: Optional[UUID] = None
    eligible_comments: Optional[int] = Field(default=None, ge=0)
    analysis_snapshot: Optional[AnalysisSnapshot] = None
    bucket: Literal["hour", "day"] = "day"
    start_at: Optional[datetime] = None
    end_at: Optional[datetime] = None
    min_bucket_comments: int = Field(default=3, ge=1, le=10_000)

    @model_validator(mode="after")
    def validate_range(self) -> "TrendQuery":
        if self.start_at is not None and self.end_at is not None and self.start_at >= self.end_at:
            raise ValueError("start_at must be earlier than end_at")
        return self


class TrendPoint(DomainModel):
    bucket_start: datetime
    total_comments: int = Field(ge=0)
    analyzed_comments: int = Field(ge=0)
    analysis_coverage: float = Field(ge=0.0, le=1.0)
    positive_count: int = Field(ge=0)
    neutral_count: int = Field(ge=0)
    negative_count: int = Field(ge=0)
    low_risk_count: int = Field(ge=0)
    medium_risk_count: int = Field(ge=0)
    high_risk_count: int = Field(ge=0)
    mean_sentiment_score: Optional[float] = Field(default=None, ge=-1.0, le=1.0)
    negative_ratio: float = Field(ge=0.0, le=1.0)
    high_risk_ratio: float = Field(ge=0.0, le=1.0)
    risk_score: float = Field(ge=0.0, le=100.0)


class TrendInflection(DomainModel):
    bucket_start: datetime
    kind: Literal["volume_surge", "risk_spike", "recovery"]
    severity: Literal["low", "medium", "high"]
    score_delta: float
    volume_change_ratio: Optional[float] = None
    reasons: list[str] = Field(default_factory=list)


class TrendSummary(DomainModel):
    scope: PublicOpinionScope
    bucket: Literal["hour", "day"]
    points: list[TrendPoint] = Field(default_factory=list)
    inflections: list[TrendInflection] = Field(default_factory=list)
    overall_risk_score: float = Field(ge=0.0, le=100.0)
    peak_risk_at: Optional[datetime] = None
    total_comments: int = Field(ge=0)
    analyzed_comments: int = Field(ge=0)
    analysis_coverage: float = Field(ge=0.0, le=1.0)
    data_quality: Literal["no_data", "no_analyses", "partial", "sufficient"]
    analysis_snapshot: Optional[AnalysisSnapshot] = None
