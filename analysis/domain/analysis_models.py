"""Validated structured outputs and persisted metadata for AI analysis."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from uuid import UUID

from pydantic import Field

from .models import DomainModel


class Sentiment(str, Enum):
    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class Emotion(str, Enum):
    JOY = "joy"
    ANGER = "anger"
    SADNESS = "sadness"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    NEUTRAL = "neutral"


class Stance(str, Enum):
    SUPPORT = "support"
    OPPOSE = "oppose"
    NEUTRAL = "neutral"
    UNCLEAR = "unclear"


class RiskLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class CommentAnalysisPayload(DomainModel):
    """The exact structured object an LLM is allowed to return."""

    sentiment: Sentiment
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    emotion: Emotion
    topics: list[str] = Field(max_length=10)
    stance: Stance
    risk_level: RiskLevel
    risk_reasons: list[str] = Field(max_length=10)
    keywords: list[str] = Field(max_length=15)
    summary: str = Field(min_length=1, max_length=200)


class CommentAnalysis(CommentAnalysisPayload):
    """One versioned analysis result persisted for one normalized comment."""

    id: UUID
    comment_id: UUID
    model: str = Field(min_length=1, max_length=255)
    provider: str = Field(min_length=1, max_length=64)
    prompt_version: str = Field(min_length=1, max_length=64)
    analysis_version: str = Field(min_length=1, max_length=64)
    input_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at: datetime
