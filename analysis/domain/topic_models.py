"""Deterministic topic-clustering domain contracts."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Literal, Optional
from uuid import UUID

from pydantic import Field

from .analysis_models import RiskLevel, Sentiment, Stance
from .models import DomainModel


class TopicStageStatus(str, Enum):
    NOT_STARTED = "not_started"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class SemanticRunItemStatus(str, Enum):
    EMBEDDED = "embedded"
    MISSING_EMBEDDING = "missing_embedding"
    COMPLETED = "completed"
    EXCLUDED = "excluded"


class TopicClusterConfig(DomainModel):
    algorithm: Literal["agglomerative"] = "agglomerative"
    algorithm_version: str = "agglomerative_v1"
    distance_threshold: float = Field(default=0.35, gt=0.0, le=2.0)
    min_cluster_size: int = Field(default=3, ge=1, le=5000)
    metric: Literal["cosine"] = "cosine"
    linkage: Literal["average"] = "average"
    representative_top_k: int = Field(default=3, ge=1, le=10)
    representative_mmr_lambda: float = Field(default=0.75, ge=0.0, le=1.0)
    metadata_top_k: int = Field(default=10, ge=1, le=50)
    max_input: int = Field(default=5000, ge=1, le=5000)
    quality_gate_enabled: bool = True
    min_informative_characters: int = Field(default=2, ge=1, le=20)


class TopicMetadataItem(DomainModel):
    value: str
    count: int = Field(ge=1)


class SemanticRunItem(DomainModel):
    id: UUID
    semantic_run_id: UUID
    comment_id: UUID
    embedding_id: Optional[UUID] = None
    analysis_id: Optional[UUID] = None
    status: SemanticRunItemStatus
    topic_cluster_id: Optional[UUID] = None
    topic_similarity: Optional[float] = None
    topic_representative_rank: Optional[int] = Field(default=None, ge=1, le=10)
    topic_representative_score: Optional[float] = None
    error_message: Optional[str] = None


class TopicClusterCandidate(DomainModel):
    member_comment_ids: list[UUID]
    centroid: list[float]
    dimension: int = Field(gt=0)
    medoid_comment_id: UUID


class TopicClusteringResult(DomainModel):
    clusters: list[TopicClusterCandidate]
    noise_comment_ids: list[UUID]
    input_comments: int = Field(ge=0)
    clustered_comments: int = Field(ge=0)
    dimension: Optional[int] = Field(default=None, gt=0)
    excluded_comments: dict[UUID, str] = Field(default_factory=dict)


class RepresentativeCommentScore(DomainModel):
    comment_id: UUID
    rank: int = Field(ge=1, le=10)
    score: float
    centroid_similarity: float
    engagement_score: float = Field(ge=0.0, le=1.0)
    information_score: float = Field(ge=0.0, le=1.0)


class TopicMembership(DomainModel):
    comment_id: UUID
    topic_cluster_id: Optional[UUID] = None
    similarity: Optional[float] = None
    representative_rank: Optional[int] = Field(default=None, ge=1, le=10)
    representative_score: Optional[float] = None
    status: SemanticRunItemStatus = SemanticRunItemStatus.COMPLETED
    error_message: Optional[str] = None


class TopicRepresentativeComment(DomainModel):
    comment_id: UUID
    text: str
    like_count: Optional[int] = None
    similarity: float
    rank: int = Field(ge=1, le=10)
    score: float


class TopicCluster(DomainModel):
    id: UUID
    semantic_run_id: UUID
    cluster_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    name: str
    summary: str
    size: int = Field(ge=1)
    percentage: float = Field(ge=0.0, le=100.0)
    top_topics: list[TopicMetadataItem] = Field(default_factory=list)
    top_keywords: list[TopicMetadataItem] = Field(default_factory=list)
    centroid: list[float]
    dimension: int = Field(gt=0)
    medoid_comment_id: UUID
    analyzed_comments: int = Field(default=0, ge=0)
    dominant_sentiment: Optional[Sentiment] = None
    dominant_stance: Optional[Stance] = None
    dominant_risk: Optional[RiskLevel] = None
    created_at: datetime
