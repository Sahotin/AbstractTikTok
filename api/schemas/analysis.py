"""Public response contracts for normalized analysis data."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Optional
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from analysis.domain import (
    AnalysisJob,
    AnalysisJobStatus,
    Emotion,
    Platform,
    PublicOpinionSummary,
    RiskLevel,
    SemanticAnalysisRun,
    SemanticSearchQuery,
    SemanticSearchResult,
    SemanticRunStatus,
    SemanticScopeType,
    TopicMetadataItem,
    Sentiment,
    Stance,
    TrendSummary,
)
from analysis.agent import AgentRequest, AgentResponse


class AnalysisResponseModel(BaseModel):
    model_config = ConfigDict(from_attributes=True)


class ContentResponse(AnalysisResponseModel):
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
    like_count: Optional[int] = None
    comment_count: Optional[int] = None
    share_count: Optional[int] = None
    favorite_count: Optional[int] = None
    view_count: Optional[int] = None
    source_keyword: Optional[str] = None
    media_urls: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    collected_at: datetime


class CommentResponse(AnalysisResponseModel):
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
    depth: int
    text: str
    published_at: Optional[datetime] = None
    like_count: Optional[int] = None
    reply_count: Optional[int] = None
    ip_location: Optional[str] = None
    media_urls: list[str] = Field(default_factory=list)
    text_hash: str
    language: Optional[str] = None
    collected_at: datetime


class AuthorResponse(AnalysisResponseModel):
    id: UUID
    platform: Platform
    native_author_id: str
    nickname: Optional[str] = None
    profile_url: Optional[str] = None
    avatar_url: Optional[str] = None
    description: Optional[str] = None
    gender: Optional[str] = None
    ip_location: Optional[str] = None
    following_count: Optional[int] = None
    follower_count: Optional[int] = None
    content_count: Optional[int] = None
    verified: Optional[bool] = None
    attributes: dict[str, Any] = Field(default_factory=dict)
    collected_at: datetime


class ContentListResponse(BaseModel):
    items: list[ContentResponse]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class CommentListResponse(BaseModel):
    items: list[CommentResponse]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class AuthorListResponse(BaseModel):
    items: list[AuthorResponse]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class ContentStatisticsResponse(BaseModel):
    content_count: int = Field(ge=0)


class CommentStatisticsResponse(BaseModel):
    comment_count: int = Field(ge=0)
    root_comment_count: int = Field(ge=0)
    reply_comment_count: int = Field(ge=0)
    earliest_comment_time: Optional[datetime] = None
    latest_comment_time: Optional[datetime] = None


class AuthorStatisticsResponse(BaseModel):
    author_count: int = Field(ge=0)


class AnalysisStatisticsResponse(BaseModel):
    contents: ContentStatisticsResponse
    comments: CommentStatisticsResponse
    authors: AuthorStatisticsResponse


class CommentAnalysisResponse(AnalysisResponseModel):
    id: UUID
    comment_id: UUID
    sentiment: Sentiment
    sentiment_score: float = Field(ge=-1.0, le=1.0)
    emotion: Emotion
    topics: list[str]
    stance: Stance
    risk_level: RiskLevel
    risk_reasons: list[str]
    keywords: list[str]
    summary: str
    model: str
    provider: str
    prompt_version: str
    analysis_version: str
    input_hash: str
    created_at: datetime


class AnalysisJobCreateRequest(BaseModel):
    platform: Platform = Platform.DOUYIN
    run_id: Optional[UUID] = None
    content_id: Optional[UUID] = None
    limit: int = Field(ge=1, le=10000)
    batch_size: int = Field(default=20, ge=1, le=500)
    max_concurrency: int = Field(default=5, ge=1, le=20)
    max_requests_per_minute: Optional[int] = Field(default=None, ge=1)


class AnalysisJobResponse(AnalysisResponseModel):
    id: UUID
    parent_job_id: Optional[UUID] = None
    status: AnalysisJobStatus
    platform: Platform
    run_id: Optional[UUID] = None
    content_id: Optional[UUID] = None
    requested_limit: int
    total_comments: int
    pending_count: int
    processing_count: int
    completed_count: int
    failed_count: int
    skipped_count: int
    progress_percent: float
    provider: str
    model: str
    prompt_version: str
    analysis_version: str
    max_concurrency: int
    batch_size: int
    max_requests_per_minute: Optional[int] = None
    estimated_input_tokens: int
    estimated_output_tokens: int
    estimated_cost: Optional[float] = None
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None

    @classmethod
    def from_job(cls, job: AnalysisJob) -> "AnalysisJobResponse":
        terminal = job.completed_count + job.failed_count + job.skipped_count
        progress = (terminal / job.total_comments * 100) if job.total_comments else 100.0
        return cls(
            **job.model_dump(mode="python", exclude={
                "input_price_per_1m_tokens",
                "output_price_per_1m_tokens",
            }),
            progress_percent=round(progress, 2),
        )


class PublicOpinionResponse(PublicOpinionSummary):
    """Read-only aggregate response; intentionally contains no raw comments."""


class TrendResponse(TrendSummary):
    """Read-only, explainable time-series response."""


class SemanticSearchRequest(SemanticSearchQuery):
    pass


class SemanticSearchResponse(SemanticSearchResult):
    pass


class AgentChatRequest(AgentRequest):
    pass


class AgentChatResponse(AgentResponse):
    pass


class SemanticRunCreateRequest(BaseModel):
    platform: Platform = Platform.DOUYIN
    run_id: Optional[UUID] = None
    content_id: Optional[UUID] = None
    limit: int = Field(ge=1, le=100000)


class SemanticRunResponse(AnalysisResponseModel):
    id: UUID
    status: SemanticRunStatus
    scope_type: SemanticScopeType
    platform: Platform
    run_id: Optional[UUID] = None
    content_id: Optional[UUID] = None
    requested_limit: int
    scope_hash: str
    input_snapshot_hash: str
    analysis_snapshot_hash: Optional[str] = None
    embedding_provider: str
    embedding_model: str
    embedding_version: str
    embedding_dimension: Optional[int] = None
    embedding_config: dict[str, Any]
    algorithm: str
    algorithm_version: str
    algorithm_config: dict[str, Any]
    config_hash: str
    total_comments: int
    embedded_comments: int
    failed_comments: int
    analyzed_comments: int
    noise_comments: int
    topic_count: int
    opinion_count: int
    created_at: datetime
    started_at: Optional[datetime] = None
    finished_at: Optional[datetime] = None
    error_message: Optional[str] = None
    topic_status: str
    clustered_comments: int
    topic_started_at: Optional[datetime] = None
    topic_finished_at: Optional[datetime] = None
    topic_error_message: Optional[str] = None

    @classmethod
    def from_run(cls, run: SemanticAnalysisRun) -> "SemanticRunResponse":
        return cls.model_validate(run)


class TopicClusterResponse(BaseModel):
    id: UUID
    semantic_run_id: UUID
    cluster_key: str
    name: str
    summary: str
    size: int
    percentage: float
    top_topics: list[TopicMetadataItem]
    top_keywords: list[TopicMetadataItem]
    dominant_sentiment: Optional[Sentiment] = None
    dominant_stance: Optional[Stance] = None
    dominant_risk: Optional[RiskLevel] = None
    analyzed_comments: int
    analysis_coverage: float = Field(ge=0.0, le=1.0)


class TopicListResponse(BaseModel):
    items: list[TopicClusterResponse]
    total: int = Field(ge=0)
    limit: int = Field(ge=1)
    offset: int = Field(ge=0)


class TopicRepresentativeResponse(BaseModel):
    comment_id: UUID
    text: str
    like_count: Optional[int] = None
    similarity: float
    rank: int
    score: float


class TopicDetailResponse(TopicClusterResponse):
    representative_comments: list[TopicRepresentativeResponse]
    algorithm: str
    algorithm_version: str
    config_hash: str


AnalysisItem = Literal[
    "sentiment",
    "emotion",
    "stance",
    "keywords",
    "risk",
    "trend",
    "topics",
    "semantic_search",
    "agent",
]


class CollectionAnalysisWorkflowRequest(BaseModel):
    """Start AI analysis from files produced by one crawler task."""

    platform: Platform
    source_files: list[str] = Field(min_length=1, max_length=20)
    analysis_items: list[AnalysisItem] = Field(min_length=1)
    crawler_type: str = Field(default="import", min_length=1, max_length=32)
    limit: int = Field(default=100, ge=1, le=10_000)
    batch_size: int = Field(default=20, ge=1, le=500)
    max_concurrency: int = Field(default=2, ge=1, le=20)


class CollectionAnalysisWorkflowResponse(AnalysisResponseModel):
    id: UUID
    status: Literal["pending", "running", "completed", "failed"]
    stage: Literal["queued", "ingesting", "analyzing", "embedding", "clustering", "completed", "failed"]
    progress_percent: float = Field(ge=0.0, le=100.0)
    message: str
    platform: Platform
    source_files: list[str]
    analysis_items: list[AnalysisItem]
    requested_limit: int
    run_id: Optional[UUID] = None
    analysis_job_id: Optional[UUID] = None
    semantic_run_id: Optional[UUID] = None
    content_count: int = Field(default=0, ge=0)
    comment_count: int = Field(default=0, ge=0)
    analyzed_count: int = Field(default=0, ge=0)
    topic_count: int = Field(default=0, ge=0)
    created_at: datetime
    updated_at: datetime
    error_message: Optional[str] = None
    dashboard_url: Optional[str] = None
