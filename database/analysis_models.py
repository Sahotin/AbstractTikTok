"""SQLAlchemy persistence models for normalized, analysis-ready data."""

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    Column,
    DateTime,
    ForeignKey,
    Float,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import relationship

from .models import Base


class CollectionRunModel(Base):
    __tablename__ = "collection_runs"

    id = Column(String(36), primary_key=True)
    platform = Column(String(32), nullable=False, index=True)
    crawler_type = Column(String(32), nullable=False)
    query = Column(Text)
    specified_ids = Column(JSON, nullable=False, default=list)
    status = Column(String(32), nullable=False, index=True)
    started_at = Column(DateTime(timezone=True), nullable=False)
    finished_at = Column(DateTime(timezone=True))
    config_snapshot = Column(JSON, nullable=False, default=dict)
    output_location = Column(JSON, nullable=False, default=list)
    content_count = Column(Integer, nullable=False, default=0)
    comment_count = Column(Integer, nullable=False, default=0)
    error_message = Column(Text)

    contents = relationship("NormalizedContentModel", back_populates="run")
    comments = relationship("NormalizedCommentModel", back_populates="run")


class NormalizedAuthorModel(Base):
    __tablename__ = "normalized_authors"
    __table_args__ = (
        UniqueConstraint("platform", "native_author_id", name="uq_normalized_author_platform_native"),
    )

    id = Column(String(36), primary_key=True)
    platform = Column(String(32), nullable=False)
    native_author_id = Column(String(255), nullable=False)
    nickname = Column(Text)
    profile_url = Column(Text)
    avatar_url = Column(Text)
    description = Column(Text)
    gender = Column(String(32))
    ip_location = Column(String(128))
    following_count = Column(BigInteger)
    follower_count = Column(BigInteger)
    content_count = Column(BigInteger)
    verified = Column(Boolean)
    attributes = Column(JSON, nullable=False, default=dict)
    raw_payload = Column(JSON, nullable=False, default=dict)
    collected_at = Column(DateTime(timezone=True), nullable=False)

    contents = relationship("NormalizedContentModel", back_populates="author")
    comments = relationship("NormalizedCommentModel", back_populates="author")


class NormalizedContentModel(Base):
    __tablename__ = "normalized_contents"
    __table_args__ = (
        UniqueConstraint("platform", "native_content_id", name="uq_normalized_content_platform_native"),
        Index("ix_normalized_content_run_published", "run_id", "published_at"),
    )

    id = Column(String(36), primary_key=True)
    run_id = Column(String(36), ForeignKey("collection_runs.id", ondelete="RESTRICT"), nullable=False, index=True)
    platform = Column(String(32), nullable=False)
    native_content_id = Column(String(255), nullable=False)
    content_type = Column(String(64), nullable=False)
    title = Column(Text)
    body = Column(Text)
    url = Column(Text)
    author_id = Column(String(36), ForeignKey("normalized_authors.id", ondelete="SET NULL"), index=True)
    native_author_id = Column(String(255))
    published_at = Column(DateTime(timezone=True), index=True)
    like_count = Column(BigInteger)
    comment_count = Column(BigInteger)
    share_count = Column(BigInteger)
    favorite_count = Column(BigInteger)
    view_count = Column(BigInteger)
    source_keyword = Column(Text)
    media_urls = Column(JSON, nullable=False, default=list)
    tags = Column(JSON, nullable=False, default=list)
    raw_payload = Column(JSON, nullable=False)
    collected_at = Column(DateTime(timezone=True), nullable=False)

    run = relationship("CollectionRunModel", back_populates="contents")
    author = relationship("NormalizedAuthorModel", back_populates="contents")
    comments = relationship("NormalizedCommentModel", back_populates="content")


class NormalizedCommentModel(Base):
    __tablename__ = "normalized_comments"
    __table_args__ = (
        UniqueConstraint("platform", "native_comment_id", name="uq_normalized_comment_platform_native"),
        Index("ix_normalized_comment_content_published", "content_id", "published_at"),
        Index("ix_normalized_comment_run_depth", "run_id", "depth"),
        Index("ix_normalized_comment_text_hash", "text_hash"),
        Index("ix_normalized_comment_platform_run_content", "platform", "run_id", "content_id"),
    )

    id = Column(String(36), primary_key=True)
    run_id = Column(String(36), ForeignKey("collection_runs.id", ondelete="RESTRICT"), nullable=False, index=True)
    platform = Column(String(32), nullable=False)
    native_comment_id = Column(String(255), nullable=False)
    content_id = Column(String(36), ForeignKey("normalized_contents.id", ondelete="CASCADE"), nullable=False, index=True)
    native_content_id = Column(String(255), nullable=False)
    author_id = Column(String(36), ForeignKey("normalized_authors.id", ondelete="SET NULL"), index=True)
    native_author_id = Column(String(255))
    parent_comment_id = Column(String(255))
    root_comment_id = Column(String(255))
    depth = Column(Integer, nullable=False, default=0)
    text = Column(Text, nullable=False)
    published_at = Column(DateTime(timezone=True), index=True)
    like_count = Column(BigInteger)
    reply_count = Column(BigInteger)
    ip_location = Column(String(128))
    media_urls = Column(JSON, nullable=False, default=list)
    text_hash = Column(String(64), nullable=False)
    language = Column(String(32))
    raw_payload = Column(JSON, nullable=False)
    collected_at = Column(DateTime(timezone=True), nullable=False)

    run = relationship("CollectionRunModel", back_populates="comments")
    content = relationship("NormalizedContentModel", back_populates="comments")
    author = relationship("NormalizedAuthorModel", back_populates="comments")
    analyses = relationship("CommentAnalysisModel", back_populates="comment")


class CommentAnalysisModel(Base):
    __tablename__ = "comment_analyses"
    __table_args__ = (
        UniqueConstraint(
            "comment_id",
            "input_hash",
            "provider",
            "model",
            "prompt_version",
            "analysis_version",
            name="uq_comment_analysis_cache_key",
        ),
        Index("ix_comment_analysis_comment_created", "comment_id", "created_at"),
    )

    id = Column(String(36), primary_key=True)
    comment_id = Column(
        String(36),
        ForeignKey("normalized_comments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    sentiment = Column(String(16), nullable=False)
    sentiment_score = Column(Float, nullable=False)
    emotion = Column(String(16), nullable=False)
    topics = Column(JSON, nullable=False, default=list)
    stance = Column(String(16), nullable=False)
    risk_level = Column(String(16), nullable=False)
    risk_reasons = Column(JSON, nullable=False, default=list)
    keywords = Column(JSON, nullable=False, default=list)
    summary = Column(Text, nullable=False)
    model = Column(String(255), nullable=False)
    provider = Column(String(64), nullable=False)
    prompt_version = Column(String(64), nullable=False)
    analysis_version = Column(String(64), nullable=False)
    input_hash = Column(String(64), nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)

    comment = relationship("NormalizedCommentModel", back_populates="analyses")


class AnalysisJobModel(Base):
    __tablename__ = "analysis_jobs"
    __table_args__ = (
        Index("ix_analysis_job_status_created", "status", "created_at"),
        Index("ix_analysis_job_parent", "parent_job_id"),
    )

    id = Column(String(36), primary_key=True)
    parent_job_id = Column(String(36), ForeignKey("analysis_jobs.id", ondelete="SET NULL"))
    status = Column(String(32), nullable=False)
    platform = Column(String(32), nullable=False)
    run_id = Column(String(36), ForeignKey("collection_runs.id", ondelete="SET NULL"), index=True)
    content_id = Column(String(36), ForeignKey("normalized_contents.id", ondelete="SET NULL"), index=True)
    requested_limit = Column(Integer, nullable=False)
    total_comments = Column(Integer, nullable=False, default=0)
    pending_count = Column(Integer, nullable=False, default=0)
    processing_count = Column(Integer, nullable=False, default=0)
    completed_count = Column(Integer, nullable=False, default=0)
    failed_count = Column(Integer, nullable=False, default=0)
    skipped_count = Column(Integer, nullable=False, default=0)
    provider = Column(String(64), nullable=False)
    model = Column(String(255), nullable=False)
    prompt_version = Column(String(64), nullable=False)
    analysis_version = Column(String(64), nullable=False)
    max_concurrency = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    max_requests_per_minute = Column(Integer)
    estimated_input_tokens = Column(BigInteger, nullable=False, default=0)
    estimated_output_tokens = Column(BigInteger, nullable=False, default=0)
    input_price_per_1m_tokens = Column(Float)
    output_price_per_1m_tokens = Column(Float)
    estimated_cost = Column(Float)
    created_at = Column(DateTime(timezone=True), nullable=False)
    started_at = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True))
    error_message = Column(Text)

    items = relationship("AnalysisJobItemModel", back_populates="job", cascade="all, delete-orphan")


class AnalysisJobItemModel(Base):
    __tablename__ = "analysis_job_items"
    __table_args__ = (
        UniqueConstraint("job_id", "comment_id", name="uq_analysis_job_item_comment"),
        Index("ix_analysis_job_item_job_status", "job_id", "status"),
    )

    id = Column(String(36), primary_key=True)
    job_id = Column(
        String(36),
        ForeignKey("analysis_jobs.id", ondelete="CASCADE"),
        nullable=False,
    )
    comment_id = Column(
        String(36),
        ForeignKey("normalized_comments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    status = Column(String(32), nullable=False)
    attempt_count = Column(Integer, nullable=False, default=0)
    error_type = Column(String(128))
    error_message = Column(Text)
    started_at = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True))

    job = relationship("AnalysisJobModel", back_populates="items")
