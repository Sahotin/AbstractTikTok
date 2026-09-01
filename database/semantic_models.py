"""SQLAlchemy tables for Phase 3B embedding and semantic-run persistence."""

from sqlalchemy import (
    JSON,
    Boolean,
    CheckConstraint,
    Column,
    DateTime,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    Float,
    String,
    Text,
    UniqueConstraint,
)

from .models import Base


class CommentEmbeddingModel(Base):
    __tablename__ = "comment_embeddings"
    __table_args__ = (
        UniqueConstraint(
            "comment_id",
            "input_hash",
            "provider",
            "model",
            "embedding_version",
            name="uq_comment_embedding_cache_key",
        ),
        Index(
            "ix_comment_embedding_content_cache",
            "input_hash",
            "provider",
            "model",
            "embedding_version",
        ),
        CheckConstraint("dimension > 0", name="ck_comment_embedding_dimension_positive"),
    )

    id = Column(String(36), primary_key=True)
    comment_id = Column(
        String(36),
        ForeignKey("normalized_comments.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    input_hash = Column(String(64), nullable=False)
    provider = Column(String(64), nullable=False)
    model = Column(String(255), nullable=False)
    embedding_version = Column(String(64), nullable=False)
    dimension = Column(Integer, nullable=False)
    dtype = Column(String(32), nullable=False)
    normalized = Column(Boolean, nullable=False)
    vector_blob = Column(LargeBinary, nullable=False)
    created_at = Column(DateTime(timezone=True), nullable=False)


class SemanticAnalysisRunModel(Base):
    __tablename__ = "semantic_analysis_runs"
    __table_args__ = (
        Index("ix_semantic_run_status_created", "status", "created_at"),
        Index("ix_semantic_run_scope", "platform", "run_id", "content_id"),
    )

    id = Column(String(36), primary_key=True)
    status = Column(String(32), nullable=False)
    scope_type = Column(String(32), nullable=False)
    platform = Column(String(32), nullable=False)
    run_id = Column(String(36), ForeignKey("collection_runs.id", ondelete="SET NULL"), index=True)
    content_id = Column(String(36), ForeignKey("normalized_contents.id", ondelete="SET NULL"), index=True)
    requested_limit = Column(Integer, nullable=False)
    scope_hash = Column(String(64), nullable=False)
    input_snapshot_hash = Column(String(64), nullable=False)
    analysis_snapshot_hash = Column(String(64))
    embedding_provider = Column(String(64), nullable=False)
    embedding_model = Column(String(255), nullable=False)
    embedding_version = Column(String(64), nullable=False)
    embedding_dimension = Column(Integer)
    embedding_config = Column(JSON, nullable=False, default=dict)
    algorithm = Column(String(64), nullable=False)
    algorithm_version = Column(String(64), nullable=False)
    algorithm_config = Column(JSON, nullable=False, default=dict)
    config_hash = Column(String(64), nullable=False)
    total_comments = Column(Integer, nullable=False, default=0)
    embedded_comments = Column(Integer, nullable=False, default=0)
    failed_comments = Column(Integer, nullable=False, default=0)
    analyzed_comments = Column(Integer, nullable=False, default=0)
    noise_comments = Column(Integer, nullable=False, default=0)
    topic_count = Column(Integer, nullable=False, default=0)
    opinion_count = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime(timezone=True), nullable=False)
    started_at = Column(DateTime(timezone=True))
    finished_at = Column(DateTime(timezone=True))
    error_message = Column(Text)
    topic_status = Column(String(32), nullable=False, default="not_started")
    clustered_comments = Column(Integer, nullable=False, default=0)
    topic_started_at = Column(DateTime(timezone=True))
    topic_finished_at = Column(DateTime(timezone=True))
    topic_error_message = Column(Text)


class TopicClusterModel(Base):
    __tablename__ = "topic_clusters"
    __table_args__ = (
        UniqueConstraint("semantic_run_id", "cluster_key", name="uq_topic_cluster_run_key"),
        Index("ix_topic_cluster_run_size", "semantic_run_id", "size"),
        CheckConstraint("size > 0", name="ck_topic_cluster_size_positive"),
        CheckConstraint("dimension > 0", name="ck_topic_cluster_dimension_positive"),
    )

    id = Column(String(36), primary_key=True)
    semantic_run_id = Column(
        String(36), ForeignKey("semantic_analysis_runs.id", ondelete="CASCADE"), nullable=False
    )
    cluster_key = Column(String(64), nullable=False)
    name = Column(String(255), nullable=False)
    summary = Column(Text, nullable=False)
    size = Column(Integer, nullable=False)
    percentage = Column(Float, nullable=False)
    top_topics = Column(JSON, nullable=False, default=list)
    top_keywords = Column(JSON, nullable=False, default=list)
    centroid_blob = Column(LargeBinary, nullable=False)
    dimension = Column(Integer, nullable=False)
    medoid_comment_id = Column(
        String(36), ForeignKey("normalized_comments.id", ondelete="RESTRICT"), nullable=False
    )
    analyzed_comments = Column(Integer, nullable=False, default=0)
    dominant_sentiment = Column(String(16))
    dominant_stance = Column(String(16))
    dominant_risk = Column(String(16))
    created_at = Column(DateTime(timezone=True), nullable=False)


class SemanticRunItemModel(Base):
    __tablename__ = "semantic_run_items"
    __table_args__ = (
        UniqueConstraint("semantic_run_id", "comment_id", name="uq_semantic_run_item_comment"),
        Index("ix_semantic_run_item_run_cluster", "semantic_run_id", "topic_cluster_id"),
    )

    id = Column(String(36), primary_key=True)
    semantic_run_id = Column(
        String(36), ForeignKey("semantic_analysis_runs.id", ondelete="CASCADE"), nullable=False, index=True
    )
    comment_id = Column(
        String(36), ForeignKey("normalized_comments.id", ondelete="CASCADE"), nullable=False
    )
    embedding_id = Column(
        String(36), ForeignKey("comment_embeddings.id", ondelete="RESTRICT")
    )
    analysis_id = Column(
        String(36), ForeignKey("comment_analyses.id", ondelete="SET NULL")
    )
    status = Column(String(32), nullable=False)
    topic_cluster_id = Column(
        String(36), ForeignKey("topic_clusters.id", ondelete="SET NULL")
    )
    topic_similarity = Column(Float)
    topic_representative_rank = Column(Integer)
    topic_representative_score = Column(Float)
    error_message = Column(Text)
