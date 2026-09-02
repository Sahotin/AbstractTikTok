"""Add Phase 3B embedding cache and semantic analysis runs."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0002_phase3b_embedding"
down_revision: Union[str, Sequence[str], None] = "0001_phase1_to_phase3a"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_index(
        "ix_normalized_comment_platform_run_content",
        "normalized_comments",
        ["platform", "run_id", "content_id"],
        unique=False,
    )
    op.create_table(
        "comment_embeddings",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("comment_id", sa.String(36), sa.ForeignKey("normalized_comments.id", ondelete="CASCADE"), nullable=False),
        sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("model", sa.String(255), nullable=False),
        sa.Column("embedding_version", sa.String(64), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column("dtype", sa.String(32), nullable=False),
        sa.Column("normalized", sa.Boolean(), nullable=False),
        sa.Column("vector_blob", sa.LargeBinary(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.CheckConstraint("dimension > 0", name="ck_comment_embedding_dimension_positive"),
        sa.UniqueConstraint("comment_id", "input_hash", "provider", "model", "embedding_version", name="uq_comment_embedding_cache_key"),
    )
    op.create_index("ix_comment_embeddings_comment_id", "comment_embeddings", ["comment_id"])
    op.create_index("ix_comment_embedding_content_cache", "comment_embeddings", ["input_hash", "provider", "model", "embedding_version"])

    op.create_table(
        "semantic_analysis_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("scope_type", sa.String(32), nullable=False),
        sa.Column("platform", sa.String(32), nullable=False),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("collection_runs.id", ondelete="SET NULL")),
        sa.Column("content_id", sa.String(36), sa.ForeignKey("normalized_contents.id", ondelete="SET NULL")),
        sa.Column("requested_limit", sa.Integer(), nullable=False),
        sa.Column("scope_hash", sa.String(64), nullable=False),
        sa.Column("input_snapshot_hash", sa.String(64), nullable=False),
        sa.Column("analysis_snapshot_hash", sa.String(64)),
        sa.Column("embedding_provider", sa.String(64), nullable=False),
        sa.Column("embedding_model", sa.String(255), nullable=False),
        sa.Column("embedding_version", sa.String(64), nullable=False),
        sa.Column("embedding_dimension", sa.Integer()),
        sa.Column("embedding_config", sa.JSON(), nullable=False),
        sa.Column("algorithm", sa.String(64), nullable=False),
        sa.Column("algorithm_version", sa.String(64), nullable=False),
        sa.Column("algorithm_config", sa.JSON(), nullable=False),
        sa.Column("config_hash", sa.String(64), nullable=False),
        sa.Column("total_comments", sa.Integer(), nullable=False),
        sa.Column("embedded_comments", sa.Integer(), nullable=False),
        sa.Column("failed_comments", sa.Integer(), nullable=False),
        sa.Column("analyzed_comments", sa.Integer(), nullable=False),
        sa.Column("noise_comments", sa.Integer(), nullable=False),
        sa.Column("topic_count", sa.Integer(), nullable=False),
        sa.Column("opinion_count", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("error_message", sa.Text()),
    )
    op.create_index("ix_semantic_analysis_runs_run_id", "semantic_analysis_runs", ["run_id"])
    op.create_index("ix_semantic_analysis_runs_content_id", "semantic_analysis_runs", ["content_id"])
    op.create_index("ix_semantic_run_status_created", "semantic_analysis_runs", ["status", "created_at"])
    op.create_index("ix_semantic_run_scope", "semantic_analysis_runs", ["platform", "run_id", "content_id"])


def downgrade() -> None:
    op.drop_table("semantic_analysis_runs")
    op.drop_table("comment_embeddings")
    op.drop_index("ix_normalized_comment_platform_run_content", table_name="normalized_comments")
