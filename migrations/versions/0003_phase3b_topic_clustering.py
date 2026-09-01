"""Add fixed semantic snapshots and deterministic topic clusters."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0003_phase3b_topic_clustering"
down_revision: Union[str, Sequence[str], None] = "0002_phase3b_embedding"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "semantic_analysis_runs",
        sa.Column("topic_status", sa.String(32), nullable=False, server_default="not_started"),
    )
    op.add_column(
        "semantic_analysis_runs",
        sa.Column("clustered_comments", sa.Integer(), nullable=False, server_default="0"),
    )
    op.add_column("semantic_analysis_runs", sa.Column("topic_started_at", sa.DateTime(timezone=True)))
    op.add_column("semantic_analysis_runs", sa.Column("topic_finished_at", sa.DateTime(timezone=True)))
    op.add_column("semantic_analysis_runs", sa.Column("topic_error_message", sa.Text()))

    op.create_table(
        "topic_clusters",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "semantic_run_id",
            sa.String(36),
            sa.ForeignKey("semantic_analysis_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column("cluster_key", sa.String(64), nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False),
        sa.Column("size", sa.Integer(), nullable=False),
        sa.Column("percentage", sa.Float(), nullable=False),
        sa.Column("top_topics", sa.JSON(), nullable=False),
        sa.Column("top_keywords", sa.JSON(), nullable=False),
        sa.Column("centroid_blob", sa.LargeBinary(), nullable=False),
        sa.Column("dimension", sa.Integer(), nullable=False),
        sa.Column(
            "medoid_comment_id",
            sa.String(36),
            sa.ForeignKey("normalized_comments.id", ondelete="RESTRICT"),
            nullable=False,
        ),
        sa.Column("analyzed_comments", sa.Integer(), nullable=False),
        sa.Column("dominant_sentiment", sa.String(16)),
        sa.Column("dominant_stance", sa.String(16)),
        sa.Column("dominant_risk", sa.String(16)),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("semantic_run_id", "cluster_key", name="uq_topic_cluster_run_key"),
        sa.CheckConstraint("size > 0", name="ck_topic_cluster_size_positive"),
        sa.CheckConstraint("dimension > 0", name="ck_topic_cluster_dimension_positive"),
    )
    op.create_index("ix_topic_cluster_run_size", "topic_clusters", ["semantic_run_id", "size"])

    op.create_table(
        "semantic_run_items",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column(
            "semantic_run_id",
            sa.String(36),
            sa.ForeignKey("semantic_analysis_runs.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "comment_id",
            sa.String(36),
            sa.ForeignKey("normalized_comments.id", ondelete="CASCADE"),
            nullable=False,
        ),
        sa.Column(
            "embedding_id",
            sa.String(36),
            sa.ForeignKey("comment_embeddings.id", ondelete="RESTRICT"),
        ),
        sa.Column(
            "analysis_id",
            sa.String(36),
            sa.ForeignKey("comment_analyses.id", ondelete="SET NULL"),
        ),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column(
            "topic_cluster_id",
            sa.String(36),
            sa.ForeignKey("topic_clusters.id", ondelete="SET NULL"),
        ),
        sa.Column("topic_similarity", sa.Float()),
        sa.Column("topic_representative_rank", sa.Integer()),
        sa.Column("topic_representative_score", sa.Float()),
        sa.Column("error_message", sa.Text()),
        sa.UniqueConstraint("semantic_run_id", "comment_id", name="uq_semantic_run_item_comment"),
    )
    op.create_index("ix_semantic_run_items_semantic_run_id", "semantic_run_items", ["semantic_run_id"])
    op.create_index(
        "ix_semantic_run_item_run_cluster",
        "semantic_run_items",
        ["semantic_run_id", "topic_cluster_id"],
    )


def downgrade() -> None:
    op.drop_table("semantic_run_items")
    op.drop_table("topic_clusters")
    op.drop_column("semantic_analysis_runs", "topic_error_message")
    op.drop_column("semantic_analysis_runs", "topic_finished_at")
    op.drop_column("semantic_analysis_runs", "topic_started_at")
    op.drop_column("semantic_analysis_runs", "clustered_comments")
    op.drop_column("semantic_analysis_runs", "topic_status")
