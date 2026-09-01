"""Phase 1A through Phase 3A analysis schema baseline."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0001_phase1_to_phase3a"
down_revision: Union[str, Sequence[str], None] = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table(
        "collection_runs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("platform", sa.String(32), nullable=False),
        sa.Column("crawler_type", sa.String(32), nullable=False),
        sa.Column("query", sa.Text()),
        sa.Column("specified_ids", sa.JSON(), nullable=False),
        sa.Column("status", sa.String(32), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("config_snapshot", sa.JSON(), nullable=False),
        sa.Column("output_location", sa.JSON(), nullable=False),
        sa.Column("content_count", sa.Integer(), nullable=False),
        sa.Column("comment_count", sa.Integer(), nullable=False),
        sa.Column("error_message", sa.Text()),
    )
    op.create_index("ix_collection_runs_platform", "collection_runs", ["platform"])
    op.create_index("ix_collection_runs_status", "collection_runs", ["status"])

    op.create_table(
        "normalized_authors",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("platform", sa.String(32), nullable=False),
        sa.Column("native_author_id", sa.String(255), nullable=False),
        sa.Column("nickname", sa.Text()),
        sa.Column("profile_url", sa.Text()),
        sa.Column("avatar_url", sa.Text()),
        sa.Column("description", sa.Text()),
        sa.Column("gender", sa.String(32)),
        sa.Column("ip_location", sa.String(128)),
        sa.Column("following_count", sa.BigInteger()),
        sa.Column("follower_count", sa.BigInteger()),
        sa.Column("content_count", sa.BigInteger()),
        sa.Column("verified", sa.Boolean()),
        sa.Column("attributes", sa.JSON(), nullable=False),
        sa.Column("raw_payload", sa.JSON(), nullable=False),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("platform", "native_author_id", name="uq_normalized_author_platform_native"),
    )

    op.create_table(
        "normalized_contents",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("collection_runs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("platform", sa.String(32), nullable=False),
        sa.Column("native_content_id", sa.String(255), nullable=False),
        sa.Column("content_type", sa.String(64), nullable=False),
        sa.Column("title", sa.Text()), sa.Column("body", sa.Text()), sa.Column("url", sa.Text()),
        sa.Column("author_id", sa.String(36), sa.ForeignKey("normalized_authors.id", ondelete="SET NULL")),
        sa.Column("native_author_id", sa.String(255)),
        sa.Column("published_at", sa.DateTime(timezone=True)),
        sa.Column("like_count", sa.BigInteger()), sa.Column("comment_count", sa.BigInteger()),
        sa.Column("share_count", sa.BigInteger()), sa.Column("favorite_count", sa.BigInteger()),
        sa.Column("view_count", sa.BigInteger()), sa.Column("source_keyword", sa.Text()),
        sa.Column("media_urls", sa.JSON(), nullable=False), sa.Column("tags", sa.JSON(), nullable=False),
        sa.Column("raw_payload", sa.JSON(), nullable=False),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("platform", "native_content_id", name="uq_normalized_content_platform_native"),
    )
    op.create_index("ix_normalized_contents_run_id", "normalized_contents", ["run_id"])
    op.create_index("ix_normalized_contents_author_id", "normalized_contents", ["author_id"])
    op.create_index("ix_normalized_contents_published_at", "normalized_contents", ["published_at"])
    op.create_index("ix_normalized_content_run_published", "normalized_contents", ["run_id", "published_at"])

    op.create_table(
        "normalized_comments",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("collection_runs.id", ondelete="RESTRICT"), nullable=False),
        sa.Column("platform", sa.String(32), nullable=False),
        sa.Column("native_comment_id", sa.String(255), nullable=False),
        sa.Column("content_id", sa.String(36), sa.ForeignKey("normalized_contents.id", ondelete="CASCADE"), nullable=False),
        sa.Column("native_content_id", sa.String(255), nullable=False),
        sa.Column("author_id", sa.String(36), sa.ForeignKey("normalized_authors.id", ondelete="SET NULL")),
        sa.Column("native_author_id", sa.String(255)),
        sa.Column("parent_comment_id", sa.String(255)), sa.Column("root_comment_id", sa.String(255)),
        sa.Column("depth", sa.Integer(), nullable=False), sa.Column("text", sa.Text(), nullable=False),
        sa.Column("published_at", sa.DateTime(timezone=True)), sa.Column("like_count", sa.BigInteger()),
        sa.Column("reply_count", sa.BigInteger()), sa.Column("ip_location", sa.String(128)),
        sa.Column("media_urls", sa.JSON(), nullable=False), sa.Column("text_hash", sa.String(64), nullable=False),
        sa.Column("language", sa.String(32)), sa.Column("raw_payload", sa.JSON(), nullable=False),
        sa.Column("collected_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("platform", "native_comment_id", name="uq_normalized_comment_platform_native"),
    )
    for name, columns in [
        ("ix_normalized_comments_run_id", ["run_id"]),
        ("ix_normalized_comments_content_id", ["content_id"]),
        ("ix_normalized_comments_author_id", ["author_id"]),
        ("ix_normalized_comments_published_at", ["published_at"]),
        ("ix_normalized_comment_content_published", ["content_id", "published_at"]),
        ("ix_normalized_comment_run_depth", ["run_id", "depth"]),
        ("ix_normalized_comment_text_hash", ["text_hash"]),
    ]:
        op.create_index(name, "normalized_comments", columns)

    op.create_table(
        "comment_analyses",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("comment_id", sa.String(36), sa.ForeignKey("normalized_comments.id", ondelete="CASCADE"), nullable=False),
        sa.Column("sentiment", sa.String(16), nullable=False), sa.Column("sentiment_score", sa.Float(), nullable=False),
        sa.Column("emotion", sa.String(16), nullable=False), sa.Column("topics", sa.JSON(), nullable=False),
        sa.Column("stance", sa.String(16), nullable=False), sa.Column("risk_level", sa.String(16), nullable=False),
        sa.Column("risk_reasons", sa.JSON(), nullable=False), sa.Column("keywords", sa.JSON(), nullable=False),
        sa.Column("summary", sa.Text(), nullable=False), sa.Column("model", sa.String(255), nullable=False),
        sa.Column("provider", sa.String(64), nullable=False), sa.Column("prompt_version", sa.String(64), nullable=False),
        sa.Column("analysis_version", sa.String(64), nullable=False), sa.Column("input_hash", sa.String(64), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("comment_id", "input_hash", "provider", "model", "prompt_version", "analysis_version", name="uq_comment_analysis_cache_key"),
    )
    op.create_index("ix_comment_analyses_comment_id", "comment_analyses", ["comment_id"])
    op.create_index("ix_comment_analysis_comment_created", "comment_analyses", ["comment_id", "created_at"])

    op.create_table(
        "analysis_jobs",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("parent_job_id", sa.String(36), sa.ForeignKey("analysis_jobs.id", ondelete="SET NULL")),
        sa.Column("status", sa.String(32), nullable=False), sa.Column("platform", sa.String(32), nullable=False),
        sa.Column("run_id", sa.String(36), sa.ForeignKey("collection_runs.id", ondelete="SET NULL")),
        sa.Column("content_id", sa.String(36), sa.ForeignKey("normalized_contents.id", ondelete="SET NULL")),
        sa.Column("requested_limit", sa.Integer(), nullable=False), sa.Column("total_comments", sa.Integer(), nullable=False),
        sa.Column("pending_count", sa.Integer(), nullable=False), sa.Column("processing_count", sa.Integer(), nullable=False),
        sa.Column("completed_count", sa.Integer(), nullable=False), sa.Column("failed_count", sa.Integer(), nullable=False),
        sa.Column("skipped_count", sa.Integer(), nullable=False), sa.Column("provider", sa.String(64), nullable=False),
        sa.Column("model", sa.String(255), nullable=False), sa.Column("prompt_version", sa.String(64), nullable=False),
        sa.Column("analysis_version", sa.String(64), nullable=False), sa.Column("max_concurrency", sa.Integer(), nullable=False),
        sa.Column("batch_size", sa.Integer(), nullable=False), sa.Column("max_requests_per_minute", sa.Integer()),
        sa.Column("estimated_input_tokens", sa.BigInteger(), nullable=False),
        sa.Column("estimated_output_tokens", sa.BigInteger(), nullable=False),
        sa.Column("input_price_per_1m_tokens", sa.Float()), sa.Column("output_price_per_1m_tokens", sa.Float()),
        sa.Column("estimated_cost", sa.Float()), sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True)), sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.Column("error_message", sa.Text()),
    )
    for name, columns in [
        ("ix_analysis_jobs_run_id", ["run_id"]), ("ix_analysis_jobs_content_id", ["content_id"]),
        ("ix_analysis_job_status_created", ["status", "created_at"]), ("ix_analysis_job_parent", ["parent_job_id"]),
    ]:
        op.create_index(name, "analysis_jobs", columns)

    op.create_table(
        "analysis_job_items",
        sa.Column("id", sa.String(36), primary_key=True),
        sa.Column("job_id", sa.String(36), sa.ForeignKey("analysis_jobs.id", ondelete="CASCADE"), nullable=False),
        sa.Column("comment_id", sa.String(36), sa.ForeignKey("normalized_comments.id", ondelete="CASCADE"), nullable=False),
        sa.Column("status", sa.String(32), nullable=False), sa.Column("attempt_count", sa.Integer(), nullable=False),
        sa.Column("error_type", sa.String(128)), sa.Column("error_message", sa.Text()),
        sa.Column("started_at", sa.DateTime(timezone=True)), sa.Column("finished_at", sa.DateTime(timezone=True)),
        sa.UniqueConstraint("job_id", "comment_id", name="uq_analysis_job_item_comment"),
    )
    op.create_index("ix_analysis_job_items_comment_id", "analysis_job_items", ["comment_id"])
    op.create_index("ix_analysis_job_item_job_status", "analysis_job_items", ["job_id", "status"])


def downgrade() -> None:
    for table in ["analysis_job_items", "analysis_jobs", "comment_analyses", "normalized_comments", "normalized_contents", "normalized_authors", "collection_runs"]:
        op.drop_table(table)
