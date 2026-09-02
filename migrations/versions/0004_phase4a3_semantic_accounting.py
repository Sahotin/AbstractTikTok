"""Add unambiguous semantic-run accounting fields."""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "0004_phase4a3_semantic_accounting"
down_revision: Union[str, Sequence[str], None] = "0003_phase3b_topic_clustering"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column("semantic_analysis_runs", sa.Column("eligible_comments", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("semantic_analysis_runs", sa.Column("excluded_comments", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("semantic_analysis_runs", sa.Column("embedding_failed_comments", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("semantic_analysis_runs", sa.Column("clustering_input_comments", sa.Integer(), nullable=False, server_default="0"))
    op.add_column("semantic_analysis_runs", sa.Column("clustering_failed_comments", sa.Integer(), nullable=False, server_default="0"))


def downgrade() -> None:
    for column in ("clustering_failed_comments", "clustering_input_comments", "embedding_failed_comments", "excluded_comments", "eligible_comments"):
        op.drop_column("semantic_analysis_runs", column)
