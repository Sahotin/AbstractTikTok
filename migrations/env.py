from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import MetaData, engine_from_config, pool

from database.models import Base
import database.analysis_models  # noqa: F401
import database.semantic_models  # noqa: F401


config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

ANALYSIS_TABLES = {
    "collection_runs",
    "normalized_authors",
    "normalized_contents",
    "normalized_comments",
    "comment_analyses",
    "analysis_jobs",
    "analysis_job_items",
    "comment_embeddings",
    "semantic_analysis_runs",
    "semantic_run_items",
    "topic_clusters",
}
target_metadata = MetaData()
for table_name in ANALYSIS_TABLES:
    Base.metadata.tables[table_name].to_metadata(target_metadata)


def run_migrations_offline() -> None:
    context.configure(
        url=config.get_main_option("sqlalchemy.url"),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata, compare_type=True)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
