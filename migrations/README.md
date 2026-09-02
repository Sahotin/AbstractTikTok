# Analysis database migrations

Fresh database:

```bash
alembic upgrade head
```

Existing pre-Alembic `database/analysis.db`:

1. Back up the database.
2. Verify the seven Phase 1A–3A tables and their row counts.
3. Run `alembic stamp 0001_phase1_to_phase3a`.
4. Run `alembic upgrade head`.
5. Run SQLite integrity and foreign-key checks.

The application never runs migrations automatically.

Phase 3B-2 upgrade from an already migrated analysis database:

```bash
alembic upgrade 0003_phase3b_topic_clustering
```

This adds fixed `semantic_run_items`, persistent `topic_clusters`, and topic-stage
fields without generating embeddings or clustering data during migration.
