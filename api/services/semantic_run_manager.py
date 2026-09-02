"""Single-process launcher for persistent embedding-only semantic runs."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import UUID

from analysis.domain import SemanticRunStatus
from analysis.embeddings import EmbeddingConfig, create_embedding_provider
from analysis.preprocessing import utc_now
from analysis.repositories import AnalysisRepository, SemanticRepository
from analysis.services import EmbeddingService, SemanticRunService


logger = logging.getLogger("analysis.semantic_run_manager")


class InProcessSemanticRunManager:
    """Keep task references while the database remains the source of truth."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self._tasks: dict[str, asyncio.Task[None]] = {}

    def start(self, run_id: UUID | str) -> bool:
        run_id_string = str(run_id)
        existing = self._tasks.get(run_id_string)
        if existing is not None and not existing.done():
            return False
        task = asyncio.create_task(self._run(run_id_string), name=f"semantic-run-{run_id_string}")
        self._tasks[run_id_string] = task
        task.add_done_callback(lambda completed, key=run_id_string: self._on_done(key, completed))
        return True

    async def _run(self, run_id: str) -> None:
        analysis_repository = AnalysisRepository(self.database_path)
        semantic_repository = SemanticRepository(self.database_path)
        try:
            config = EmbeddingConfig.from_environment()
            provider = create_embedding_provider(config)
            service = SemanticRunService(
                analysis_repository,
                semantic_repository,
                EmbeddingService(semantic_repository, provider, config),
                config,
            )
            await service.execute_run(run_id)
        except asyncio.CancelledError:
            await semantic_repository.update_semantic_run(
                run_id,
                status=SemanticRunStatus.INTERRUPTED,
                finished_at=utc_now(),
                error_message="Semantic run task was interrupted",
            )
            raise
        except Exception as exc:
            try:
                await semantic_repository.update_semantic_run(
                    run_id,
                    status=SemanticRunStatus.FAILED,
                    finished_at=utc_now(),
                    error_message=f"{type(exc).__name__}: {str(exc)[:500]}",
                )
            except Exception:
                logger.exception("semantic_run_failure_status_write_failed run_id=%s", run_id)
            raise
        finally:
            await semantic_repository.close()
            await analysis_repository.close()

    def _on_done(self, run_id: str, task: asyncio.Task[None]) -> None:
        self._tasks.pop(run_id, None)
        if task.cancelled():
            logger.warning("semantic_run_task_cancelled run_id=%s", run_id)
            return
        error = task.exception()
        if error is not None:
            logger.error("semantic_run_task_failed run_id=%s error_type=%s", run_id, type(error).__name__)
