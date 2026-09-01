"""Single-process task launcher backed by persistent AnalysisJob state."""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from uuid import UUID

from analysis.domain import AnalysisJobStatus
from analysis.llm import LLMConfig, create_provider
from analysis.preprocessing import utc_now
from analysis.repositories import AnalysisRepository
from analysis.services import AnalysisJobRunner, CommentAnalysisService


logger = logging.getLogger("analysis.job_manager")


class InProcessAnalysisJobManager:
    """Keep task references; persistence and recovery remain database-driven."""

    def __init__(self, database_path: Path):
        self.database_path = database_path
        self._tasks: dict[str, asyncio.Task] = {}

    def start(self, job_id: UUID | str) -> bool:
        job_id_string = str(job_id)
        existing = self._tasks.get(job_id_string)
        if existing is not None and not existing.done():
            return False
        task = asyncio.create_task(self._run(job_id_string), name=f"analysis-job-{job_id_string}")
        self._tasks[job_id_string] = task
        task.add_done_callback(lambda completed, key=job_id_string: self._on_done(key, completed))
        return True

    async def _run(self, job_id: str) -> None:
        repository = AnalysisRepository(self.database_path)
        try:
            await repository.initialize()
            config = LLMConfig.from_environment()
            provider = create_provider(config)
            analysis_service = CommentAnalysisService(repository, provider, config)
            runner = AnalysisJobRunner(repository, analysis_service)
            await runner.run(job_id)
        except Exception as exc:
            await repository.update_analysis_job_status(
                job_id,
                AnalysisJobStatus.FAILED,
                finished_at=utc_now(),
                error_message=f"{type(exc).__name__}: {str(exc)[:500]}",
            )
            raise
        finally:
            await repository.close()

    def _on_done(self, job_id: str, task: asyncio.Task) -> None:
        self._tasks.pop(job_id, None)
        if task.cancelled():
            logger.warning("analysis_job_task_cancelled job_id=%s", job_id)
            return
        error = task.exception()
        if error is not None:
            logger.error("analysis_job_task_failed job_id=%s error_type=%s", job_id, type(error).__name__)
