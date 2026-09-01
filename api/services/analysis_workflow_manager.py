"""In-process orchestration from one crawler output to the AI dashboard."""

from __future__ import annotations

import asyncio
from pathlib import Path
from uuid import UUID, uuid4

from analysis.domain import TopicClusterConfig
from analysis.embeddings import EmbeddingConfig, create_embedding_provider
from analysis.llm import LLMConfig, create_provider
from analysis.preprocessing import utc_now
from analysis.repositories import AnalysisRepository, SemanticRepository
from analysis.services import (
    AnalysisJobRunner,
    AnalysisJobService,
    AnalysisQueryService,
    BatchAnalysisConfig,
    CommentAnalysisService,
    EmbeddingService,
    IngestionService,
    SemanticRunService,
    TopicService,
)
from api.runtime_env import PROJECT_ROOT
from api.schemas.analysis import (
    CollectionAnalysisWorkflowRequest,
    CollectionAnalysisWorkflowResponse,
)


DATA_ROOT = PROJECT_ROOT / "data"
ANALYSIS_DATABASE = PROJECT_ROOT / "database" / "analysis.db"
LLM_ITEMS = {"sentiment", "emotion", "stance", "keywords", "risk", "agent"}
SEMANTIC_ITEMS = {"topics", "semantic_search", "agent"}


class AnalysisWorkflowManager:
    """Run bounded ingestion, LLM analysis, embedding, and clustering jobs."""

    def __init__(self) -> None:
        self._records: dict[UUID, CollectionAnalysisWorkflowResponse] = {}
        self._tasks: dict[UUID, asyncio.Task] = {}

    def start(self, request: CollectionAnalysisWorkflowRequest) -> CollectionAnalysisWorkflowResponse:
        workflow_id = uuid4()
        now = utc_now()
        record = CollectionAnalysisWorkflowResponse(
            id=workflow_id,
            status="pending",
            stage="queued",
            progress_percent=0,
            message="分析任务已创建",
            platform=request.platform,
            source_files=request.source_files,
            analysis_items=list(dict.fromkeys(request.analysis_items)),
            requested_limit=request.limit,
            created_at=now,
            updated_at=now,
        )
        self._records[workflow_id] = record
        self._tasks[workflow_id] = asyncio.create_task(self._execute(workflow_id, request))
        return record

    def get(self, workflow_id: UUID) -> CollectionAnalysisWorkflowResponse | None:
        return self._records.get(workflow_id)

    def _update(self, workflow_id: UUID, **values) -> CollectionAnalysisWorkflowResponse:
        record = self._records[workflow_id].model_copy(update={**values, "updated_at": utc_now()})
        self._records[workflow_id] = record
        return record

    @staticmethod
    def _resolve_sources(source_files: list[str]) -> list[Path]:
        data_root = DATA_ROOT.resolve()
        resolved: list[Path] = []
        for relative in source_files:
            path = (DATA_ROOT / relative).resolve()
            try:
                path.relative_to(data_root)
            except ValueError as exc:
                raise ValueError(f"数据文件不在允许的 data 目录中: {relative}") from exc
            if not path.is_file():
                raise FileNotFoundError(f"采集数据文件不存在: {relative}")
            if path.suffix.lower() not in {".json", ".jsonl", ".csv", ".xlsx"}:
                raise ValueError(f"暂不支持该采集文件格式: {path.suffix}")
            resolved.append(path)
        return resolved

    async def _execute(self, workflow_id: UUID, request: CollectionAnalysisWorkflowRequest) -> None:
        analysis_repository = AnalysisRepository(ANALYSIS_DATABASE)
        semantic_repository = SemanticRepository(ANALYSIS_DATABASE)
        try:
            sources = self._resolve_sources(request.source_files)
            self._update(
                workflow_id,
                status="running",
                stage="ingesting",
                progress_percent=5,
                message="正在标准化本次采集数据",
            )
            ingestion = IngestionService(analysis_repository, batch_size=500)
            report = await ingestion.ingest(
                sources,
                platform=request.platform.value,
                crawler_type=request.crawler_type,
            )
            run_comment_count = report.first_level_comment_count + report.second_level_comment_count
            self._update(
                workflow_id,
                run_id=report.run_id,
                content_count=report.video_count,
                comment_count=run_comment_count,
                progress_percent=25,
                message=f"已导入 {run_comment_count} 条评论",
            )
            if run_comment_count == 0:
                raise ValueError("本次采集文件中没有可分析的评论")

            items = set(request.analysis_items)
            if items & LLM_ITEMS:
                self._update(
                    workflow_id,
                    stage="analyzing",
                    progress_percent=30,
                    message="DeepSeek 正在分析情感、立场与风险",
                )
                llm_config = LLMConfig.from_environment()
                llm_provider = create_provider(llm_config)
                comment_service = CommentAnalysisService(analysis_repository, llm_provider, llm_config)
                job_service = AnalysisJobService(
                    analysis_repository,
                    AnalysisQueryService(analysis_repository),
                    comment_service,
                )
                batch_config = BatchAnalysisConfig.from_environment().model_copy(update={
                    "batch_size": request.batch_size,
                    "max_concurrency": request.max_concurrency,
                })
                job = await job_service.create_job(
                    platform=request.platform.value,
                    run_id=report.run_id,
                    limit=min(request.limit, run_comment_count),
                    config=batch_config,
                )
                self._update(workflow_id, analysis_job_id=job.id)
                job = await AnalysisJobRunner(analysis_repository, comment_service).run(job.id)
                self._update(
                    workflow_id,
                    analyzed_count=job.completed_count + job.skipped_count,
                    progress_percent=65 if items & SEMANTIC_ITEMS else 90,
                    message=f"评论 AI 分析完成 {job.completed_count + job.skipped_count} 条",
                )

            if items & SEMANTIC_ITEMS:
                self._update(
                    workflow_id,
                    stage="embedding",
                    progress_percent=68,
                    message="正在生成评论 Embedding",
                )
                embedding_config = EmbeddingConfig.from_environment()
                embedding_provider = create_embedding_provider(embedding_config)
                semantic_service = SemanticRunService(
                    analysis_repository,
                    semantic_repository,
                    EmbeddingService(semantic_repository, embedding_provider, embedding_config),
                    embedding_config,
                )
                semantic_run = await semantic_service.create_run(
                    platform=request.platform.value,
                    run_id=report.run_id,
                    limit=min(request.limit, run_comment_count),
                )
                self._update(workflow_id, semantic_run_id=semantic_run.id)
                semantic_run, _report = await semantic_service.execute_run(semantic_run.id)
                if "topics" in items or "agent" in items:
                    self._update(
                        workflow_id,
                        stage="clustering",
                        progress_percent=88,
                        message="正在聚类评论主题与代表观点",
                    )
                    semantic_run, result = await TopicService(semantic_repository).cluster_run(
                        semantic_run.id,
                        TopicClusterConfig(),
                    )
                    self._update(workflow_id, topic_count=len(result.clusters))

            completed = self._records[workflow_id]
            query = [f"workflow_id={workflow_id}", f"platform={request.platform.value}", f"run_id={report.run_id}"]
            if completed.semantic_run_id:
                query.append(f"semantic_run_id={completed.semantic_run_id}")
            self._update(
                workflow_id,
                status="completed",
                stage="completed",
                progress_percent=100,
                message="分析完成，正在加载智能分析页面",
                dashboard_url="/dashboard?" + "&".join(query),
            )
        except Exception as exc:
            self._update(
                workflow_id,
                status="failed",
                stage="failed",
                message="分析任务失败",
                error_message=str(exc),
            )
        finally:
            await semantic_repository.close()
            await analysis_repository.close()
            self._tasks.pop(workflow_id, None)


analysis_workflow_manager = AnalysisWorkflowManager()
