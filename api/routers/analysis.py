"""Read-only API for normalized content, comments, authors, and statistics."""

from __future__ import annotations

import asyncio
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, AsyncIterator, Optional
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from analysis.domain import AnalysisJobItemStatus, AnalysisJobStatus, Platform, PublicOpinionQuery, TrendQuery
from analysis.agent import (
    AgentLimitError,
    AgentToolRegistry,
    OpenAICompatibleToolCallingProvider,
    PublicOpinionAgentService,
)
from analysis.embeddings import EmbeddingConfig, EmbeddingProviderError, create_embedding_provider
from analysis.evaluation import CommentQualityArtifact
from analysis.llm import LLMConfig, create_provider
from analysis.llm.base import LLMProviderError, StructuredOutputError, TransientLLMError
from analysis.preprocessing import prepare_analysis_text, utc_now
from analysis.repositories import AnalysisRepository, SemanticRepository
from analysis.services import (
    AnalysisJobService,
    AnalysisQueryService,
    BatchAnalysisConfig,
    CommentAnalysisError,
    CommentAnalysisService,
    EmbeddingService,
    PublicOpinionService,
    SemanticRunService,
    SemanticSearchService,
    TrendAnalysisService,
    present_topic,
    present_topic_set,
)
from api.services.analysis_job_manager import InProcessAnalysisJobManager
from api.services.semantic_run_manager import InProcessSemanticRunManager
from api.services.analysis_workflow_manager import analysis_workflow_manager
from api.schemas.analysis import (
    AnalysisStatisticsResponse,
    AnalysisJobCreateRequest,
    AnalysisJobResponse,
    AuthorListResponse,
    AuthorResponse,
    AuthorStatisticsResponse,
    CommentListResponse,
    CommentAnalysisResponse,
    CommentResponse,
    CommentStatisticsResponse,
    ContentListResponse,
    ContentResponse,
    ContentStatisticsResponse,
    PublicOpinionResponse,
    SemanticRunCreateRequest,
    SemanticRunResponse,
    SemanticSearchRequest,
    SemanticSearchResponse,
    TopicClusterResponse,
    TopicDetailResponse,
    TopicListResponse,
    TopicRepresentativeResponse,
    TrendResponse,
    AgentChatRequest,
    AgentChatResponse,
    AgentChatFailureResponse,
    CollectionAnalysisWorkflowRequest,
    CollectionAnalysisWorkflowResponse,
    AnalysisResultScopeResponse,
    AnalysisResultScopeListResponse,
)


router = APIRouter(prefix="/analysis", tags=["analysis"])
DEFAULT_ANALYSIS_DATABASE = Path(__file__).parents[2] / "database" / "analysis.db"
DEFAULT_QUALITY_REPORT_DIR = Path(__file__).parents[2] / "benchmarks" / "comment_quality"
analysis_job_manager = InProcessAnalysisJobManager(DEFAULT_ANALYSIS_DATABASE)
semantic_run_manager = InProcessSemanticRunManager(DEFAULT_ANALYSIS_DATABASE)


@dataclass(frozen=True)
class AgentRuntime:
    analysis_repository: AnalysisRepository
    semantic_repository: SemanticRepository
    llm_provider: OpenAICompatibleToolCallingProvider
    embedding_provider: object
    llm_config: LLMConfig


async def get_analysis_query_service() -> AsyncIterator[AnalysisQueryService]:
    """Provide one short-lived repository per request and always close it."""

    repository = AnalysisRepository(DEFAULT_ANALYSIS_DATABASE)
    try:
        yield AnalysisQueryService(repository)
    finally:
        await repository.close()


QueryServiceDependency = Annotated[AnalysisQueryService, Depends(get_analysis_query_service)]
LimitParameter = Annotated[int, Query(ge=1, le=500)]
OffsetParameter = Annotated[int, Query(ge=0)]


async def get_comment_analysis_service() -> AsyncIterator[CommentAnalysisService]:
    """Build the remote provider only for the explicit single-comment endpoint."""

    repository = AnalysisRepository(DEFAULT_ANALYSIS_DATABASE)
    try:
        await repository.initialize()
        config = LLMConfig.from_environment()
        try:
            provider = create_provider(config)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        yield CommentAnalysisService(repository, provider, config)
    finally:
        await repository.close()


CommentAnalysisServiceDependency = Annotated[
    CommentAnalysisService,
    Depends(get_comment_analysis_service),
]


async def get_analysis_repository() -> AsyncIterator[AnalysisRepository]:
    repository = AnalysisRepository(DEFAULT_ANALYSIS_DATABASE)
    try:
        await repository.initialize()
        yield repository
    finally:
        await repository.close()


async def get_public_opinion_service() -> AsyncIterator[PublicOpinionService]:
    """Provide a read-only aggregate service without constructing an LLM provider."""

    repository = AnalysisRepository(DEFAULT_ANALYSIS_DATABASE)
    try:
        await repository.initialize()
        yield PublicOpinionService(repository)
    finally:
        await repository.close()


async def get_trend_service() -> AsyncIterator[TrendAnalysisService]:
    repository = AnalysisRepository(DEFAULT_ANALYSIS_DATABASE)
    try:
        await repository.initialize()
        yield TrendAnalysisService(repository)
    finally:
        await repository.close()


async def get_analysis_job_service() -> AsyncIterator[AnalysisJobService]:
    repository = AnalysisRepository(DEFAULT_ANALYSIS_DATABASE)
    try:
        await repository.initialize()
        llm_config = LLMConfig.from_environment()
        try:
            provider = create_provider(llm_config)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        analysis_service = CommentAnalysisService(repository, provider, llm_config)
        yield AnalysisJobService(
            repository,
            AnalysisQueryService(repository),
            analysis_service,
        )
    finally:
        await repository.close()


def get_analysis_job_manager() -> InProcessAnalysisJobManager:
    return analysis_job_manager


async def get_semantic_run_service() -> AsyncIterator[SemanticRunService]:
    analysis_repository = AnalysisRepository(DEFAULT_ANALYSIS_DATABASE)
    semantic_repository = SemanticRepository(DEFAULT_ANALYSIS_DATABASE)
    try:
        config = EmbeddingConfig.from_environment()
        try:
            provider = create_embedding_provider(config)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        yield SemanticRunService(
            analysis_repository,
            semantic_repository,
            EmbeddingService(semantic_repository, provider, config),
            config,
        )
    finally:
        await semantic_repository.close()
        await analysis_repository.close()


def get_semantic_run_manager() -> InProcessSemanticRunManager:
    return semantic_run_manager


async def get_semantic_repository() -> AsyncIterator[SemanticRepository]:
    repository = SemanticRepository(DEFAULT_ANALYSIS_DATABASE)
    try:
        yield repository
    finally:
        await repository.close()


async def get_semantic_search_service() -> AsyncIterator[SemanticSearchService]:
    repository = SemanticRepository(DEFAULT_ANALYSIS_DATABASE)
    try:
        config = EmbeddingConfig.from_environment()
        try:
            provider = create_embedding_provider(config)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        yield SemanticSearchService(repository, provider)
    finally:
        await repository.close()


async def get_agent_runtime() -> AsyncIterator[AgentRuntime]:
    analysis_repository = AnalysisRepository(DEFAULT_ANALYSIS_DATABASE)
    semantic_repository = SemanticRepository(DEFAULT_ANALYSIS_DATABASE)
    try:
        await analysis_repository.initialize()
        llm_config = LLMConfig.from_environment()
        embedding_config = EmbeddingConfig.from_environment()
        try:
            llm_provider = OpenAICompatibleToolCallingProvider(llm_config)
            embedding_provider = create_embedding_provider(embedding_config)
        except ValueError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        yield AgentRuntime(
            analysis_repository=analysis_repository,
            semantic_repository=semantic_repository,
            llm_provider=llm_provider,
            embedding_provider=embedding_provider,
            llm_config=llm_config,
        )
    finally:
        await semantic_repository.close()
        await analysis_repository.close()


RepositoryDependency = Annotated[AnalysisRepository, Depends(get_analysis_repository)]
PublicOpinionServiceDependency = Annotated[PublicOpinionService, Depends(get_public_opinion_service)]
TrendServiceDependency = Annotated[TrendAnalysisService, Depends(get_trend_service)]
JobServiceDependency = Annotated[AnalysisJobService, Depends(get_analysis_job_service)]
JobManagerDependency = Annotated[InProcessAnalysisJobManager, Depends(get_analysis_job_manager)]
SemanticRunServiceDependency = Annotated[SemanticRunService, Depends(get_semantic_run_service)]
SemanticRunManagerDependency = Annotated[InProcessSemanticRunManager, Depends(get_semantic_run_manager)]
SemanticRepositoryDependency = Annotated[SemanticRepository, Depends(get_semantic_repository)]
SemanticSearchServiceDependency = Annotated[
    SemanticSearchService, Depends(get_semantic_search_service)
]
AgentRuntimeDependency = Annotated[AgentRuntime, Depends(get_agent_runtime)]


async def _semantic_snapshot_context(
    repository: SemanticRepository,
    semantic_run_id: UUID | None,
):
    """Resolve a semantic run's frozen analysis identity for read endpoints."""

    if semantic_run_id is None:
        return None, None
    run = await repository.get_semantic_run(str(semantic_run_id))
    if run is None:
        raise HTTPException(status_code=404, detail="Semantic analysis run not found")
    try:
        snapshot = await repository.get_analysis_snapshot(str(semantic_run_id))
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    if snapshot is None:
        raise HTTPException(status_code=409, detail="Semantic run has no frozen comment-analysis snapshot")
    return run, snapshot


@router.get("/quality/latest", response_model=CommentQualityArtifact)
async def get_latest_comment_quality() -> CommentQualityArtifact:
    """Return the newest valid machine-readable quality report without raw comments."""

    candidates = sorted(DEFAULT_QUALITY_REPORT_DIR.glob("comment_quality_report_*.json"), reverse=True)
    for path in candidates:
        try:
            return CommentQualityArtifact.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
    raise HTTPException(status_code=404, detail="No comment quality evaluation report is available")


def _topic_response(topic) -> TopicClusterResponse:
    values = topic.model_dump(mode="python")
    values["analysis_coverage"] = topic.analyzed_comments / topic.size if topic.size else 0.0
    return TopicClusterResponse.model_validate(values)


@router.get("/topics", response_model=TopicListResponse)
async def list_topics(
    repository: SemanticRepositoryDependency,
    semantic_run_id: Optional[UUID] = None,
    platform: Optional[Platform] = None,
    run_id: Optional[UUID] = None,
    content_id: Optional[UUID] = None,
    limit: LimitParameter = 50,
    offset: OffsetParameter = 0,
    min_size: Annotated[int, Query(ge=1)] = 1,
) -> TopicListResponse:
    page = await repository.list_topic_clusters(
        semantic_run_id=str(semantic_run_id) if semantic_run_id else None,
        platform=platform.value if platform else None,
        run_id=str(run_id) if run_id else None,
        content_id=str(content_id) if content_id else None,
        min_size=min_size,
        limit=limit,
        offset=offset,
    )
    return TopicListResponse(
        items=[_topic_response(item) for item in page.items],
        total=page.total,
        limit=page.limit,
        offset=page.offset,
    )


@router.get("/topics/{topic_id}", response_model=TopicDetailResponse)
async def get_topic_detail(
    topic_id: UUID,
    repository: SemanticRepositoryDependency,
    representative_limit: Annotated[int, Query(ge=1, le=10)] = 3,
) -> TopicDetailResponse:
    topic = await repository.get_topic_cluster(str(topic_id))
    if topic is None:
        raise HTTPException(status_code=404, detail="Semantic topic not found")
    run = await repository.get_semantic_run(str(topic.semantic_run_id))
    if run is None:
        raise HTTPException(status_code=404, detail="Semantic analysis run not found")
    representatives = await repository.get_topic_representatives(
        str(topic.id), limit=representative_limit
    )
    # A semantic run may contain several distinct clusters with the same
    # broad category. Resolve only their display names as one set; membership
    # and persisted cluster metadata remain immutable.
    siblings = (await repository.list_topic_clusters(
        semantic_run_id=str(topic.semantic_run_id), limit=5_000
    )).items
    sibling_representatives = await asyncio.gather(*(
        repository.get_topic_representatives(str(item.id), limit=3) for item in siblings
    ))
    presentation = present_topic_set(list(zip(siblings, sibling_representatives)))[str(topic.id)]
    distributions = await repository.get_topic_analysis_distributions(str(topic.id))

    def distribution(values: dict[str, int]) -> dict[str, dict[str, int | float]]:
        denominator = sum(values.values())
        if not denominator:
            return {}
        return {
            key: {"count": count, "percentage": round(count / denominator * 100, 2)}
            for key, count in values.items()
        }

    return TopicDetailResponse(
        **_topic_response(topic).model_dump(mode="python", exclude={
            "topic_name", "topic_summary", "topic_naming_version", "naming_provider",
            "naming_model", "naming_prompt_version", "sentiment_distribution", "risk_distribution",
        }),
        topic_name=presentation.name,
        topic_summary=presentation.summary,
        topic_naming_version=presentation.naming_version,
        naming_provider=presentation.provider,
        naming_model=presentation.model,
        naming_prompt_version=presentation.prompt_version,
        sentiment_distribution=distribution(distributions["sentiment"]),
        risk_distribution=distribution(distributions["risk"]),
        representative_comments=[
            TopicRepresentativeResponse.model_validate({
                **item.model_dump(mode="python"),
                # Raw crawler text remains in storage. API consumers receive
                # the same clean text used by semantic analysis.
                "text": prepare_analysis_text(item.text),
            })
            for item in representatives
        ],
        algorithm=run.algorithm,
        algorithm_version=run.algorithm_version,
        config_hash=run.config_hash,
    )


@router.post("/semantic-runs", response_model=SemanticRunResponse, status_code=202)
async def create_semantic_run(
    request: SemanticRunCreateRequest,
    service: SemanticRunServiceDependency,
    manager: SemanticRunManagerDependency,
) -> SemanticRunResponse:
    try:
        run = await service.create_run(
            platform=request.platform,
            run_id=request.run_id,
            content_id=request.content_id,
            limit=request.limit,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    manager.start(run.id)
    return SemanticRunResponse.from_run(run)


@router.get("/semantic-runs/{run_id}", response_model=SemanticRunResponse)
async def get_semantic_run(
    run_id: UUID,
    repository: SemanticRepositoryDependency,
) -> SemanticRunResponse:
    run = await repository.get_semantic_run(str(run_id))
    if run is None:
        raise HTTPException(status_code=404, detail="Semantic analysis run not found")
    return SemanticRunResponse.from_run(run)


@router.get("/contents", response_model=ContentListResponse)
async def list_contents(
    service: QueryServiceDependency,
    platform: Optional[Platform] = None,
    native_content_id: Optional[str] = None,
    run_id: Optional[UUID] = None,
    limit: LimitParameter = 50,
    offset: OffsetParameter = 0,
) -> ContentListResponse:
    page = await service.list_contents(
        platform=platform,
        native_content_id=native_content_id,
        run_id=run_id,
        limit=limit,
        offset=offset,
    )
    return ContentListResponse(items=page.items, total=page.total, limit=page.limit, offset=page.offset)


@router.get("/contents/{content_id}", response_model=ContentResponse)
async def get_content(content_id: UUID, service: QueryServiceDependency) -> ContentResponse:
    content = await service.get_content(content_id)
    if content is None:
        raise HTTPException(status_code=404, detail="Content not found")
    return ContentResponse.model_validate(content)


@router.get("/comments", response_model=CommentListResponse)
async def list_comments(
    service: QueryServiceDependency,
    platform: Optional[Platform] = None,
    content_id: Optional[UUID] = None,
    native_content_id: Optional[str] = None,
    run_id: Optional[UUID] = None,
    parent_comment_id: Optional[str] = None,
    depth: Annotated[Optional[int], Query(ge=0)] = None,
    limit: LimitParameter = 50,
    offset: OffsetParameter = 0,
) -> CommentListResponse:
    page = await service.list_comments(
        platform=platform,
        content_id=content_id,
        native_content_id=native_content_id,
        run_id=run_id,
        parent_comment_id=parent_comment_id,
        depth=depth,
        limit=limit,
        offset=offset,
    )
    return CommentListResponse(items=page.items, total=page.total, limit=page.limit, offset=page.offset)


@router.get("/comments/{comment_id}", response_model=CommentResponse)
async def get_comment(comment_id: UUID, service: QueryServiceDependency) -> CommentResponse:
    comment = await service.get_comment(comment_id)
    if comment is None:
        raise HTTPException(status_code=404, detail="Comment not found")
    return CommentResponse.model_validate(comment)


@router.post("/comments/{comment_id}/analyze", response_model=CommentAnalysisResponse)
async def analyze_comment(
    comment_id: UUID,
    query_service: QueryServiceDependency,
    analysis_service: CommentAnalysisServiceDependency,
) -> CommentAnalysisResponse:
    comment = await query_service.get_comment(comment_id)
    if comment is None:
        raise HTTPException(status_code=404, detail="Comment not found")
    try:
        analysis = await analysis_service.analyze_comment(comment)
    except CommentAnalysisError as exc:
        raise HTTPException(status_code=502, detail="Comment analysis failed") from exc
    return CommentAnalysisResponse.model_validate(analysis)


@router.get("/authors", response_model=AuthorListResponse)
async def list_authors(
    service: QueryServiceDependency,
    platform: Optional[Platform] = None,
    native_author_id: Optional[str] = None,
    limit: LimitParameter = 50,
    offset: OffsetParameter = 0,
) -> AuthorListResponse:
    page = await service.list_authors(
        platform=platform,
        native_author_id=native_author_id,
        limit=limit,
        offset=offset,
    )
    return AuthorListResponse(items=page.items, total=page.total, limit=page.limit, offset=page.offset)


@router.get("/authors/{author_id}", response_model=AuthorResponse)
async def get_author(author_id: UUID, service: QueryServiceDependency) -> AuthorResponse:
    author = await service.get_author(author_id)
    if author is None:
        raise HTTPException(status_code=404, detail="Author not found")
    return AuthorResponse.model_validate(author)


@router.get("/statistics", response_model=AnalysisStatisticsResponse)
async def get_statistics(
    service: QueryServiceDependency,
    platform: Optional[Platform] = None,
    run_id: Optional[UUID] = None,
    content_id: Optional[UUID] = None,
    native_content_id: Optional[str] = None,
) -> AnalysisStatisticsResponse:
    content_statistics = await service.get_content_statistics(platform=platform, run_id=run_id)
    comment_statistics = await service.get_comment_statistics(
        platform=platform,
        run_id=run_id,
        content_id=content_id,
        native_content_id=native_content_id,
    )
    author_statistics = await service.get_author_statistics(platform=platform)
    return AnalysisStatisticsResponse(
        contents=ContentStatisticsResponse(**vars(content_statistics)),
        comments=CommentStatisticsResponse(**vars(comment_statistics)),
        authors=AuthorStatisticsResponse(**vars(author_statistics)),
    )


@router.get("/public-opinion", response_model=PublicOpinionResponse)
async def get_public_opinion(
    service: PublicOpinionServiceDependency,
    semantic_repository: SemanticRepositoryDependency,
    platform: Optional[Platform] = None,
    run_id: Optional[UUID] = None,
    content_id: Optional[UUID] = None,
    semantic_run_id: Optional[UUID] = None,
    top_k: Annotated[int, Query(ge=1, le=100)] = 20,
) -> PublicOpinionResponse:
    semantic_run, analysis_snapshot = await _semantic_snapshot_context(
        semantic_repository, semantic_run_id
    )
    summary = await service.get_public_opinion(
        PublicOpinionQuery(
            platform=platform,
            run_id=run_id,
            content_id=content_id,
            semantic_run_id=semantic_run_id,
            eligible_comments=semantic_run.eligible_comments if semantic_run else None,
            analysis_snapshot=analysis_snapshot,
            top_k=top_k,
        )
    )
    return PublicOpinionResponse(**summary.model_dump(mode="python"))


@router.get("/trends", response_model=TrendResponse)
async def get_trends(
    service: TrendServiceDependency,
    semantic_repository: SemanticRepositoryDependency,
    platform: Optional[Platform] = None,
    run_id: Optional[UUID] = None,
    content_id: Optional[UUID] = None,
    semantic_run_id: Optional[UUID] = None,
    bucket: Annotated[str, Query(pattern="^(hour|day)$")] = "day",
    start_at: Optional[datetime] = None,
    end_at: Optional[datetime] = None,
    min_bucket_comments: Annotated[int, Query(ge=1, le=10000)] = 3,
) -> TrendResponse:
    try:
        semantic_run, analysis_snapshot = await _semantic_snapshot_context(
            semantic_repository, semantic_run_id
        )
        summary = await service.get_trend(TrendQuery(
            platform=platform,
            run_id=run_id,
            content_id=content_id,
            semantic_run_id=semantic_run_id,
            eligible_comments=semantic_run.eligible_comments if semantic_run else None,
            analysis_snapshot=analysis_snapshot,
            bucket=bucket,
            start_at=start_at,
            end_at=end_at,
            min_bucket_comments=min_bucket_comments,
        ))
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return TrendResponse(**summary.model_dump(mode="python"))


@router.post("/semantic-search", response_model=SemanticSearchResponse)
async def semantic_search(
    request: SemanticSearchRequest,
    service: SemanticSearchServiceDependency,
) -> SemanticSearchResponse:
    try:
        result = await service.search(request)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except EmbeddingProviderError as exc:
        raise HTTPException(status_code=502, detail="Embedding provider request failed") from exc
    return SemanticSearchResponse(**result.model_dump(mode="python"))


@router.post("/agent/query", response_model=AgentChatResponse | AgentChatFailureResponse)
async def query_agent(
    request: AgentChatRequest,
    runtime: AgentRuntimeDependency,
) -> AgentChatResponse:
    semantic_run, analysis_snapshot = await _semantic_snapshot_context(
        runtime.semantic_repository, request.semantic_run_id
    )
    if semantic_run is not None:
        request = request.model_copy(update={
            "eligible_comments": semantic_run.eligible_comments,
            "analysis_snapshot": analysis_snapshot,
        })
    search_service = SemanticSearchService(
        runtime.semantic_repository, runtime.embedding_provider
    )
    tools = AgentToolRegistry(
        runtime.analysis_repository,
        runtime.semantic_repository,
        search_service,
        request,
    )
    service = PublicOpinionAgentService(
        runtime.llm_provider,
        tools,
        max_retries=runtime.llm_config.max_retries,
        retry_base_delay=runtime.llm_config.retry_base_delay,
    )
    try:
        response = await asyncio.wait_for(service.run(request), timeout=90)
    except (AgentLimitError, StructuredOutputError, TransientLLMError, LLMProviderError, asyncio.TimeoutError) as exc:
        failure = AgentChatFailureResponse(
            error_type="timeout" if isinstance(exc, asyncio.TimeoutError) else type(exc).__name__,
            message="Agent request timed out or could not produce a complete response." if isinstance(exc, asyncio.TimeoutError) else str(exc)[:300],
        )
        return JSONResponse(status_code=502, content=failure.model_dump(mode="json"))
    except Exception as exc:
        failure = AgentChatFailureResponse(error_type=type(exc).__name__, message=str(exc)[:300])
        return JSONResponse(status_code=500, content=failure.model_dump(mode="json"))
    return AgentChatResponse(**response.model_dump(mode="python"))


@router.post("/jobs", response_model=AnalysisJobResponse, status_code=202)
async def create_analysis_job(
    request: AnalysisJobCreateRequest,
    job_service: JobServiceDependency,
    manager: JobManagerDependency,
) -> AnalysisJobResponse:
    environment_config = BatchAnalysisConfig.from_environment()
    config = environment_config.model_copy(update={
        "batch_size": request.batch_size,
        "max_concurrency": request.max_concurrency,
        "max_requests_per_minute": (
            request.max_requests_per_minute
            if request.max_requests_per_minute is not None
            else environment_config.max_requests_per_minute
        ),
    })
    try:
        job = await job_service.create_job(
            platform=request.platform,
            run_id=request.run_id,
            content_id=request.content_id,
            limit=request.limit,
            config=config,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    manager.start(job.id)
    return AnalysisJobResponse.from_job(job)


@router.get("/jobs/{job_id}", response_model=AnalysisJobResponse)
async def get_analysis_job(job_id: UUID, repository: RepositoryDependency) -> AnalysisJobResponse:
    job = await repository.get_analysis_job(str(job_id))
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return AnalysisJobResponse.from_job(job)


@router.post("/jobs/{job_id}/cancel", response_model=AnalysisJobResponse)
async def cancel_analysis_job(job_id: UUID, repository: RepositoryDependency) -> AnalysisJobResponse:
    job = await repository.cancel_analysis_job(str(job_id), utc_now())
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    return AnalysisJobResponse.from_job(job)


@router.post("/jobs/{job_id}/retry-failed", response_model=AnalysisJobResponse, status_code=202)
async def retry_failed_analysis_job(
    job_id: UUID,
    job_service: JobServiceDependency,
    manager: JobManagerDependency,
) -> AnalysisJobResponse:
    try:
        job = await job_service.retry_failed_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Analysis job not found") from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    manager.start(job.id)
    return AnalysisJobResponse.from_job(job)


@router.post("/jobs/{job_id}/resume", response_model=AnalysisJobResponse, status_code=202)
async def resume_analysis_job(
    job_id: UUID,
    repository: RepositoryDependency,
    _job_service: JobServiceDependency,
    manager: JobManagerDependency,
) -> AnalysisJobResponse:
    job = await repository.get_analysis_job(str(job_id))
    if job is None:
        raise HTTPException(status_code=404, detail="Analysis job not found")
    if job.status in {AnalysisJobStatus.COMPLETED, AnalysisJobStatus.CANCELLED}:
        raise HTTPException(status_code=409, detail=f"Cannot resume a {job.status.value} job")
    if not manager.start(job.id):
        raise HTTPException(status_code=409, detail="Analysis job is already running in this process")
    return AnalysisJobResponse.from_job(job)


@router.post(
    "/workflows",
    response_model=CollectionAnalysisWorkflowResponse,
    status_code=202,
)
async def create_collection_analysis_workflow(
    request: CollectionAnalysisWorkflowRequest,
) -> CollectionAnalysisWorkflowResponse:
    """Start the selected AI pipeline for one completed crawler output."""

    return analysis_workflow_manager.start(request)


@router.get("/result-scopes", response_model=AnalysisResultScopeListResponse)
async def list_analysis_result_scopes(
    repository: RepositoryDependency,
    semantic_repository: SemanticRepositoryDependency,
    platform: Optional[Platform] = None,
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
) -> AnalysisResultScopeListResponse:
    """List durable collection scopes that can be opened without rerunning AI."""

    runs = await repository.list_collection_runs(
        platform=platform.value if platform else None,
        limit=limit,
    )
    items = []
    for run in runs:
        analyzed_count = await repository.count_analyzed_comments_for_run(str(run.id))
        job = await repository.latest_analysis_job_for_run(str(run.id))
        failed_items = (
            await repository.list_job_items(
                str(job.id), statuses=[AnalysisJobItemStatus.FAILED], limit=1
            )
            if job and job.failed_count
            else []
        )
        semantic_run = await semantic_repository.latest_semantic_run_for_collection(str(run.id))
        query = [f"platform={run.platform.value}", f"run_id={run.id}"]
        if semantic_run:
            query.append(f"semantic_run_id={semantic_run.id}")
        items.append(AnalysisResultScopeResponse(
            run_id=run.id,
            platform=run.platform,
            crawler_type=run.crawler_type,
            content_count=run.content_count,
            comment_count=run.comment_count,
            analyzed_count=analyzed_count,
            analysis_coverage=round(analyzed_count / run.comment_count, 4) if run.comment_count else 0.0,
            source_files=run.output_location,
            started_at=run.started_at,
            finished_at=run.finished_at,
            analysis_job_id=job.id if job else None,
            analysis_job_status=job.status.value if job else None,
            analysis_failed_count=job.failed_count if job else 0,
            analysis_error_message=failed_items[0].error_message if failed_items else (job.error_message if job else None),
            semantic_run_id=semantic_run.id if semantic_run else None,
            semantic_status=semantic_run.status.value if semantic_run else None,
            topic_count=semantic_run.topic_count if semantic_run else 0,
            dashboard_url="/dashboard?" + "&".join(query),
        ))
    return AnalysisResultScopeListResponse(items=items, total=len(items))


@router.get(
    "/workflows/{workflow_id}",
    response_model=CollectionAnalysisWorkflowResponse,
)
async def get_collection_analysis_workflow(
    workflow_id: UUID,
) -> CollectionAnalysisWorkflowResponse:
    workflow = analysis_workflow_manager.get(workflow_id)
    if workflow is None:
        raise HTTPException(status_code=404, detail="Analysis workflow not found")
    return workflow
