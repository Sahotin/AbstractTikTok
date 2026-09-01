"""Standalone CLI for normalized data ingestion.

Example:
    python -m analysis.cli ingest --platform douyin --input data/douyin/json
"""

from __future__ import annotations

import asyncio
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import typer

from analysis.repositories import AnalysisRepository
from analysis.repositories import SemanticRepository
from analysis.embeddings import EmbeddingConfig, FakeEmbeddingProvider, create_embedding_provider
from analysis.domain import SemanticSearchQuery, TopicClusterConfig
from analysis.experiments.topic_quality import TopicQualityExperimentService, write_topic_quality_report
from analysis.evaluation import (
    CommentQualityEvaluator,
    CommentQualityArtifact,
    EvaluationSampleItem,
    add_cost_range,
    deterministic_stratified_sample,
    write_annotation_template,
    write_comment_quality_report,
)
from analysis.agent import (
    AgentRequest,
    AgentToolRegistry,
    OpenAICompatibleToolCallingProvider,
    PublicOpinionAgentService,
)
from analysis.llm import FakeLLMProvider, LLMConfig, create_provider
from analysis.services import (
    AnalysisJobRunner,
    AnalysisJobService,
    AnalysisQueryService,
    BatchAnalysisConfig,
    CommentAnalysisService,
    IngestionService,
    EmbeddingService,
    SemanticRunService,
    SemanticSearchService,
    TopicService,
)


app = typer.Typer(add_completion=False, help="Analysis data foundation commands")


@app.callback()
def main() -> None:
    """Run standalone analysis data commands."""


def _print_progress(message: str) -> None:
    typer.echo(f"[ingest] {message}")


@app.command()
def ingest(
    inputs: List[Path] = typer.Option(..., "--input", "-i", help="JSON/JSONL file or directory; repeat for multiple inputs"),
    platform: str = typer.Option("douyin", help="Supported values: douyin, bilibili"),
    database: Path = typer.Option(Path("database/analysis.db"), help="Target SQLite database"),
    batch_size: int = typer.Option(500, min=1, max=5000, help="Rows written per transaction batch"),
    crawler_type: str = typer.Option("import", help="Source crawler mode for run metadata"),
    query: Optional[str] = typer.Option(None, help="Optional source search query"),
    specified_ids: Optional[str] = typer.Option(None, help="Optional comma-separated source content IDs"),
) -> None:
    """Import MediaCrawler Douyin JSON/JSONL into normalized SQLite tables."""

    async def _run() -> None:
        repository = AnalysisRepository(database)
        service = IngestionService(repository, batch_size=batch_size, progress=_print_progress)
        try:
            report = await service.ingest(
                inputs=inputs,
                platform=platform,
                crawler_type=crawler_type,
                query=query,
                specified_ids=[item.strip() for item in specified_ids.split(",") if item.strip()] if specified_ids else [],
            )
        finally:
            await repository.close()

        typer.echo("\n=== Normalized ingestion report ===")
        typer.echo(report.model_dump_json(indent=2))

    try:
        asyncio.run(_run())
    except (FileNotFoundError, ValueError) as exc:
        typer.secho(f"Import failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command("analyze-comments")
def analyze_comments(
    limit: int = typer.Option(..., min=1, max=100000, help="Explicit maximum comments to analyze"),
    database: Path = typer.Option(Path("database/analysis.db"), help="Analysis SQLite database"),
    platform: str = typer.Option("douyin", help="Normalized platform value"),
    content_id: Optional[UUID] = typer.Option(None, help="Optional internal content UUID"),
    run_id: Optional[UUID] = typer.Option(None, help="Optional collection run UUID"),
    batch_size: int = typer.Option(20, min=1, max=500, help="Comments loaded per processing batch"),
    max_concurrency: int = typer.Option(5, min=1, max=20, help="Maximum concurrent LLM requests"),
    max_requests_per_minute: Optional[int] = typer.Option(None, min=1, help="Optional provider request rate limit"),
    fake_provider: bool = typer.Option(False, help="Use deterministic Fake Provider without network or API key"),
) -> None:
    """Create and synchronously run one persistent batch analysis job."""

    async def _run() -> None:
        repository = AnalysisRepository(database)
        try:
            await repository.initialize()
            llm_config = LLMConfig.from_environment()
            if fake_provider:
                provider = FakeLLMProvider()
                llm_config = llm_config.model_copy(update={
                    "provider": provider.provider_name,
                    "model": provider.model,
                })
            else:
                provider = create_provider(llm_config)
            analysis_service = CommentAnalysisService(repository, provider, llm_config)
            job_service = AnalysisJobService(
                repository,
                AnalysisQueryService(repository),
                analysis_service,
            )
            environment_config = BatchAnalysisConfig.from_environment()
            batch_config = environment_config.model_copy(update={
                "batch_size": batch_size,
                "max_concurrency": max_concurrency,
                "max_requests_per_minute": (
                    max_requests_per_minute
                    if max_requests_per_minute is not None
                    else environment_config.max_requests_per_minute
                ),
            })
            job = await job_service.create_job(
                platform=platform,
                limit=limit,
                content_id=content_id,
                run_id=run_id,
                config=batch_config,
            )
            typer.echo(f"[analysis-job] created job_id={job.id} total={job.total_comments}")
            result = await AnalysisJobRunner(repository, analysis_service).run(job.id)
            typer.echo("\n=== Phase 2B analysis job ===")
            typer.echo(result.model_dump_json(indent=2))
        finally:
            await repository.close()

    try:
        asyncio.run(_run())
    except (FileNotFoundError, KeyError, ValueError) as exc:
        typer.secho(f"Analysis job failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command("comment-quality")
def comment_quality(
    database: Path = typer.Option(..., help="Explicit non-production SQLite validation database"),
    limit: int = typer.Option(100, min=1, max=100, help="Maximum deterministic evaluation sample"),
    platform: str = typer.Option("douyin", help="Normalized platform value"),
    seed: str = typer.Option("phase4a-v1", help="Stable deterministic sampling seed"),
    output: Path = typer.Option(Path("benchmarks/comment_quality"), help="Local evaluation artifact directory"),
    annotations: Optional[Path] = typer.Option(None, help="Existing labeled JSONL template to audit"),
    execute: bool = typer.Option(False, "--execute", help="Run the sampled comments through the configured LLM"),
    fake_provider: bool = typer.Option(False, help="Use deterministic Fake Provider without network or API key"),
    batch_size: int = typer.Option(20, min=1, max=100),
    max_concurrency: int = typer.Option(2, min=1, max=10),
    max_requests_per_minute: Optional[int] = typer.Option(None, min=1),
    reuse_telemetry_from: Optional[Path] = typer.Option(None, help="Reuse provider telemetry from a prior quality JSON artifact"),
    cache_hit_input_price_per_1m: Optional[float] = typer.Option(None, min=0),
    cache_miss_input_price_per_1m: Optional[float] = typer.Option(None, min=0),
    output_price_per_1m: Optional[float] = typer.Option(None, min=0),
    pricing_label: str = typer.Option("manual pricing", help="Human-readable source/time basis for supplied prices"),
) -> None:
    """Prepare or execute a reproducible comment-analysis quality evaluation."""

    async def _run() -> tuple[Path, Path, dict]:
        resolved_database = database.expanduser().resolve()
        production_database = Path("database/analysis.db").resolve()
        if resolved_database == production_database:
            raise ValueError("comment-quality refuses database/analysis.db; use a validation copy")
        if not resolved_database.is_file():
            raise FileNotFoundError(f"Validation database does not exist: {resolved_database}")
        repository = AnalysisRepository(resolved_database)
        started = time.perf_counter()
        try:
            query_service = AnalysisQueryService(repository)
            if annotations is not None:
                annotation_path = annotations.expanduser().resolve()
                if not annotation_path.is_file():
                    raise FileNotFoundError(f"Annotation JSONL does not exist: {annotation_path}")
                sample = [
                    EvaluationSampleItem.model_validate_json(line)
                    for line in annotation_path.read_text(encoding="utf-8").splitlines()
                    if line.strip()
                ][:limit]
            else:
                comments = []
                offset = 0
                while True:
                    page = await query_service.list_comments(
                        platform=platform,
                        limit=1000,
                        offset=offset,
                    )
                    comments.extend(page.items)
                    offset += len(page.items)
                    if not page.items or offset >= page.total:
                        break
                sample = deterministic_stratified_sample(comments, limit=min(limit, len(comments)), seed=seed)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                annotation_path = write_annotation_template(sample, output, stamp)
            if not sample:
                raise ValueError("No comments matched the evaluation filters")

            llm_config = LLMConfig.from_environment()
            if fake_provider:
                provider = FakeLLMProvider()
                llm_config = llm_config.model_copy(update={
                    "provider": provider.provider_name,
                    "model": provider.model,
                })
            else:
                provider = create_provider(llm_config)
            analysis_service = CommentAnalysisService(repository, provider, llm_config)
            job_dump = None
            reported_elapsed_seconds = None
            if execute:
                environment_config = BatchAnalysisConfig.from_environment()
                batch_config = environment_config.model_copy(update={
                    "batch_size": batch_size,
                    "max_concurrency": max_concurrency,
                    "max_requests_per_minute": (
                        max_requests_per_minute
                        if max_requests_per_minute is not None
                        else environment_config.max_requests_per_minute
                    ),
                })
                job_service = AnalysisJobService(repository, query_service, analysis_service)
                job = await job_service.create_job(
                    platform=platform,
                    limit=len(sample),
                    config=batch_config,
                    comment_ids=[item.comment_id for item in sample],
                )
                typer.echo(f"[comment-quality] job_id={job.id} total={job.total_comments}")
                job = await AnalysisJobRunner(repository, analysis_service).run(job.id)
                job_dump = job.model_dump(mode="json")

            comments_by_id = {
                str(comment.id): comment
                for comment in await repository.get_comments_by_ids([item.comment_id for item in sample])
            }
            analyses = {}
            for item in sample:
                comment = comments_by_id.get(item.comment_id)
                if comment is not None:
                    cached = await analysis_service.get_cached_analysis(comment)
                    if cached is not None:
                        analyses[item.comment_id] = cached
            if reuse_telemetry_from is not None:
                telemetry_path = reuse_telemetry_from.expanduser().resolve()
                if not telemetry_path.is_file():
                    raise FileNotFoundError(f"Telemetry artifact does not exist: {telemetry_path}")
                prior = CommentQualityArtifact.model_validate_json(telemetry_path.read_text(encoding="utf-8"))
                usage = prior.evaluation.provider_usage
                if job_dump is None:
                    job_dump = prior.job
                reported_elapsed_seconds = prior.elapsed_seconds
            else:
                usage = provider.usage_snapshot() if hasattr(provider, "usage_snapshot") else {
                    "request_count": getattr(provider, "call_count", 0)
                }
            prices = (
                cache_hit_input_price_per_1m,
                cache_miss_input_price_per_1m,
                output_price_per_1m,
            )
            if any(price is not None for price in prices):
                if any(price is None for price in prices):
                    raise ValueError("All three token price options are required for cost reporting")
                usage = add_cost_range(
                    usage,
                    cache_hit_input_price_per_1m=cache_hit_input_price_per_1m,
                    cache_miss_input_price_per_1m=cache_miss_input_price_per_1m,
                    output_price_per_1m=output_price_per_1m,
                    pricing_label=pricing_label,
                )
            evaluation = CommentQualityEvaluator().evaluate(sample, analyses, provider_usage=usage)
            report_stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
            report = write_comment_quality_report(
                evaluation,
                output_dir=output,
                stamp=report_stamp,
                database=resolved_database,
                seed=seed,
                job=job_dump,
                annotation_path=annotation_path,
                elapsed_seconds=(
                    reported_elapsed_seconds
                    if reported_elapsed_seconds is not None
                    else time.perf_counter() - started
                ),
            )
            return report, annotation_path, evaluation.model_dump(mode="json")
        finally:
            await repository.close()

    try:
        report, annotation_path, result = asyncio.run(_run())
        typer.echo(f"[comment-quality] annotations={annotation_path}")
        typer.echo(f"[comment-quality] report={report}")
        typer.echo(json.dumps(result, ensure_ascii=False, indent=2))
    except (FileNotFoundError, KeyError, ValueError) as exc:
        typer.secho(f"Comment quality evaluation failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command()
def embed(
    limit: int = typer.Option(..., min=1, max=100000, help="Explicit maximum comments to embed"),
    database: Path = typer.Option(Path("database/analysis.db"), help="Migrated analysis SQLite database"),
    platform: str = typer.Option("douyin", help="Normalized platform value"),
    content_id: Optional[UUID] = typer.Option(None, help="Optional internal content UUID"),
    run_id: Optional[UUID] = typer.Option(None, help="Optional collection run UUID"),
    provider: Optional[str] = typer.Option(None, help="Provider override; use fake for local validation"),
    batch_size: Optional[int] = typer.Option(None, min=1, max=500, help="Embedding request batch size"),
    max_concurrency: Optional[int] = typer.Option(None, min=1, max=20, help="Maximum concurrent requests"),
) -> None:
    """Create and synchronously execute one embedding-only semantic run."""

    async def _run() -> None:
        analysis_repository = AnalysisRepository(database)
        semantic_repository = SemanticRepository(database)
        try:
            config = EmbeddingConfig.from_environment()
            updates = {}
            if provider is not None:
                updates["provider"] = provider
            if batch_size is not None:
                updates["batch_size"] = batch_size
            if max_concurrency is not None:
                updates["max_concurrency"] = max_concurrency
            config = config.model_copy(update=updates)
            embedding_provider = (
                FakeEmbeddingProvider(
                    dimension=config.fake_dimension,
                    model=config.model or "fake-embedding-v1",
                )
                if config.provider.lower() == "fake"
                else create_embedding_provider(config)
            )
            service = SemanticRunService(
                analysis_repository,
                semantic_repository,
                EmbeddingService(semantic_repository, embedding_provider, config),
                config,
            )
            semantic_run = await service.create_run(
                platform=platform,
                run_id=run_id,
                content_id=content_id,
                limit=limit,
            )
            typer.echo(f"[semantic-run] created run_id={semantic_run.id} total={semantic_run.total_comments}")
            result, report = await service.execute_run(semantic_run.id)
            typer.echo("\n=== Phase 3B embedding run ===")
            typer.echo(result.model_dump_json(indent=2))
            typer.echo("\n=== Embedding report ===")
            typer.echo(report.model_dump_json(indent=2))
        finally:
            await semantic_repository.close()
            await analysis_repository.close()

    try:
        asyncio.run(_run())
    except (FileNotFoundError, KeyError, ValueError) as exc:
        typer.secho(f"Embedding run failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command("cluster-topics")
def cluster_topics(
    semantic_run_id: UUID = typer.Option(..., help="Completed semantic run UUID"),
    database: Path = typer.Option(Path("database/analysis.db"), help="Migrated analysis SQLite database"),
    distance_threshold: float = typer.Option(0.35, min=0.000001, max=2.0),
    min_cluster_size: int = typer.Option(3, min=1, max=5000),
    representative_top_k: int = typer.Option(3, min=1, max=10),
) -> None:
    """Cluster one fixed SemanticRun snapshot without generating new embeddings."""

    async def _run() -> None:
        repository = SemanticRepository(database)
        try:
            config = TopicClusterConfig(
                distance_threshold=distance_threshold,
                min_cluster_size=min_cluster_size,
                representative_top_k=representative_top_k,
            )
            run, result = await TopicService(repository).cluster_run(semantic_run_id, config)
            typer.echo("\n=== Phase 3B-2 topic clustering ===")
            typer.echo(run.model_dump_json(indent=2))
            typer.echo(json.dumps({
                "cluster_count": len(result.clusters),
                "noise_count": len(result.noise_comment_ids),
                "clustered_comments": result.clustered_comments,
            }, ensure_ascii=False, indent=2))
        finally:
            await repository.close()

    try:
        asyncio.run(_run())
    except (FileNotFoundError, KeyError, ValueError) as exc:
        typer.secho(f"Topic clustering failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command("semantic-search")
def semantic_search(
    semantic_run_id: UUID = typer.Option(..., help="Completed semantic run UUID"),
    query: str = typer.Option(..., help="Natural-language comment search query"),
    database: Path = typer.Option(..., help="Migrated analysis SQLite database"),
    top_k: int = typer.Option(10, min=1, max=50),
    min_similarity: float = typer.Option(0.0, min=-1.0, max=1.0),
) -> None:
    """Embed one query and search an immutable semantic-run snapshot."""

    async def _run() -> None:
        repository = SemanticRepository(database)
        try:
            config = EmbeddingConfig.from_environment()
            service = SemanticSearchService(
                repository, create_embedding_provider(config)
            )
            result = await service.search(SemanticSearchQuery(
                semantic_run_id=semantic_run_id,
                query=query,
                top_k=top_k,
                min_similarity=min_similarity,
            ))
            typer.echo(result.model_dump_json(indent=2))
        finally:
            await repository.close()

    try:
        asyncio.run(_run())
    except (FileNotFoundError, KeyError, ValueError) as exc:
        typer.secho(f"Semantic search failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


@app.command("ask-agent")
def ask_agent(
    question: str = typer.Option(..., help="Natural-language public-opinion question"),
    database: Path = typer.Option(..., help="Migrated analysis SQLite database"),
    platform: str = typer.Option("douyin"),
    run_id: Optional[UUID] = typer.Option(None),
    content_id: Optional[UUID] = typer.Option(None),
    semantic_run_id: Optional[UUID] = typer.Option(None),
    max_steps: int = typer.Option(5, min=1, max=8),
) -> None:
    """Run the bounded read-only Tool Calling Agent."""

    async def _run() -> None:
        analysis_repository = AnalysisRepository(database)
        semantic_repository = SemanticRepository(database)
        try:
            llm_config = LLMConfig.from_environment()
            embedding_config = EmbeddingConfig.from_environment()
            request = AgentRequest(
                question=question,
                platform=platform,
                run_id=run_id,
                content_id=content_id,
                semantic_run_id=semantic_run_id,
                max_steps=max_steps,
            )
            search_service = SemanticSearchService(
                semantic_repository, create_embedding_provider(embedding_config)
            )
            tools = AgentToolRegistry(
                analysis_repository, semantic_repository, search_service, request
            )
            result = await PublicOpinionAgentService(
                OpenAICompatibleToolCallingProvider(llm_config),
                tools,
                max_retries=llm_config.max_retries,
                retry_base_delay=llm_config.retry_base_delay,
            ).run(request)
            typer.echo(result.model_dump_json(indent=2))
        finally:
            await semantic_repository.close()
            await analysis_repository.close()

    try:
        asyncio.run(_run())
    except (FileNotFoundError, KeyError, ValueError) as exc:
        typer.secho(f"Agent failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


def _csv_floats(value: str, *, option_name: str) -> list[float]:
    try:
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{option_name} must be a comma-separated list of numbers") from exc
    if not values:
        raise ValueError(f"{option_name} must contain at least one value")
    return values


def _csv_ints(value: str, *, option_name: str) -> list[int]:
    try:
        values = [int(item.strip()) for item in value.split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError(f"{option_name} must be a comma-separated list of integers") from exc
    if not values:
        raise ValueError(f"{option_name} must contain at least one value")
    return values


@app.command("topic-quality")
def topic_quality(
    semantic_run_id: UUID = typer.Option(..., help="Completed source semantic run UUID in the validation database"),
    database: Path = typer.Option(..., help="Explicit non-production SQLite validation database"),
    thresholds: str = typer.Option("0.15,0.20,0.25,0.30", help="Comma-separated cosine distance thresholds"),
    min_cluster_sizes: str = typer.Option("3,5,10", help="Comma-separated minimum cluster sizes"),
    sample_size: int = typer.Option(50, min=1, max=100, help="Deterministic human-review sample size"),
    output: Path = typer.Option(Path("benchmarks/topic_quality"), help="Local Markdown report directory"),
) -> None:
    """Run isolated topic parameter experiments in an explicit validation database."""

    async def _run() -> Path:
        resolved_database = database.expanduser().resolve()
        production_database = Path("database/analysis.db").resolve()
        if resolved_database == production_database:
            raise ValueError("topic-quality refuses to write to database/analysis.db; use a validation copy")
        if not resolved_database.is_file():
            raise FileNotFoundError(f"Validation database does not exist: {resolved_database}")
        config = EmbeddingConfig.from_environment()
        embedding_provider = (
            FakeEmbeddingProvider(
                dimension=config.fake_dimension,
                model=config.model or "fake-embedding-v1",
            )
            if config.provider.lower() == "fake"
            else create_embedding_provider(config)
        )
        analysis_repository = AnalysisRepository(resolved_database)
        semantic_repository = SemanticRepository(resolved_database)
        try:
            experiments = await TopicQualityExperimentService(
                analysis_repository,
                semantic_repository,
                EmbeddingService(semantic_repository, embedding_provider, config),
                config,
            ).run_matrix(
                semantic_run_id,
                thresholds=_csv_floats(thresholds, option_name="--thresholds"),
                min_cluster_sizes=_csv_ints(min_cluster_sizes, option_name="--min-cluster-sizes"),
                evaluation_sample_size=sample_size,
            )
            return write_topic_quality_report(experiments, output)
        finally:
            await semantic_repository.close()
            await analysis_repository.close()

    try:
        report = asyncio.run(_run())
        typer.echo(f"[topic-quality] report={report}")
    except (FileNotFoundError, KeyError, ValueError) as exc:
        typer.secho(f"Topic quality experiment failed: {exc}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    app()
