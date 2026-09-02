"""Reproducible, local-only quality evaluation for topic-clustering experiments.

This module deliberately does not infer semantic correctness.  It records
quantitative properties and produces a local review document containing a
deterministically selected set of comments for human evaluation.
"""

from __future__ import annotations

import hashlib
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence
from uuid import UUID

import numpy as np
from pydantic import BaseModel, Field

from analysis.domain import SemanticAnalysisRun, TopicCluster, TopicClusterConfig, TopicClusteringResult
from analysis.preprocessing import utc_now
from analysis.repositories import AnalysisRepository, SemanticRepository
from analysis.repositories.semantic_sqlalchemy import TopicSnapshotRow
from analysis.services import EmbeddingService, SemanticRunService, TopicService


def canonical_config_hash(value: object) -> str:
    """Return a stable SHA-256 for persisted/reportable experiment configuration."""

    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def deterministic_comment_sample(
    comment_ids: Iterable[UUID],
    input_snapshot_hash: str,
    limit: int,
) -> list[UUID]:
    """Select comments without mutable random state or input-order dependence."""

    if limit < 0:
        raise ValueError("Sample limit must not be negative")
    ranked = sorted(
        set(comment_ids),
        key=lambda item: (
            hashlib.sha256(f"{item}:{input_snapshot_hash}".encode("utf-8")).hexdigest(),
            str(item),
        ),
    )
    return ranked[:limit]


class TopicQualityMetrics(BaseModel):
    total_comments: int = Field(ge=0)
    cluster_count: int = Field(ge=0)
    noise_count: int = Field(ge=0)
    noise_ratio: float = Field(ge=0.0, le=1.0)
    excluded_count: int = Field(default=0, ge=0)
    excluded_ratio: float = Field(default=0.0, ge=0.0, le=1.0)
    largest_cluster_size: int | None = Field(default=None, ge=1)
    median_cluster_size: float | None = Field(default=None, ge=1)
    smallest_cluster_size: int | None = Field(default=None, ge=1)
    silhouette_score: float | None = None
    silhouette_sample_count: int = Field(default=0, ge=0)
    silhouette_sampling: str = "sha256(comment_id:input_snapshot_hash)"


def evaluate_topic_quality(
    *,
    comment_ids: Sequence[UUID],
    vectors: Sequence[Sequence[float]],
    result: TopicClusteringResult,
    input_snapshot_hash: str,
    silhouette_max_samples: int = 2_000,
) -> TopicQualityMetrics:
    """Calculate deterministic, non-semantic quality metrics.

    Noise is intentionally excluded from size statistics and silhouette.  A
    sampled silhouette is only used above ``silhouette_max_samples`` and its
    sample is selected by the immutable comment IDs and snapshot hash.
    """

    if len(comment_ids) != len(vectors):
        raise ValueError("Comment and vector counts do not match")
    if silhouette_max_samples < 1:
        raise ValueError("silhouette_max_samples must be positive")
    if len(set(comment_ids)) != len(comment_ids):
        raise ValueError("Comment IDs must be unique")

    total = len(comment_ids)
    known_ids = set(comment_ids)
    memberships: dict[UUID, int] = {}
    sizes: list[int] = []
    for cluster_index, cluster in enumerate(result.clusters):
        members = cluster.member_comment_ids
        if not members or any(item not in known_ids for item in members):
            raise ValueError("Topic cluster members do not match input comments")
        if any(item in memberships for item in members):
            raise ValueError("A comment belongs to more than one topic cluster")
        memberships.update({item: cluster_index for item in members})
        sizes.append(len(members))
    noise = set(result.noise_comment_ids)
    if any(item not in known_ids for item in noise) or noise.intersection(memberships):
        raise ValueError("Noise membership does not match topic clustering result")
    excluded = set(result.excluded_comments)
    if any(item not in known_ids for item in excluded):
        raise ValueError("Excluded membership does not match topic clustering result")
    if excluded.intersection(memberships) or excluded.intersection(noise):
        raise ValueError("Excluded comments overlap topic clusters or noise")
    if len(memberships) + len(noise) + len(excluded) != total:
        raise ValueError("Topic clusters, noise, and exclusions do not account for every input comment")

    silhouette, sample_count = _silhouette(
        comment_ids, vectors, memberships, input_snapshot_hash, silhouette_max_samples
    )
    return TopicQualityMetrics(
        total_comments=total,
        cluster_count=len(sizes),
        noise_count=len(noise),
        noise_ratio=(len(noise) / total) if total else 0.0,
        excluded_count=len(excluded),
        excluded_ratio=(len(excluded) / total) if total else 0.0,
        largest_cluster_size=max(sizes) if sizes else None,
        median_cluster_size=float(statistics.median(sizes)) if sizes else None,
        smallest_cluster_size=min(sizes) if sizes else None,
        silhouette_score=silhouette,
        silhouette_sample_count=sample_count,
    )


def _silhouette(
    comment_ids: Sequence[UUID],
    vectors: Sequence[Sequence[float]],
    memberships: dict[UUID, int],
    input_snapshot_hash: str,
    max_samples: int,
) -> tuple[float | None, int]:
    clustered_ids = [item for item in comment_ids if item in memberships]
    labels = {memberships[item] for item in clustered_ids}
    if len(labels) < 2 or len(clustered_ids) <= len(labels):
        return None, 0
    selected_ids = deterministic_comment_sample(clustered_ids, input_snapshot_hash, max_samples)
    selected_labels = [memberships[item] for item in selected_ids]
    if len(set(selected_labels)) < 2 or len(selected_ids) <= len(set(selected_labels)):
        return None, 0
    vector_by_id = {comment_id: vector for comment_id, vector in zip(comment_ids, vectors)}
    matrix = np.asarray([vector_by_id[item] for item in selected_ids], dtype="<f4")
    if matrix.ndim != 2 or not np.isfinite(matrix).all():
        raise ValueError("Silhouette vectors must be finite two-dimensional data")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Silhouette vectors must have non-zero norm")
    from sklearn.metrics import silhouette_score

    return float(silhouette_score(matrix / norms, selected_labels, metric="cosine")), len(selected_ids)


@dataclass(frozen=True)
class TopicQualityExperiment:
    semantic_run: SemanticAnalysisRun
    config: TopicClusterConfig
    config_hash: str
    metrics: TopicQualityMetrics
    runtime_seconds: float
    topics: list[TopicCluster]
    representatives: dict[UUID, list[str]]
    medoid_text: dict[UUID, str]
    evaluation_rows: list[tuple[UUID, str, str, str | None]]
    noise_samples: list[tuple[UUID, str]]
    excluded_samples: list[tuple[UUID, str, str]]


class TopicQualityExperimentService:
    """Create isolated runs and evaluate each clustering configuration locally."""

    def __init__(
        self,
        analysis_repository: AnalysisRepository,
        semantic_repository: SemanticRepository,
        embedding_service: EmbeddingService,
        embedding_config,
    ):
        self.analysis_repository = analysis_repository
        self.semantic_repository = semantic_repository
        self.embedding_service = embedding_service
        self.embedding_config = embedding_config
        self.semantic_service = SemanticRunService(
            analysis_repository, semantic_repository, embedding_service, embedding_config
        )
        self.topic_service = TopicService(semantic_repository)

    async def run_matrix(
        self,
        source_run_id: UUID | str,
        *,
        thresholds: Sequence[float],
        min_cluster_sizes: Sequence[int],
        evaluation_sample_size: int = 50,
        silhouette_max_samples: int = 2_000,
    ) -> list[TopicQualityExperiment]:
        source = await self.semantic_service.get_run(source_run_id)
        if source.status.value != "completed":
            raise ValueError("Topic quality experiments require a completed source semantic run")
        self.semantic_service._validate_runtime(source)
        if not thresholds or not min_cluster_sizes:
            raise ValueError("At least one threshold and minimum cluster size are required")

        experiments: list[TopicQualityExperiment] = []
        for threshold in sorted(set(thresholds)):
            for min_cluster_size in sorted(set(min_cluster_sizes)):
                experiment_run = await self.semantic_service.create_run(
                    platform=source.platform,
                    limit=source.requested_limit,
                    run_id=source.run_id,
                    content_id=source.content_id,
                )
                if experiment_run.input_snapshot_hash != source.input_snapshot_hash:
                    raise ValueError("Source run input snapshot changed before experiment creation")
                started = time.perf_counter()
                experiment_run, _ = await self.semantic_service.execute_run(experiment_run.id)
                if experiment_run.embedded_comments != experiment_run.total_comments:
                    raise ValueError(
                        "Topic quality experiments require embeddings for every snapshot comment"
                    )
                config = TopicClusterConfig(
                    distance_threshold=threshold,
                    min_cluster_size=min_cluster_size,
                )
                experiment_run, result = await self.topic_service.cluster_run(experiment_run.id, config)
                runtime_seconds = time.perf_counter() - started
                snapshot = await self.semantic_repository.get_topic_snapshot(str(experiment_run.id))
                embedded = [row for row in snapshot if row.embedding is not None]
                metrics = evaluate_topic_quality(
                    comment_ids=[row.comment.id for row in embedded],
                    vectors=[row.embedding.vector for row in embedded],
                    result=result,
                    input_snapshot_hash=experiment_run.input_snapshot_hash,
                    silhouette_max_samples=silhouette_max_samples,
                )
                experiments.append(
                    await self._artifact(experiment_run, config, metrics, runtime_seconds, snapshot, evaluation_sample_size)
                )
        return experiments

    async def _artifact(
        self,
        run: SemanticAnalysisRun,
        config: TopicClusterConfig,
        metrics: TopicQualityMetrics,
        runtime_seconds: float,
        snapshot: Sequence[TopicSnapshotRow],
        evaluation_sample_size: int,
    ) -> TopicQualityExperiment:
        page = await self.semantic_repository.list_topic_clusters(
            semantic_run_id=str(run.id), limit=5_000
        )
        topics = page.items
        text_by_id = {row.comment.id: row.comment.text for row in snapshot}
        items = await self.semantic_repository.list_semantic_run_items(str(run.id))
        topic_by_id = {topic.id: topic for topic in topics}
        representatives: dict[UUID, list[str]] = {}
        for topic in topics:
            representatives[topic.id] = [
                representative.text
                for representative in await self.semantic_repository.get_topic_representatives(str(topic.id), limit=3)
            ]
        item_by_comment = {item.comment_id: item for item in items}
        sample_ids = deterministic_comment_sample(text_by_id, run.input_snapshot_hash, evaluation_sample_size)
        rows: list[tuple[UUID, str, str, str | None]] = []
        for comment_id in sample_ids:
            item = item_by_comment[comment_id]
            topic = topic_by_id.get(item.topic_cluster_id) if item.topic_cluster_id else None
            representative = representatives.get(topic.id, [None])[0] if topic else None
            if topic:
                assignment = topic.name
            elif item.status.value == "excluded":
                assignment = f"Excluded ({item.error_message or 'semantic_quality'})"
            else:
                assignment = "Noise"
            rows.append((comment_id, text_by_id[comment_id], assignment, representative))
        noise_ids = [
            item.comment_id
            for item in items
            if item.topic_cluster_id is None and item.status.value == "completed"
        ]
        noise_samples = [
            (comment_id, text_by_id[comment_id])
            for comment_id in deterministic_comment_sample(noise_ids, run.input_snapshot_hash, 10)
        ]
        excluded_items = [item for item in items if item.status.value == "excluded"]
        excluded_by_id = {item.comment_id: item for item in excluded_items}
        excluded_samples = [
            (comment_id, text_by_id[comment_id], excluded_by_id[comment_id].error_message or "semantic_quality")
            for comment_id in deterministic_comment_sample(
                excluded_by_id, run.input_snapshot_hash, 10
            )
        ]
        metadata = {
            "embedding_provider": run.embedding_provider,
            "embedding_model": run.embedding_model,
            "embedding_version": run.embedding_version,
            "embedding_dimension": run.embedding_dimension,
            "algorithm_config": config.model_dump(mode="json"),
            "input_snapshot_hash": run.input_snapshot_hash,
        }
        return TopicQualityExperiment(
            semantic_run=run,
            config=config,
            config_hash=canonical_config_hash(metadata),
            metrics=metrics,
            runtime_seconds=runtime_seconds,
            topics=topics,
            representatives=representatives,
            medoid_text={topic.id: text_by_id[topic.medoid_comment_id] for topic in topics},
            evaluation_rows=rows,
            noise_samples=noise_samples,
            excluded_samples=excluded_samples,
        )


def write_topic_quality_report(
    experiments: Sequence[TopicQualityExperiment],
    output_directory: Path,
) -> Path:
    """Write the sole intentionally text-bearing local artifact for human review."""

    if not experiments:
        raise ValueError("Cannot write a topic quality report with no experiments")
    output_directory.mkdir(parents=True, exist_ok=True)
    timestamp = utc_now().strftime("%Y%m%d_%H%M%S")
    path = output_directory / f"phase3b3_topic_quality_{timestamp}.md"
    first = experiments[0].semantic_run
    lines = [
        "# Phase 3B-3 Topic Clustering Quality Evaluation",
        "",
        "## Dataset",
        "",
        f"- scope: `{first.scope_type.value}`",
        f"- total comments: {first.total_comments}",
        f"- input snapshot hash: `{first.input_snapshot_hash}`",
        f"- embedding provider: `{first.embedding_provider}`",
        f"- embedding model: `{first.embedding_model}`",
        f"- embedding version: `{first.embedding_version}`",
        f"- embedding dimension: {first.embedding_dimension}",
        "",
        "## Experiment Matrix",
        "",
        "| threshold | min cluster size | clusters | noise | noise ratio | excluded | excluded ratio | largest | median | smallest | silhouette | runtime (s) |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for experiment in experiments:
        metric = experiment.metrics
        silhouette = "null" if metric.silhouette_score is None else f"{metric.silhouette_score:.6f}"
        lines.append(
            f"| {experiment.config.distance_threshold:.4f} | {experiment.config.min_cluster_size} | "
            f"{metric.cluster_count} | {metric.noise_count} | {metric.noise_ratio:.6f} | "
            f"{metric.excluded_count} | {metric.excluded_ratio:.6f} | "
            f"{metric.largest_cluster_size or '-'} | {metric.median_cluster_size or '-'} | "
            f"{metric.smallest_cluster_size or '-'} | {silhouette} | {experiment.runtime_seconds:.3f} |"
        )
    lines.extend([
        "",
        "## Interpretation",
        "",
        "These values describe cluster count, size distribution, noise, and cosine silhouette only. "
        "They do not establish best semantic quality; use the review samples below for human judgment.",
    ])
    for number, experiment in enumerate(experiments, start=1):
        lines.extend([
            "",
            f"## Experiment {number}: threshold={experiment.config.distance_threshold}, min_cluster_size={experiment.config.min_cluster_size}",
            "",
            f"- semantic run: `{experiment.semantic_run.id}`",
            f"- experiment config hash: `{experiment.config_hash}`",
            f"- silhouette sample: {experiment.metrics.silhouette_sample_count} ({experiment.metrics.silhouette_sampling})",
            "",
            "### Largest Clusters",
        ])
        for index, topic in enumerate(experiment.topics[:10], start=1):
            lines.extend([
                "",
                f"#### Cluster {index}: {topic.name}",
                "",
                f"- id: `{topic.id}`",
                f"- size: {topic.size} ({topic.percentage:.6f}%)",
                f"- medoid: {experiment.medoid_text[topic.id]}",
                "- representative comments:",
                *[f"  {rank}. {text}" for rank, text in enumerate(experiment.representatives[topic.id], start=1)],
            ])
        lines.extend(["", "### Noise Samples"])
        lines.extend([f"{index}. `{comment_id}` — {text}" for index, (comment_id, text) in enumerate(experiment.noise_samples, start=1)] or ["No noise comments."])
        lines.extend(["", "### Excluded Low-Information Samples"])
        lines.extend([
            f"{index}. `{comment_id}` — {text} (`{reason}`)"
            for index, (comment_id, text, reason) in enumerate(experiment.excluded_samples, start=1)
        ] or ["No comments excluded by the semantic quality gate."])
        lines.extend(["", "### Deterministic Human Evaluation Set", ""])
        lines.append("| comment id | assigned topic | comment | first representative |")
        lines.append("| --- | --- | --- | --- |")
        for comment_id, text, assigned, representative in experiment.evaluation_rows:
            escaped = lambda value: (value or "").replace("|", "\\|").replace("\n", " ")
            lines.append(f"| `{comment_id}` | {escaped(assigned)} | {escaped(text)} | {escaped(representative)} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
