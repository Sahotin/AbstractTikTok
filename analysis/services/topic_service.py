"""Orchestrate fixed-snapshot topic clustering and atomic persistence."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections import Counter
from typing import Iterable, Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

import numpy as np

from analysis.clustering import RepresentativeInput, cluster_topic_vectors, select_representatives
from analysis.domain import (
    CommentAnalysis,
    SemanticAnalysisRun,
    SemanticRunItemStatus,
    SemanticRunStatus,
    TopicCluster,
    TopicClusterConfig,
    TopicClusteringResult,
    TopicMembership,
    TopicMetadataItem,
    TopicStageStatus,
)
from analysis.preprocessing import utc_now
from analysis.preprocessing import assess_semantic_quality
from analysis.repositories import SemanticRepository
from analysis.repositories.semantic_sqlalchemy import TopicSnapshotRow


def _sha256_json(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _frequencies(values: Iterable[str], limit: int) -> list[TopicMetadataItem]:
    counts = Counter(value.strip() for value in values if value and value.strip())
    ordered = sorted(counts.items(), key=lambda item: (-item[1], item[0]))[:limit]
    return [TopicMetadataItem(value=value, count=count) for value, count in ordered]


def _dominant(values: Iterable[object]):
    counts = Counter(value for value in values if value is not None)
    if not counts:
        return None
    return min(counts, key=lambda value: (-counts[value], str(getattr(value, "value", value))))


class TopicService:
    def __init__(self, repository: SemanticRepository):
        self.repository = repository

    async def cluster_run(
        self,
        semantic_run_id: UUID | str,
        config: TopicClusterConfig,
    ) -> tuple[SemanticAnalysisRun, TopicClusteringResult]:
        run = await self.repository.get_semantic_run(str(semantic_run_id))
        if run is None:
            raise KeyError(f"Semantic analysis run not found: {semantic_run_id}")
        if run.status != SemanticRunStatus.COMPLETED:
            raise ValueError("Topic clustering requires a completed embedding run")
        algorithm_config = config.model_dump(mode="json")
        # Persist a canonical, experiment-comparable configuration.  The
        # snapshot is part of the identity: identical hyperparameters over a
        # different comment population are not the same experiment.
        config_hash = _sha256_json({
            "input_snapshot_hash": run.input_snapshot_hash,
            "embedding": {
                "provider": run.embedding_provider,
                "model": run.embedding_model,
                "embedding_version": run.embedding_version,
                "embedding_dimension": run.embedding_dimension,
            },
            "algorithm": algorithm_config,
        })
        await self.repository.update_semantic_run(
            str(run.id),
            topic_status=TopicStageStatus.RUNNING.value,
            topic_started_at=utc_now(),
            topic_finished_at=None,
            topic_error_message=None,
            algorithm=config.algorithm,
            algorithm_version=config.algorithm_version,
            algorithm_config=algorithm_config,
            config_hash=config_hash,
        )
        try:
            snapshot = await self.repository.get_topic_snapshot(str(run.id))
            if len(snapshot) != run.total_comments:
                raise ValueError("Semantic run item snapshot is incomplete")
            embedded = [row for row in snapshot if row.embedding is not None]
            if not embedded:
                raise ValueError("Semantic run snapshot has no embeddings")
            if len(embedded) > config.max_input:
                raise ValueError(f"Topic clustering input exceeds maximum {config.max_input}")
            self._validate_snapshot(run, embedded)
            eligible = []
            excluded: dict[UUID, str] = {}
            for row in embedded:
                decision = assess_semantic_quality(
                    row.comment.text,
                    min_informative_characters=config.min_informative_characters,
                )
                if config.quality_gate_enabled and not decision.eligible:
                    excluded[row.comment.id] = decision.reason.value
                else:
                    eligible.append(row)
            result = await asyncio.to_thread(
                cluster_topic_vectors,
                [row.comment.id for row in eligible],
                [row.embedding.vector for row in eligible],
                config,
            )
            result = result.model_copy(update={
                "input_comments": len(embedded),
                "excluded_comments": excluded,
            })
            clusters, memberships = self._build_results(run.id, embedded, result, config)
            finished_at = utc_now()
            await self.repository.replace_topic_results(
                str(run.id),
                clusters,
                memberships,
                run_updates={
                    "topic_status": TopicStageStatus.COMPLETED.value,
                    "topic_finished_at": finished_at,
                    "topic_error_message": None,
                    "algorithm": config.algorithm,
                    "algorithm_version": config.algorithm_version,
                    "algorithm_config": algorithm_config,
                    "config_hash": config_hash,
                    "clustered_comments": result.clustered_comments,
                    "noise_comments": len(result.noise_comment_ids),
                    "topic_count": len(clusters),
                },
            )
            stored = await self.repository.get_semantic_run(str(run.id))
            if stored is None:
                raise RuntimeError("Semantic run topic state was not persisted")
            return stored, result
        except Exception as exc:
            await self.repository.fail_topic_run(
                str(run.id),
                finished_at=utc_now(),
                error_message=f"{type(exc).__name__}: {str(exc)[:500]}",
            )
            raise

    @staticmethod
    def _validate_snapshot(run, rows: Sequence[TopicSnapshotRow]) -> None:
        dimensions = set()
        for row in rows:
            embedding = row.embedding
            if embedding is None:
                continue
            if (
                embedding.provider != run.embedding_provider
                or embedding.model != run.embedding_model
                or embedding.embedding_version != run.embedding_version
            ):
                raise ValueError("Semantic run embedding snapshot configuration mismatch")
            vector = np.asarray(embedding.vector, dtype="<f4")
            if len(vector) != embedding.dimension or not np.isfinite(vector).all():
                raise ValueError("Semantic run embedding snapshot contains invalid vectors")
            if embedding.normalized:
                if not np.isclose(np.linalg.norm(vector), 1.0, rtol=1e-4, atol=1e-5):
                    raise ValueError("Embedding marked normalized is not unit length")
            dimensions.add(embedding.dimension)
        if len(dimensions) != 1:
            raise ValueError("Semantic run embedding dimensions do not match")

    def _build_results(
        self,
        run_id: UUID,
        rows: Sequence[TopicSnapshotRow],
        result: TopicClusteringResult,
        config: TopicClusterConfig,
    ) -> tuple[list[TopicCluster], list[TopicMembership]]:
        by_comment = {row.comment.id: row for row in rows}
        clusters: list[TopicCluster] = []
        memberships: list[TopicMembership] = []
        denominator = max(result.clustered_comments, 1)

        for display_index, candidate in enumerate(result.clusters, start=1):
            member_ids = sorted(candidate.member_comment_ids, key=str)
            cluster_key = hashlib.sha256(
                f"{run_id}:{','.join(str(item) for item in member_ids)}".encode("utf-8")
            ).hexdigest()
            cluster_id = uuid5(NAMESPACE_URL, f"semantic-topic:{run_id}:{cluster_key}")
            member_rows = [by_comment[comment_id] for comment_id in member_ids]
            analyses: list[CommentAnalysis] = [row.analysis for row in member_rows if row.analysis is not None]
            top_topics = _frequencies(
                (topic for analysis in analyses for topic in analysis.topics), config.metadata_top_k
            )
            top_keywords = _frequencies(
                (keyword for analysis in analyses for keyword in analysis.keywords), config.metadata_top_k
            )
            name = " / ".join(item.value for item in top_topics[:3]) or f"Semantic Topic #{display_index}"
            summary = f"{len(member_ids)} comments; {len(analyses)} with fixed CommentAnalysis metadata."
            representatives = select_representatives(
                [
                    RepresentativeInput(
                        comment_id=row.comment.id,
                        text=row.comment.text,
                        like_count=row.comment.like_count,
                        vector=row.embedding.vector,
                    )
                    for row in member_rows
                    if row.embedding is not None
                ],
                candidate.centroid,
                top_k=config.representative_top_k,
                mmr_lambda=config.representative_mmr_lambda,
            )
            representative_by_id = {item.comment_id: item for item in representatives}
            centroid = np.asarray(candidate.centroid, dtype="<f4")
            for row in member_rows:
                vector = np.asarray(row.embedding.vector, dtype="<f4")
                vector = vector / np.linalg.norm(vector)
                representative = representative_by_id.get(row.comment.id)
                memberships.append(
                    TopicMembership(
                        comment_id=row.comment.id,
                        topic_cluster_id=cluster_id,
                        similarity=float(vector @ centroid),
                        representative_rank=representative.rank if representative else None,
                        representative_score=representative.score if representative else None,
                    )
                )
            clusters.append(
                TopicCluster(
                    id=cluster_id,
                    semantic_run_id=run_id,
                    cluster_key=cluster_key,
                    name=name,
                    summary=summary,
                    size=len(member_ids),
                    percentage=round(len(member_ids) / denominator * 100.0, 6),
                    top_topics=top_topics,
                    top_keywords=top_keywords,
                    centroid=candidate.centroid,
                    dimension=candidate.dimension,
                    medoid_comment_id=candidate.medoid_comment_id,
                    analyzed_comments=len(analyses),
                    dominant_sentiment=_dominant(analysis.sentiment for analysis in analyses),
                    dominant_stance=_dominant(analysis.stance for analysis in analyses),
                    dominant_risk=_dominant(analysis.risk_level for analysis in analyses),
                    created_at=utc_now(),
                )
            )

        for comment_id in result.noise_comment_ids:
            memberships.append(TopicMembership(comment_id=comment_id))
        for comment_id, reason in result.excluded_comments.items():
            memberships.append(TopicMembership(
                comment_id=comment_id,
                status=SemanticRunItemStatus.EXCLUDED,
                error_message=f"semantic_quality:{reason}",
            ))
        memberships.sort(key=lambda item: str(item.comment_id))
        return clusters, memberships
