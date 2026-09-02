from __future__ import annotations

from uuid import uuid4

import pytest

from analysis.clustering import cluster_topic_vectors
from analysis.domain import TopicClusterConfig
from analysis.embeddings import EmbeddingConfig, FakeEmbeddingProvider
from analysis.experiments.topic_quality import (
    TopicQualityExperimentService,
    canonical_config_hash,
    deterministic_comment_sample,
    evaluate_topic_quality,
    write_topic_quality_report,
)
from analysis.services import EmbeddingService, SemanticRunService


def _separated_input():
    ids = [uuid4() for _ in range(8)]
    vectors = [
        [1.0, 0.0], [0.99, 0.01], [0.98, -0.02],
        [0.0, 1.0], [0.01, 0.99], [-0.02, 0.98],
        [-1.0, 0.0], [0.0, -1.0],
    ]
    return ids, vectors


def test_metrics_are_deterministic_and_exclude_noise() -> None:
    ids, vectors = _separated_input()
    result = cluster_topic_vectors(ids, vectors, TopicClusterConfig(distance_threshold=0.08, min_cluster_size=3))
    first = evaluate_topic_quality(
        comment_ids=ids, vectors=vectors, result=result, input_snapshot_hash="a" * 64
    )
    second = evaluate_topic_quality(
        comment_ids=list(reversed(ids)), vectors=list(reversed(vectors)), result=result, input_snapshot_hash="a" * 64
    )

    assert first == second
    assert first.cluster_count == 2
    assert first.noise_count == 2
    assert first.noise_ratio == pytest.approx(0.25)
    assert first.largest_cluster_size == 3
    assert first.median_cluster_size == 3
    assert first.silhouette_score is not None and first.silhouette_score > 0


def test_metrics_account_for_quality_gate_exclusions() -> None:
    ids, vectors = _separated_input()
    result = cluster_topic_vectors(
        ids[:6], vectors[:6], TopicClusterConfig(distance_threshold=0.08, min_cluster_size=3)
    ).model_copy(update={
        "input_comments": 8,
        "excluded_comments": {ids[6]: "numeric_only", ids[7]: "emoji_only"},
    })
    metric = evaluate_topic_quality(
        comment_ids=ids,
        vectors=vectors,
        result=result,
        input_snapshot_hash="e" * 64,
    )
    assert metric.cluster_count == 2
    assert metric.noise_count == 0
    assert metric.excluded_count == 2
    assert metric.excluded_ratio == pytest.approx(0.25)


def test_parameter_sensitivity_one_cluster_and_empty_input() -> None:
    ids, vectors = _separated_input()
    tight = cluster_topic_vectors(ids, vectors, TopicClusterConfig(distance_threshold=0.08, min_cluster_size=3))
    loose = cluster_topic_vectors(ids, vectors, TopicClusterConfig(distance_threshold=2.0, min_cluster_size=1))
    assert (len(tight.clusters), len(tight.noise_comment_ids)) != (len(loose.clusters), len(loose.noise_comment_ids))

    one = cluster_topic_vectors(ids[:3], vectors[:3], TopicClusterConfig(distance_threshold=0.2, min_cluster_size=1))
    one_metric = evaluate_topic_quality(
        comment_ids=ids[:3], vectors=vectors[:3], result=one, input_snapshot_hash="b" * 64
    )
    assert one_metric.silhouette_score is None

    empty = cluster_topic_vectors([], [], TopicClusterConfig())
    empty_metric = evaluate_topic_quality(
        comment_ids=[], vectors=[], result=empty, input_snapshot_hash="c" * 64
    )
    assert empty_metric.total_comments == 0
    assert empty_metric.noise_ratio == 0
    assert empty_metric.silhouette_score is None


def test_deterministic_human_sampling() -> None:
    ids = [uuid4() for _ in range(10)]
    assert deterministic_comment_sample(ids, "d" * 64, 5) == deterministic_comment_sample(
        list(reversed(ids)), "d" * 64, 5
    )


def test_quality_configuration_changes_canonical_hash() -> None:
    first = TopicClusterConfig(min_informative_characters=2)
    second = TopicClusterConfig(min_informative_characters=3)
    assert canonical_config_hash(first.model_dump(mode="json")) != canonical_config_hash(
        second.model_dump(mode="json")
    )


@pytest.mark.asyncio
async def test_matrix_creates_isolated_runs_and_a_local_report(phase1b_context, semantic_repository, tmp_path) -> None:
    config = EmbeddingConfig(provider="fake", model="quality-test", embedding_version="quality-v1")
    provider = FakeEmbeddingProvider(dimension=8, model="quality-test")
    semantic_service = SemanticRunService(
        phase1b_context.repository,
        semantic_repository,
        EmbeddingService(semantic_repository, provider, config),
        config,
    )
    source = await semantic_service.create_run(platform="douyin", limit=2)
    source, _ = await semantic_service.execute_run(source.id)
    experiments = await TopicQualityExperimentService(
        phase1b_context.repository,
        semantic_repository,
        EmbeddingService(semantic_repository, provider, config),
        config,
    ).run_matrix(
        source.id,
        thresholds=[0.2, 2.0],
        min_cluster_sizes=[1],
        evaluation_sample_size=2,
    )

    assert len(experiments) == 2
    assert {item.semantic_run.id for item in experiments}.isdisjoint({source.id})
    assert all(item.semantic_run.input_snapshot_hash == source.input_snapshot_hash for item in experiments)
    assert await semantic_repository.get_semantic_run(str(source.id)) == source
    report = write_topic_quality_report(experiments, tmp_path / "topic_quality")
    content = report.read_text(encoding="utf-8")
    assert "Experiment Matrix" in content
    assert "Deterministic Human Evaluation Set" in content
