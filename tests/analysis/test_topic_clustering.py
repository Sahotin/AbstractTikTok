from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from analysis.clustering import cluster_topic_vectors
from analysis.domain import TopicClusterConfig


def _config(**updates):
    values = {"distance_threshold": 0.2, "min_cluster_size": 2}
    values.update(updates)
    return TopicClusterConfig(**values)


def _signature(result):
    return [
        (tuple(map(str, cluster.member_comment_ids)), tuple(np.round(cluster.centroid, 6)))
        for cluster in result.clusters
    ], tuple(map(str, result.noise_comment_ids))


def test_two_separated_groups_are_deterministic_and_order_independent() -> None:
    ids = [uuid4() for _ in range(4)]
    vectors = [[1, 0], [0.95, 0.05], [0, 1], [0.05, 0.95]]

    first = cluster_topic_vectors(ids, vectors, _config())
    second = cluster_topic_vectors(list(reversed(ids)), list(reversed(vectors)), _config())

    assert len(first.clusters) == 2
    assert _signature(first) == _signature(second)
    assert all(len(cluster.member_comment_ids) == 2 for cluster in first.clusters)


def test_small_clusters_become_noise() -> None:
    ids = [uuid4(), uuid4()]
    result = cluster_topic_vectors(ids, [[1, 0], [0, 1]], _config(min_cluster_size=2))
    assert result.clusters == []
    assert result.noise_comment_ids == sorted(ids, key=str)


def test_single_cluster_centroid_and_medoid_are_stable() -> None:
    ids = sorted([uuid4(), uuid4(), uuid4()], key=str)
    result = cluster_topic_vectors(ids, [[1, 0], [1, 0], [0.9, 0.1]], _config())
    cluster = result.clusters[0]

    assert len(result.clusters) == 1
    assert np.linalg.norm(cluster.centroid) == pytest.approx(1.0)
    assert cluster.medoid_comment_id == ids[0]


def test_empty_and_single_comment_inputs() -> None:
    empty = cluster_topic_vectors([], [], _config())
    one_id = uuid4()
    one = cluster_topic_vectors([one_id], [[1, 0]], _config(min_cluster_size=1))
    noise = cluster_topic_vectors([one_id], [[1, 0]], _config(min_cluster_size=2))

    assert empty.clusters == []
    assert one.clusters[0].member_comment_ids == [one_id]
    assert noise.noise_comment_ids == [one_id]


@pytest.mark.parametrize(
    "vectors",
    [
        [[1, 0], [1, 0, 0]],
        [[1, 0], [float("nan"), 1]],
        [[1, 0], [float("inf"), 1]],
        [[1, 0], [0, 0]],
    ],
)
def test_invalid_vectors_are_rejected(vectors) -> None:
    with pytest.raises(ValueError):
        cluster_topic_vectors([uuid4(), uuid4()], vectors, _config())


def test_cluster_members_produce_stable_sha256_input() -> None:
    ids = [uuid4(), uuid4()]
    first = cluster_topic_vectors(ids, [[1, 0], [0.99, 0.01]], _config())
    second = cluster_topic_vectors(ids, [[1, 0], [0.99, 0.01]], _config())
    assert _signature(first) == _signature(second)


def test_input_above_mvp_limit_is_rejected_without_sampling() -> None:
    identifier = uuid4()
    ids = [uuid4() for _ in range(5001)]
    with pytest.raises(ValueError, match="maximum"):
        cluster_topic_vectors(ids, [[1, 0]] * len(ids), _config())
