"""Deterministic agglomerative clustering over unit-normalized vectors."""

from __future__ import annotations

import inspect
from collections import defaultdict
from typing import Sequence
from uuid import UUID

import numpy as np

from analysis.domain import TopicClusterCandidate, TopicClusterConfig, TopicClusteringResult


def _unit_matrix(vectors: Sequence[Sequence[float]]) -> np.ndarray:
    matrix = np.asarray(vectors, dtype="<f4")
    if matrix.ndim != 2 or matrix.shape[1] == 0:
        raise ValueError("Topic vectors must be a non-empty two-dimensional matrix")
    if not np.isfinite(matrix).all():
        raise ValueError("Topic vectors contain NaN or Infinity")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("Topic vectors must have non-zero norm")
    if np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
        return matrix
    return (matrix / norms).astype("<f4")


def cluster_topic_vectors(
    comment_ids: Sequence[UUID],
    vectors: Sequence[Sequence[float]],
    config: TopicClusterConfig,
    *,
    enforce_max_input: bool = True,
) -> TopicClusteringResult:
    if len(comment_ids) != len(vectors):
        raise ValueError("Comment and vector counts do not match")
    if not comment_ids:
        return TopicClusteringResult(
            clusters=[], noise_comment_ids=[], input_comments=0, clustered_comments=0
        )
    if enforce_max_input and len(comment_ids) > config.max_input:
        raise ValueError(f"Topic clustering input exceeds maximum {config.max_input}")
    if len(set(comment_ids)) != len(comment_ids):
        raise ValueError("Topic clustering comment IDs must be unique")

    ordered = sorted(zip(comment_ids, vectors), key=lambda item: str(item[0]))
    ordered_ids = [item[0] for item in ordered]
    dimensions = {len(item[1]) for item in ordered}
    if len(dimensions) != 1:
        raise ValueError("Topic vector dimensions do not match")
    matrix = _unit_matrix([item[1] for item in ordered])

    if len(ordered_ids) == 1:
        labels = np.asarray([0], dtype=int)
    else:
        from sklearn.cluster import AgglomerativeClustering

        arguments = {
            "n_clusters": None,
            "distance_threshold": config.distance_threshold,
            "linkage": config.linkage,
        }
        if "metric" in inspect.signature(AgglomerativeClustering).parameters:
            arguments["metric"] = config.metric
        else:
            arguments["affinity"] = config.metric
        labels = AgglomerativeClustering(**arguments).fit_predict(matrix)

    grouped: dict[int, list[int]] = defaultdict(list)
    for index, label in enumerate(labels.tolist()):
        grouped[int(label)].append(index)

    clusters: list[TopicClusterCandidate] = []
    noise_ids: list[UUID] = []
    for indices in grouped.values():
        member_ids = sorted((ordered_ids[index] for index in indices), key=str)
        if len(indices) < config.min_cluster_size:
            noise_ids.extend(member_ids)
            continue
        member_matrix = matrix[indices]
        centroid = member_matrix.mean(axis=0)
        centroid_norm = float(np.linalg.norm(centroid))
        if centroid_norm == 0:
            raise ValueError("Topic cluster centroid has zero norm")
        centroid = (centroid / centroid_norm).astype("<f4")
        similarities = member_matrix @ centroid
        best_similarity = float(similarities.max())
        medoid_candidates = [
            ordered_ids[index]
            for index, similarity in zip(indices, similarities.tolist())
            if abs(float(similarity) - best_similarity) <= 1e-7
        ]
        medoid_id = min(medoid_candidates, key=str)
        clusters.append(
            TopicClusterCandidate(
                member_comment_ids=member_ids,
                centroid=centroid.astype(float).tolist(),
                dimension=int(matrix.shape[1]),
                medoid_comment_id=medoid_id,
            )
        )

    clusters.sort(key=lambda cluster: (-len(cluster.member_comment_ids), str(cluster.medoid_comment_id)))
    noise_ids.sort(key=str)
    return TopicClusteringResult(
        clusters=clusters,
        noise_comment_ids=noise_ids,
        input_comments=len(ordered_ids),
        clustered_comments=sum(len(cluster.member_comment_ids) for cluster in clusters),
        dimension=int(matrix.shape[1]),
    )
