"""Opt-in 10k synthetic benchmark; intentionally excluded from normal tests.

Agglomerative clustering is quadratic in memory. Run only on a machine with
ample RAM:
    python benchmarks/topic_clustering_10k.py --execute
"""

from __future__ import annotations

import argparse
import time
import tracemalloc
from uuid import NAMESPACE_URL, uuid5

import numpy as np

from analysis.clustering import cluster_topic_vectors
from analysis.domain import TopicClusterConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    arguments = parser.parse_args()
    if not arguments.execute:
        parser.error("Pass --execute to acknowledge the quadratic memory cost")

    generator = np.random.default_rng(20260831)
    vectors = generator.normal(size=(10_000, 16)).astype("<f4")
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    identifiers = [uuid5(NAMESPACE_URL, f"topic-benchmark:{index}") for index in range(10_000)]
    config = TopicClusterConfig(distance_threshold=0.65, min_cluster_size=5)

    tracemalloc.start()
    started = time.perf_counter()
    result = cluster_topic_vectors(
        identifiers,
        vectors.tolist(),
        config,
        enforce_max_input=False,
    )
    elapsed = time.perf_counter() - started
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    print({
        "comments": 10_000,
        "clusters": len(result.clusters),
        "noise": len(result.noise_comment_ids),
        "elapsed_seconds": round(elapsed, 3),
        "python_peak_memory_mib": round(peak / 1024 / 1024, 3),
    })


if __name__ == "__main__":
    main()
