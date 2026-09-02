"""Opt-in synthetic benchmark for topic-quality metrics.

Agglomerative clustering has quadratic memory characteristics.  The 10K case
is intentionally explicit and may be blocked by available machine memory; it
does not change the production service limit of 5,000 comments.
"""

from __future__ import annotations

import argparse
import sys
import time
import tracemalloc
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

import numpy as np

# Support the documented direct invocation from the repository root without
# requiring callers to set PYTHONPATH themselves.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from analysis.clustering import cluster_topic_vectors
from analysis.domain import TopicClusterConfig
from analysis.experiments import evaluate_topic_quality


def synthetic_vectors(count: int, dimension: int = 32) -> tuple[list, list[list[float]]]:
    generator = np.random.default_rng(31_403)
    centers = generator.normal(size=(8, dimension)).astype("<f4")
    centers /= np.linalg.norm(centers, axis=1, keepdims=True)
    labels = np.arange(count) % len(centers)
    matrix = centers[labels] + generator.normal(scale=0.10, size=(count, dimension)).astype("<f4")
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    ids = [uuid5(NAMESPACE_URL, f"phase3b3-benchmark:{index}") for index in range(count)]
    return ids, matrix.astype(float).tolist()


def run_case(count: int) -> dict[str, object]:
    ids, vectors = synthetic_vectors(count)
    config = TopicClusterConfig(distance_threshold=0.35, min_cluster_size=3)
    tracemalloc.start()
    started = time.perf_counter()
    result = cluster_topic_vectors(ids, vectors, config, enforce_max_input=False)
    clustering_seconds = time.perf_counter() - started
    metric_started = time.perf_counter()
    metrics = evaluate_topic_quality(
        comment_ids=ids,
        vectors=vectors,
        result=result,
        input_snapshot_hash="benchmark-synthetic-v1",
        silhouette_max_samples=2_000,
    )
    silhouette_seconds = time.perf_counter() - metric_started
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return {
        "n": count,
        "cluster_count": metrics.cluster_count,
        "noise_ratio": metrics.noise_ratio,
        "runtime_seconds": round(clustering_seconds, 6),
        "peak_memory_mib": round(peak_bytes / 1024 / 1024, 3),
        "silhouette": metrics.silhouette_score,
        "silhouette_runtime_seconds": round(silhouette_seconds, 6),
        "silhouette_sample_count": metrics.silhouette_sample_count,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="Run the potentially memory-intensive benchmark")
    parser.add_argument("--sizes", default="1000,5000,10000", help="Comma-separated synthetic input sizes")
    arguments = parser.parse_args()
    if not arguments.execute:
        print("Dry run only. Re-run with --execute to benchmark 1000/5000/10000 vectors.")
        return
    sizes = [int(item.strip()) for item in arguments.sizes.split(",") if item.strip()]
    for count in sizes:
        print(run_case(count))


if __name__ == "__main__":
    main()
