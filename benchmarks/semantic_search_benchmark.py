"""Opt-in local benchmark for fixed-snapshot cosine retrieval (no API calls)."""

from __future__ import annotations

import argparse
import asyncio
import json
import time
from pathlib import Path

import numpy as np

from analysis.domain import SemanticRunItemStatus
from analysis.repositories import SemanticRepository


async def benchmark(database: Path, semantic_run_id: str, iterations: int) -> dict[str, object]:
    repository = SemanticRepository(database)
    try:
        started = time.perf_counter()
        snapshot = await repository.get_topic_snapshot(semantic_run_id)
        load_seconds = time.perf_counter() - started
    finally:
        await repository.close()
    eligible = [
        row for row in snapshot
        if row.embedding is not None and row.item.status != SemanticRunItemStatus.EXCLUDED
    ]
    if not eligible:
        raise ValueError("Semantic run has no eligible embeddings")
    matrix_started = time.perf_counter()
    matrix = np.asarray([row.embedding.vector for row in eligible], dtype="<f4")
    matrix_seconds = time.perf_counter() - matrix_started
    query = matrix[0]
    _ = matrix @ query
    timings = []
    for _index in range(iterations):
        started = time.perf_counter()
        scores = matrix @ query
        top = np.argpartition(scores, -min(10, len(scores)))[-min(10, len(scores)):]
        _ = top[np.argsort(scores[top])[::-1]]
        timings.append((time.perf_counter() - started) * 1000)
    return {
        "semantic_run_id": semantic_run_id,
        "eligible_comments": len(eligible),
        "dimension": int(matrix.shape[1]),
        "matrix_mebibytes": round(matrix.nbytes / 1024 / 1024, 3),
        "database_snapshot_load_seconds": round(load_seconds, 6),
        "matrix_build_seconds": round(matrix_seconds, 6),
        "retrieval_iterations": iterations,
        "retrieval_mean_ms": round(float(np.mean(timings)), 6),
        "retrieval_p95_ms": round(float(np.percentile(timings, 95)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--database", type=Path, required=True)
    parser.add_argument("--semantic-run-id", required=True)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()
    if not 1 <= args.iterations <= 10_000:
        raise ValueError("iterations must be between 1 and 10000")
    print(json.dumps(
        asyncio.run(benchmark(args.database, args.semantic_run_id, args.iterations)),
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
