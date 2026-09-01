"""In-memory cosine search over a fixed, versioned semantic-run snapshot."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from analysis.domain import (
    SemanticRunItemStatus,
    SemanticRunStatus,
    SemanticSearchHit,
    SemanticSearchQuery,
    SemanticSearchResult,
)
from analysis.embeddings import EmbeddingProvider, normalize_vector
from analysis.preprocessing import normalize_text
from analysis.repositories import SemanticRepository


@dataclass(frozen=True)
class _Candidate:
    similarity: float
    row: object


class SemanticSearchService:
    def __init__(
        self,
        repository: SemanticRepository,
        provider: EmbeddingProvider,
        *,
        max_snapshot_size: int = 5_000,
    ):
        self.repository = repository
        self.provider = provider
        self.max_snapshot_size = max_snapshot_size

    async def search(self, query: SemanticSearchQuery) -> SemanticSearchResult:
        run = await self.repository.get_semantic_run(str(query.semantic_run_id))
        if run is None:
            raise KeyError(f"Semantic analysis run not found: {query.semantic_run_id}")
        if run.status != SemanticRunStatus.COMPLETED:
            raise ValueError("Semantic search requires a completed embedding run")
        if self.provider.provider_name != run.embedding_provider or self.provider.model != run.embedding_model:
            raise ValueError("Query embedding provider/model does not match the semantic run")
        snapshot = await self.repository.get_topic_snapshot(str(run.id))
        if len(snapshot) > self.max_snapshot_size:
            raise ValueError(f"Semantic search snapshot exceeds maximum {self.max_snapshot_size}")

        query_text = normalize_text(query.query)
        if not query_text:
            raise ValueError("Semantic search query cannot be empty")
        batch = await self.provider.embed_texts([query_text])
        if batch.model != run.embedding_model or len(batch.vectors) != 1:
            raise ValueError("Query embedding response does not match the semantic run")
        query_vector = np.asarray(normalize_vector(batch.vectors[0]), dtype="<f4")
        if run.embedding_dimension is None or len(query_vector) != run.embedding_dimension:
            raise ValueError("Query embedding dimension does not match the semantic run")

        rows = []
        excluded = 0
        for row in snapshot:
            if row.embedding is None:
                excluded += 1
                continue
            if query.exclude_low_information and row.item.status == SemanticRunItemStatus.EXCLUDED:
                excluded += 1
                continue
            if query.risk_level is not None and (
                row.analysis is None or row.analysis.risk_level != query.risk_level
            ):
                excluded += 1
                continue
            rows.append(row)
        if rows:
            matrix = np.asarray([row.embedding.vector for row in rows], dtype="<f4")
            if matrix.ndim != 2 or matrix.shape[1] != len(query_vector):
                raise ValueError("Stored embedding dimensions are inconsistent")
            scores = matrix @ query_vector
            candidates = [
                _Candidate(float(score), row)
                for score, row in zip(scores, rows)
                if float(score) >= query.min_similarity
            ]
        else:
            candidates = []
        candidates.sort(key=lambda item: (-item.similarity, str(item.row.comment.id)))

        hits = []
        seen_text: set[str] = set()
        for candidate in candidates:
            row = candidate.row
            normalized = normalize_text(row.comment.text)
            if query.deduplicate_text and normalized in seen_text:
                continue
            seen_text.add(normalized)
            hits.append(SemanticSearchHit(
                comment_id=row.comment.id,
                text=row.comment.text,
                similarity=round(candidate.similarity, 6),
                like_count=row.comment.like_count,
                published_at=row.comment.published_at,
                content_id=row.comment.content_id,
                topic_cluster_id=row.item.topic_cluster_id,
                sentiment=row.analysis.sentiment if row.analysis else None,
                risk_level=row.analysis.risk_level if row.analysis else None,
                summary=row.analysis.summary if row.analysis else None,
            ))
            if len(hits) >= query.top_k:
                break
        return SemanticSearchResult(
            semantic_run_id=run.id,
            query=query_text,
            embedding_provider=run.embedding_provider,
            embedding_model=run.embedding_model,
            embedding_version=run.embedding_version,
            searched_comments=len(rows),
            excluded_comments=excluded,
            hits=hits,
        )
