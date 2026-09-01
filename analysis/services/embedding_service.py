"""Bounded, cached, versioned batch embedding pipeline."""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Sequence
from uuid import NAMESPACE_URL, UUID, uuid5

from pydantic import BaseModel, Field

from analysis.domain import EmbeddingRecord, NormalizedComment
from analysis.embeddings import (
    EmbeddingConfig,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingResponseError,
    normalize_vector,
)
from analysis.preprocessing import compute_text_hash, normalize_text, utc_now
from analysis.repositories import SemanticRepository


class EmbeddingFailure(BaseModel):
    comment_id: UUID
    error_type: str
    error_message: str


class EmbeddingReport(BaseModel):
    total_comments: int = Field(ge=0)
    embedded_comments: int = Field(ge=0)
    failed_comments: int = Field(ge=0)
    exact_cache_hits: int = Field(ge=0)
    content_cache_hits: int = Field(ge=0)
    provider_inputs: int = Field(ge=0)
    provider_requests: int = Field(ge=0)
    dimension: int | None = None
    failures: list[EmbeddingFailure] = Field(default_factory=list)


@dataclass(frozen=True)
class _PreparedGroup:
    input_hash: str
    text: str
    comments: list[NormalizedComment]


class EmbeddingService:
    def __init__(
        self,
        repository: SemanticRepository,
        provider: EmbeddingProvider,
        config: EmbeddingConfig,
    ):
        self.repository = repository
        self.provider = provider
        self.config = config
        self._provider_requests = 0
        self._provider_inputs = 0
        self._dimension: int | None = None
        self._dimension_lock = asyncio.Lock()

    async def embed_comments(self, comments: Sequence[NormalizedComment]) -> EmbeddingReport:
        self._provider_requests = 0
        self._provider_inputs = 0
        self._dimension = None
        if not comments:
            return EmbeddingReport(total_comments=0, embedded_comments=0, failed_comments=0,
                                   exact_cache_hits=0, content_cache_hits=0,
                                   provider_inputs=0, provider_requests=0)
        prepared: dict[str, _PreparedGroup] = {}
        input_hash_by_comment: dict[str, str] = {}
        failures: list[EmbeddingFailure] = []
        for comment in comments:
            text = normalize_text(comment.text)
            if not text:
                failures.append(self._failure(comment, ValueError("Cannot embed an empty comment")))
                continue
            input_hash = compute_text_hash(text)
            input_hash_by_comment[str(comment.id)] = input_hash
            group = prepared.get(input_hash)
            if group is None:
                prepared[input_hash] = _PreparedGroup(input_hash, text, [comment])
            else:
                group.comments.append(comment)

        key = {
            "provider": self.provider.provider_name,
            "model": self.provider.model,
            "embedding_version": self.config.embedding_version,
        }
        exact_records = await self.repository.get_embeddings_for_comments(
            list(input_hash_by_comment), **key
        )
        exact_by_comment = {
            str(record.comment_id): record
            for record in exact_records
            if input_hash_by_comment.get(str(record.comment_id)) == record.input_hash
        }
        exact_hits = len(exact_by_comment)
        for record in exact_by_comment.values():
            self._accept_dimension(record.dimension)

        missing_groups: dict[str, _PreparedGroup] = {}
        for input_hash, group in prepared.items():
            missing_comments = [comment for comment in group.comments if str(comment.id) not in exact_by_comment]
            if missing_comments:
                missing_groups[input_hash] = _PreparedGroup(input_hash, group.text, missing_comments)

        content_cached = await self.repository.get_embeddings_by_content_hashes(
            list(missing_groups), **key
        )
        reused_records: list[EmbeddingRecord] = []
        content_hits = 0
        provider_groups: list[_PreparedGroup] = []
        for input_hash, group in missing_groups.items():
            cached = content_cached.get(input_hash)
            if cached is None:
                provider_groups.append(group)
                continue
            self._accept_dimension(cached.dimension)
            for comment in group.comments:
                reused_records.append(self._record(comment, input_hash, cached.vector, cached.dimension))
                content_hits += 1

        generated_records, generated_failures = await self._embed_groups(provider_groups)
        failures.extend(generated_failures)
        await self.repository.save_embeddings([*reused_records, *generated_records])
        embedded = exact_hits + len(reused_records) + len(generated_records)
        return EmbeddingReport(
            total_comments=len(comments),
            embedded_comments=embedded,
            failed_comments=len(failures),
            exact_cache_hits=exact_hits,
            content_cache_hits=content_hits,
            provider_inputs=self._provider_inputs,
            provider_requests=self._provider_requests,
            dimension=self._dimension,
            failures=failures,
        )

    async def _embed_groups(
        self,
        groups: Sequence[_PreparedGroup],
    ) -> tuple[list[EmbeddingRecord], list[EmbeddingFailure]]:
        batches = [list(groups[offset : offset + self.config.batch_size]) for offset in range(0, len(groups), self.config.batch_size)]
        next_batch = 0
        batch_lock = asyncio.Lock()
        records: list[EmbeddingRecord] = []
        failures: list[EmbeddingFailure] = []

        async def worker() -> None:
            nonlocal next_batch
            while True:
                async with batch_lock:
                    if next_batch >= len(batches):
                        return
                    batch = batches[next_batch]
                    next_batch += 1
                batch_records, batch_failures = await self._embed_with_isolation(batch)
                records.extend(batch_records)
                failures.extend(batch_failures)

        worker_count = min(self.config.max_concurrency, len(batches))
        if worker_count:
            await asyncio.gather(*(worker() for _ in range(worker_count)))
        return records, failures

    async def _embed_with_isolation(
        self,
        groups: list[_PreparedGroup],
    ) -> tuple[list[EmbeddingRecord], list[EmbeddingFailure]]:
        try:
            self._provider_requests += 1
            self._provider_inputs += len(groups)
            batch = await self.provider.embed_texts([group.text for group in groups])
            if batch.model != self.provider.model:
                raise EmbeddingResponseError("Embedding response model does not match configured model")
            if len(batch.vectors) != len(groups):
                raise EmbeddingResponseError("Embedding response count mismatch")
            await self._accept_dimension_async(batch.dimension)
            records = []
            for group, vector in zip(groups, batch.vectors):
                normalized = normalize_vector(vector)
                if len(normalized) != batch.dimension:
                    raise EmbeddingResponseError("Embedding vector dimension mismatch")
                records.extend(
                    self._record(comment, group.input_hash, normalized, batch.dimension)
                    for comment in group.comments
                )
            return records, []
        except EmbeddingProviderError as exc:
            if len(groups) == 1:
                return [], [self._failure(comment, exc) for comment in groups[0].comments]
            midpoint = len(groups) // 2
            # Keep failure isolation inside the bounded worker. Recursive
            # gather would expand concurrency when many inputs fail.
            left = await self._embed_with_isolation(groups[:midpoint])
            right = await self._embed_with_isolation(groups[midpoint:])
            return [*left[0], *right[0]], [*left[1], *right[1]]

    async def _accept_dimension_async(self, dimension: int) -> None:
        async with self._dimension_lock:
            self._accept_dimension(dimension)

    def _accept_dimension(self, dimension: int) -> None:
        if self._dimension is None:
            self._dimension = dimension
        elif self._dimension != dimension:
            raise EmbeddingResponseError("Embedding dimension changed within one service run")

    def _record(
        self,
        comment: NormalizedComment,
        input_hash: str,
        vector: Sequence[float],
        dimension: int,
    ) -> EmbeddingRecord:
        cache_key = {
            "comment_id": str(comment.id),
            "input_hash": input_hash,
            "provider": self.provider.provider_name,
            "model": self.provider.model,
            "embedding_version": self.config.embedding_version,
        }
        identity = json.dumps(cache_key, sort_keys=True)
        return EmbeddingRecord(
            id=uuid5(NAMESPACE_URL, f"mediacrawler:embedding:{identity}"),
            comment_id=comment.id,
            input_hash=input_hash,
            provider=self.provider.provider_name,
            model=self.provider.model,
            embedding_version=self.config.embedding_version,
            dimension=dimension,
            vector=list(vector),
            created_at=utc_now(),
        )

    @staticmethod
    def _failure(comment: NormalizedComment, exc: Exception) -> EmbeddingFailure:
        return EmbeddingFailure(
            comment_id=comment.id,
            error_type=type(exc).__name__,
            error_message=str(exc)[:500],
        )
