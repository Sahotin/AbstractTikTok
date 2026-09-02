"""Deterministic no-network embedding provider for tests and local validation."""

from __future__ import annotations

import hashlib
import math
from typing import Mapping, Optional, Sequence

from .base import EmbeddingBatch, EmbeddingProvider, EmbeddingProviderError


class FakeEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        mapping: Optional[Mapping[str, Sequence[float]]] = None,
        *,
        dimension: Optional[int] = None,
        model: str = "fake-embedding-v1",
        provider_name: str = "fake",
        failure_texts: Optional[set[str]] = None,
    ):
        self.mapping = {text: [float(value) for value in vector] for text, vector in (mapping or {}).items()}
        if any(not math.isfinite(value) for vector in self.mapping.values() for value in vector):
            raise ValueError("Explicit fake embedding vectors must contain only finite values")
        mapped_dimensions = {len(vector) for vector in self.mapping.values()}
        if len(mapped_dimensions) > 1:
            raise ValueError("Explicit fake embedding vectors must have one dimension")
        inferred_dimension = next(iter(mapped_dimensions), None)
        if dimension is not None and inferred_dimension is not None and dimension != inferred_dimension:
            raise ValueError("Configured fake dimension does not match explicit vectors")
        self.dimension = dimension or inferred_dimension or 16
        if self.dimension <= 0:
            raise ValueError("Fake embedding dimension must be positive")
        self._model = model
        self._provider_name = provider_name
        self.failure_texts = set(failure_texts or set())
        self.call_count = 0
        self.input_count = 0
        self.batch_sizes: list[int] = []

    @property
    def provider_name(self) -> str:
        return self._provider_name

    @property
    def model(self) -> str:
        return self._model

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        self.call_count += 1
        self.input_count += len(texts)
        self.batch_sizes.append(len(texts))
        failing = next((text for text in texts if text in self.failure_texts), None)
        if failing is not None:
            raise EmbeddingProviderError("Simulated embedding failure")
        vectors = [self.mapping.get(text, self._stable_vector(text)) for text in texts]
        if any(len(vector) != self.dimension for vector in vectors):
            raise EmbeddingProviderError("Fake embedding vector dimension mismatch")
        return EmbeddingBatch(vectors=vectors, model=self.model, dimension=self.dimension)

    def _stable_vector(self, text: str) -> list[float]:
        values = []
        for index in range(self.dimension):
            digest = hashlib.sha256(f"{index}:{text}".encode("utf-8")).digest()
            integer = int.from_bytes(digest[:8], "little", signed=False)
            values.append((integer / (2**64 - 1)) * 2.0 - 1.0)
        norm = math.sqrt(sum(value * value for value in values))
        return [value / norm for value in values]
