"""OpenAI-compatible batch embedding provider implemented with httpx."""

from __future__ import annotations

import asyncio
import math
from typing import Any, Sequence

import httpx

from .base import (
    EmbeddingBatch,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingResponseError,
    TransientEmbeddingError,
)
from .config import EmbeddingConfig


class OpenAICompatibleEmbeddingProvider(EmbeddingProvider):
    def __init__(
        self,
        config: EmbeddingConfig,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ):
        config.validate_for_remote_provider()
        self.config = config
        self.transport = transport

    @property
    def provider_name(self) -> str:
        return self.config.provider

    @property
    def model(self) -> str:
        return self.config.model

    def _endpoint(self) -> str:
        base = self.config.base_url.rstrip("/")
        return base if base.endswith("/embeddings") else f"{base}/embeddings"

    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        if not texts:
            raise ValueError("Embedding input cannot be empty")
        for attempt in range(self.config.max_retries + 1):
            try:
                return await self._request(texts)
            except TransientEmbeddingError:
                if attempt >= self.config.max_retries:
                    raise
                delay = self.config.retry_base_delay * (2**attempt)
                if delay:
                    await asyncio.sleep(delay)
        raise AssertionError("unreachable embedding retry state")

    async def _request(self, texts: Sequence[str]) -> EmbeddingBatch:
        key = self.config.api_key
        if key is None:
            raise EmbeddingProviderError("Embedding API key is not configured")
        try:
            async with httpx.AsyncClient(timeout=self.config.timeout, transport=self.transport) as client:
                response = await client.post(
                    self._endpoint(),
                    headers={"Authorization": f"Bearer {key.get_secret_value()}"},
                    json={"model": self.model, "input": list(texts)},
                )
                response.raise_for_status()
        except httpx.TimeoutException as exc:
            raise TransientEmbeddingError("Embedding request timed out") from exc
        except httpx.HTTPStatusError as exc:
            status = exc.response.status_code
            if status == 429 or status >= 500:
                raise TransientEmbeddingError(f"Embedding provider returned retryable HTTP {status}") from exc
            raise EmbeddingProviderError(f"Embedding provider returned HTTP {status}") from exc
        except httpx.RequestError as exc:
            raise TransientEmbeddingError("Embedding provider network request failed") from exc

        try:
            body: dict[str, Any] = response.json()
            data = body["data"]
            if not isinstance(data, list) or len(data) != len(texts):
                raise TypeError("embedding count mismatch")
            ordered: list[list[float] | None] = [None] * len(texts)
            for item in data:
                index = item["index"]
                vector = item["embedding"]
                if not isinstance(index, int) or not 0 <= index < len(texts) or ordered[index] is not None:
                    raise TypeError("invalid embedding index")
                if not isinstance(vector, list) or not vector:
                    raise TypeError("invalid embedding vector")
                converted = [float(value) for value in vector]
                if not all(math.isfinite(value) for value in converted):
                    raise TypeError("non-finite embedding")
                ordered[index] = converted
            vectors = [vector for vector in ordered if vector is not None]
            dimensions = {len(vector) for vector in vectors}
            if len(vectors) != len(texts) or len(dimensions) != 1:
                raise TypeError("embedding dimension mismatch")
            return EmbeddingBatch(
                vectors=vectors,
                model=str(body.get("model") or self.model),
                dimension=next(iter(dimensions)),
                usage=body.get("usage") if isinstance(body.get("usage"), dict) else None,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise EmbeddingResponseError("Embedding provider returned malformed data") from exc
