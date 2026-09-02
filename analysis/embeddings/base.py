"""Provider-neutral batch embedding contracts and safe error types."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional, Sequence

from pydantic import BaseModel, Field


class EmbeddingProviderError(Exception):
    """A permanent provider or response failure."""


class TransientEmbeddingError(EmbeddingProviderError):
    """A retryable network, timeout, rate-limit, or server failure."""


class EmbeddingResponseError(EmbeddingProviderError):
    """The provider returned malformed or unsafe vector data."""


class EmbeddingBatch(BaseModel):
    vectors: list[list[float]]
    model: str
    dimension: int = Field(gt=0)
    usage: Optional[dict[str, Any]] = None


class EmbeddingProvider(ABC):
    @property
    @abstractmethod
    def provider_name(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def model(self) -> str:
        raise NotImplementedError

    @abstractmethod
    async def embed_texts(self, texts: Sequence[str]) -> EmbeddingBatch:
        raise NotImplementedError
