"""Embedding provider abstractions and vector storage helpers."""

from .base import (
    EmbeddingBatch,
    EmbeddingProvider,
    EmbeddingProviderError,
    EmbeddingResponseError,
    TransientEmbeddingError,
)
from .codec import FLOAT32_LE, decode_vector, encode_vector, normalize_vector
from .config import EmbeddingConfig
from .fake import FakeEmbeddingProvider
from .openai_compatible import OpenAICompatibleEmbeddingProvider


def create_embedding_provider(config: EmbeddingConfig) -> EmbeddingProvider:
    provider = config.provider.lower()
    if provider == "fake":
        return FakeEmbeddingProvider(dimension=config.fake_dimension, model=config.model or "fake-embedding-v1")
    if provider in {"openai", "openai-compatible", "openai_compatible"}:
        return OpenAICompatibleEmbeddingProvider(config)
    raise ValueError(f"Unsupported embedding provider: {config.provider}")


__all__ = [
    "EmbeddingBatch",
    "EmbeddingConfig",
    "EmbeddingProvider",
    "EmbeddingProviderError",
    "EmbeddingResponseError",
    "FakeEmbeddingProvider",
    "FLOAT32_LE",
    "OpenAICompatibleEmbeddingProvider",
    "TransientEmbeddingError",
    "create_embedding_provider",
    "decode_vector",
    "encode_vector",
    "normalize_vector",
]
