"""Portable little-endian float32 vector encoding for SQLite BLOB storage."""

from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import EmbeddingResponseError


FLOAT32_LE = "float32-le"


def encode_vector(vector: Sequence[float]) -> bytes:
    array = np.asarray(vector, dtype="<f4")
    if array.ndim != 1 or array.size == 0:
        raise EmbeddingResponseError("Embedding vector must be a non-empty one-dimensional array")
    if not np.isfinite(array).all():
        raise EmbeddingResponseError("Embedding vector contains NaN or Infinity")
    return array.tobytes(order="C")


def decode_vector(blob: bytes, *, dimension: int, dtype: str) -> list[float]:
    if dtype != FLOAT32_LE:
        raise EmbeddingResponseError(f"Unsupported embedding dtype: {dtype}")
    if dimension <= 0:
        raise EmbeddingResponseError("Embedding dimension must be positive")
    expected_length = dimension * np.dtype("<f4").itemsize
    if len(blob) != expected_length:
        raise EmbeddingResponseError(
            f"Embedding byte length mismatch: expected {expected_length}, got {len(blob)}"
        )
    array = np.frombuffer(blob, dtype="<f4")
    if not np.isfinite(array).all():
        raise EmbeddingResponseError("Stored embedding contains NaN or Infinity")
    return array.astype(float).tolist()


def normalize_vector(vector: Sequence[float]) -> list[float]:
    array = np.asarray(vector, dtype="<f4")
    if array.ndim != 1 or array.size == 0 or not np.isfinite(array).all():
        raise EmbeddingResponseError("Embedding vector is empty or non-finite")
    norm = float(np.linalg.norm(array))
    if norm == 0:
        raise EmbeddingResponseError("Embedding vector has zero norm")
    return (array / norm).astype("<f4").astype(float).tolist()
