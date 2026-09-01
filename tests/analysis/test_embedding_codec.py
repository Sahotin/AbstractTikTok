from __future__ import annotations

import struct

import pytest

from analysis.embeddings import EmbeddingResponseError, decode_vector, encode_vector


def test_float32_little_endian_round_trip() -> None:
    vector = [1.25, -2.5, 0.125]
    blob = encode_vector(vector)

    assert blob == struct.pack("<fff", *vector)
    assert decode_vector(blob, dimension=3, dtype="float32-le") == pytest.approx(vector)


@pytest.mark.parametrize(
    ("blob", "dimension", "dtype"),
    [
        (b"\x00" * 8, 0, "float32-le"),
        (b"\x00" * 7, 2, "float32-le"),
        (b"\x00" * 8, 2, "float64"),
        (struct.pack("<ff", float("nan"), 1.0), 2, "float32-le"),
        (struct.pack("<ff", float("inf"), 1.0), 2, "float32-le"),
    ],
)
def test_decode_rejects_invalid_metadata_or_data(blob: bytes, dimension: int, dtype: str) -> None:
    with pytest.raises(EmbeddingResponseError):
        decode_vector(blob, dimension=dimension, dtype=dtype)


@pytest.mark.parametrize("vector", [[], [[1.0]], [1.0, float("nan")], [1.0, float("inf")]])
def test_encode_rejects_invalid_vectors(vector) -> None:
    with pytest.raises(EmbeddingResponseError):
        encode_vector(vector)
