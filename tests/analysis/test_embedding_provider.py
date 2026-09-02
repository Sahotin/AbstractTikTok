from __future__ import annotations

import json

import httpx
import pytest

from analysis.embeddings import (
    EmbeddingConfig,
    EmbeddingProviderError,
    EmbeddingResponseError,
    FakeEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
    TransientEmbeddingError,
)


@pytest.mark.asyncio
async def test_fake_provider_is_deterministic_and_preserves_order() -> None:
    provider = FakeEmbeddingProvider({"价格太贵": [1, 0, 0]}, dimension=3)

    first = await provider.embed_texts(["未知文本", "价格太贵"])
    second = await provider.embed_texts(["未知文本", "价格太贵"])

    assert first.vectors == second.vectors
    assert first.vectors[1] == [1.0, 0.0, 0.0]
    assert first.dimension == 3


def _config(**updates) -> EmbeddingConfig:
    values = {
        "provider": "openai-compatible",
        "model": "embed-test",
        "base_url": "https://embedding.invalid/v1",
        "api_key": "secret-value",
        "max_retries": 2,
        "retry_base_delay": 0,
    }
    values.update(updates)
    return EmbeddingConfig(**values)


@pytest.mark.asyncio
async def test_openai_provider_parses_and_reorders_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer secret-value"
        return httpx.Response(200, json={
            "model": "embed-test",
            "data": [
                {"index": 1, "embedding": [0, 1]},
                {"index": 0, "embedding": [1, 0]},
            ],
            "usage": {"total_tokens": 2},
        })

    provider = OpenAICompatibleEmbeddingProvider(_config(), transport=httpx.MockTransport(handler))
    result = await provider.embed_texts(["a", "b"])

    assert result.vectors == [[1.0, 0.0], [0.0, 1.0]]
    assert result.usage == {"total_tokens": 2}


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "connection", 429, 500])
async def test_openai_provider_retries_transient_failures(failure) -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            if failure == "timeout":
                raise httpx.ReadTimeout("timeout", request=request)
            if failure == "connection":
                raise httpx.ConnectError("connection failed", request=request)
            return httpx.Response(failure, request=request)
        return httpx.Response(200, json={"data": [{"index": 0, "embedding": [1, 0]}]})

    provider = OpenAICompatibleEmbeddingProvider(_config(), transport=httpx.MockTransport(handler))
    result = await provider.embed_texts(["a"])

    assert attempts == 3
    assert result.dimension == 2


@pytest.mark.asyncio
async def test_openai_provider_stops_after_bounded_retries() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(503, request=request)

    provider = OpenAICompatibleEmbeddingProvider(_config(max_retries=1), transport=httpx.MockTransport(handler))
    with pytest.raises(TransientEmbeddingError):
        await provider.embed_texts(["a"])
    assert attempts == 2


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {},
        {"data": [{"index": 0, "embedding": [1, 0]}]},
        {"data": [{"index": 0, "embedding": [1, 0]}, {"index": 1, "embedding": [1]}]},
        {"data": [{"index": 0, "embedding": [float("nan"), 0]}, {"index": 1, "embedding": [1, 0]}]},
        {"data": [{"index": 0, "embedding": [float("inf"), 0]}, {"index": 1, "embedding": [1, 0]}]},
    ],
)
async def test_openai_provider_rejects_malformed_vectors(body) -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=json.dumps(body).encode("utf-8"))

    provider = OpenAICompatibleEmbeddingProvider(_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(EmbeddingResponseError):
        await provider.embed_texts(["a", "b"])


@pytest.mark.asyncio
async def test_openai_provider_rejects_malformed_json() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not-json")

    provider = OpenAICompatibleEmbeddingProvider(_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(EmbeddingResponseError):
        await provider.embed_texts(["a"])


@pytest.mark.asyncio
async def test_openai_provider_does_not_retry_permanent_4xx() -> None:
    attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal attempts
        attempts += 1
        return httpx.Response(400, request=request)

    provider = OpenAICompatibleEmbeddingProvider(_config(), transport=httpx.MockTransport(handler))
    with pytest.raises(EmbeddingProviderError):
        await provider.embed_texts(["a"])
    assert attempts == 1


def test_embedding_config_does_not_expose_secret() -> None:
    config = _config()
    assert "secret-value" not in repr(config)
    assert "api_key" not in config.safe_snapshot()
