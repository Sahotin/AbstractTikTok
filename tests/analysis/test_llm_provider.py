from __future__ import annotations

import json

import httpx
import pytest

from analysis.domain import CommentAnalysisPayload, Sentiment
from analysis.llm import (
    FakeLLMProvider,
    LLMConfig,
    OpenAICompatibleProvider,
    StructuredOutputError,
)
from analysis.llm.fake import DEFAULT_FAKE_OUTPUT


@pytest.mark.asyncio
async def test_fake_provider_returns_validated_structured_result() -> None:
    provider = FakeLLMProvider()

    result = await provider.generate_structured("private comment text", CommentAnalysisPayload)

    assert isinstance(result, CommentAnalysisPayload)
    assert result.sentiment == Sentiment.POSITIVE
    assert provider.call_count == 1


@pytest.mark.asyncio
async def test_openai_compatible_provider_parses_structured_response() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["Authorization"] == "Bearer test-secret"
        payload = json.loads(request.content)
        assert payload["response_format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": json.dumps(DEFAULT_FAKE_OUTPUT, ensure_ascii=False)}}],
                "usage": {
                    "prompt_tokens": 12,
                    "prompt_cache_hit_tokens": 4,
                    "prompt_cache_miss_tokens": 8,
                    "completion_tokens": 8,
                    "total_tokens": 20,
                },
            },
        )

    config = LLMConfig(
        provider="openai-compatible",
        model="test-model",
        base_url="https://llm.invalid/v1",
        api_key="test-secret",
    )
    provider = OpenAICompatibleProvider(config, transport=httpx.MockTransport(handler))

    result = await provider.generate_structured("comment", CommentAnalysisPayload)

    assert result.summary == DEFAULT_FAKE_OUTPUT["summary"]
    assert provider.usage_snapshot()["total_tokens"] == 20
    assert provider.usage_snapshot()["request_count"] == 1
    assert provider.usage_snapshot()["prompt_cache_hit_tokens"] == 4
    assert provider.usage_snapshot()["prompt_cache_miss_tokens"] == 8


@pytest.mark.asyncio
async def test_json_object_mode_sends_schema_in_prompt_and_validates_result() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["response_format"] == {"type": "json_object"}
        assert "JSON Schema" in payload["messages"][0]["content"]
        assert "sentiment_score" in payload["messages"][0]["content"]
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(DEFAULT_FAKE_OUTPUT)}}]},
        )

    config = LLMConfig(
        model="deepseek-test",
        base_url="https://llm.invalid/v1",
        api_key="test-secret",
        structured_output_mode="json_object",
    )
    result = await OpenAICompatibleProvider(
        config, transport=httpx.MockTransport(handler)
    ).generate_structured("return JSON", CommentAnalysisPayload)

    assert result.sentiment == Sentiment.POSITIVE


@pytest.mark.asyncio
async def test_openai_compatible_provider_rejects_invalid_output() -> None:
    async def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "not-json"}}]})

    config = LLMConfig(
        model="test-model",
        base_url="https://llm.invalid/v1",
        api_key="test-secret",
    )
    provider = OpenAICompatibleProvider(config, transport=httpx.MockTransport(handler))

    with pytest.raises(StructuredOutputError):
        await provider.generate_structured("comment", CommentAnalysisPayload)


def test_deepseek_environment_defaults_to_json_object(monkeypatch) -> None:
    monkeypatch.setenv("LLM_BASE_URL", "https://api.deepseek.com")
    monkeypatch.delenv("LLM_STRUCTURED_OUTPUT_MODE", raising=False)

    assert LLMConfig.from_environment().structured_output_mode == "json_object"


def test_deepseek_rejects_explicit_json_schema_mode() -> None:
    config = LLMConfig(
        model="deepseek-v4-flash",
        base_url="https://api.deepseek.com",
        api_key="test-secret",
        structured_output_mode="json_schema",
    )

    with pytest.raises(ValueError, match="requires LLM_STRUCTURED_OUTPUT_MODE=json_object"):
        OpenAICompatibleProvider(config)
