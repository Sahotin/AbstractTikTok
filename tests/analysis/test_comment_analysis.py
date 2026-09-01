from __future__ import annotations

import asyncio
import logging
from uuid import uuid4

import httpx
import pytest
from pydantic import ValidationError

from analysis.domain import CommentAnalysisPayload
from analysis.llm import FakeLLMProvider, LLMConfig
from analysis.llm.fake import DEFAULT_FAKE_OUTPUT
from analysis.services import CommentAnalysisError, CommentAnalysisService
from api.main import app
from api.routers.analysis import get_analysis_query_service, get_comment_analysis_service


def _config(**updates) -> LLMConfig:
    values = {
        "provider": "fake",
        "model": "fake-model-v1",
        "timeout": 0.2,
        "max_retries": 2,
        "retry_base_delay": 0,
    }
    values.update(updates)
    return LLMConfig(**values)


async def _one_comment(context):
    return (await context.service.list_comments(limit=1)).items[0]


def test_comment_analysis_payload_accepts_valid_structure() -> None:
    result = CommentAnalysisPayload.model_validate(DEFAULT_FAKE_OUTPUT)
    assert result.sentiment_score == 0.8


@pytest.mark.asyncio
async def test_input_estimate_includes_prompt_and_schema(phase1b_context) -> None:
    provider = FakeLLMProvider()
    service = CommentAnalysisService(phase1b_context.repository, provider, _config())
    comment = await _one_comment(phase1b_context)

    assert service.estimate_request_input_tokens(comment) > len(comment.text)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("sentiment", "mixed"),
        ("sentiment_score", 1.1),
        ("risk_level", "critical"),
    ],
)
def test_comment_analysis_payload_rejects_invalid_values(field, value) -> None:
    payload = dict(DEFAULT_FAKE_OUTPUT)
    payload[field] = value
    with pytest.raises(ValidationError):
        CommentAnalysisPayload.model_validate(payload)


@pytest.mark.asyncio
async def test_first_call_misses_cache_and_second_call_hits(phase1b_context) -> None:
    provider = FakeLLMProvider()
    service = CommentAnalysisService(phase1b_context.repository, provider, _config())
    comment = await _one_comment(phase1b_context)

    first = await service.analyze_comment(comment)
    second = await service.analyze_comment(comment)

    assert first == second
    assert provider.call_count == 1
    assert await phase1b_context.repository.count_comment_analyses() == 1


@pytest.mark.asyncio
async def test_prompt_version_change_causes_cache_miss(phase1b_context) -> None:
    provider = FakeLLMProvider()
    comment = await _one_comment(phase1b_context)
    first_service = CommentAnalysisService(
        phase1b_context.repository,
        provider,
        _config(),
        prompt_version="comment_analysis_v1",
    )
    second_service = CommentAnalysisService(
        phase1b_context.repository,
        provider,
        _config(),
        prompt_version="comment_analysis_v2",
    )

    await first_service.analyze_comment(comment)
    await second_service.analyze_comment(comment)

    assert provider.call_count == 2
    assert await phase1b_context.repository.count_comment_analyses() == 2


@pytest.mark.asyncio
async def test_model_change_causes_cache_miss(phase1b_context) -> None:
    comment = await _one_comment(phase1b_context)
    first_provider = FakeLLMProvider(model="fake-model-v1")
    second_provider = FakeLLMProvider(model="fake-model-v2")

    await CommentAnalysisService(phase1b_context.repository, first_provider, _config()).analyze_comment(comment)
    await CommentAnalysisService(phase1b_context.repository, second_provider, _config()).analyze_comment(comment)

    assert first_provider.call_count == 1
    assert second_provider.call_count == 1
    assert await phase1b_context.repository.count_comment_analyses() == 2


@pytest.mark.asyncio
async def test_timeout_is_retried_then_succeeds(phase1b_context) -> None:
    provider = FakeLLMProvider(delays=[0.05, 0])
    service = CommentAnalysisService(
        phase1b_context.repository,
        provider,
        _config(timeout=0.01, max_retries=1),
    )

    result = await service.analyze_comment(await _one_comment(phase1b_context))

    assert result.sentiment.value == "positive"
    assert provider.call_count == 2


@pytest.mark.asyncio
async def test_retry_limit_raises_failure(phase1b_context) -> None:
    provider = FakeLLMProvider(delays=[0.05, 0.05, 0.05])
    service = CommentAnalysisService(
        phase1b_context.repository,
        provider,
        _config(timeout=0.01, max_retries=2),
    )

    with pytest.raises(CommentAnalysisError):
        await service.analyze_comment(await _one_comment(phase1b_context))

    assert provider.call_count == 3
    assert await phase1b_context.repository.count_comment_analyses() == 0


@pytest.mark.asyncio
async def test_invalid_structured_output_is_retried_and_rejected(phase1b_context) -> None:
    invalid_output = dict(DEFAULT_FAKE_OUTPUT)
    invalid_output["sentiment"] = "mixed"
    provider = FakeLLMProvider(output=invalid_output)
    service = CommentAnalysisService(
        phase1b_context.repository,
        provider,
        _config(max_retries=1),
    )

    with pytest.raises(CommentAnalysisError):
        await service.analyze_comment(await _one_comment(phase1b_context))

    assert provider.call_count == 2
    assert await phase1b_context.repository.count_comment_analyses() == 0


@pytest.mark.asyncio
async def test_api_key_and_comment_text_are_not_logged(phase1b_context, caplog) -> None:
    secret = "do-not-log-this-secret"
    comment = await _one_comment(phase1b_context)
    provider = FakeLLMProvider()
    service = CommentAnalysisService(
        phase1b_context.repository,
        provider,
        _config(api_key=secret),
    )

    with caplog.at_level(logging.INFO, logger="analysis.comment_analysis"):
        await service.analyze_comment(comment)

    logs = caplog.text
    assert secret not in logs
    assert comment.text not in logs
    assert str(comment.id) in logs


@pytest.mark.asyncio
async def test_ten_comments_pass_through_sequential_pipeline(phase1b_context) -> None:
    source = await _one_comment(phase1b_context)
    comments = [
        source.model_copy(update={"id": uuid4(), "native_comment_id": f"phase2-comment-{index}"})
        for index in range(10)
    ]
    await phase1b_context.repository.upsert_comments(comments)
    provider = FakeLLMProvider()
    service = CommentAnalysisService(phase1b_context.repository, provider, _config())

    results = await service.analyze_comments(comments)

    assert len(results) == 10
    assert provider.call_count == 1
    assert await phase1b_context.repository.count_comment_analyses() == 10


@pytest.mark.asyncio
async def test_identical_text_across_comments_is_reused_concurrently(phase1b_context) -> None:
    source = await _one_comment(phase1b_context)
    comments = [
        source.model_copy(update={"id": uuid4(), "native_comment_id": f"duplicate-text-{index}"})
        for index in range(5)
    ]
    await phase1b_context.repository.upsert_comments(comments)
    provider = FakeLLMProvider(delays=[0.03])
    service = CommentAnalysisService(phase1b_context.repository, provider, _config())

    results = await asyncio.gather(*(service.analyze_comment(comment) for comment in comments))

    assert provider.call_count == 1
    assert len({result.id for result in results}) == 5
    assert {result.comment_id for result in results} == {comment.id for comment in comments}
    assert await phase1b_context.repository.count_comment_analyses() == 5


@pytest.mark.asyncio
async def test_single_comment_analysis_api_uses_service(phase1b_context) -> None:
    comment = await _one_comment(phase1b_context)
    provider = FakeLLMProvider()
    analysis_service = CommentAnalysisService(phase1b_context.repository, provider, _config())

    async def override_query_service():
        yield phase1b_context.service

    async def override_analysis_service():
        yield analysis_service

    app.dependency_overrides[get_analysis_query_service] = override_query_service
    app.dependency_overrides[get_comment_analysis_service] = override_analysis_service
    try:
        async with httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://test") as client:
            response = await client.post(f"/api/analysis/comments/{comment.id}/analyze")
    finally:
        app.dependency_overrides.clear()

    assert response.status_code == 200
    assert response.json()["comment_id"] == str(comment.id)
    assert response.json()["prompt_version"] == "comment_analysis_v1"
    assert provider.call_count == 1
