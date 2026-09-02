from __future__ import annotations

from uuid import NAMESPACE_URL, uuid5

import pytest

from analysis.domain import CommentAnalysis, CommentAnalysisPayload
from analysis.llm.fake import DEFAULT_FAKE_OUTPUT
from analysis.preprocessing import compute_text_hash, utc_now


def _analysis(comment_id, *, prompt_version: str = "comment_analysis_v1") -> CommentAnalysis:
    payload = CommentAnalysisPayload.model_validate(DEFAULT_FAKE_OUTPUT)
    input_hash = compute_text_hash("测试评论")
    identity = f"{comment_id}|{input_hash}|fake|fake-model-v1|{prompt_version}|schema-v1"
    return CommentAnalysis(
        id=uuid5(NAMESPACE_URL, identity),
        comment_id=comment_id,
        **payload.model_dump(mode="python"),
        model="fake-model-v1",
        provider="fake",
        prompt_version=prompt_version,
        analysis_version="schema-v1",
        input_hash=input_hash,
        created_at=utc_now(),
    )


@pytest.mark.asyncio
async def test_repository_saves_and_reads_exact_cache_key(phase1b_context) -> None:
    comments = await phase1b_context.service.list_comments(limit=1)
    analysis = _analysis(comments.items[0].id)

    stored = await phase1b_context.repository.save_analysis(analysis)
    cached = await phase1b_context.repository.get_cached_analysis(
        comment_id=str(analysis.comment_id),
        input_hash=analysis.input_hash,
        provider=analysis.provider,
        model=analysis.model,
        prompt_version=analysis.prompt_version,
        analysis_version=analysis.analysis_version,
    )

    assert stored == cached
    assert cached is not None
    assert cached.created_at.tzinfo is not None
    assert await phase1b_context.repository.count_comment_analyses() == 1


@pytest.mark.asyncio
async def test_repository_allows_prompt_versions_to_coexist(phase1b_context) -> None:
    comments = await phase1b_context.service.list_comments(limit=1)
    first = _analysis(comments.items[0].id, prompt_version="comment_analysis_v1")
    second = _analysis(comments.items[0].id, prompt_version="comment_analysis_v2")

    await phase1b_context.repository.save_analysis(first)
    await phase1b_context.repository.save_analysis(second)

    assert await phase1b_context.repository.count_comment_analyses(comment_id=str(first.comment_id)) == 2


@pytest.mark.asyncio
async def test_repository_duplicate_cache_save_is_idempotent(phase1b_context) -> None:
    comments = await phase1b_context.service.list_comments(limit=1)
    analysis = _analysis(comments.items[0].id)

    await phase1b_context.repository.save_analysis(analysis)
    await phase1b_context.repository.save_analysis(analysis)

    assert await phase1b_context.repository.count_comment_analyses() == 1
