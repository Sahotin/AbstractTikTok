from __future__ import annotations

from uuid import uuid4

import pytest


@pytest.mark.asyncio
async def test_get_content_by_internal_id(phase1b_context) -> None:
    page = await phase1b_context.service.list_contents(limit=10)
    content = await phase1b_context.service.get_content(page.items[0].id)

    assert content is not None
    assert content.native_content_id == "7634004863847910656"


@pytest.mark.asyncio
async def test_list_contents_supports_native_id_and_run_filters(phase1b_context) -> None:
    page = await phase1b_context.service.list_contents(
        platform="douyin",
        native_content_id="7634004863847910656",
        run_id=phase1b_context.run_id,
    )

    assert page.total == 1
    assert len(page.items) == 1


@pytest.mark.asyncio
async def test_list_comments_by_content(phase1b_context) -> None:
    contents = await phase1b_context.service.list_contents(limit=1)
    page = await phase1b_context.service.list_comments(content_id=contents.items[0].id)
    comment = await phase1b_context.service.get_comment(page.items[0].id)

    assert page.total == 2
    assert {comment.depth for comment in page.items} == {0, 1}
    assert comment is not None
    assert comment.content_id == contents.items[0].id


@pytest.mark.asyncio
async def test_list_comments_filters_root_and_reply_depth(phase1b_context) -> None:
    roots = await phase1b_context.service.list_comments(depth=0)
    replies = await phase1b_context.service.list_comments(depth=1)

    assert roots.total == 1
    assert replies.total == 1
    assert roots.items[0].parent_comment_id is None
    assert replies.items[0].parent_comment_id == roots.items[0].native_comment_id


@pytest.mark.asyncio
async def test_get_and_list_author(phase1b_context) -> None:
    authors = await phase1b_context.service.list_authors(platform="douyin", limit=10)
    author = await phase1b_context.service.get_author(authors.items[0].id)

    assert authors.total == 3
    assert author is not None
    assert author.native_author_id == authors.items[0].native_author_id


@pytest.mark.asyncio
async def test_comment_statistics_are_computed(phase1b_context) -> None:
    statistics = await phase1b_context.service.get_comment_statistics(run_id=phase1b_context.run_id)

    assert statistics.comment_count == 2
    assert statistics.root_comment_count == 1
    assert statistics.reply_comment_count == 1
    assert statistics.earliest_comment_time is not None
    assert statistics.latest_comment_time is not None


@pytest.mark.asyncio
async def test_content_and_author_statistics_are_computed(phase1b_context) -> None:
    contents = await phase1b_context.service.get_content_statistics(platform="douyin")
    authors = await phase1b_context.service.get_author_statistics(platform="douyin")

    assert contents.content_count == 1
    assert authors.author_count == 3


@pytest.mark.asyncio
async def test_limit_and_offset_are_applied_in_query(phase1b_context) -> None:
    first = await phase1b_context.service.list_comments(limit=1, offset=0)
    second = await phase1b_context.service.list_comments(limit=1, offset=1)

    assert first.total == second.total == 2
    assert len(first.items) == len(second.items) == 1
    assert first.items[0].id != second.items[0].id


@pytest.mark.asyncio
async def test_missing_entities_return_none_or_empty_page(phase1b_context) -> None:
    missing = await phase1b_context.service.get_content(uuid4())
    page = await phase1b_context.service.list_comments(native_content_id="missing")

    assert missing is None
    assert page.total == 0
    assert page.items == []
