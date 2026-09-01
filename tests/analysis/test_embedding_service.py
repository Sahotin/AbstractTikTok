from __future__ import annotations

import pytest

from analysis.embeddings import EmbeddingConfig, FakeEmbeddingProvider
from analysis.services import EmbeddingService


def _config(**updates) -> EmbeddingConfig:
    values = {"provider": "fake", "model": "fake-model", "embedding_version": "v1", "batch_size": 10, "max_concurrency": 2}
    values.update(updates)
    return EmbeddingConfig(**values)


def _unique(comments):
    return [comment.model_copy(update={"text": f"{comment.text} #{index}"}) for index, comment in enumerate(comments)]


@pytest.mark.asyncio
@pytest.mark.parametrize("count", [1, 10, 50])
async def test_service_embeds_bounded_batches(semantic_repository, job_comments, count) -> None:
    comments = _unique(job_comments[:count])
    provider = FakeEmbeddingProvider(dimension=8, model="fake-model")
    report = await EmbeddingService(semantic_repository, provider, _config()).embed_comments(comments)

    assert report.embedded_comments == count
    assert report.failed_comments == 0
    assert provider.batch_sizes and max(provider.batch_sizes) <= 10
    assert await semantic_repository.count_embeddings() == count


@pytest.mark.asyncio
async def test_exact_cache_hit_avoids_provider(semantic_repository, job_comments) -> None:
    comments = _unique(job_comments[:10])
    provider = FakeEmbeddingProvider(dimension=8, model="fake-model")
    service = EmbeddingService(semantic_repository, provider, _config())
    first = await service.embed_comments(comments)
    calls_after_first = provider.call_count
    second = await service.embed_comments(comments)

    assert first.provider_inputs == 10
    assert second.exact_cache_hits == 10
    assert second.provider_inputs == 0
    assert provider.call_count == calls_after_first


@pytest.mark.asyncio
async def test_duplicate_text_is_requested_once_and_reused(semantic_repository, job_comments) -> None:
    comments = [comment.model_copy(update={"text": "重复正文"}) for comment in job_comments[:10]]
    provider = FakeEmbeddingProvider(dimension=8, model="fake-model")
    report = await EmbeddingService(semantic_repository, provider, _config()).embed_comments(comments)

    assert report.embedded_comments == 10
    assert report.provider_inputs == 1
    assert provider.input_count == 1


@pytest.mark.asyncio
async def test_cross_comment_content_cache_reuse(semantic_repository, job_comments) -> None:
    source = job_comments[0].model_copy(update={"text": "可复用正文"})
    target = job_comments[1].model_copy(update={"text": "可复用正文"})
    provider = FakeEmbeddingProvider(dimension=8, model="fake-model")
    service = EmbeddingService(semantic_repository, provider, _config())
    await service.embed_comments([source])
    report = await service.embed_comments([target])

    assert report.content_cache_hits == 1
    assert report.provider_inputs == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("provider_name", "model", "version"),
    [("other", "fake-model", "v1"), ("fake", "other-model", "v1"), ("fake", "fake-model", "v2")],
)
async def test_cache_miss_when_identity_changes(semantic_repository, job_comments, provider_name, model, version) -> None:
    first_provider = FakeEmbeddingProvider(dimension=8, model="fake-model")
    await EmbeddingService(semantic_repository, first_provider, _config()).embed_comments(job_comments[:1])
    changed = FakeEmbeddingProvider(dimension=8, model=model, provider_name=provider_name)
    report = await EmbeddingService(
        semantic_repository,
        changed,
        _config(provider=provider_name, model=model, embedding_version=version),
    ).embed_comments(job_comments[:1])

    assert report.provider_inputs == 1
    assert report.exact_cache_hits == 0


@pytest.mark.asyncio
async def test_batch_failure_isolated_to_one_text(semantic_repository, job_comments) -> None:
    comments = _unique(job_comments[:50])
    failing_text = comments[17].text
    provider = FakeEmbeddingProvider(dimension=8, model="fake-model", failure_texts={failing_text})
    report = await EmbeddingService(
        semantic_repository,
        provider,
        _config(batch_size=50),
    ).embed_comments(comments)

    assert report.embedded_comments == 49
    assert report.failed_comments == 1
    assert report.failures[0].comment_id == comments[17].id
    assert provider.call_count > 1


@pytest.mark.asyncio
async def test_fake_results_are_deterministic_across_runs(semantic_repository, job_comments) -> None:
    one = FakeEmbeddingProvider(dimension=8, model="fake-model")
    two = FakeEmbeddingProvider(dimension=8, model="fake-model")
    assert (await one.embed_texts([job_comments[0].text])).vectors == (await two.embed_texts([job_comments[0].text])).vectors
