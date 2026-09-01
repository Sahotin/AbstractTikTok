from __future__ import annotations

from uuid import uuid4

import pytest

from analysis.domain import (
    CommentAnalysis,
    Emotion,
    RiskLevel,
    Sentiment,
    Stance,
    TopicClusterConfig,
    TopicStageStatus,
)
from analysis.embeddings import EmbeddingConfig, FakeEmbeddingProvider
from analysis.preprocessing import compute_text_hash, normalize_text, utc_now
from analysis.services import EmbeddingService, SemanticRunService, TopicService


async def _prepared_run(phase1b_context, semantic_repository, *, include_low_quality=False):
    originals = (await phase1b_context.service.list_comments(limit=10)).items
    copies = [
        originals[0].model_copy(update={
            "id": uuid4(), "native_comment_id": "topic-copy-a", "text": "价格还是太高了",
            "text_hash": compute_text_hash("价格还是太高了"),
        }),
        originals[1].model_copy(update={
            "id": uuid4(), "native_comment_id": "topic-copy-b", "text": "质量和做工很好",
            "text_hash": compute_text_hash("质量和做工很好"),
        }),
    ]
    if include_low_quality:
        copies.append(originals[0].model_copy(update={
            "id": uuid4(), "native_comment_id": "topic-copy-excluded", "text": "666",
            "text_hash": compute_text_hash("666"),
        }))
    await phase1b_context.repository.upsert_comments(copies)
    comments = (await phase1b_context.service.list_comments(limit=10)).items
    original_comments = [comment for comment in comments if not comment.native_comment_id.startswith("topic-copy-")]
    vectors = {
        original_comments[0].text: [1, 0],
        original_comments[1].text: [0, 1],
        "价格还是太高了": [0.98, 0.02],
        "质量和做工很好": [0.02, 0.98],
        "666": [-1.0, 0.0],
    }
    for comment in comments:
        price = vectors[comment.text][0] > vectors[comment.text][1]
        await phase1b_context.repository.save_analysis(CommentAnalysis(
            id=uuid4(),
            comment_id=comment.id,
            sentiment=Sentiment.NEGATIVE if price else Sentiment.POSITIVE,
            sentiment_score=-0.8 if price else 0.8,
            emotion=Emotion.NEUTRAL,
            topics=["价格"] if price else ["质量"],
            stance=Stance.OPPOSE if price else Stance.SUPPORT,
            risk_level=RiskLevel.MEDIUM if price else RiskLevel.LOW,
            risk_reasons=[],
            keywords=["贵"] if price else ["做工"],
            summary="fixed metadata",
            model="fake-analysis",
            provider="fake",
            prompt_version="v1",
            analysis_version="v1",
            input_hash=compute_text_hash(normalize_text(comment.text)),
            created_at=utc_now(),
        ))
    embedding_config = EmbeddingConfig(
        provider="fake", model="topic-explicit", embedding_version="topic-v1",
        batch_size=10, max_concurrency=1,
    )
    provider = FakeEmbeddingProvider(vectors, dimension=2, model="topic-explicit")
    semantic_service = SemanticRunService(
        phase1b_context.repository,
        semantic_repository,
        EmbeddingService(semantic_repository, provider, embedding_config),
        embedding_config,
    )
    run = await semantic_service.create_run(platform="douyin", limit=len(comments))
    run, _ = await semantic_service.execute_run(run.id)
    return run, comments


@pytest.mark.asyncio
async def test_embedding_snapshot_to_topics_integration(phase1b_context, semantic_repository) -> None:
    run, comments = await _prepared_run(phase1b_context, semantic_repository)
    config = TopicClusterConfig(distance_threshold=0.2, min_cluster_size=2)
    completed, result = await TopicService(semantic_repository).cluster_run(run.id, config)

    assert completed.topic_status == TopicStageStatus.COMPLETED
    assert completed.status.value == "completed"
    assert len(result.clusters) == 2
    assert completed.topic_count == 2
    assert completed.noise_comments == 0
    assert sum(len(cluster.member_comment_ids) for cluster in result.clusters) == len(comments)
    topics = await semantic_repository.list_topic_clusters(semantic_run_id=str(run.id))
    assert {topic.name for topic in topics.items} == {"价格", "质量"}
    assert all(topic.analyzed_comments == 2 for topic in topics.items)
    assert {topic.dominant_sentiment.value for topic in topics.items} == {"positive", "negative"}


@pytest.mark.asyncio
async def test_same_run_rerun_is_idempotent(phase1b_context, semantic_repository) -> None:
    run, _ = await _prepared_run(phase1b_context, semantic_repository)
    service = TopicService(semantic_repository)
    config = TopicClusterConfig(distance_threshold=0.2, min_cluster_size=2)
    await service.cluster_run(run.id, config)
    first_topics = (await semantic_repository.list_topic_clusters(semantic_run_id=str(run.id))).items
    first_items = await semantic_repository.list_semantic_run_items(str(run.id))
    await service.cluster_run(run.id, config)
    second_topics = (await semantic_repository.list_topic_clusters(semantic_run_id=str(run.id))).items
    second_items = await semantic_repository.list_semantic_run_items(str(run.id))

    assert [(item.cluster_key, item.centroid) for item in first_topics] == [
        (item.cluster_key, item.centroid) for item in second_topics
    ]
    assert [
        (item.comment_id, item.topic_cluster_id, item.topic_representative_rank)
        for item in first_items
    ] == [
        (item.comment_id, item.topic_cluster_id, item.topic_representative_rank)
        for item in second_items
    ]


@pytest.mark.asyncio
async def test_quality_gate_persists_excluded_membership(phase1b_context, semantic_repository) -> None:
    run, comments = await _prepared_run(
        phase1b_context, semantic_repository, include_low_quality=True
    )
    completed, result = await TopicService(semantic_repository).cluster_run(
        run.id, TopicClusterConfig(distance_threshold=0.2, min_cluster_size=2)
    )

    excluded_comment = next(item for item in comments if item.text == "666")
    items = await semantic_repository.list_semantic_run_items(str(run.id))
    excluded_item = next(item for item in items if item.comment_id == excluded_comment.id)
    assert result.excluded_comments == {excluded_comment.id: "numeric_only"}
    assert excluded_item.status.value == "excluded"
    assert excluded_item.error_message == "semantic_quality:numeric_only"
    assert excluded_item.topic_cluster_id is None
    assert completed.clustered_comments + completed.noise_comments + 1 == len(comments)


@pytest.mark.asyncio
async def test_no_comment_analysis_produces_empty_metadata(phase1b_context, semantic_repository) -> None:
    comments = (await phase1b_context.service.list_comments(limit=10)).items
    config = EmbeddingConfig(provider="fake", model="no-analysis", embedding_version="v1")
    semantic = SemanticRunService(
        phase1b_context.repository,
        semantic_repository,
        EmbeddingService(semantic_repository, FakeEmbeddingProvider(dimension=2, model="no-analysis"), config),
        config,
    )
    run = await semantic.create_run(platform="douyin", limit=len(comments))
    await semantic.execute_run(run.id)
    await TopicService(semantic_repository).cluster_run(
        run.id, TopicClusterConfig(distance_threshold=2, min_cluster_size=1)
    )
    topic = (await semantic_repository.list_topic_clusters(semantic_run_id=str(run.id))).items[0]
    assert topic.top_topics == []
    assert topic.top_keywords == []
    assert topic.analyzed_comments == 0
    assert topic.dominant_sentiment is None


@pytest.mark.asyncio
async def test_missing_snapshot_items_are_not_regenerated(phase1b_context, semantic_repository) -> None:
    run, _ = await _prepared_run(phase1b_context, semantic_repository)
    items = await semantic_repository.list_semantic_run_items(str(run.id))
    await semantic_repository.replace_semantic_run_items(
        str(run.id), items[:-1], analysis_snapshot_hash=run.analysis_snapshot_hash, analyzed_comments=len(items)-1
    )
    with pytest.raises(ValueError, match="incomplete"):
        await TopicService(semantic_repository).cluster_run(run.id, TopicClusterConfig())
    failed = await semantic_repository.get_semantic_run(str(run.id))
    assert failed.topic_status == TopicStageStatus.FAILED
