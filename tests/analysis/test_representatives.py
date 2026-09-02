from __future__ import annotations

from uuid import UUID, uuid4

import pytest

from analysis.clustering import RepresentativeInput, information_score, select_representatives


def _candidate(identifier: UUID, text: str, likes: int, vector):
    return RepresentativeInput(identifier, text, likes, vector)


def test_information_score_penalizes_emoji_and_repetition() -> None:
    assert information_score("这是一条包含明确产品价格信息的评论") > information_score("哈哈哈哈哈哈哈哈")
    assert information_score("这是一条正常评论") > information_score("😀😀😀")


def test_representative_scores_similarity_likes_and_information() -> None:
    ids = [uuid4() for _ in range(3)]
    result = select_representatives([
        _candidate(ids[0], "价格确实偏高，但产品功能完整", 100, [1, 0]),
        _candidate(ids[1], "贵", 0, [0.99, 0.01]),
        _candidate(ids[2], "质量不错而且售后响应很快", 10, [0.8, 0.2]),
    ], [1, 0], top_k=3)

    assert result[0].comment_id == ids[0]
    assert {item.rank for item in result} == {1, 2, 3}
    assert all(0 <= item.engagement_score <= 1 for item in result)


def test_equal_likes_have_zero_engagement_and_duplicates_are_filtered() -> None:
    ids = sorted([uuid4(), uuid4(), uuid4()], key=str)
    result = select_representatives([
        _candidate(ids[0], "相同正文", 5, [1, 0]),
        _candidate(ids[1], "相同正文", 5, [0.99, 0.01]),
        _candidate(ids[2], "不同但有信息的正文", 5, [0.8, 0.2]),
    ], [1, 0], top_k=3)

    assert len(result) == 2
    assert all(item.engagement_score == 0 for item in result)
    assert result[0].comment_id == ids[0]


def test_mmr_promotes_diversity_after_first_result() -> None:
    ids = [uuid4() for _ in range(3)]
    result = select_representatives([
        _candidate(ids[0], "价格高但是功能不错", 100, [1, 0]),
        _candidate(ids[1], "价格真的比较高", 90, [0.999, 0.001]),
        _candidate(ids[2], "售后服务响应及时", 80, [0.7, 0.7]),
    ], [1, 0], top_k=2, mmr_lambda=0.2)
    assert result[0].comment_id == ids[0]
    assert result[1].comment_id == ids[2]


def test_tie_break_and_top_k_validation() -> None:
    ids = sorted([uuid4(), uuid4()], key=str)
    result = select_representatives([
        _candidate(ids[1], "文本甲", 0, [1, 0]),
        _candidate(ids[0], "文本乙", 0, [1, 0]),
    ], [1, 0], top_k=1)
    assert result[0].comment_id == ids[0]
    with pytest.raises(ValueError):
        select_representatives([], [1, 0], top_k=11)
