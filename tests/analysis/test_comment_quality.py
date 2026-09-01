from analysis.preprocessing import SemanticExclusionReason, assess_semantic_quality


def test_semantic_quality_reasons_are_explainable() -> None:
    cases = {
        "": SemanticExclusionReason.EMPTY,
        "111": SemanticExclusionReason.NUMERIC_ONLY,
        "[玫瑰][比心]": SemanticExclusionReason.EMOJI_ONLY,
        "我": SemanticExclusionReason.TOO_SHORT,
        "哈哈哈": SemanticExclusionReason.REPETITIVE,
        "我在": SemanticExclusionReason.GENERIC_REPLY,
    }
    for text, expected in cases.items():
        decision = assess_semantic_quality(text)
        assert decision.eligible is False
        assert decision.reason == expected


def test_meaningful_short_chinese_and_technical_text_are_eligible() -> None:
    assert assess_semantic_quality("涨价").eligible is True
    assert assess_semantic_quality("Claude Code 登录失败").eligible is True


def test_quality_assessment_is_deterministic() -> None:
    text = "  价格虽然高，但是质量很好  "
    assert assess_semantic_quality(text) == assess_semantic_quality(text)
