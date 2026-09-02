"""Deterministic, explainable semantic-eligibility checks for comments."""

from __future__ import annotations

import re
from enum import Enum

from pydantic import BaseModel, Field

from .text import normalize_text


class SemanticExclusionReason(str, Enum):
    EMPTY = "empty"
    NUMERIC_ONLY = "numeric_only"
    EMOJI_ONLY = "emoji_only"
    TOO_SHORT = "too_short"
    REPETITIVE = "repetitive"
    GENERIC_REPLY = "generic_reply"


class SemanticQualityDecision(BaseModel):
    eligible: bool
    reason: SemanticExclusionReason | None = None
    informative_characters: int = Field(ge=0)


_BRACKET_EMOJI_RE = re.compile(r"\[[^\]\r\n]{1,16}\]")
_INFORMATIVE_RE = re.compile(r"[A-Za-z0-9\u3400-\u4dbf\u4e00-\u9fff]")
_DIGIT_OR_SEPARATOR_RE = re.compile(r"^[\d\s.,，。!！?？:：;；+\-_/\\]+$")
_GENERIC_REPLIES = {
    "好的", "好吧", "可以", "可以的", "是的", "不是", "来了", "来吧", "我在",
    "在吗", "有吗", "没有", "真的吗", "真的", "支持", "收到", "知道了", "明白了",
}


def assess_semantic_quality(
    text: str,
    *,
    min_informative_characters: int = 2,
) -> SemanticQualityDecision:
    """Classify low-information text without mutating or rewriting it."""

    if min_informative_characters < 1:
        raise ValueError("min_informative_characters must be positive")
    raw_text = str(text or "").strip()
    if raw_text and _DIGIT_OR_SEPARATOR_RE.fullmatch(raw_text):
        informative = _INFORMATIVE_RE.findall(raw_text)
        return SemanticQualityDecision(
            eligible=False,
            reason=SemanticExclusionReason.NUMERIC_ONLY,
            informative_characters=len(informative),
        )
    normalized = normalize_text(text)
    if not normalized:
        return SemanticQualityDecision(
            eligible=False,
            reason=SemanticExclusionReason.EMPTY,
            informative_characters=0,
        )
    without_bracket_emoji = _BRACKET_EMOJI_RE.sub("", normalized).strip()
    informative = _INFORMATIVE_RE.findall(without_bracket_emoji)
    if not informative:
        return SemanticQualityDecision(
            eligible=False,
            reason=SemanticExclusionReason.EMOJI_ONLY,
            informative_characters=0,
        )
    if _DIGIT_OR_SEPARATOR_RE.fullmatch(without_bracket_emoji):
        return SemanticQualityDecision(
            eligible=False,
            reason=SemanticExclusionReason.NUMERIC_ONLY,
            informative_characters=len(informative),
        )
    if len(informative) < min_informative_characters:
        return SemanticQualityDecision(
            eligible=False,
            reason=SemanticExclusionReason.TOO_SHORT,
            informative_characters=len(informative),
        )
    folded = without_bracket_emoji.casefold()
    if folded in _GENERIC_REPLIES:
        return SemanticQualityDecision(
            eligible=False,
            reason=SemanticExclusionReason.GENERIC_REPLY,
            informative_characters=len(informative),
        )
    if len(set(character.casefold() for character in informative)) == 1 and len(informative) <= 8:
        return SemanticQualityDecision(
            eligible=False,
            reason=SemanticExclusionReason.REPETITIVE,
            informative_characters=len(informative),
        )
    return SemanticQualityDecision(eligible=True, informative_characters=len(informative))
