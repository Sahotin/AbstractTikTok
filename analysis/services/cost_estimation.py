"""Explicitly approximate token and cost totals without claiming real usage."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Optional, Sequence

from analysis.domain import NormalizedComment


_CJK_RE = re.compile(r"[\u3400-\u4dbf\u4e00-\u9fff]")


def estimate_text_tokens(text: str) -> int:
    """Estimate mixed Chinese/Latin tokens more honestly than a flat chars/4 rule."""

    cjk_count = len(_CJK_RE.findall(text))
    other_count = sum(not character.isspace() and not _CJK_RE.match(character) for character in text)
    return max(1, cjk_count + math.ceil(other_count / 4))


@dataclass(frozen=True)
class CostEstimate:
    input_tokens: int
    output_tokens: int
    cost: Optional[float]


def estimate_comment_costs(
    comments: Sequence[NormalizedComment],
    *,
    output_tokens_per_comment: int,
    input_price_per_1m_tokens: Optional[float],
    output_price_per_1m_tokens: Optional[float],
    input_token_estimates: Optional[Sequence[int]] = None,
) -> CostEstimate:
    if input_token_estimates is not None and len(input_token_estimates) != len(comments):
        raise ValueError("input_token_estimates must match comments length")
    input_tokens = (
        sum(input_token_estimates)
        if input_token_estimates is not None
        else sum(estimate_text_tokens(comment.text) for comment in comments)
    )
    output_tokens = len(comments) * output_tokens_per_comment
    cost = None
    if input_price_per_1m_tokens is not None and output_price_per_1m_tokens is not None:
        cost = (
            input_tokens * input_price_per_1m_tokens
            + output_tokens * output_price_per_1m_tokens
        ) / 1_000_000
    return CostEstimate(input_tokens=input_tokens, output_tokens=output_tokens, cost=cost)
