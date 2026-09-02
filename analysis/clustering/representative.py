"""Deterministic relevance/MMR representative-comment selection."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Sequence
from uuid import UUID

import numpy as np

from analysis.domain import RepresentativeCommentScore
from analysis.preprocessing import prepare_analysis_text


_INFORMATIVE_CHARACTER = re.compile(r"[\w\u3400-\u9fff]", re.UNICODE)


@dataclass(frozen=True)
class RepresentativeInput:
    comment_id: UUID
    text: str
    like_count: int | None
    vector: Sequence[float]


def information_score(text: str) -> float:
    normalized = prepare_analysis_text(text)
    if not normalized:
        return 0.0
    characters = [character for character in normalized if not character.isspace()]
    informative = [character for character in characters if _INFORMATIVE_CHARACTER.match(character)]
    if not informative:
        return 0.0
    effective_length = min(len(informative), 80) / 80.0
    valid_ratio = len(informative) / max(len(characters), 1)
    max_repetition = max(characters.count(character) for character in set(characters)) / len(characters)
    repetition_penalty = max(0.0, (max_repetition - 0.35) / 0.65)
    score = (0.55 * effective_length + 0.45 * valid_ratio) * (1.0 - repetition_penalty)
    return float(min(1.0, max(0.0, score)))


def select_representatives(
    candidates: Sequence[RepresentativeInput],
    centroid: Sequence[float],
    *,
    top_k: int = 3,
    mmr_lambda: float = 0.75,
) -> list[RepresentativeCommentScore]:
    if not 1 <= top_k <= 10:
        raise ValueError("Representative top_k must be between 1 and 10")
    if not 0.0 <= mmr_lambda <= 1.0:
        raise ValueError("MMR lambda must be between 0 and 1")
    if not candidates:
        return []

    ordered = sorted(candidates, key=lambda candidate: str(candidate.comment_id))
    dimensions = {len(candidate.vector) for candidate in ordered} | {len(centroid)}
    if len(dimensions) != 1:
        raise ValueError("Representative vector dimensions do not match")
    matrix = np.asarray([candidate.vector for candidate in ordered], dtype="<f4")
    centroid_array = np.asarray(centroid, dtype="<f4")
    if not np.isfinite(matrix).all() or not np.isfinite(centroid_array).all():
        raise ValueError("Representative vectors contain NaN or Infinity")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    centroid_norm = float(np.linalg.norm(centroid_array))
    if np.any(norms == 0) or centroid_norm == 0:
        raise ValueError("Representative vectors must have non-zero norm")
    matrix = matrix / norms
    centroid_array = centroid_array / centroid_norm
    similarities = matrix @ centroid_array

    logs = np.asarray([math.log1p(max(candidate.like_count or 0, 0)) for candidate in ordered])
    spread = float(logs.max() - logs.min())
    engagement = np.zeros_like(logs) if spread == 0 else (logs - logs.min()) / spread
    information = np.asarray([information_score(candidate.text) for candidate in ordered])
    relevance = 0.65 * similarities + 0.20 * engagement + 0.15 * information

    deduplicated: list[int] = []
    seen_text: set[str] = set()
    for index, candidate in enumerate(ordered):
        key = prepare_analysis_text(candidate.text).casefold()
        if key in seen_text:
            continue
        seen_text.add(key)
        deduplicated.append(index)

    selected: list[int] = []
    remaining = set(deduplicated)
    while remaining and len(selected) < min(top_k, len(deduplicated)):
        scored = []
        for index in remaining:
            redundancy = max((float(matrix[index] @ matrix[item]) for item in selected), default=0.0)
            mmr = float(relevance[index]) if not selected else (
                mmr_lambda * float(relevance[index]) - (1.0 - mmr_lambda) * redundancy
            )
            scored.append((mmr, float(relevance[index]), str(ordered[index].comment_id), index))
        chosen = min(scored, key=lambda item: (-item[0], -item[1], item[2]))[3]
        selected.append(chosen)
        remaining.remove(chosen)

    return [
        RepresentativeCommentScore(
            comment_id=ordered[index].comment_id,
            rank=rank,
            score=float(relevance[index]),
            centroid_similarity=float(similarities[index]),
            engagement_score=float(engagement[index]),
            information_score=float(information[index]),
        )
        for rank, index in enumerate(selected, start=1)
    ]
