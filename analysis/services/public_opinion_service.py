"""Deterministic public-opinion aggregation over persisted comment analyses."""

from __future__ import annotations

import asyncio
from datetime import date
from uuid import UUID

from analysis.domain import (
    CommentVolumeByDate,
    DistributionValue,
    FrequencyItem,
    Platform,
    PublicOpinionQuery,
    PublicOpinionScope,
    PublicOpinionSummary,
    SentimentScoreStatistics,
)
from analysis.repositories import AnalysisRepository


def _value(value: UUID | Platform | str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, Platform):
        return value.value
    return str(value)


class PublicOpinionService:
    """Expose one Tool-ready, read-only boundary for aggregate public opinion."""

    def __init__(self, repository: AnalysisRepository):
        self.repository = repository

    async def get_public_opinion(self, query: PublicOpinionQuery) -> PublicOpinionSummary:
        platform = _value(query.platform)
        run_id = _value(query.run_id)
        content_id = _value(query.content_id)
        scope_kwargs = {"platform": platform, "run_id": run_id, "content_id": content_id}

        total_comments, analyzed_comments = await self.repository.get_public_opinion_counts(**scope_kwargs)
        denominator = analyzed_comments

        sentiment, emotion, stance, risk, score_statistics, topics, keywords, risk_reasons, topic_sentiment, topic_risk, volume = (
            await self._read_aggregates(scope_kwargs, query.top_k)
        )
        coverage = analyzed_comments / total_comments if total_comments else 0.0
        if analyzed_comments == 0:
            analysis_status = "no_analyses"
        elif analyzed_comments == total_comments:
            analysis_status = "complete"
        else:
            analysis_status = "partial"

        return PublicOpinionSummary(
            scope=PublicOpinionScope(
                platform=query.platform,
                run_id=query.run_id,
                content_id=query.content_id,
            ),
            total_comments=total_comments,
            analyzed_comments=analyzed_comments,
            unanalyzed_comments=total_comments - analyzed_comments,
            analysis_coverage=round(coverage, 4),
            analysis_status=analysis_status,
            sentiment_distribution=self._distribution(sentiment, denominator),
            sentiment_score_statistics=SentimentScoreStatistics(
                minimum=score_statistics.minimum,
                maximum=score_statistics.maximum,
                mean=round(score_statistics.mean, 4) if score_statistics.mean is not None else None,
            ),
            emotion_distribution=self._distribution(emotion, denominator),
            stance_distribution=self._distribution(stance, denominator),
            risk_distribution=self._distribution(risk, denominator),
            high_risk_count=next((item.count for item in risk if item.value == "high"), 0),
            risk_reason_frequency=self._frequency(risk_reasons, denominator),
            topics=self._frequency(topics, denominator),
            keywords=self._frequency(keywords, denominator),
            topic_sentiment_distribution=self._topic_dimension(topic_sentiment),
            topic_risk_distribution=self._topic_dimension(topic_risk),
            comment_volume_by_date=[
                CommentVolumeByDate(date=date.fromisoformat(item.date), count=item.count)
                for item in volume
            ],
        )

    async def _read_aggregates(self, scope_kwargs: dict[str, str | None], top_k: int):
        """Execute a fixed number of SQL aggregate queries, never per comment."""

        return await asyncio.gather(
            self.repository.get_analysis_distribution("sentiment", **scope_kwargs),
            self.repository.get_analysis_distribution("emotion", **scope_kwargs),
            self.repository.get_analysis_distribution("stance", **scope_kwargs),
            self.repository.get_analysis_distribution("risk_level", **scope_kwargs),
            self.repository.get_sentiment_score_statistics(**scope_kwargs),
            self.repository.get_json_frequency("topics", top_k=top_k, **scope_kwargs),
            self.repository.get_json_frequency("keywords", top_k=top_k, **scope_kwargs),
            self.repository.get_json_frequency("risk_reasons", top_k=top_k, **scope_kwargs),
            self.repository.get_topic_dimension_distribution("sentiment", top_k=top_k, **scope_kwargs),
            self.repository.get_topic_dimension_distribution("risk_level", top_k=top_k, **scope_kwargs),
            self.repository.get_comment_volume_by_date(**scope_kwargs),
        )

    @staticmethod
    def _distribution(values, denominator: int) -> dict[str, DistributionValue]:
        if denominator == 0:
            return {}
        return {
            item.value: DistributionValue(
                count=item.count,
                percentage=round(item.count / denominator * 100, 2),
            )
            for item in values
        }

    @staticmethod
    def _frequency(values, denominator: int) -> list[FrequencyItem]:
        if denominator == 0:
            return []
        return [
            FrequencyItem(
                value=item.value,
                count=item.count,
                percentage=round(item.count / denominator * 100, 2),
            )
            for item in values
        ]

    @staticmethod
    def _topic_dimension(values) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for item in values:
            result.setdefault(item.topic, {})[item.dimension] = item.count
        return result
