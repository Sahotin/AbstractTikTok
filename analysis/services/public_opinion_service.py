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
    RiskComponent,
    RiskEvidence,
    RiskExplanation,
    SentimentScoreStatistics,
)
from analysis.preprocessing import prepare_analysis_text
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
        semantic_run_id = _value(query.semantic_run_id)
        scope_kwargs = {
            "platform": platform,
            "run_id": run_id,
            "content_id": content_id,
            "semantic_run_id": semantic_run_id,
        }

        total_comments, analyzed_comments = await self.repository.get_public_opinion_counts(**scope_kwargs)
        eligible_comments = query.eligible_comments if semantic_run_id is not None else total_comments
        if eligible_comments is None:
            eligible_comments = analyzed_comments
        eligible_comments = min(eligible_comments, total_comments)
        excluded_comments = total_comments - eligible_comments
        denominator = analyzed_comments

        sentiment, emotion, stance, risk, score_statistics, topics, keywords, risk_reasons, topic_sentiment, topic_risk, volume, evidence = (
            await self._read_aggregates(scope_kwargs, query.top_k)
        )
        coverage = analyzed_comments / eligible_comments if eligible_comments else 0.0
        if analyzed_comments == 0:
            analysis_status = "no_analyses"
        elif analyzed_comments == eligible_comments:
            analysis_status = "complete"
        else:
            analysis_status = "partial"

        sentiment_distribution = self._distribution(sentiment, denominator)
        risk_distribution = self._distribution(risk, denominator)
        cleaned_topics = self._frequency(topics, denominator)
        cleaned_keywords = self._frequency(keywords, denominator)
        risk_explanation = self._risk_explanation(
            sentiment_distribution, risk_distribution, topic_risk, evidence, denominator
        )
        conclusion, conclusion_points = self._conclusion(
            total_comments,
            analyzed_comments,
            sentiment_distribution,
            cleaned_topics,
            risk_explanation,
            eligible_comments,
            excluded_comments,
        )
        return PublicOpinionSummary(
            scope=PublicOpinionScope(
                platform=query.platform,
                run_id=query.run_id,
                content_id=query.content_id,
                semantic_run_id=query.semantic_run_id,
            ),
            total_comments=total_comments,
            eligible_comments=eligible_comments,
            excluded_comments=excluded_comments,
            analyzed_comments=analyzed_comments,
            unanalyzed_eligible_comments=max(eligible_comments - analyzed_comments, 0),
            unanalyzed_comments=max(eligible_comments - analyzed_comments, 0),
            analysis_coverage=round(coverage, 4),
            ai_analysis_coverage=round(coverage, 4),
            analysis_status=analysis_status,
            analysis_snapshot=query.analysis_snapshot,
            conclusion=conclusion,
            conclusion_points=conclusion_points,
            sentiment_distribution=sentiment_distribution,
            sentiment_score_statistics=SentimentScoreStatistics(
                minimum=score_statistics.minimum,
                maximum=score_statistics.maximum,
                mean=round(score_statistics.mean, 4) if score_statistics.mean is not None else None,
            ),
            emotion_distribution=self._distribution(emotion, denominator),
            stance_distribution={},
            risk_distribution=risk_distribution,
            high_risk_count=next((item.count for item in risk if item.value == "high"), 0),
            risk_explanation=risk_explanation,
            risk_reason_frequency=self._frequency(risk_reasons, denominator),
            topics=cleaned_topics,
            keywords=cleaned_keywords,
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
            self.repository.list_risk_evidence_comments(**scope_kwargs, limit=5),
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
        # Preserve meaningful phrases such as "链接失效", while excluding
        # standalone platform/transport labels from dashboard aggregates.
        ignored = {
            "bilibili", "b站", "url", "网址", "链接", "视频", "视频链接", "链接分享",
            "回复", "展开回复", "分享链接", "复制链接", "复制", "share_source",
            "timestamp", "spmid", "amp",
        }
        merged: dict[str, int] = {}
        for item in values:
            cleaned = prepare_analysis_text(item.value).strip()
            folded = cleaned.casefold()
            if (
                not cleaned
                or folded in ignored
                or (folded.startswith("b站") and any(token in folded for token in {"链接", "搜索"}))
                or folded == "agreement"
                or len(cleaned) > 80
            ):
                continue
            if folded.isdigit() or (len(folded) >= 16 and all(char in "0123456789abcdef-" for char in folded)):
                continue
            merged[cleaned] = merged.get(cleaned, 0) + item.count
        return [
            FrequencyItem(value=value, count=count, percentage=round(count / denominator * 100, 2))
            for value, count in sorted(merged.items(), key=lambda pair: (-pair[1], pair[0]))
        ]

    @staticmethod
    def _risk_explanation(sentiment, risk, topic_risk, evidence, analyzed: int) -> RiskExplanation:
        if analyzed <= 0:
            return RiskExplanation(risk_level="unknown", risk_score=0)
        negative_ratio = sentiment.get("negative", DistributionValue(count=0, percentage=0)).percentage
        medium_ratio = risk.get("medium", DistributionValue(count=0, percentage=0)).percentage
        high_ratio = risk.get("high", DistributionValue(count=0, percentage=0)).percentage
        risky_topics: dict[str, int] = {}
        for item in topic_risk:
            if item.dimension in {"medium", "high"}:
                risky_topics[item.topic] = risky_topics.get(item.topic, 0) + item.count
        risky_topic_total = sum(risky_topics.values())
        topic_concentration = (
            max(risky_topics.values(), default=0) / risky_topic_total * 100
            if risky_topic_total else 0.0
        )
        concentration_label = "高" if topic_concentration >= 60 else "中" if topic_concentration >= 30 else "低"
        contributions = {
            "negative_ratio": round(negative_ratio * 0.5, 2),
            "high_risk_ratio": round(high_ratio * 0.35, 2),
            "medium_risk_ratio": round(medium_ratio * 0.15, 2),
        }
        score = min(100, max(0, round(sum(contributions.values()))))
        level = "high" if score >= 60 else "medium" if score >= 30 else "low"
        rules = []
        if negative_ratio >= 30:
            rules.append("负面评论比例达到关注阈值（≥30%）")
        if high_ratio >= 10:
            rules.append("高风险评论比例达到关注阈值（≥10%）")
        if not rules:
            rules.append("未触发负面或高风险比例阈值")
        return RiskExplanation(
            risk_level=level,
            risk_score=score,
            components=[
                RiskComponent(key="negative_ratio", label="负面评论比例", value=negative_ratio, display_value=f"{negative_ratio:.1f}%", weight=0.5, contribution=contributions["negative_ratio"]),
                RiskComponent(key="high_risk_ratio", label="高风险评论比例", value=high_ratio, display_value=f"{high_ratio:.1f}%", weight=0.35, contribution=contributions["high_risk_ratio"]),
                RiskComponent(key="medium_risk_ratio", label="中风险评论比例", value=medium_ratio, display_value=f"{medium_ratio:.1f}%", weight=0.15, contribution=contributions["medium_risk_ratio"]),
                RiskComponent(key="risk_topic_concentration", label="风险主题集中度", value=round(topic_concentration, 2), display_value=concentration_label, weight=0.0, contribution=0.0),
            ],
            triggered_rules=rules,
            representative_evidence=[
                RiskEvidence(
                    comment_id=row.comment.id,
                    text=row.comment.text,
                    sentiment=row.sentiment,
                    risk_level=row.risk_level,
                    risk_reasons=row.risk_reasons,
                    like_count=row.comment.like_count,
                )
                for row in evidence
            ],
        )

    @staticmethod
    def _conclusion(total, analyzed, sentiment, topics, risk, eligible=None, excluded=0):
        if total == 0:
            return "当前范围没有可分析评论。", []
        eligible = total if eligible is None else eligible
        if analyzed == 0:
            return f"当前已采集 {total} 条评论，其中 {eligible} 条有效评论尚未生成 AI 分析结果。", ["可先查看评论量趋势和已生成的语义主题。"]
        dominant_key, dominant = max(sentiment.items(), key=lambda item: item[1].count, default=("neutral", DistributionValue(count=0, percentage=0)))
        sentiment_label = {"positive": "正面", "neutral": "中性", "negative": "负面"}.get(dominant_key, dominant_key)
        risk_label = {"low": "低", "medium": "中", "high": "高"}.get(risk.risk_level, "未知")
        scope_note = (
            f"其中 {eligible} 条为有效评论，{excluded} 条低信息或清洗后为空已排除"
            if excluded else f"其中 {eligible} 条为有效评论"
        )
        conclusion = f"基于当前采集的 {total} 条评论，{scope_note}，{analyzed} 条完成 AI 分析；整体以{sentiment_label}讨论为主，当前为{risk_label}风险。"
        points = [f"{sentiment_label}评论占已分析有效样本的 {dominant.percentage:.1f}%"]
        if topics:
            points.append("主要讨论：" + "、".join(item.value for item in topics[:3]))
        points.append(risk.triggered_rules[0])
        return conclusion, points

    @staticmethod
    def _topic_dimension(values) -> dict[str, dict[str, int]]:
        result: dict[str, dict[str, int]] = {}
        for item in values:
            result.setdefault(item.topic, {})[item.dimension] = item.count
        return result
