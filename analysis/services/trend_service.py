"""Explainable trend and inflection analysis over SQL time buckets."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from statistics import median

from analysis.domain import (
    Platform,
    PublicOpinionScope,
    TrendInflection,
    TrendPoint,
    TrendQuery,
    TrendSummary,
)
from analysis.repositories import AnalysisRepository


def _value(value):
    if value is None:
        return None
    if isinstance(value, Platform):
        return value.value
    return str(value)


def _risk_score(negative: int, medium: int, high: int, analyzed: int) -> float:
    if analyzed <= 0:
        return 0.0
    # Transparent fixed weights: negative tone 50%, explicit high risk 35%,
    # and medium risk 15%. This is a ranking signal, not a probability.
    score = 50 * negative / analyzed + 35 * high / analyzed + 15 * medium / analyzed
    return round(min(max(score, 0.0), 100.0), 2)


class TrendAnalysisService:
    def __init__(self, repository: AnalysisRepository, *, max_inflections: int = 20):
        self.repository = repository
        self.max_inflections = max_inflections

    async def get_trend(self, query: TrendQuery) -> TrendSummary:
        rows = await self.repository.get_trend_buckets(
            bucket=query.bucket,
            platform=_value(query.platform),
            run_id=_value(query.run_id),
            content_id=_value(query.content_id),
            start_at=query.start_at,
            end_at=query.end_at,
        )
        points = [self._point(row) for row in rows]
        total = sum(point.total_comments for point in points)
        analyzed = sum(point.analyzed_comments for point in points)
        coverage = analyzed / total if total else 0.0
        if total == 0:
            quality = "no_data"
        elif analyzed == 0:
            quality = "no_analyses"
        elif coverage < 0.8:
            quality = "partial"
        else:
            quality = "sufficient"
        overall = _risk_score(
            sum(point.negative_count for point in points),
            sum(point.medium_risk_count for point in points),
            sum(point.high_risk_count for point in points),
            analyzed,
        )
        peak = max(points, key=lambda point: (point.risk_score, point.bucket_start), default=None)
        return TrendSummary(
            scope=PublicOpinionScope(
                platform=query.platform, run_id=query.run_id, content_id=query.content_id
            ),
            bucket=query.bucket,
            points=points,
            inflections=self._inflections(
                points, query.min_bucket_comments, query.bucket, self.max_inflections
            ),
            overall_risk_score=overall,
            peak_risk_at=peak.bucket_start if peak and analyzed else None,
            total_comments=total,
            analyzed_comments=analyzed,
            analysis_coverage=round(coverage, 4),
            data_quality=quality,
        )

    @staticmethod
    def _point(row) -> TrendPoint:
        analyzed = row.analyzed_comments
        return TrendPoint(
            bucket_start=datetime.fromisoformat(row.bucket_start).replace(tzinfo=timezone.utc),
            total_comments=row.total_comments,
            analyzed_comments=analyzed,
            analysis_coverage=round(analyzed / row.total_comments, 4) if row.total_comments else 0.0,
            positive_count=row.positive_count,
            neutral_count=row.neutral_count,
            negative_count=row.negative_count,
            low_risk_count=row.low_risk_count,
            medium_risk_count=row.medium_risk_count,
            high_risk_count=row.high_risk_count,
            mean_sentiment_score=(
                round(row.mean_sentiment_score, 4)
                if row.mean_sentiment_score is not None
                else None
            ),
            negative_ratio=round(row.negative_count / analyzed, 4) if analyzed else 0.0,
            high_risk_ratio=round(row.high_risk_count / analyzed, 4) if analyzed else 0.0,
            risk_score=_risk_score(
                row.negative_count, row.medium_risk_count, row.high_risk_count, analyzed
            ),
        )

    @staticmethod
    def _inflections(
        points: list[TrendPoint],
        min_bucket_comments: int,
        bucket: str,
        max_inflections: int,
    ) -> list[TrendInflection]:
        if len(points) < 2:
            return []
        baseline_volume = max(float(median(point.total_comments for point in points)), 1.0)
        expected_gap = timedelta(hours=1) if bucket == "hour" else timedelta(days=1)
        output: list[TrendInflection] = []
        for previous, current in zip(points, points[1:]):
            # Sparse historical data must not compare two non-consecutive
            # active buckets as if there were no quiet interval between them.
            if current.bucket_start - previous.bucket_start > expected_gap * 1.5:
                continue
            if current.total_comments < min_bucket_comments:
                continue
            score_delta = round(current.risk_score - previous.risk_score, 2)
            volume_ratio = round((current.total_comments - previous.total_comments) / max(previous.total_comments, 1), 4)
            volume_surge = (
                current.total_comments >= max(min_bucket_comments, baseline_volume * 3)
                and current.total_comments - previous.total_comments >= max(5, baseline_volume * 2)
            )
            if score_delta >= 15:
                kind = "risk_spike"
                reasons = [f"risk score increased by {score_delta:.2f}"]
            elif score_delta <= -15:
                kind = "recovery"
                reasons = [f"risk score decreased by {abs(score_delta):.2f}"]
            elif volume_surge and volume_ratio >= 0.5:
                kind = "volume_surge"
                reasons = [f"comment volume increased by {volume_ratio * 100:.1f}%"]
            else:
                continue
            magnitude = abs(score_delta) if kind != "volume_surge" else volume_ratio * 20
            severity = "high" if magnitude >= 30 else "medium" if magnitude >= 15 else "low"
            output.append(TrendInflection(
                bucket_start=current.bucket_start,
                kind=kind,
                severity=severity,
                score_delta=score_delta,
                volume_change_ratio=volume_ratio,
                reasons=reasons,
            ))
        if len(output) <= max_inflections:
            return output
        ranked = sorted(
            output,
            key=lambda item: (
                -max(abs(item.score_delta), abs(item.volume_change_ratio or 0) * 20),
                item.bucket_start,
            ),
        )[:max_inflections]
        return sorted(ranked, key=lambda item: item.bucket_start)
