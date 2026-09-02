"""Deterministic samples, output audits, and human-label-ready reports."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal, Sequence

from pydantic import BaseModel, Field

from analysis.domain import CommentAnalysis, NormalizedComment
from analysis.preprocessing import assess_semantic_quality, compute_text_hash, prepare_analysis_text


class EvaluationSampleItem(BaseModel):
    comment_id: str
    text: str
    text_hash: str
    depth: int
    like_count: int | None
    quality_eligible: bool
    exclusion_reason: str | None
    length_bucket: str
    stratum: str
    sample_rank: str
    gold_sentiment: Literal["positive", "neutral", "negative"] | None = None
    gold_emotion: Literal["joy", "anger", "sadness", "fear", "surprise", "disgust", "neutral"] | None = None
    gold_stance: Literal["support", "oppose", "neutral", "unclear"] | None = None
    gold_risk_level: Literal["low", "medium", "high"] | None = None
    annotator_notes: str | None = None


class CommentQualityEvaluation(BaseModel):
    sample_size: int
    analyzed_count: int
    missing_count: int
    structure_success_rate: float
    semantic_eligible_count: int
    duplicate_group_count: int
    duplicate_consistency_rate: float | None
    duplicate_consistency_by_dimension: dict[str, float | None] = Field(default_factory=dict)
    consistency_violation_count: int
    consistency_violations: dict[str, int]
    sentiment_distribution: dict[str, int]
    stance_distribution: dict[str, int]
    risk_distribution: dict[str, int]
    supervised_metrics: dict[str, Any] = Field(default_factory=dict)
    provider_usage: dict[str, Any] = Field(default_factory=dict)


class CommentQualityArtifact(BaseModel):
    generated_at: datetime
    database_name: str
    seed: str
    annotation_file: str
    elapsed_seconds: float = Field(ge=0)
    job: dict[str, Any] | None = None
    evaluation: CommentQualityEvaluation


def add_cost_range(
    usage: dict[str, Any],
    *,
    cache_hit_input_price_per_1m: float,
    cache_miss_input_price_per_1m: float,
    output_price_per_1m: float,
    pricing_label: str,
) -> dict[str, Any]:
    """Attach a transparent cost or range without pretending unknown cache splits are known."""

    result = dict(usage)
    prompt_tokens = int(result.get("prompt_tokens") or 0)
    completion_tokens = int(result.get("completion_tokens") or 0)
    cache_hit_tokens = int(result.get("prompt_cache_hit_tokens") or 0)
    cache_miss_tokens = int(result.get("prompt_cache_miss_tokens") or 0)
    output_cost = completion_tokens * output_price_per_1m / 1_000_000
    if cache_hit_tokens + cache_miss_tokens == prompt_tokens and prompt_tokens > 0:
        actual = (
            cache_hit_tokens * cache_hit_input_price_per_1m
            + cache_miss_tokens * cache_miss_input_price_per_1m
        ) / 1_000_000 + output_cost
        minimum = maximum = actual
        assumption = "provider cache split available"
    else:
        minimum = prompt_tokens * cache_hit_input_price_per_1m / 1_000_000 + output_cost
        maximum = prompt_tokens * cache_miss_input_price_per_1m / 1_000_000 + output_cost
        assumption = "input cache split unavailable; reported as all-hit to all-miss range"
    result.update({
        "cost_usd_min": round(minimum, 6),
        "cost_usd_max": round(maximum, 6),
        "pricing_label": pricing_label,
        "pricing_assumption": assumption,
        "cache_hit_input_price_per_1m": cache_hit_input_price_per_1m,
        "cache_miss_input_price_per_1m": cache_miss_input_price_per_1m,
        "output_price_per_1m": output_price_per_1m,
    })
    return result


def _length_bucket(informative_characters: int) -> str:
    if informative_characters <= 5:
        return "short"
    if informative_characters <= 20:
        return "medium"
    return "long"


def _sample_item(comment: NormalizedComment, seed: str) -> EvaluationSampleItem:
    normalized = prepare_analysis_text(comment.text)
    quality = assess_semantic_quality(normalized)
    family = "eligible" if quality.eligible else f"excluded:{quality.reason.value}"
    depth = "root" if comment.depth == 0 else "reply"
    length = _length_bucket(quality.informative_characters)
    rank = hashlib.sha256(f"{seed}:{comment.id}".encode()).hexdigest()
    return EvaluationSampleItem(
        comment_id=str(comment.id),
        text=comment.text,
        text_hash=compute_text_hash(normalized),
        depth=comment.depth,
        like_count=comment.like_count,
        quality_eligible=quality.eligible,
        exclusion_reason=quality.reason.value if quality.reason else None,
        length_bucket=length,
        stratum=f"{family}|{length}|{depth}",
        sample_rank=rank,
    )


def _round_robin(groups: dict[str, list[EvaluationSampleItem]]) -> Iterable[EvaluationSampleItem]:
    ordered = {key: sorted(value, key=lambda item: item.sample_rank) for key, value in sorted(groups.items())}
    while ordered:
        for key in list(ordered):
            values = ordered[key]
            if values:
                yield values.pop(0)
            if not values:
                del ordered[key]


def deterministic_stratified_sample(
    comments: Sequence[NormalizedComment],
    *,
    limit: int,
    seed: str,
) -> list[EvaluationSampleItem]:
    """Select a reproducible sample with quality, length, depth, and duplicate coverage."""

    if limit < 1 or limit > 100:
        raise ValueError("evaluation sample limit must be between 1 and 100")
    items = [_sample_item(comment, seed) for comment in comments]
    if not items:
        return []
    selected: list[EvaluationSampleItem] = []
    selected_ids: set[str] = set()

    # Reserve up to 10% of the sample for deterministic duplicate-text pairs.
    by_hash: dict[str, list[EvaluationSampleItem]] = defaultdict(list)
    for item in items:
        by_hash[item.text_hash].append(item)
    pair_budget = min(10, (limit // 10) * 2)
    duplicate_groups = sorted(
        (sorted(group, key=lambda item: item.sample_rank) for group in by_hash.values() if len(group) >= 2),
        key=lambda group: group[0].sample_rank,
    )
    for group in duplicate_groups:
        if len(selected) + 2 > pair_budget:
            break
        for item in group[:2]:
            selected.append(item)
            selected_ids.add(item.comment_id)

    remaining = [item for item in items if item.comment_id not in selected_ids]
    eligible = defaultdict(list)
    excluded = defaultdict(list)
    for item in remaining:
        (eligible if item.quality_eligible else excluded)[item.stratum].append(item)
    eligible_target = max(0, round(limit * 0.8) - sum(item.quality_eligible for item in selected))
    for item in _round_robin(eligible):
        if len(selected) >= limit or eligible_target <= 0:
            break
        selected.append(item)
        selected_ids.add(item.comment_id)
        eligible_target -= 1
    for item in _round_robin(excluded):
        if len(selected) >= limit:
            break
        selected.append(item)
        selected_ids.add(item.comment_id)
    if len(selected) < limit:
        leftovers = sorted(
            (item for item in items if item.comment_id not in selected_ids),
            key=lambda item: item.sample_rank,
        )
        selected.extend(leftovers[: limit - len(selected)])
    return selected[:limit]


def write_annotation_template(items: Sequence[EvaluationSampleItem], output_dir: Path, stamp: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"comment_quality_annotations_{stamp}.jsonl"
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        for item in items:
            handle.write(item.model_dump_json() + "\n")
    return path


class CommentQualityEvaluator:
    def evaluate(
        self,
        sample: Sequence[EvaluationSampleItem],
        analyses: dict[str, CommentAnalysis],
        *,
        provider_usage: dict[str, Any] | None = None,
    ) -> CommentQualityEvaluation:
        violations: Counter[str] = Counter()
        sentiment: Counter[str] = Counter()
        stance: Counter[str] = Counter()
        risk: Counter[str] = Counter()
        by_hash: dict[str, list[CommentAnalysis]] = defaultdict(list)
        for item in sample:
            analysis = analyses.get(item.comment_id)
            if analysis is None:
                continue
            sentiment[analysis.sentiment.value] += 1
            stance[analysis.stance.value] += 1
            risk[analysis.risk_level.value] += 1
            by_hash[item.text_hash].append(analysis)
            if analysis.sentiment.value == "positive" and analysis.sentiment_score < 0:
                violations["positive_with_negative_score"] += 1
            if analysis.sentiment.value == "negative" and analysis.sentiment_score > 0:
                violations["negative_with_positive_score"] += 1
            if analysis.sentiment.value == "neutral" and abs(analysis.sentiment_score) > 0.5:
                violations["neutral_with_extreme_score"] += 1
            if analysis.risk_level.value == "high" and not analysis.risk_reasons:
                violations["high_risk_without_reason"] += 1

        duplicate_groups = [group for group in by_hash.values() if len(group) >= 2]
        consistent_groups = 0
        dimension_consistency: dict[str, float | None] = {}
        for field in ("sentiment", "emotion", "stance", "risk_level"):
            consistent = sum(
                len({getattr(item, field).value for item in group}) == 1
                for group in duplicate_groups
            )
            dimension_consistency[field] = (
                round(consistent / len(duplicate_groups), 4) if duplicate_groups else None
            )
        for group in duplicate_groups:
            labels = {
                (item.sentiment.value, item.emotion.value, item.stance.value, item.risk_level.value)
                for item in group
            }
            consistent_groups += len(labels) == 1
        analyzed_count = len(analyses)
        return CommentQualityEvaluation(
            sample_size=len(sample),
            analyzed_count=analyzed_count,
            missing_count=len(sample) - analyzed_count,
            structure_success_rate=round(analyzed_count / len(sample), 4) if sample else 0.0,
            semantic_eligible_count=sum(item.quality_eligible for item in sample),
            duplicate_group_count=len(duplicate_groups),
            duplicate_consistency_rate=(
                round(consistent_groups / len(duplicate_groups), 4) if duplicate_groups else None
            ),
            duplicate_consistency_by_dimension=dimension_consistency,
            consistency_violation_count=sum(violations.values()),
            consistency_violations=dict(sorted(violations.items())),
            sentiment_distribution=dict(sorted(sentiment.items())),
            stance_distribution=dict(sorted(stance.items())),
            risk_distribution=dict(sorted(risk.items())),
            supervised_metrics=_supervised_metrics(sample, analyses),
            provider_usage=provider_usage or {},
        )


def _supervised_metrics(
    sample: Sequence[EvaluationSampleItem], analyses: dict[str, CommentAnalysis]
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for field in ("sentiment", "emotion", "stance", "risk_level"):
        pairs = []
        for item in sample:
            gold = getattr(item, f"gold_{field}")
            analysis = analyses.get(item.comment_id)
            if gold is not None and analysis is not None:
                pairs.append((gold, getattr(analysis, field).value))
        if pairs:
            labels = sorted({label for pair in pairs for label in pair})
            confusion = {
                gold: {predicted: sum(pair == (gold, predicted) for pair in pairs) for predicted in labels}
                for gold in labels
            }
            f1_by_label = {}
            for label in labels:
                true_positive = sum(gold == label and predicted == label for gold, predicted in pairs)
                false_positive = sum(gold != label and predicted == label for gold, predicted in pairs)
                false_negative = sum(gold == label and predicted != label for gold, predicted in pairs)
                denominator = 2 * true_positive + false_positive + false_negative
                f1_by_label[label] = round(2 * true_positive / denominator, 4) if denominator else 0.0
            result[field] = {
                "labeled_count": len(pairs),
                "accuracy": round(sum(gold == predicted for gold, predicted in pairs) / len(pairs), 4),
                "macro_f1": round(sum(f1_by_label.values()) / len(f1_by_label), 4),
                "f1_by_label": f1_by_label,
                "confusion_matrix": confusion,
            }
    return result


def write_comment_quality_report(
    evaluation: CommentQualityEvaluation,
    *,
    output_dir: Path,
    stamp: str,
    database: Path,
    seed: str,
    job: dict[str, Any] | None,
    annotation_path: Path,
    elapsed_seconds: float,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"comment_quality_report_{stamp}.md"
    job = job or {}
    actual_tokens = evaluation.provider_usage.get("total_tokens")
    cost_min = evaluation.provider_usage.get("cost_usd_min")
    cost_max = evaluation.provider_usage.get("cost_usd_max")
    if cost_min is None:
        actual_cost = "n/a (pricing not supplied)"
    elif cost_min == cost_max:
        actual_cost = f"${cost_min:.6f}"
    else:
        actual_cost = f"${cost_min:.6f}–${cost_max:.6f}"
    generated_at = datetime.now(timezone.utc)
    artifact = CommentQualityArtifact(
        generated_at=generated_at,
        database_name=database.name,
        seed=seed,
        annotation_file=annotation_path.name,
        elapsed_seconds=elapsed_seconds,
        job=job or None,
        evaluation=evaluation,
    )
    artifact_path = output_dir / f"comment_quality_report_{stamp}.json"
    artifact_path.write_text(artifact.model_dump_json(indent=2), encoding="utf-8")
    lines = [
        "# Comment Analysis Quality Evaluation",
        "",
        f"- Generated at: {generated_at.isoformat()}",
        f"- Validation database: `{database}`",
        f"- Deterministic seed: `{seed}`",
        f"- Annotation template: `{annotation_path}`",
        f"- Machine-readable artifact: `{artifact_path}`",
        f"- Sample / analyzed / missing: {evaluation.sample_size} / {evaluation.analyzed_count} / {evaluation.missing_count}",
        f"- Structured-output success rate: {evaluation.structure_success_rate:.2%}",
        f"- Wall-clock elapsed: {elapsed_seconds:.2f}s",
        f"- Actual provider requests: {evaluation.provider_usage.get('request_count', 'n/a')}",
        f"- Actual total tokens: {actual_tokens if actual_tokens else 'n/a'}",
        f"- Actual token cost estimate: {actual_cost}",
        f"- Pricing basis: {evaluation.provider_usage.get('pricing_label', 'n/a')}",
        f"- Cost assumption: {evaluation.provider_usage.get('pricing_assumption', 'n/a')}",
        f"- Persisted preflight input/output estimate: {job.get('estimated_input_tokens', 'n/a')} / {job.get('estimated_output_tokens', 'n/a')}",
        f"- Persisted preflight cost: {job.get('estimated_cost') if job.get('estimated_cost') is not None else 'n/a (prices not configured)'}",
        "- Note: this run predates full-prompt estimation; current jobs include the fixed prompt and JSON Schema.",
        "",
        "## Reliability checks",
        "",
        f"- Semantic-eligible samples: {evaluation.semantic_eligible_count}",
        f"- Duplicate groups checked: {evaluation.duplicate_group_count}",
        f"- Duplicate-label consistency: {evaluation.duplicate_consistency_rate if evaluation.duplicate_consistency_rate is not None else 'n/a'}",
        f"- Sentiment / stance / risk consistency: "
        f"{evaluation.duplicate_consistency_by_dimension.get('sentiment', 'n/a')} / "
        f"{evaluation.duplicate_consistency_by_dimension.get('stance', 'n/a')} / "
        f"{evaluation.duplicate_consistency_by_dimension.get('risk_level', 'n/a')}",
        f"- Rule consistency violations: {evaluation.consistency_violation_count}",
        f"- Violation details: `{json.dumps(evaluation.consistency_violations, ensure_ascii=False)}`",
        "",
        "## Output distributions",
        "",
        f"- Sentiment: `{json.dumps(evaluation.sentiment_distribution, ensure_ascii=False)}`",
        f"- Stance: `{json.dumps(evaluation.stance_distribution, ensure_ascii=False)}`",
        f"- Risk: `{json.dumps(evaluation.risk_distribution, ensure_ascii=False)}`",
        "",
        "## Human evaluation",
        "",
        "Fill the `gold_*` fields in the JSONL template and rerun the audit to obtain supervised metrics. "
        "Unlabeled output distributions are monitoring evidence, not accuracy claims.",
        "",
        f"Current supervised metrics: `{json.dumps(evaluation.supervised_metrics, ensure_ascii=False)}`",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
