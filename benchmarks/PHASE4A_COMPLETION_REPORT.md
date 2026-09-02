# Phase 4A Completion Report

## Scope and isolation

- Validation database: `database/phase3b3_real_validation_20260901_105826.db`
- Production database: `database/analysis.db` (kept at 0 comment analyses)
- Frozen sample: `comment_quality/comment_quality_annotations_20260901_055749.jsonl`
- Seed: `phase4a-v1`
- Sample design: 100 comments; 80 semantic-eligible and 20 low-information; includes root/reply, length strata, and five duplicate-text pairs.
- Successful Job: `9ac53ec7-7cd7-4980-bf32-91d901bb82f0`

## Real DeepSeek result

| Metric | Result |
| --- | ---: |
| Sample coverage | 100 / 100 |
| Job completed / skipped / failed | 98 / 2 / 0 |
| Structured-output success | 100% |
| Duplicate groups | 5 |
| Sentiment consistency | 100% |
| Stance consistency | 100% |
| Risk consistency | 100% |
| Rule consistency violations | 0 |
| Provider requests | 91 |
| Prompt / completion / total tokens | 68,250 / 50,477 / 118,727 |
| Mean / P95 latency | 5.94s / 15.08s |
| Wall-clock runtime | 287.47s |
| Estimated token cost | $0.067585–$0.096660 |

The cost range uses the official DeepSeek V4 Flash peak rates checked on 2026-09-01: cache-hit input $0.014/1M, cache-miss input $0.44/1M, and output $1.32/1M. The original responses did not retain the input cache split, so the report shows the transparent all-hit to all-miss range rather than a false point estimate. Source: <https://api-docs.deepseek.com/quick_start/pricing/>.

## Output distribution

- Sentiment: 23 negative, 51 neutral, 26 positive.
- Stance: 6 oppose, 7 support, 87 unclear.
- Risk: 2 high, 2 medium, 96 low.

These are sample distributions, not model-accuracy measurements. Accuracy, macro-F1, per-class F1, and confusion matrices remain intentionally empty until a human fills the `gold_*` fields in the annotation JSONL.

## Engineering findings

1. DeepSeek Chat Completions rejected `json_schema` with HTTP 400; provider configuration now defaults DeepSeek URLs to `json_object` and rejects an explicit incompatible mode before creating a paid Job.
2. Identical normalized text across different comment IDs now reuses one versioned result while persisting an independent analysis row for every comment.
3. Provider telemetry records request count, prompt/completion tokens, cache hit/miss tokens when available, average latency, and P95 latency.
4. Preflight token estimation now includes the fixed prompt and JSON Schema and uses a mixed Chinese/Latin heuristic. The successful historical Job's persisted estimate predates this correction; the final report uses actual provider usage.
5. `GET /api/analysis/quality/latest` exposes the newest report without raw comments, and the Dashboard displays reliability, usage, latency, and cost.

## Final artifacts

- This report is the repository-safe, aggregate-only artifact.
- Human-label JSONL and generated runtime reports under `benchmarks/comment_quality/` are intentionally gitignored because the annotation file contains real comment text.
