# Demo Script (2–3 minutes)

This script is designed for a portfolio walkthrough. Use a small, authorized sample and avoid exposing account credentials or raw sensitive text.

## 0:00–0:25 — Problem and scope

“This project starts with multi-platform video-comment collection, but the core value is what happens after collection: I turn exported comments into a versioned analysis dataset for sentiment, risk, topics, trends, semantic search and a tool-using Agent.”

Show the collection console. Point out that the AI layer is separate from the crawler and works with historical data as well.

## 0:25–0:50 — Collection-to-analysis handoff

Show a completed task’s **采集完成 · AI 分析** action. Then open **数据管理**, select a historical record and choose **分析所选记录**.

Say: “Navigation does not silently call a model. I can inspect an existing analysis snapshot first; a refresh is an explicit cost-bearing operation.”

## 0:50–1:25 — Dashboard semantics

Open the Dashboard and identify the selected platform/content/run scope. Explain the counts in this order:

1. collected comments;
2. eligible versus quality-gate excluded;
3. analyzed coverage for the selected analysis version;
4. semantic clustering input, clustered records and noise.

Then show sentiment/risk distribution, topic cards and representative comments. Explain that a topic’s noise count is not an embedding failure.

## 1:25–1:55 — Reliability and reproducibility

Show the analysis version and semantic run. Say: “My aggregate APIs and Agent tools use a snapshot, not the latest result. A partial rerun cannot silently change this dashboard.”

Mention the guardrails: deterministic cleaning, duplicate caching, concurrency/rate limits, structured outputs and failure isolation.

## 1:55–2:30 — Agent

Ask a scoped question such as: “高风险评论集中在哪些主题？”

Show the tool trace and minimal evidence returned by the tools. Explain the control boundary: “The LLM has no direct database access. It chooses among read-only tools, each tool receives scope and result limits, and the Agent has a maximum number of steps.”

## If a provider is unavailable

Do not fabricate an online response. Say: “The local tool-chain regression test passes. In the final external acceptance attempt the provider returned HTTP 502 before tool execution, so I record that as a provider availability block. The UI and the tool behavior can still be demonstrated with prepared/local test data.”

## Screenshots for a portfolio

Capture only aggregate dashboards and tool traces. Before publishing screenshots, remove API keys, account identifiers, URLs containing tokens and raw comments that are not authorized for public display. See [assets/README.md](assets/README.md).
