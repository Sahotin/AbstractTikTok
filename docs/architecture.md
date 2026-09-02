# Architecture

## Positioning

AbstractTikTok is a Python AI application layered on top of the existing MediaCrawler collection capability. The crawler remains responsible for acquiring data; the AI application reads normalized records through explicit repository boundaries.

```mermaid
flowchart LR
  C[MediaCrawler\nplatform collectors] --> I[Import adapters\nJSON / JSONL / CSV / Excel]
  I --> N[(Normalized SQLite\ncontent · comment · author)]
  N --> Q[Text cleaning &\nquality gate]
  Q --> L[CommentAnalysisService]
  L --> DP[DeepSeek-compatible LLM\nstructured JSON]
  L --> A[(Versioned\nCommentAnalysis)]
  A --> P[Public opinion /\nrisk / trend services]
  Q --> E[EmbeddingService]
  E --> EP[OpenAI-compatible\nEmbedding provider]
  E --> S[(Versioned Embeddings\nand SemanticRun)]
  S --> T[Topic & similarity\nservices]
  P --> API[FastAPI routers]
  T --> API
  API --> UI[Collection console\nDashboard]
  API --> AG[Tool Calling Agent]
  AG --> TOOLS[8 read-only tools]
  TOOLS --> P
  TOOLS --> T
```

## Module boundaries

| Boundary | Responsibility | Must not do |
| --- | --- | --- |
| `media_platform/`, `store/` | Collect and export source-platform data | Depend on LLM or Agent logic |
| `analysis/ingest` and adapters | Convert exports into normalized records | Expose raw platform schema to analytics |
| `analysis/domain/` | Stable entities, snapshots and query results | Call HTTP or SQL directly |
| `analysis/services/` | Orchestrate cleaning, providers, aggregation and workflows | Render HTML or own request state |
| `analysis/repositories/` | SQLAlchemy persistence/query boundary | Infer product policy or call a provider |
| `analysis/agent/` | Tool schemas, safety limits and answer orchestration | Query tables directly |
| `api/` | HTTP schemas, routers, workflow status and WebUI | Re-implement domain calculations |

## Snapshot consistency

A dashboard request is bound to an `AnalysisSnapshot`, not to “whatever analysis happened last”. The snapshot records versions for analysis, prompt and embedding as well as a `semantic_run_id`. Public-opinion, topic, trend, search and Agent tool calls carry this scope forward.

This prevents a familiar analytics failure: a new model run for one comment silently changing totals or mixing representative comments from another run.

## Agent safety boundary

```mermaid
sequenceDiagram
  participant U as User
  participant G as Agent
  participant T as Read-only Tool
  participant S as Domain Service
  participant R as Repository
  U->>G: Question + scoped collection/snapshot
  G->>T: Validated tool arguments
  T->>S: Bounded query
  S->>R: Scoped read
  R-->>S: Aggregates / minimal evidence
  S-->>T: Typed result + provenance
  T-->>G: Compact JSON result
  G-->>U: Answer + tool trace
```

The Agent has a maximum tool-step budget, duplicate-call protection, tool argument validation and bounded result sizes. It cannot mutate collection data, analysis records or database state.

## Provider roles

The project intentionally keeps the LLM and embedding providers separate:

- A DeepSeek-compatible chat completion endpoint produces structured comment labels and answers Tool Calling requests.
- An OpenAI-compatible embedding endpoint supplies vectors for semantic retrieval and topic clustering.

Both are configured only in local environment variables. The provider interface is isolated in services, so a provider migration does not require changing crawler code or API contracts.
