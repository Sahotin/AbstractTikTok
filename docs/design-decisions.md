# Design Decisions

## Keep the crawler and AI application decoupled

**Decision:** build the analysis layer around normalized records and repositories rather than alter crawler internals.

**Why:** platform collectors change frequently, while analytics needs a stable contract. The separation allows an export from a historical crawl to be analyzed without rerunning a crawler.

## Use versioned snapshots instead of “latest result” joins

**Decision:** dashboard and Agent queries carry analysis/prompt/embedding versions and semantic-run identity.

**Why:** “latest” is convenient but produces irreproducible analytics when a partial rerun happens. Snapshot selection is explicit and testable.

## Apply a quality gate before expensive model calls

**Decision:** clean and classify low-information text before LLM or embedding requests, while retaining raw text for source traceability.

**Why:** it lowers cost and reduces semantic noise without making cleaning irreversible.

## Separate LLM analysis from embedding

**Decision:** chat completions and embeddings are distinct providers with independent configuration and failure accounting.

**Why:** they have different APIs, availability profiles, costs and evaluation methods. A failure of one should not silently masquerade as a result from the other.

## Do not add LangChain for this MVP

**Decision:** use explicit service classes, Pydantic/domain schemas and direct tool registration.

**Why:** the tool set is small and safety-critical. Explicit schemas make the execution path, scope validation, test surface and provider errors easier to inspect in an interview and in production debugging.

## Do not add a vector database yet

**Decision:** persist versioned embeddings and perform local similarity/search for the current project scale.

**Why:** a separate vector database would add deployment, indexing and consistency complexity before it solves a demonstrated bottleneck. A vector-store adapter is a future scaling seam, not a present dependency.

## Make the Agent read-only and bounded

**Decision:** Agent tools are read-only and operate through services/repositories with maximum tool steps, duplicate-call detection and compact result limits.

**Why:** this makes an LLM useful for question routing and synthesis without granting it arbitrary SQL, mutation authority or unbounded context.

## Record external failure honestly

**Decision:** provider HTTP 502 is surfaced as a blocked provider validation, not converted into synthetic success.

**Why:** reliability claims must distinguish code-level regression evidence from third-party availability. This is part of the product contract, not merely test reporting.
