# Interview Notes

## 1. Why not make the crawler call the LLM directly?

Platform crawling is volatile and I/O-heavy, whereas model analysis needs stable data, retry/cost controls and reproducibility. A normalized persistence boundary lets either side evolve independently and allows historical crawls to be analyzed later.

## 2. How do you avoid `for comment: call_llm(comment)`?

The analysis service cleans text first, skips low-information entries, uses duplicate-text caching, processes bounded batches with a concurrency cap, rate limits provider requests, retries transient errors and stores structured results independently. The workflow can report partial failure instead of losing all progress.

## 3. What is a snapshot and why is it important?

A snapshot fixes the analysis/prompt/embedding versions plus a semantic-run identity. Every dashboard aggregate and Agent tool query is scoped to it. It prevents an accidental partial rerun from mixing old and new model outputs.

## 4. Why preserve raw comment text after cleaning?

Cleaning is a model-input transformation, not a replacement for source data. Retaining raw text supports traceability, display and debugging; the cleaned representation supports deterministic cache keys and quality checks.

## 5. How do you distinguish noise from failure in clustering?

An embedding failure means no usable vector. A clustering input is a valid vector admitted into a semantic run. Noise is a valid clusterer decision that the vector does not belong to a stable non-noise topic. These counts are reported separately.

## 6. Why no vector database?

The current scale can use versioned local embeddings and local retrieval, which keeps deployment and consistency simpler. The repository/service boundary is the scaling seam; a vector DB becomes justified when data volume, latency or concurrent retrieval shows a real bottleneck.

## 7. Why no LangChain?

The Agent has a small fixed set of safety-critical read-only tools. Explicit schemas and orchestration make argument validation, tool-step limits, failure handling and testing easier to reason about than adding framework abstraction at this stage.

## 8. How does the Agent avoid unlimited context and calls?

Each tool is scoped and result-limited; tool outputs return compact aggregates plus minimal evidence. The Agent validates arguments, has a maximum step budget and detects duplicate calls. It never receives direct SQL access.

## 9. What happens when a provider fails?

The workflow records a provider error and preserves successfully persisted prior work. It does not create a fake analysis or claim success. The final real Agent acceptance attempt encountered provider HTTP 502; this is documented separately from passing local tool-chain tests.

## 10. What would you build next?

Add a provider-independent evaluation dataset, background job persistence/queueing for long runs, cancellation and resumability, role-based access for a shared deployment, and a vector-store adapter after measuring local-retrieval limits.

## A concise project introduction

“I turned a multi-platform crawler into a video-comment intelligence application. I designed the normalized data contract, versioned LLM and embedding pipelines, snapshot-consistent aggregation, semantic topics/search and a bounded read-only Tool Calling Agent, then connected it to a FastAPI dashboard and tested the core analysis services.”
