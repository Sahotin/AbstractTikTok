# Resume Facts

Use only claims that match the code and the documented validation state. Replace bracketed fields with your own personal contribution wording where necessary.

## Chinese bullets

- 基于 Python、FastAPI 与 SQLAlchemy，在 MediaCrawler 多平台采集能力之上设计并实现视频评论智能分析与舆情 Agent，形成“采集—标准化—分析—语义检索—看板”的闭环。
- 设计统一内容、作者与评论数据模型及幂等导入链路，隔离平台原始字段差异，使历史采集结果可复用到 AI 分析流程。
- 实现版本化评论分析 Pipeline：文本清洗与低信息过滤、重复文本缓存、批处理、并发/限流、重试、结构化输出和失败隔离，支持情感、立场、主题、关键词与风险标签。
- 设计分析快照机制，以 analysis/prompt/embedding 版本和 semantic run 约束聚合查询，避免部分重跑导致 Dashboard 与 Agent 混用不同批次结果。
- 实现 Embedding、相似评论检索、主题聚类和代表性评论选择；将有效、已嵌入、入聚类、归主题和噪声计数拆分展示，提升结果可解释性。
- 实现受限 Tool Calling Agent：8 个只读工具、Pydantic 参数校验、最大调用步数、重复调用保护、最小证据返回与工具轨迹展示，LLM 不直接访问数据库。
- 为 AI 服务层编写回归测试，覆盖 Agent 作用域、快照聚合、主题持久化、趋势与公共舆情查询等关键一致性逻辑。

## English bullets

- Built a Python/FastAPI video-comment intelligence application on top of MediaCrawler, connecting collection, normalized storage, LLM analysis, semantic retrieval, and an analytics dashboard.
- Designed versioned analysis snapshots across prompt, LLM analysis, embeddings, and semantic runs to keep dashboard and Agent aggregates reproducible after partial reruns.
- Implemented a bounded Tool Calling Agent with eight read-only tools, typed schemas, scoped queries, duplicate-call protection, and compact evidence traces instead of direct database access.

## Metrics that can be cited with context

For the documented Bilibili acceptance snapshot: 131 collected comments; 104 quality-gate eligible and analyzed for the selected version; 96 semantic clustering inputs; 30 clustered comments; 66 noise assignments; 5 topic clusters. State this as a **local acceptance snapshot**, not a benchmark for all platforms or datasets.

Do not claim that final live Agent acceptance passed: the documented external call was blocked by a provider HTTP 502 before tool execution. Local regression coverage for the bounded tool chain passed.
