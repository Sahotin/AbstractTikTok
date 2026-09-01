# AbstractTikTok AI 应用架构与运行手册

## 1. 定位

该项目不是继续堆叠爬虫平台，而是把 MediaCrawler 作为数据入口，构建“视频评论智能分析与舆情 Agent”。`media_platform/`、`store/` 等采集模块继续负责获取公开数据；新增 `analysis/` 只消费标准化数据，两者通过数据库边界解耦。

## 2. 架构

```text
┌─────────────────────────────────────────────────────────────┐
│ MediaCrawler: Playwright / Chrome CDP / 多平台采集           │
└──────────────────────────────┬──────────────────────────────┘
                               │ JSON / JSONL / SQLite
┌──────────────────────────────▼──────────────────────────────┐
│ Normalization                                                │
│ CollectionRun · NormalizedContent · Comment · Author         │
└───────────────┬────────────────────────────┬─────────────────┘
                │                            │
┌───────────────▼──────────────┐ ┌──────────▼─────────────────┐
│ Comment Analysis Pipeline    │ │ Semantic Pipeline           │
│ DeepSeek JSON Object         │ │ Qwen Embedding + cache      │
│ Pydantic validation/cache    │ │ quality gate + clustering   │
│ retry/rate limit/job state   │ │ representatives/search      │
└───────────────┬──────────────┘ └──────────┬─────────────────┘
                └──────────────┬─────────────┘
┌──────────────────────────────▼──────────────────────────────┐
│ Public Opinion Services                                      │
│ distributions · risk · topics · trend · inflection           │
└──────────────────────────────┬──────────────────────────────┘
┌──────────────────────────────▼──────────────────────────────┐
│ Tool Calling Agent                                            │
│ 8 read-only tools · scope enforcement · bounded loop/context │
└──────────────────────────────┬──────────────────────────────┘
                               ▼
                 FastAPI / OpenAPI / Dashboard
```

## 3. 目录职责

- `analysis/domain/`：严格 Pydantic 领域模型，不依赖 Web 和 ORM。
- `analysis/normalization/`：平台原始字段到统一模型的映射。
- `analysis/repositories/`：SQLite 查询、缓存与原子写入。
- `analysis/llm/`：DeepSeek/OpenAI-compatible 结构化输出适配。
- `analysis/embeddings/`：批量 Embedding、复用缓存、向量编码。
- `analysis/clustering/`：确定性聚类和代表评论选择。
- `analysis/services/`：分析任务、聚合、趋势、语义检索编排。
- `analysis/evaluation/`：确定性分层抽样、人工标注模板、质量与一致性评测。
- `analysis/agent/`：Tool schema、只读工具、Provider 和受控循环。
- `api/routers/analysis.py`：HTTP 边界；后台任务与依赖生命周期。
- `api/webui/dashboard.html`：不依赖前端构建链的 AI Dashboard。
- `migrations/`：仅管理分析层表，不接管 MediaCrawler 原表。

## 4. 数据与版本策略

评论分析的持久化唯一键由 `comment_id + input_hash + provider + model + prompt_version + analysis_version` 组成；相同标准化文本还会按去掉 `comment_id` 的内容键跨评论复用，并为每条评论保存独立结果。Embedding 同样支持同评论精确命中与相同文本跨评论复用。语义运行固化评论集合、Embedding 配置、输入快照哈希和算法配置哈希，保证实验可重复。

主题质量门只改变语义运行项状态：低信息评论标记为 `excluded` 并记录原因，不删除评论和向量。当前规则覆盖空文本、纯数字、纯表情、极短、重复字符和泛化回复。

## 5. Agent 工具

| Tool | 用途 | 主要边界 |
| --- | --- | --- |
| `get_video_info` | 视频元数据 | 必须提供内容 UUID |
| `get_comments` | 评论样本 | 最多 50 条 |
| `get_comment_statistics` | 评论与回复数量、时间范围 | SQL 聚合 |
| `get_sentiment_distribution` | 情感/立场/风险/主题分布 | 返回分析覆盖率 |
| `get_topics` | 主题簇和代表评论 | 最多 20 个簇 |
| `get_trend` | 时间趋势、风险分和拐点 | 最多返回最近 48 个点给 Agent |
| `search_similar_comments` | 自然语言语义检索 | 最多 10 条结果 |
| `get_high_risk_comments` | 最新分析中的高风险评论 | 最多 20 条 |

Agent 最多 8 次工具调用、请求最多 8 个模型步骤；重复调用会被阻断并强制总结。单工具结果和总工具上下文均有字符上限。评论文本被明确视为不可信数据，不能覆盖系统指令。

## 6. 常用命令

```powershell
# 数据迁移
uv run alembic upgrade head

# 评论分析（先用 Fake Provider 验证）
uv run python -m analysis.cli analyze-comments --database database/analysis.db --platform douyin --limit 20 --fake-provider

# 真实评论分析（会产生 API 成本）
uv run python -m analysis.cli analyze-comments --database database/analysis.db --platform douyin --limit 20 --batch-size 10 --max-concurrency 3

# 质量评测：先冻结不超过 100 条样本，再显式执行真实模型
uv run python -m analysis.cli comment-quality --database database/validation.db --limit 100 --seed phase4a-v1
uv run python -m analysis.cli comment-quality --database database/validation.db --annotations benchmarks/comment_quality/<annotations>.jsonl --execute --max-concurrency 2

# Embedding 与主题聚类
uv run python -m analysis.cli embed --database database/analysis.db --platform douyin --limit 2000
uv run python -m analysis.cli cluster-topics --database database/analysis.db --semantic-run-id <UUID> --distance-threshold 0.30 --min-cluster-size 5

# 语义检索与 Agent
uv run python -m analysis.cli semantic-search --database database/analysis.db --semantic-run-id <UUID> --query "如何报名"
uv run python -m analysis.cli ask-agent --database database/analysis.db --semantic-run-id <UUID> --question "用户最关心什么？"

# 测试
uv run pytest -q tests/analysis
```

## 7. API

主要接口：

- `GET /api/analysis/public-opinion`
- `GET /api/analysis/quality/latest`
- `GET /api/analysis/trends`
- `GET /api/analysis/topics`
- `POST /api/analysis/semantic-search`
- `POST /api/analysis/agent/query`
- `POST /api/analysis/jobs`
- `POST /api/analysis/semantic-runs`

所有筛选条件采用 AND 语义。聚合和 Agent 工具不会隐式触发付费分析；只有显式分析 Job、语义运行、语义检索查询和 Agent 模型步骤会访问远程 Provider。

## 8. Docker

```powershell
Copy-Item .env.example .env
# 编辑 .env，填入模型和 Embedding 配置；不要提交该文件。
docker compose up --build
```

容器启动时对挂载的 `database/analysis.db` 执行 Alembic 升级。不要把真实 `.env` 或数据库提交到 Git。采集端依赖本机 Chrome 登录态，推荐仍在本机运行；Docker 主要承载分析 API 与 Dashboard。

## 9. 当前边界

- 统一导入 MVP 目前以抖音为主；其他平台需实现各自 normalizer，不能只改枚举。
- SQLite 内存余弦检索限定单语义快照最多 5,000 条，足以演示和中小数据集；更大规模才考虑 pgvector/Qdrant。
- 风险分是透明排序信号，不是事件发生概率，也不替代人工判断。
- Phase 4A 已在独立验证库完成固定 seed 的 100 条 DeepSeek 真实样本评测；这是分层样本基线，不能替代 1,907 条全量分析，也不能在缺少人工 gold label 时宣称分类准确率。
- In-process 后台任务适合单实例 MVP；多实例部署应换成 Redis + Celery/RQ/Arq 等持久队列。

## 10. 评论分析质量评测

`comment-quality` 只接受显式验证数据库，并拒绝写入 `database/analysis.db`。默认仅生成冻结样本和报告；只有增加 `--execute` 才会把样本发送给配置的 LLM。抽样固定 seed，覆盖语义有效/低信息、长短文本、一级/二级评论，并预留重复文本对。

评测同时生成 Markdown 和不含原始评论的 JSON 产物。评测报告区分三类证据：Pydantic 结构化成功率、标签与分数/风险原因的规则一致性、重复文本标签一致性。`GET /api/analysis/quality/latest` 读取最新有效 JSON，Dashboard 以独立质量卡片展示；损坏或缺失的报告不会影响其他分析接口。JSONL 中的 `gold_*` 字段供人工标注；填好后使用 `--annotations` 重跑，会计算 accuracy、macro-F1、逐类 F1 和混淆矩阵。没有人工标签时，分布和规则检查只能证明管线可靠性，不能宣称模型准确率。

OpenAI-compatible Provider 会在进程内聚合响应中的 token usage、请求数、平均延迟和 P95 延迟；价格通过 `LLM_INPUT_PRICE_PER_1M_TOKENS` 与 `LLM_OUTPUT_PRICE_PER_1M_TOKENS` 显式配置，未配置时报告保留 token 数并将费用标为不可用。

当前脱敏基线见 `benchmarks/PHASE4A_COMPLETION_REPORT.md`：100/100 结构成功，5 组重复文本在 sentiment、stance、risk_level 三个维度均为 100% 一致，规则冲突为 0；91 次真实请求共使用 118,727 tokens，平均延迟 5.94s、P95 15.08s。按执行时 DeepSeek V4 Flash 官方峰时价格和未知输入缓存拆分计算，成本区间为 $0.067585–$0.096660。包含真实评论的标注与运行时评测产物不会提交到 Git。
