# AbstractTikTok · 视频评论智能分析与舆情 Agent

> 基于 MediaCrawler 多平台采集能力构建的 Python AI 应用：将视频评论转为可追溯的情感、主题、风险、趋势与语义洞察，并提供受限 Tool Calling Agent。

AbstractTikTok 保留了 MediaCrawler 的采集能力，但将项目重点放在采集后的 AI 应用工程：数据标准化、版本化分析、快照一致性、语义主题、只读工具与可视化看板。

```text
多平台采集结果
  → 标准化导入（内容 / 作者 / 一级评论 / 二级评论）
  → 文本清洗与质量门
  → DeepSeek 结构化评论分析
  → 聚合舆情 / 趋势 / 风险
  → Embedding / 主题聚类 / 语义检索
  → Tool Calling Agent
  → FastAPI Dashboard
```

## 项目亮点

- **采集与 AI 解耦**：不重构 MediaCrawler 核心；AI 层从标准化 SQLite 数据读取，支持历史采集数据的导入和分析。
- **统一数据模型**：以 `NormalizedContent`、`NormalizedComment`、`NormalizedAuthor` 连接不同平台的原始字段差异。
- **可复现分析快照**：分析结果由 `analysis_version + prompt_version + embedding_version + semantic_run_id` 定位；聚合不混用“最新一条”分析。
- **成本可控的 LLM Pipeline**：清洗、低信息过滤、去重缓存、并发/速率限制、重试、结构化输出与失败隔离。
- **语义层而非关键词堆砌**：Embedding、相似评论检索、聚类、代表性评论和主题覆盖率分开统计。
- **真正受约束的 Agent**：Agent 只能调用 8 个只读工具；工具返回最小证据、数据范围和调用轨迹，避免直接访问数据库或无限调用。
- **产品闭环**：采集结束和历史数据管理都可以进入“智能分析”，分析页面展示进度、数据范围与可解释的结果。

## 当前冻结状态

当前版本用于作品集和面试演示，状态为 **`FROZEN_FOR_CAREER_PACKAGING`**。最终验收摘要见 [reports/final_acceptance_status.md](reports/final_acceptance_status.md)。

一组 B 站验收快照（仅统计，不含评论原文）：131 条采集评论，104 条有效、27 条被质量门排除；104 条均存在指定版本分析结果；语义层 96 条入聚类、30 条归入主题、66 条为噪声，生成 5 个主题。

真实 Agent 外部调用曾在模型提供商侧返回 HTTP 502，已按“失败不伪造成功”的原则记录为 provider-blocked；本地受控工具链回归测试通过。不要将这一状态表述为“线上真实 Agent 已验收成功”。

## 快速启动（Windows）

### 1. 前置条件

- Python `>=3.11`
- [uv](https://docs.astral.sh/uv/)
- Chromium / Chrome（采集登录或浏览器自动化时需要）

### 2. 配置环境变量

首次启动会将 `.env.example` 复制为本地 `.env`。密钥只保存在 `.env`，该文件已被 Git 忽略。

```dotenv
# DeepSeek：评论结构化分析与 Agent
LLM_PROVIDER=openai-compatible
LLM_MODEL=<your_deepseek_model>
LLM_BASE_URL=https://api.deepseek.com
LLM_API_KEY=<your_deepseek_api_key>
LLM_STRUCTURED_OUTPUT_MODE=json_object

# SiliconFlow（或兼容 OpenAI Embeddings 的服务）：语义检索与聚类
EMBEDDING_PROVIDER=openai-compatible
EMBEDDING_MODEL=<your_embedding_model>
EMBEDDING_BASE_URL=https://api.siliconflow.cn/v1
EMBEDDING_API_KEY=<your_embedding_api_key>
EMBEDDING_VERSION=embedding_v1
```

不配置密钥时，采集控制台仍可使用；依赖模型的分析会明确提示配置缺失。请不要把 `.env`、真实评论导出或模型响应日志提交到仓库。

### 3. 一键启动

双击 `start.bat`，或在 PowerShell 中运行：

```powershell
.\start.bat

# 常用开发选项
.\start.ps1 -SkipSync -NoBrowser
.\start.ps1 -Port 8090
```

脚本会使用项目内 `.uv-cache` 和 `.uv-python`，避免 Windows 用户目录缓存无权限时启动失败；随后同步锁定依赖、执行 Alembic 迁移并启动 FastAPI。

启动后访问：

- 采集控制台：`http://127.0.0.1:8080/`
- 智能分析 Dashboard：`http://127.0.0.1:8080/dashboard`
- OpenAPI：`http://127.0.0.1:8080/docs`

## 演示路径

1. 在采集控制台选择平台、模式与评论采集，完成一次采集。
2. 点击右下角 **“采集完成 · AI 分析”**，选择分析项与样本上限；或在 **数据管理** 中勾选历史记录后选择 **“分析所选记录”**。
3. Dashboard 展示当前集合、快照版本、有效/排除/已分析计数、情感分布、风险、趋势、主题和语义检索。
4. 在 Agent 面板用自然语言提问，例如“高风险评论集中在哪些主题？”；Agent 根据问题选择只读工具并展示证据来源。

完整的 2–3 分钟演示脚本见 [docs/demo-script.md](docs/demo-script.md)。

## 技术架构与设计取舍

- [架构总览](docs/architecture.md)
- [数据流与快照一致性](docs/data-flow.md)
- [关键设计决策](docs/design-decisions.md)
- [面试问答](docs/interview-notes.md)
- [可直接用于简历的事实清单](docs/resume-facts.md)
- [最终验收状态](reports/final_acceptance_status.md)

### 目录导览

```text
analysis/                 # 独立 AI 领域层、服务层、仓储层、Agent 与语义能力
  domain/                 # 统一模型与快照/聚合领域对象
  services/               # LLM、Embedding、主题、趋势、检索等编排服务
  repositories/           # SQLAlchemy 持久化与查询边界
  agent/                  # 只读 Tool Calling Agent、schema 与工具
api/                      # FastAPI routers、schema、工作流管理与 WebUI
database/                 # SQLAlchemy 模型（本地 *.db 被忽略）
migrations/               # Alembic 数据库迁移
tests/analysis/           # AI 应用层回归测试
docs/                     # 架构、演示、面试和运行说明
```

## 测试与验证

```powershell
# AI 核心回归测试
uv run pytest tests/analysis/test_agent.py tests/analysis/test_semantic_repository.py tests/analysis/test_topic_service.py tests/analysis/test_public_opinion.py tests/analysis/test_trend_service.py -q

# 静态语法检查
uv run python -m compileall analysis api
node --check api/webui/assets/dashboard.js
```

项目中的本地数据库、原始评论、模型密钥和包含原文的质量报告均不会提交。仓库保留的是代码、迁移、测试、示例配置和仅包含聚合指标的验收说明。

## 已知边界

- 采集功能仅用于学习、研究和遵守平台规则的个人用途；请自行确认账号、数据和平台许可。
- 主题聚类和风险评分是辅助决策信号，不应替代人工审核。
- 向第三方 LLM/Embedding 服务发送文本前，需要自行确认数据处理合规性、最小化原则和费用。
- 当前语义检索以版本化 Embedding 和本地持久化为主，尚未引入独立向量数据库；这是当前数据规模下保持可演示、可测试、可控的取舍。
- Docker 相关配置保留在仓库中，但冻结验收以 Windows `start.bat` 本地启动链路为准。

## 上游致谢与许可

本项目基于 [NanmiCoder/MediaCrawler](https://github.com/NanmiCoder/MediaCrawler) 的多平台采集基础进行二次开发。保留原项目许可证、版权和使用限制；本仓库新增的 AI 应用层不改变上游许可约束。

> 本项目仅用于学习、研究与作品集演示，不用于商业化爬取、规避平台限制、侵犯隐私或其他违法违规用途。使用者应对其账号、访问频率、数据来源、存储与后续处理承担责任。
