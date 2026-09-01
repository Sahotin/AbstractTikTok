# Phase 3B–Agent 基础完成验收报告

验收日期：2026-09-01。所有远程验证均在 `database/phase3b3_real_validation_20260901_105826.db` 完成；正式 `database/analysis.db` 未用于实验写入。

## 已完成

- DeepSeek `json_object` 兼容：3 条真实评论全部成功，结构化字段通过 Pydantic 并持久化。
- Qwen3 Embedding：1,907/1,907 成功，1,209 个唯一文本，13 个 Provider 请求，0 次重试，维度 2,560，耗时 17.808 秒。
- 质量门：排除 814 条低信息评论（42.6848%），其中纯数字 423、过短 189、纯表情 153、重复字符 31、泛化回复 18。
- 主题聚类候选参数：cosine distance threshold 0.30，minimum cluster size 5；29 个主题簇、846 条噪声、最大簇 20。
- 语义检索：查询“如何报名学习”在 1,093 条有效评论中 Top-1 为“怎么报名”，相似度 0.905221。
- 本地检索基准：1,093 × 2,560 float32 矩阵占 10.674 MiB；数据库快照解码 0.335 秒，矩阵构建 0.065 秒，200 次 Top-10 余弦检索平均 0.174 ms、P95 0.275 ms。
- Tool Calling Agent：DeepSeek 真实执行 3 个模型步骤，自主调用主题、评论统计和情感分布工具；正确指出仅 3/1,907 条具备 LLM 分析（0.16%），没有把缺失数据解释为零风险。
- Dashboard：真实浏览器加载 `/dashboard`，API、布局、趋势 SVG 和空覆盖率降级正常，控制台无错误。
- 自动化测试：最终 `153 passed`；Python 编译、OpenAPI 35 条路径、editable install、uv lock 和 Alembic fresh-database migration 均通过。
- Docker：Compose 配置和 Dockerfile 静态检查已完成；本机 Docker Desktop daemon 未运行，因此未执行镜像构建。

## 质量结论

质量门解决了“低信息文本形成 418 条巨大主题簇”的主要问题，但不能掩盖语料异质性：候选参数下，有效评论中的噪声比例仍为 `846 / 1093 = 77.4%`。因此当前主题输出适合探索和代表评论展示，不应包装成高精度自动主题标签系统。

## 未执行

未对剩余约 1,904 条评论执行全量 DeepSeek 分析。原因是这会产生外部付费调用，且不影响 Pipeline、聚合、语义检索、Agent 和 Dashboard 的工程验收。执行前应先确定预算、速率和抽样人工评测方案。

完整主题实验矩阵见 `benchmarks/topic_quality/phase3b3_topic_quality_20260901_031911.md`。
