# Benchmark 体系说明

核心原则：**"同项目 = 相似"只是弱先验，不能作为结构相似性的 Ground Truth。**
同一个项目内可能存在不同楼栋/户型/高度/柱网/结构布置；不同项目也可能采用
非常接近的标准户型与结构方案。

## Benchmark A：Type Classification（类型分类）

- 方法：Statistics（11D 最近邻，无 GNN）
- 指标：Accuracy / Precision / Recall / F1 / Confusion Matrix
- 回答："结构类型判断是否可靠？"

## Benchmark B：Weak Project Retrieval（弱项目检索）

- 文件：`weak_project_labels.csv`（query_id, candidate_id, weak_relevance 0/1）
- 相关集定义：`weak_project_relevance` = 同工程且同类（**仅弱先验**）
- 指标：MRR / NDCG@5 / Recall@5
- 运行：`python -m src.analysis.intra_type_benchmark`
- 作用：快速回归测试，测"能否找到同项目/同类案例"，**不能**回答"是否结构上最相似"
- 注意：仅包含存在 ≥1 个同项目同类案例的查询；无正样本的查询（如某工程仅 1 张图）
  被排除，归入未来的 Cross-Project Retrieval Benchmark。

## Benchmark C：Expert Retrieval（专家结构相似性检索）——最终裁决基准

- 文件：`expert_similarity.csv`（query_id, candidate_id, score 0-3, comment）
- 评分：0 = 不相似，1 = 略有参考价值，2 = 相似，3 = 高度相似
- 指标：NDCG@5 / MRR（score >= 2 视为相关）/ Pairwise Accuracy / Spearman
- 运行：`python -m src.benchmark.expert_retrieval`
- 目标规模：第一目标 30 query、第二目标 50 query（默认生成 50：
  框架 15 / 框剪 18 / 剪力墙 17，满足框架 ≥10、框剪 ≥15、剪力墙 ≥15）
- 多项目覆盖：同项目 query 按项目轮转挑选，避免集中在单一项目；
  并刻意包含"跨项目"query（该类型下所在项目仅 1 张图，如则安_框剪_1）。
- 子集报告：对每个查询，把候选拆分为**同项目 / 跨项目**两组分别报告 NDCG@5
  （跨项目子集 = 剔除同项目候选后的候选集）。因此每个类型的查询都能贡献跨项目样本；
  可通过 `--exclude-query <文件名>` 从评估中排除指定查询。
- 候选：三条路线 Top-5 在真实类型内的并集；已填分数保留，新 query 的 score 留空。

## Benchmark C v2：Top-K 候选集评估（0-5 分盲评）——当前主基准

- 研究问题：GNN 能否在 Top3~Top5 候选集合上提供**增量价值**（而非只争 Top1）。
- 盲评文件：`expert_similarity_top3.csv`（query_id, candidate_id, expert_score 0-5），
  只暴露 query 与候选，不暴露模型来源，避免评价偏差。
- 候选登记：`expert_candidates_top3.csv`
  （statistics_rank / gnn_rank / hybrid_rank / source，程序计算）。
- 第一轮：15 个 query（框架/框剪/剪力墙各 5，各来自 5 个不同项目），
  候选 = 三路线 Top5 并集 + 每 query 2 个 Hard Negative（同类型但最不相似）。
- 评分：0 = 基本不相似，1 = 弱相关，2 = 有一定相似性，3 = 有明确参考价值，
  4 = 很相似、参考价值高，5 = 高度相似、可直接作为主要参考。
- 指标：NDCG@1/3/5、Recall@1/3/5（相关 = score>=3）、Precision@3、Hit@3/5
  （至少一个 score>=4）、MeanExpertScore@3/5。
- 方法：statistics / gnn / hybrid / **union**（Statistics Top3 ∪ GNN Top3 → Hybrid 重排）。
- 增量价值：各方法相对 Statistics 的新增高价值候选（≥3 / ≥4 / =5 计数）。
- 运行：`python -m src.benchmark.topk_retrieval` 或 `--topk-benchmark`。
- 生成：`python -m src.benchmark.prepare_data --top3`。
- 历史归档：v1（Top1、0-3 分）保留在 `expert_similarity_v1_top1.csv`。
