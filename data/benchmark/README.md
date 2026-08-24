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
- 第一版：每类 3 张共 9 个 query，候选为三条路线 Top-5 在真实类型内的并集，
  score 列留空待专家填写。
