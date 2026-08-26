# 论文准备计划（草稿）

定位：**面向工程结构案例检索的结构表征与多阶段相似性检索方法**。
GNN 作为对照实验，不承担核心方法角色。

## 核心贡献

1. **结构体系约束的两阶段检索框架**：Stage 1 统计判型（框架/框剪/剪力墙）→
   Stage 2 类型内相似检索。核心思想：结构相似性在结构体系约束下做局部检索，
   而非全空间直接比对。
2. **结构统计描述子（Structural Statistical Descriptor）**：由类型占比、面积占比、
   面积分布、空间分布组成的 11D 描述；实验证明其在该任务上强于 GNN 表征
   （Top-K 专家评估中 NDCG@3：0.72 vs GNN 0.59）。
3. **专家结构相似性基准（Expert Structural Similarity Benchmark）**：
   证明"同项目"只能作为弱先验（同项目平均 1.58 vs 跨项目 0.55），
   并建立 0-5 分专家标注 + Top-K（NDCG@3/5、Recall@3/5）评价体系。
4. **对深度图表示与工程先验表示的系统比较**：GNN 以分类目标训练导致类内嵌入坍缩
   （框架类两两相似度 std≈0.001），不适合细粒度排序——作为研究发现而非失败。

## 方法原则：Project-blind Retrieval, Project-aware Evaluation

**检索完全不知道项目名称**：参与计算的只有结构类型、构件比例、面积、面积分布、
空间统计特征与标准化特征（相似度使用 Manhattan 距离）。项目名仅作为评估元数据：

| 信息 | 检索 | 评估 |
| --- | :-: | :-: |
| 结构类型 | ✅ | ✅ |
| 构件比例 / 面积 / 空间特征 | ✅ | — |
| 项目名称 | ❌ | ✅（同/跨项目分析） |
| 专家评分 | ❌ | ✅（主结果） |

代码约束：`datasets.schemas.parse_drawing_meta` 统一解析项目/图纸标识，
检索模块（classifier / search_engine）不使用项目名；基准模块用其做
"同项目 / 跨项目"分析维度，而非相关性标签。测试见 `tests/test_project_blind.py`。

## 论文实验清单

| 实验 | 模块 | 状态 |
| --- | --- | --- |
| 结构类型识别（Stage 1） | src/classification | 完成 |
| 两阶段 vs 直接检索对比 | src/benchmark | 完成（弱基准） |
| 特征分组消融（Type/Area/Spatial） | src/benchmark/feature_ablation.py | 完成 |
| 专家基准（0-3 与 0-5 两版） | src/benchmark/expert_retrieval.py / topk_retrieval.py | 完成 |
| 同项目 vs 跨项目子集（NDCG@3/5、Recall@3/5） | expert_retrieval 子集拆分 | 完成 |
| 传统 Baseline 对比（Random/原始统计/欧氏/曼哈顿/GNN/Hybrid） | src/benchmark/baselines.py | 完成 |
| 结构空间可视化（t-SNE/PCA） | src/tools/visualize | 已有 |

## 关键结果（当前数据）

- 类型识别（11D 最近邻）：Top-1 ≈ 87.7%（106 张库内）。
- 消融：类型占比为主要信息源（单组 Top-1 86.8%），位置组当前表示贡献弱。
- Top-K 专家评估（15 query / 171 对，0-5 分）：Statistics NDCG@3 0.721 最优；
  GNN 总体最弱，但提供少量增量高分候选（集中在框架类）。
- 跨项目子集：Statistics NDCG@5 0.50 > Hybrid 0.46 > GNN 0.27。
- Baseline：标准化 Statistics 显著优于 Random/原始统计；曼哈顿距离最优
  （NDCG@3 0.746），距离度量选择本身是论文可讨论点。

## 后续可补

- Union 重排器研究（简单 Hybrid 重排未超越 Statistics，重排器是关键）。
- 更大规模专家标注与跨项目样本。
- UMAP 可视化作为辅助材料。

## 最终验收实验（2026-08-26，全部通过）

### ① Project-blind 静态代码审计

结论见 `docs/project_blind_audit.md`：检索路径（classifier / search_engine /
processor / index_builder）中的 filename 仅用于"排除自身 + 结果展示"；
`rg "project_id" src/retrieval src/classification src/pipeline` 为 0 处；
项目名解析（`parse_drawing_meta`）只出现在数据准备与基准（评估）模块。

### ② 端到端 Project-Blind 验证

`python -m src.benchmark.project_blind_verify`

同一 DXF 复制为两个随机命名副本（内容逐字节相同），跑完整两阶段流水线：
结构类型、置信度、Top-5 文件名与相似度分数**逐位一致**（CPU 下确定性比较）。

### ③ 项目名置换检验

`python -m src.benchmark.project_permutation`

对 21 个项目名做确定性置换（seed 42 改名 19 个、seed 7 改名 20 个），
Expert NDCG@3（Statistics 0.7214 / GNN 0.5947 / Hybrid 0.7135）与
weak MRR（0.5800）**完全不变**；项目感知弱指标同样不变（等价关系在置换下保持，
数学上符合预期）。结论：检索指标对项目身份置换不变，可作为论文中的
"Project identity does not participate in retrieval" 的实证证据。

## 当前结论

方法可冻结：算法、数据处理、评价标准均已稳定，33+ 项测试全绿。
下一步进入论文写作，不再进行大规模程序开发。
