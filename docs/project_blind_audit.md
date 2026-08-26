# Project-blind 静态代码审计

审计时间：2026-08-26（main 分支，commit 8448669 之后）

## 审计目标

确认"项目名称只进入评估路径、绝不进入检索路径"在代码层面成立：

- **允许**：metadata / benchmark / analysis / visualization
- **禁止**：classifier / feature vector / distance / ranking / filter / score

## 检索路径中的 filename 用法（允许项）

| 位置 | 用途 |
| --- | --- |
| `src/classification/classifier.py:86-88` | `query_name` 仅用于排除自身（`sims[...] = -1.0`） |
| `src/retrieval/search_engine.py:165-170` | `query_name` 仅用于排除自身（`-np.inf`） |
| `src/retrieval/search_engine.py:182/188/266/281` | 结果展示（filename/label 字段） |
| `src/retrieval/processor.py:83` | 仅记录 `basename` 到 Data |
| `src/retrieval/index_builder.py` | 仅保存 filenames 供展示与自排除 |

结论：检索计算（`_query_vector`、`_cosine_similarity`、GraphEncoder、11D 统计）
不读取文件名/项目名；filename 只承担"自排除标识"与"结果展示"两个职责。

## 项目名解析的使用位置（评估路径）

| 位置 | 用途 |
| --- | --- |
| `src/datasets/schemas.py` | `DrawingMeta` / `parse_drawing_meta` 定义 |
| `src/benchmark/prepare_data.py` | 查询选样与弱标签生成 |
| `src/benchmark/expert_retrieval.py:279-287` | 同项目/跨项目分析维度拆分 |
| `src/analysis/intra_type_benchmark.py:76` | weak_project_relevance 相关集 |

## 检索模块中的 project_id 引用

`rg "project_id" src/retrieval src/classification src/pipeline`：**0 处**。

## 结论

静态审计通过：项目名称不进入任何特征、距离、排序、过滤或打分逻辑；
仅作为评估/展示元数据使用。结合 `tests/test_project_blind.py` 的动态回归测试，
"Project-blind Retrieval, Project-aware Evaluation"在代码层面成立。
