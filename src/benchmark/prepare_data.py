"""生成 data/benchmark 下的标注文件。

1) weak_project_labels.csv：Benchmark B 的弱相关标签
   - 对每个存在 ≥1 个"同工程且同类"正样本的查询，列出全部同类候选，
     weak_relevance=1 表示同工程，否则 0。
   - 无同项目同类案例的查询（例如某工程只有 1 张图）被排除，
     这类查询属于未来的 Cross-Project Retrieval Benchmark。

2) expert_similarity.csv：Benchmark C 的第一轮候选模板
   - 取每类前 3 张（共 9 个 query）；候选 = statistics/gnn/hybrid 三路线
     Top-5 在真实类型内的并集；score 列留空待专家填写 0-3。

运行：python -m src.benchmark.prepare_data
"""

import csv
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from classification.classifier import StructureTypeClassifier
from datasets.structure_dataset import StructureDataset
from retrieval.config import load_retrieval_config
from retrieval.search_engine import SearchEngine

BENCH_DIR = os.path.join(ROOT_DIR, "data", "benchmark")


def _dataset(cfg):
    return StructureDataset(
        root=cfg.data_root,
        k=cfg.graph.k,
        connect_mode=cfg.graph.connect_mode,
    )


def _query_list(dataset, per_type=3):
    by_label = {}
    for i in range(len(dataset)):
        y = int(dataset[i].y.item())
        by_label.setdefault(y, []).append(dataset[i].filename)
    queries = []
    for y in sorted(by_label):
        queries.extend(sorted(by_label[y])[:per_type])
    return queries


def write_weak_labels(cfg_path=None, out_path=None):
    """生成 Benchmark B 的弱相关标签 CSV。"""
    cfg = load_retrieval_config(cfg_path)
    dataset = _dataset(cfg)
    names = [dataset[i].filename for i in range(len(dataset))]
    labels = [int(dataset[i].y.item()) for i in range(len(dataset))]
    projects = [name.split("_")[0] for name in names]

    rows = []
    for i in range(len(names)):
        positives = [
            j
            for j in range(len(names))
            if j != i and labels[j] == labels[i] and projects[j] == projects[i]
        ]
        if not positives:
            continue  # 无同项目同类正样本，不属于本项目基准
        for j in range(len(names)):
            if j == i or labels[j] != labels[i]:
                continue
            rows.append(
                (names[i], names[j], 1 if projects[j] == projects[i] else 0)
            )

    out_path = out_path or os.path.join(BENCH_DIR, "weak_project_labels.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "candidate_id", "weak_relevance"])
        writer.writerows(rows)
    print(f"[prepare] weak_project_labels.csv 已生成 {len(rows)} 行 -> {out_path}")


def write_expert_template(cfg_path=None, out_path=None, per_type=3, topk=5):
    """生成 Benchmark C 第一轮专家标注模板（score 留空）。"""
    cfg = load_retrieval_config(cfg_path)
    dataset = _dataset(cfg)
    true_label = {
        dataset[i].filename: int(dataset[i].y.item())
        for i in range(len(dataset))
    }
    queries = _query_list(dataset, per_type=per_type)

    clf = StructureTypeClassifier(cfg_path)
    engine = SearchEngine(cfg_path)

    rows = []
    for query in queries:
        pred = clf.predict_dxf(os.path.join(cfg.data_root, "raw", query))
        if pred is None:
            print(f"[prepare] 解析失败，跳过: {query}")
            continue
        type_id = true_label[query]  # 候选限定在真实类型内
        candidates = set()
        for method in ("statistics", "gnn", "hybrid"):
            result = engine.search_by_type(
                pred["data"],
                type_id=type_id,
                topk=topk,
                method=method,
            )
            candidates.update(item["filename"] for item in result["results"])
        for cand in sorted(candidates):
            rows.append([query, cand, "", ""])

    out_path = out_path or os.path.join(BENCH_DIR, "expert_similarity.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "candidate_id", "score", "comment"])
        writer.writerows(rows)
    print(
        f"[prepare] expert_similarity.csv 模板已生成 {len(rows)} 行 "
        f"（{len(queries)} 个 query）-> {out_path}"
    )


if __name__ == "__main__":
    write_weak_labels()
    write_expert_template()
