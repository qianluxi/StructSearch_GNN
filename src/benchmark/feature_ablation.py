"""论文实验一：Statistics 特征分组消融。

11D 结构统计描述子按语义分组：
- Type:      0-2   类型占比
- AreaRatio: 3-5   类型面积占比
- AreaDist:  6-8   面积分布（log1p 总面积 / CV / 熵）
- Spatial:   9-10  最近邻距离（位置）

输出：单组 / 递进 / 留一组消融 的类别检索 Top1/3/5，并保存 CSV/JSON。
运行：python -m src.benchmark.feature_ablation
"""

import csv
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from retrieval.config import load_retrieval_config

# 语义分组 -> 11D 列索引
FEATURE_GROUPS = {
    "Type": [0, 1, 2],
    "AreaRatio": [3, 4, 5],
    "AreaDist": [6, 7, 8],
    "Spatial": [9, 10],
}


def _load_stats(cfg):
    embeddings = np.load(
        os.path.join(cfg.embedding_dir, "embeddings.npy")
    ).astype(np.float64)
    labels = np.load(
        os.path.join(cfg.embedding_dir, "labels.npy")
    ).astype(np.int64)
    dim = cfg.embedding.dim
    return embeddings[:, dim:], labels


def topk_accuracy(stats_subset, labels, k):
    """leave-one-out 余弦检索的 Top-K 类别准确率。"""
    norm = stats_subset / np.linalg.norm(stats_subset, axis=1, keepdims=True)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)
    acc = 0
    for i in range(len(labels)):
        idx = np.argsort(-sim[i])[:k]
        if (labels[idx] == labels[i]).any():
            acc += 1
    return acc / len(labels)


def _row(name, cols, stats, labels):
    return {
        "group": name,
        "cols": cols,
        "top1": round(topk_accuracy(stats[:, cols], labels, 1), 4),
        "top3": round(topk_accuracy(stats[:, cols], labels, 3), 4),
        "top5": round(topk_accuracy(stats[:, cols], labels, 5), 4),
    }


def run(save_dir=None, cfg_path=None):
    """运行特征分组消融并输出结果。"""
    cfg = load_retrieval_config(cfg_path)
    stats, labels = _load_stats(cfg)
    all_cols = list(range(stats.shape[1]))
    save_dir = save_dir or os.path.join(ROOT_DIR, "fig", "benchmark", "ablation")
    os.makedirs(save_dir, exist_ok=True)

    rows = []

    # 1) 单组
    print("=== 单组检索 ===")
    for name, cols in FEATURE_GROUPS.items():
        row = _row(name, cols, stats, labels)
        rows.append(row)
        print(f'  {name:10s} Top1={row["top1"]:6.2%} Top3={row["top3"]:6.2%} Top5={row["top5"]:6.2%}')

    # 2) 递进
    print("\n=== 递进消融（逐步加入信息组） ===")
    order = list(FEATURE_GROUPS)
    progressive = []
    for i in range(len(order)):
        cols = []
        for name in order[: i + 1]:
            cols.extend(FEATURE_GROUPS[name])
        row = _row("+".join(order[: i + 1]), cols, stats, labels)
        rows.append(row)
        progressive.append(row)
        print(
            f'  {"+".join(order[: i + 1]):20s} '
            f'Top1={row["top1"]:6.2%} Top3={row["top3"]:6.2%} Top5={row["top5"]:6.2%}'
        )

    # 3) 留一组消融
    print("\n=== 留一组消融（全量去掉某组） ===")
    full = _row("Full", all_cols, stats, labels)
    rows.append(full)
    print(f'  {"Full":20s} Top1={full["top1"]:6.2%} Top3={full["top3"]:6.2%} Top5={full["top5"]:6.2%}')
    for name, cols in FEATURE_GROUPS.items():
        rest = [c for c in all_cols if c not in cols]
        row = _row(f"Full-{name}", rest, stats, labels)
        rows.append(row)
        print(
            f'  {"Full-" + name:20s} '
            f'Top1={row["top1"]:6.2%} Top3={row["top3"]:6.2%} Top5={row["top5"]:6.2%}'
        )

    # 保存
    csv_path = os.path.join(save_dir, "ablation.csv")
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group", "cols", "top1", "top3", "top5"])
        writer.writeheader()
        writer.writerows(rows)
    json_path = os.path.join(save_dir, "ablation.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {"feature_groups": FEATURE_GROUPS, "results": rows},
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[Feature Ablation] 结果已保存: {csv_path}")
    return rows


if __name__ == "__main__":
    run()
