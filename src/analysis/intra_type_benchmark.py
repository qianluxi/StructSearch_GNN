"""Stage 2 基准：类内相似检索三路线对比（Statistics / GNN / Hybrid）。

第一版无需人工标注：
1. 类内两两相似度分布（观察各路线能否把同类拉开，尤其框架类是否坍缩）；
2. 以“同工程且同类”为相关集（弱代理），计算 MRR / NDCG@5 / Recall@5。

直接运行：python -m src.analysis.intra_type_benchmark
推荐通过 CLI：python -m src.retrieval.cli --intra-benchmark
"""

import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from retrieval.config import CLASS_NAMES, load_retrieval_config


def _load_index(cfg):
    embeddings = np.load(
        os.path.join(cfg.embedding_dir, "embeddings.npy")
    ).astype(np.float64)
    filenames = np.asarray(
        np.load(
            os.path.join(cfg.embedding_dir, "filenames.npy"),
            allow_pickle=True,
        )
    ).astype(str)
    labels = np.load(
        os.path.join(cfg.embedding_dir, "labels.npy")
    ).astype(np.int64)
    return embeddings, filenames, labels


def _method_vectors(embeddings, method, dim):
    if method == "statistics":
        return embeddings[:, dim:]
    if method == "gnn":
        return embeddings[:, :dim]
    return embeddings  # hybrid


def _cosine_sim_matrix(vecs):
    norm = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    sim = norm @ norm.T
    np.fill_diagonal(sim, np.nan)
    return sim


def _dcg_at_k(rel, k):
    return sum(float(r) / np.log2(i + 2) for i, r in enumerate(rel[:k]))


def _ndcg_at_k(rel, k):
    dcg = _dcg_at_k(rel, k)
    idcg = _dcg_at_k(sorted(rel, reverse=True), k)
    return dcg / idcg if idcg > 0 else 0.0


def run(save_dir=None, cfg_path=None):
    """运行类内相似检索基准，输出三路线对比表与相似度分布。"""
    cfg = load_retrieval_config(cfg_path)
    save_dir = save_dir or os.path.join(ROOT_DIR, "fig", "benchmark", "intra_type")
    os.makedirs(save_dir, exist_ok=True)

    embeddings, filenames, labels = _load_index(cfg)
    dim = cfg.embedding.dim
    projects = np.array([name.split("_")[0] for name in filenames])
    methods = ["statistics", "gnn", "hybrid"]

    print("\n=== Benchmark B：Weak Project Retrieval（类内三路线对比） ===")
    print(
        "（相关集 = weak_project_relevance：同工程且同类；"
        "仅作弱先验，不代表结构相似性 Ground Truth；n 为有相关集的查询数）"
    )
    print(f'{"method":12s} {"MRR":>8s} {"NDCG@5":>8s} {"Recall@5":>9s} {"n":>4s}')

    summary = {}
    for method in methods:
        vecs = _method_vectors(embeddings, method, dim)
        sim = _cosine_sim_matrix(vecs)

        mrrs, ndcgs, recalls, n = [], [], [], 0
        for i in range(len(labels)):
            same_type = labels == labels[i]
            relevant = (
                same_type
                & (projects == projects[i])
                & (np.arange(len(labels)) != i)
            )
            if not relevant.any():
                continue

            candidates = np.where(same_type & (np.arange(len(labels)) != i))[0]
            order = candidates[np.argsort(-sim[i][candidates])]
            rel = np.array([1 if relevant[j] else 0 for j in order])

            mrr = 1.0 / (np.argmax(rel) + 1) if rel.any() else 0.0
            mrrs.append(mrr)
            ndcgs.append(_ndcg_at_k(rel, 5))
            recalls.append(rel[:5].sum() / rel.sum())
            n += 1

        row = {
            "mrr": float(np.mean(mrrs)) if n else 0.0,
            "ndcg5": float(np.mean(ndcgs)) if n else 0.0,
            "recall5": float(np.mean(recalls)) if n else 0.0,
            "n": n,
        }
        summary[method] = row
        print(
            f'{method:12s} {row["mrr"]:8.4f} {row["ndcg5"]:8.4f} '
            f'{row["recall5"]:9.2%} {row["n"]:4d}'
        )

    print("\n=== 类内两两余弦相似度分布（std 越小说明同类越难拉开） ===")
    spread = {}
    for type_id in sorted(CLASS_NAMES):
        mask = labels == type_id
        print(f'--- {CLASS_NAMES[type_id]}（{mask.sum()} 张） ---')
        for method in methods:
            vecs = _method_vectors(embeddings, method, dim)
            s = _cosine_sim_matrix(vecs[mask])
            row = {
                "mean": float(np.nanmean(s)),
                "std": float(np.nanstd(s)),
                "min": float(np.nanmin(s)),
                "max": float(np.nanmax(s)),
            }
            spread.setdefault(CLASS_NAMES[type_id], {})[method] = row
            print(
                f'  {method:12s} mean={row["mean"]:.4f} '
                f'std={row["std"]:.4f} min={row["min"]:.4f} max={row["max"]:.4f}'
            )

    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "weak_project_retrieval": summary,
                "similarity_spread": spread,
                "note": (
                    "weak_project_relevance 仅是同工程且同类的弱先验，"
                    "不能作为结构相似性 Ground Truth"
                ),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[Intra-Type Benchmark] 结果已保存: {metrics_path}")
    return summary


if __name__ == "__main__":
    run()
