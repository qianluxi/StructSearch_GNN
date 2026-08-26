"""论文实验三：传统 Baseline 对比（在 Expert Top-K 标注上评估）。

方法：
- Random：候选池随机排序（固定种子 42）
- Raw Statistics：未标准化的 11D 统计特征（cosine）
- Statistics（cosine / euclidean / manhattan）：标准化 11D
- GNN（cosine）：128D 嵌入
- Hybrid（cosine）：139D 组合

指标：NDCG@3/5、Recall@3/5（相关 = score>=3）、Precision@3、MeanScore@3。
运行：python -m src.benchmark.baselines
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

from benchmark.topk_retrieval import (
    load_scores,
    mean_score_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from datasets.structure_dataset import StructureDataset
from retrieval.config import load_retrieval_config

DEFAULT_LABELS = os.path.join(ROOT_DIR, "data", "benchmark", "expert_similarity_top3.csv")


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
    return embeddings, filenames


def _raw_stats_matrix(cfg):
    """从数据集缓存读取未标准化的 11D 图统计特征矩阵 (N, 11)。"""
    dataset = StructureDataset(
        root=cfg.data_root,
        k=cfg.graph.k,
        connect_mode=cfg.graph.connect_mode,
    )
    return np.vstack(
        [dataset[i].graph_feature_raw.numpy().ravel() for i in range(len(dataset))]
    ).astype(np.float64)


def _rank_pool(method, q_vec, pool_vecs, rng):
    """按方法对候选池排序，返回降序相似/升序距离的索引。"""
    if method == "random":
        scores = rng.random(len(pool_vecs))
        return np.argsort(-scores)
    if method.endswith("euclidean") or method.endswith("manhattan"):
        diff = pool_vecs - q_vec
        if method.endswith("euclidean"):
            dist = np.linalg.norm(diff, axis=1)
        else:
            dist = np.abs(diff).sum(axis=1)
        return np.argsort(dist)
    # cosine 类
    db_norm = np.linalg.norm(pool_vecs, axis=1)
    q_norm = np.linalg.norm(q_vec)
    denom = db_norm * q_norm
    denom[denom == 0] = 1e-12
    sim = (pool_vecs @ q_vec) / denom
    return np.argsort(-sim)


def _method_vectors(method, embeddings, raw_stats, dim):
    if method in ("raw_stats", "statistics_cosine", "statistics_euclidean", "statistics_manhattan"):
        return raw_stats if method == "raw_stats" else embeddings[:, dim:]
    if method == "gnn":
        return embeddings[:, :dim]
    return embeddings


def run(save_dir=None, labels_csv=None, cfg_path=None, seed=42):
    """运行传统 Baseline 对比。"""
    cfg = load_retrieval_config(cfg_path)
    save_dir = save_dir or os.path.join(ROOT_DIR, "fig", "benchmark", "baselines")
    os.makedirs(save_dir, exist_ok=True)

    labels_csv = labels_csv or DEFAULT_LABELS
    scores = load_scores(labels_csv)
    if not scores:
        print(f"[Baselines] 尚未有人工标注：{labels_csv}")
        return None

    embeddings, filenames = _load_index(cfg)
    raw_stats = _raw_stats_matrix(cfg)
    dim = cfg.embedding.dim
    name_to_idx = {name: i for i, name in enumerate(filenames)}

    queries = {}
    for (q, c), s in scores.items():
        queries.setdefault(q, {})[c] = s

    methods = [
        "random",
        "raw_stats",
        "statistics_cosine",
        "statistics_euclidean",
        "statistics_manhattan",
        "gnn",
        "hybrid",
    ]
    rng = np.random.default_rng(seed)
    per_query = {m: [] for m in methods}

    for query, cand_scores in queries.items():
        if query not in name_to_idx:
            continue
        qi = name_to_idx[query]
        cand_idx = [name_to_idx[c] for c in cand_scores if c in name_to_idx]
        if not cand_idx:
            continue

        for method in methods:
            vecs = _method_vectors(method, embeddings, raw_stats, dim)
            if method == "raw_stats":
                q_vec = raw_stats[qi]
            else:
                q_vec = vecs[qi]
            pool_vecs = vecs[cand_idx]
            order = _rank_pool(method, q_vec, pool_vecs, rng)
            rel = [cand_scores[filenames[cand_idx[i]]] for i in order]
            per_query[method].append({
                "ndcg3": ndcg_at_k(rel, 3),
                "ndcg5": ndcg_at_k(rel, 5),
                "recall3": recall_at_k(rel, 3),
                "recall5": recall_at_k(rel, 5),
                "precision3": precision_at_k(rel, 3),
                "mean_score3": mean_score_at_k(rel, 3),
            })

    def _mean(items, key):
        return float(np.mean([it[key] for it in items])) if items else 0.0

    print("\n=== 论文实验三：传统 Baseline 对比（Expert Top-K，15 query） ===")
    print(f'{"method":24s} {"NDCG@3":>8s} {"NDCG@5":>8s} {"Recall@3":>9s} {"Recall@5":>9s} {"P@3":>6s} {"Mean@3":>8s}')
    summary = {}
    for method in methods:
        row = {key: _mean(per_query[method], key) for key in per_query[method][0]}
        row["n_queries"] = len(per_query[method])
        summary[method] = row
        print(
            f'{method:24s} {row["ndcg3"]:8.4f} {row["ndcg5"]:8.4f} '
            f'{row["recall3"]:9.2%} {row["recall5"]:9.2%} '
            f'{row["precision3"]:6.2%} {row["mean_score3"]:8.3f}'
        )

    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[Baselines] 结果已保存: {metrics_path}")
    return summary


if __name__ == "__main__":
    run()
