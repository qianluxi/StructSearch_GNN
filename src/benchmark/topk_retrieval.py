"""Benchmark C v2：Top-K 候选集评估（Top3~Top5 视角）。

研究问题：GNN 是否能在 Top-K 候选集合上提供增量价值（而不仅是 Top1 排序）。

数据：
- 盲评标注：data/benchmark/expert_similarity_top3.csv
  （query_id, candidate_id, expert_score 0-5，由专家填写）
- 候选登记：data/benchmark/expert_candidates_top3.csv（各路线排名，程序用）

方法：
- statistics / gnn / hybrid：在标注候选池内按各自余弦相似度取 Top3/Top5
- union：Statistics Top3 ∪ GNN Top3，再用 Hybrid 相似度重排取 Top3

指标（k=3/5，相关阈值默认 score>=3）：
- NDCG@k（分级相关性=专家分数）
- Recall@k（相关候选覆盖率）、Precision@k、Hit@k（至少一个 score>=4）
- MeanExpertScore@k（Top-k 平均专家分）

增量价值：对每个方法，统计其 Top3 中"Statistics Top3 未覆盖"的新增候选，
并按专家分 >=3 / >=4 / ==5 计数。

运行：python -m src.benchmark.topk_retrieval
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

from benchmark.expert_retrieval import _detect_encoding
from retrieval.config import load_retrieval_config

DEFAULT_LABELS = os.path.join(ROOT_DIR, "data", "benchmark", "expert_similarity_top3.csv")


def load_scores(path):
    """读取盲评标注 -> {(query_id, candidate_id): score}。"""
    scores = {}
    with open(path, "r", encoding=_detect_encoding(path), newline="") as f:
        for row in csv.DictReader(f):
            q = (row.get("query_id") or "").strip()
            c = (row.get("candidate_id") or "").strip()
            raw = (row.get("expert_score") or "").strip()
            if not q or not c or raw == "":
                continue
            try:
                scores[(q, c)] = float(raw)
            except ValueError:
                continue
    return scores


def ndcg_at_k(rel, k):
    rel = list(rel[:k])
    idcg = sum(sorted(rel, reverse=True)[i] / np.log2(i + 2) for i in range(len(rel)))
    dcg = sum(r / np.log2(i + 2) for i, r in enumerate(rel))
    return dcg / idcg if idcg > 0 else 0.0


def recall_at_k(rel, k, threshold=3.0):
    rel = list(rel)
    total = sum(1 for r in rel if r >= threshold)
    if total == 0:
        return 0.0
    return sum(1 for r in rel[:k] if r >= threshold) / total


def precision_at_k(rel, k, threshold=3.0):
    return sum(1 for r in rel[:k] if r >= threshold) / k


def hit_at_k(rel, k, threshold=4.0):
    return 1.0 if any(r >= threshold for r in rel[:k]) else 0.0


def mean_score_at_k(rel, k):
    rel = list(rel[:k])
    return float(np.mean(rel)) if rel else 0.0


def union_top3(stats_order, gnn_order, hybrid_sim):
    """Statistics Top3 ∪ GNN Top3，按 Hybrid 相似度重排取 Top3。"""
    union = list(stats_order[:3])
    for cand in gnn_order[:3]:
        if cand not in union:
            union.append(cand)
    return sorted(union, key=lambda c: -hybrid_sim[c])[:3]


def incremental_novel(method_top3, stats_top3, scores):
    """方法 Top3 中 Statistics Top3 未覆盖的新增候选及其专家分值。"""
    novel = [c for c in method_top3 if c not in stats_top3]
    return {
        "n_novel": len(novel),
        "novel_ge3": sum(1 for c in novel if scores.get(c, 0) >= 3),
        "novel_ge4": sum(1 for c in novel if scores.get(c, 0) >= 4),
        "novel_eq5": sum(1 for c in novel if scores.get(c, 0) >= 5),
    }


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


def _method_vectors(embeddings, method, dim):
    if method == "statistics":
        return embeddings[:, dim:]
    if method == "gnn":
        return embeddings[:, :dim]
    return embeddings


def _cosine(query_vec, db_vectors):
    db_norm = np.linalg.norm(db_vectors, axis=1)
    q_norm = np.linalg.norm(query_vec)
    denom = db_norm * q_norm
    denom[denom == 0] = 1e-12
    return (db_vectors @ query_vec) / denom


def _method_order(method, q_idx, cand_idx, embeddings, dim):
    db = _method_vectors(embeddings, method, dim)
    sims = _cosine(db[q_idx], db[cand_idx])
    order = [cand_idx[i] for i in np.argsort(-sims)]
    sim = {cand_idx[i]: float(sims[i]) for i in range(len(cand_idx))}
    return order, sim


def run(
    save_dir=None,
    labels_csv=None,
    k_values=(3, 5),
    relevance_threshold=3.0,
    hit_threshold=4.0,
    cfg_path=None,
):
    """运行 Top-K 候选集评估。"""
    cfg = load_retrieval_config(cfg_path)
    save_dir = save_dir or os.path.join(ROOT_DIR, "fig", "benchmark", "topk")
    os.makedirs(save_dir, exist_ok=True)

    labels_csv = labels_csv or DEFAULT_LABELS
    scores = load_scores(labels_csv)
    if not scores:
        print(f"[TopK Benchmark] 尚未有人工标注：{labels_csv}")
        print("请先由专家填写 data/benchmark/expert_similarity_top3.csv 的 expert_score 列。")
        return None

    embeddings, filenames = _load_index(cfg)
    dim = cfg.embedding.dim
    name_to_idx = {name: i for i, name in enumerate(filenames)}

    queries = {}
    for (q, c), s in scores.items():
        queries.setdefault(q, {})[c] = s

    methods = ["statistics", "gnn", "hybrid"]
    per_query = {m: [] for m in methods + ["union"]}
    top3_store = {m: {} for m in methods + ["union"]}

    for query, cand_scores in queries.items():
        if query not in name_to_idx:
            print(f"[TopK Benchmark] 查询不在库中，跳过: {query}")
            continue
        qi = name_to_idx[query]
        cand_idx = []
        for cand in cand_scores:
            if cand not in name_to_idx:
                continue
            cand_idx.append(name_to_idx[cand])
        if not cand_idx:
            continue

        orders = {}
        sims = {}
        for m in methods:
            order, sim = _method_order(m, qi, cand_idx, embeddings, dim)
            orders[m] = order
            sims[m] = sim

        rel = {
            m: [cand_scores[filenames[i]] for i in orders[m]]
            for m in methods
        }

        # union：Statistics Top3 ∪ GNN Top3，Hybrid 重排
        union_order = union_top3(
            [filenames[i] for i in orders["statistics"]],
            [filenames[i] for i in orders["gnn"]],
            {filenames[i]: sims["hybrid"][i] for i in cand_idx},
        )
        orders["union"] = [name_to_idx[c] for c in union_order]
        rel["union"] = [cand_scores[c] for c in union_order]

        for m in methods + ["union"]:
            top3_store[m][query] = [filenames[i] for i in orders[m]][:3]
            metrics = {}
            for k in k_values:
                metrics[f"ndcg{k}"] = ndcg_at_k(rel[m], k)
                metrics[f"recall{k}"] = recall_at_k(
                    rel[m], k, relevance_threshold
                )
                metrics[f"hit{k}"] = hit_at_k(rel[m], k, hit_threshold)
                metrics[f"mean_score{k}"] = mean_score_at_k(rel[m], k)
                if k == 3:
                    metrics["precision3"] = precision_at_k(
                        rel[m], k, relevance_threshold
                    )
            metrics["ndcg1"] = ndcg_at_k(rel[m], 1)
            metrics["recall1"] = recall_at_k(rel[m], 1, relevance_threshold)
            per_query[m].append(metrics)

    def _mean(items, key):
        return float(np.mean([it[key] for it in items])) if items else 0.0

    summary = {}
    print("\n=== Benchmark C v2：Top-K 候选集评估 ===")
    print(f"（标注 {len(scores)} 对，{len(queries)} query；相关=score>={relevance_threshold:.0f}）")
    header = (
        f'{"method":12s} {"NDCG@3":>8s} {"Recall@3":>9s} {"P@3":>6s} '
        f'{"Hit@3":>7s} {"Mean@3":>8s} {"NDCG@5":>8s} {"Recall@5":>9s}'
    )
    print(header)
    for m in methods + ["union"]:
        row = {
            key: _mean(per_query[m], key)
            for key in per_query[m][0]
        } if per_query[m] else {}
        row["n_queries"] = len(per_query[m])
        summary[m] = row
        print(
            f'{m:12s} {row.get("ndcg3", 0):8.4f} {row.get("recall3", 0):9.2%} '
            f'{row.get("precision3", 0):6.2%} {row.get("hit3", 0):7.2%} '
            f'{row.get("mean_score3", 0):8.3f} {row.get("ndcg5", 0):8.4f} '
            f'{row.get("recall5", 0):9.2%}'
        )

    print("\n=== 增量价值（相对 Statistics Top3） ===")
    print(f'{"method":12s} {"新增":>5s} {"新增≥3":>7s} {"新增≥4":>7s} {"新增=5":>7s}')
    incremental = {}
    for m in ["gnn", "hybrid", "union"]:
        novel_counts = {"n_novel": 0, "novel_ge3": 0, "novel_ge4": 0, "novel_eq5": 0}
        for query, cand_scores in queries.items():
            if query not in top3_store["statistics"] or query not in top3_store[m]:
                continue
            inc = incremental_novel(
                top3_store[m][query],
                top3_store["statistics"][query],
                cand_scores,
            )
            for key in novel_counts:
                novel_counts[key] += inc[key]
        incremental[m] = novel_counts
        print(
            f'{m:12s} {novel_counts["n_novel"]:5d} {novel_counts["novel_ge3"]:7d} '
            f'{novel_counts["novel_ge4"]:7d} {novel_counts["novel_eq5"]:7d}'
        )

    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "methods": summary,
                "incremental_vs_statistics": incremental,
                "config": {
                    "relevance_threshold": relevance_threshold,
                    "hit_threshold": hit_threshold,
                    "k_values": list(k_values),
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\n[TopK Benchmark] 结果已保存: {metrics_path}")
    return summary


if __name__ == "__main__":
    run()
