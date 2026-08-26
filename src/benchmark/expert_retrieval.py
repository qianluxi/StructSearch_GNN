"""Benchmark C：专家结构相似性检索基准。

以人工标注的 (query_id, candidate_id, score 0-3) 为 Ground Truth，
对比 Statistics / GNN / Hybrid 三条类内检索路线：

- NDCG@5：分级相关性直接用专家分数（0-3）
- MRR：score >= mrr_threshold 视为相关
- Pairwise Accuracy：专家排序与模型排序一致的候选对比例
- Spearman：每查询模型相似度与专家分数的平均秩相关

查询的候选集按"是否与查询同项目"拆分为两个子集分别报告 NDCG@5：
- 同项目子集：与查询同项目的候选
- 跨项目子集：与查询不同项目的候选（评估时剔除同项目候选）

标注文件：data/benchmark/expert_similarity.csv（由专家填写）
运行：python -m src.benchmark.expert_retrieval
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

DEFAULT_LABELS = os.path.join(
    ROOT_DIR, "data", "benchmark", "expert_similarity_v1_top1.csv"
)


def _detect_encoding(path):
    """自动探测标注 CSV 编码（Excel 保存中文常为 GBK）。"""
    with open(path, "rb") as f:
        raw = f.read()
    for enc in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            raw.decode(enc)
            return enc
        except UnicodeDecodeError:
            continue
    return "utf-8"


def load_expert_labels(path):
    """读取专家标注 CSV -> {(query_id, candidate_id): score}。"""
    labels = {}
    with open(path, "r", encoding=_detect_encoding(path), newline="") as f:
        for row in csv.DictReader(f):
            query = (row.get("query_id") or "").strip()
            candidate = (row.get("candidate_id") or "").strip()
            raw = (row.get("score") or "").strip()
            if not query or not candidate or raw == "":
                continue
            try:
                score = float(raw)
            except ValueError:
                continue
            labels[(query, candidate)] = score
    return labels


def _rankdata(x):
    """平均秩（并列取均值）。"""
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(x):
        j = i
        while j + 1 < len(x) and x[order[j + 1]] == x[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        ranks[order[i : j + 1]] = avg
        i = j + 1
    return ranks


def spearman(x, y):
    """Spearman 秩相关（秩后 Pearson）。"""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    rx, ry = _rankdata(x), _rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _dcg(rel, k):
    return sum(float(r) / np.log2(i + 2) for i, r in enumerate(rel[:k]))


def ndcg_at_k(rel, k=5):
    rel = list(rel)
    idcg = _dcg(sorted(rel, reverse=True), k)
    return _dcg(rel, k) / idcg if idcg > 0 else 0.0


def pairwise_accuracy(ordered_candidates, scores):
    """按模型相似度降序的候选列表 vs 专家分数 -> 排序一致率。"""
    agree = total = 0
    for i in range(len(ordered_candidates)):
        for j in range(i + 1, len(ordered_candidates)):
            si = scores[ordered_candidates[i]]
            sj = scores[ordered_candidates[j]]
            if si == sj:
                continue
            total += 1
            if (si > sj) == (i < j):
                agree += 1
    return agree / total if total else float("nan")


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
    return embeddings


def _cosine(query_vec, db_vectors):
    db_norm = np.linalg.norm(db_vectors, axis=1)
    q_norm = np.linalg.norm(query_vec)
    denom = db_norm * q_norm
    denom[denom == 0] = 1e-12
    return (db_vectors @ query_vec) / denom


def _aggregate(items):
    """总体指标聚合：items = [(ndcg, mrr, pairwise, spearman), ...]。"""
    if not items:
        return {
            "ndcg5": 0.0,
            "mrr": 0.0,
            "pairwise_accuracy": 0.0,
            "spearman": 0.0,
            "n": 0,
        }
    return {
        "ndcg5": float(np.nanmean([it[0] for it in items])),
        "mrr": float(np.nanmean([it[1] for it in items])),
        "pairwise_accuracy": float(np.nanmean([it[2] for it in items])),
        "spearman": float(np.nanmean([it[3] for it in items])),
        "n": len(items),
    }


def _aggregate_ndcg_mrr(items):
    """子集指标聚合：items = [(ndcg, mrr), ...]。"""
    if not items:
        return {"ndcg5": 0.0, "mrr": 0.0, "n": 0}
    return {
        "ndcg5": float(np.nanmean([it[0] for it in items])),
        "mrr": float(np.nanmean([it[1] for it in items])),
        "n": len(items),
    }


def run(
    save_dir=None,
    labels_csv=None,
    mrr_threshold=2.0,
    topk=5,
    cfg_path=None,
    exclude_queries=None,
):
    """运行专家检索基准（含同项目/跨项目子集拆分）。

    Args:
        save_dir: 输出目录（默认 fig/benchmark/expert）
        labels_csv: 专家标注 CSV；默认 data/benchmark/expert_similarity.csv
        mrr_threshold: MRR 判定"相关"的分数阈值（0-3 制，默认 >=2 视为相关）
        exclude_queries: 需要从评估中排除的查询文件名列表（如则安_框剪_1）
    """
    cfg = load_retrieval_config(cfg_path)
    save_dir = save_dir or os.path.join(ROOT_DIR, "fig", "benchmark", "expert")
    os.makedirs(save_dir, exist_ok=True)
    exclude_queries = set(exclude_queries or [])

    labels_csv = labels_csv or DEFAULT_LABELS
    labels = load_expert_labels(labels_csv)
    if not labels:
        print(f"[Expert Benchmark] 尚未有人工标注：{labels_csv}")
        print("请先由专家填写 data/benchmark/expert_similarity.csv 的 score 列。")
        return None

    embeddings, filenames, labels_vec = _load_index(cfg)
    dim = cfg.embedding.dim
    projects = np.array([name.split("_")[0] for name in filenames])
    name_to_idx = {name: i for i, name in enumerate(filenames)}

    queries = {}
    for (q, c), score in labels.items():
        queries.setdefault(q, {})[c] = score

    methods = ["statistics", "gnn", "hybrid"]
    records = {m: {"all": [], "same": [], "cross": []} for m in methods}

    for query, cand_scores in queries.items():
        if query in exclude_queries:
            print(f"[Expert Benchmark] 按配置排除查询: {query}")
            continue
        if query not in name_to_idx:
            print(f"[Expert Benchmark] 查询不在库中，跳过: {query}")
            continue
        qi = name_to_idx[query]

        cand_idx = []
        for cand in cand_scores:
            if cand not in name_to_idx:
                print(f"[Expert Benchmark] 候选不在库中，跳过: {cand}")
                continue
            cand_idx.append(name_to_idx[cand])
        if not cand_idx:
            continue

        for method in methods:
            db_vecs = _method_vectors(embeddings, method, dim)
            q_vec = db_vecs[qi]
            sims = _cosine(q_vec, db_vecs[cand_idx])

            # 全量候选上的指标（总体）
            order = np.argsort(-sims)  # 模型相似度降序的候选索引
            ordered = [cand_idx[i] for i in order]
            rel = [cand_scores[filenames[i]] for i in ordered]

            ndcg = ndcg_at_k(rel, topk)

            relevant = [i for i, r in enumerate(rel) if r >= mrr_threshold]
            mrr = 1.0 / (relevant[0] + 1) if relevant else 0.0

            pair = pairwise_accuracy(
                [filenames[i] for i in ordered],
                {filenames[i]: cand_scores[filenames[i]] for i in cand_idx},
            )
            sp = spearman(sims, [cand_scores[filenames[i]] for i in cand_idx])

            records[method]["all"].append((ndcg, mrr, pair, sp))

            # 子集：同项目 / 跨项目（候选级拆分）
            q_proj = query.split("_")[0]
            pools = {
                "same": {
                    c: s for c, s in cand_scores.items()
                    if c.split("_")[0] == q_proj
                },
                "cross": {
                    c: s for c, s in cand_scores.items()
                    if c.split("_")[0] != q_proj
                },
            }
            for subset_name, pool in pools.items():
                if not pool:
                    continue
                pool_names = list(pool.keys())
                pool_idx = [name_to_idx[c] for c in pool_names]
                sims_sub = _cosine(q_vec, db_vecs[pool_idx])
                order_sub = np.argsort(-sims_sub)
                rel_sub = [pool[pool_names[k]] for k in order_sub]
                ndcg_sub = ndcg_at_k(rel_sub, topk)
                relevant_sub = [
                    i for i, r in enumerate(rel_sub) if r >= mrr_threshold
                ]
                mrr_sub = (
                    1.0 / (relevant_sub[0] + 1) if relevant_sub else 0.0
                )
                records[method][subset_name].append((ndcg_sub, mrr_sub))

    print("\n=== Benchmark C：Expert Retrieval ===")
    print(
        f"（标注 {len(labels)} 对，{len(queries)} 个 query；"
        f"MRR 相关阈值 score>={mrr_threshold:.0f}）"
    )
    print(f'{"method":12s} {"NDCG@5":>8s} {"MRR":>8s} {"Pairwise":>9s} {"Spearman":>9s}')

    summary = {}
    for method in methods:
        row = _aggregate(records[method]["all"])
        row["subsets"] = {
            "same": _aggregate_ndcg_mrr(records[method]["same"]),
            "cross": _aggregate_ndcg_mrr(records[method]["cross"]),
        }
        row["excluded_queries"] = sorted(exclude_queries)
        summary[method] = row
        print(
            f'{method:12s} {row["ndcg5"]:8.4f} {row["mrr"]:8.4f} '
            f'{row["pairwise_accuracy"]:9.2%} {row["spearman"]:9.4f}'
        )

    print("\n=== 子集拆分：NDCG@5（同项目 / 跨项目） ===")
    print(f'{"method":12s} {"同项目(n)":>12s} {"跨项目(n)":>12s}')
    for method in methods:
        s = summary[method]["subsets"]["same"]
        c = summary[method]["subsets"]["cross"]
        print(f'{method:12s} {s["ndcg5"]:8.4f}({s["n"]}) {c["ndcg5"]:8.4f}({c["n"]})')
    if exclude_queries:
        print(f"（已排除查询: {sorted(exclude_queries)}）")

    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[Expert Benchmark] 结果已保存: {metrics_path}")
    return summary


if __name__ == "__main__":
    run()
