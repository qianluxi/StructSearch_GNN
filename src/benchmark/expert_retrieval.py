"""Benchmark C：专家结构相似性检索基准。

以人工标注的 (query_id, candidate_id, score 0-3) 为 Ground Truth，
对比 Statistics / GNN / Hybrid 三条类内检索路线：

- NDCG@5：分级相关性直接用专家分数（0-3）
- MRR：score >= mrr_threshold 视为相关
- Pairwise Accuracy：专家排序与模型排序一致的候选对比例
- Spearman：每查询模型相似度与专家分数的平均秩相关

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

DEFAULT_LABELS = os.path.join(ROOT_DIR, "data", "benchmark", "expert_similarity.csv")


def _detect_encoding(path):
    """自动探测标注 CSV 编码（Excel 保存中文常为 GBK）。"""
    with open(path, "rb") as f:
        raw = f.read(4096)
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


def run(save_dir=None, labels_csv=None, mrr_threshold=2.0, topk=5, cfg_path=None):
    """运行专家检索基准。

    Args:
        save_dir: 输出目录（默认 fig/benchmark/expert）
        labels_csv: 专家标注 CSV；默认 data/benchmark/expert_similarity.csv
        mrr_threshold: MRR 判定"相关"的分数阈值（0-3 制，默认 >=2 视为相关）
    """
    cfg = load_retrieval_config(cfg_path)
    save_dir = save_dir or os.path.join(ROOT_DIR, "fig", "benchmark", "expert")
    os.makedirs(save_dir, exist_ok=True)

    labels_csv = labels_csv or DEFAULT_LABELS
    labels = load_expert_labels(labels_csv)
    if not labels:
        print(f"[Expert Benchmark] 尚未有人工标注：{labels_csv}")
        print("请先由专家填写 data/benchmark/expert_similarity.csv 的 score 列。")
        return None

    embeddings, filenames, labels_vec = _load_index(cfg)
    dim = cfg.embedding.dim

    # 按 query 分组
    queries = {}
    for (q, c), score in labels.items():
        queries.setdefault(q, {})[c] = score

    methods = ["statistics", "gnn", "hybrid"]
    agg = {m: {"ndcg": [], "mrr": [], "pairwise": [], "spearman": []} for m in methods}

    for query, cand_scores in queries.items():
        if query not in filenames:
            print(f"[Expert Benchmark] 查询不在库中，跳过: {query}")
            continue
        q_idx = int(np.where(filenames == query)[0][0])
        cand_idx = []
        for cand in cand_scores:
            hit = np.where(filenames == cand)[0]
            if len(hit) == 0:
                print(f"[Expert Benchmark] 候选不在库中，跳过: {cand}")
                continue
            cand_idx.append(int(hit[0]))
        if not cand_idx:
            continue

        for method in methods:
            db_vecs = _method_vectors(embeddings, method, dim)
            q_vec = db_vecs[q_idx]
            sims = _cosine(q_vec, db_vecs[cand_idx])

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

            agg[method]["ndcg"].append(ndcg)
            agg[method]["mrr"].append(mrr)
            agg[method]["pairwise"].append(pair)
            agg[method]["spearman"].append(sp)

    print("\n=== Benchmark C：Expert Retrieval ===")
    print(
        f"（标注 {len(labels)} 对，{len(queries)} 个 query；"
        f"MRR 相关阈值 score>={mrr_threshold:.0f}）"
    )
    print(f'{"method":12s} {"NDCG@5":>8s} {"MRR":>8s} {"Pairwise":>9s} {"Spearman":>9s}')

    summary = {}
    for method in methods:
        row = {
            "ndcg5": float(np.nanmean(agg[method]["ndcg"])),
            "mrr": float(np.nanmean(agg[method]["mrr"])),
            "pairwise_accuracy": float(np.nanmean(agg[method]["pairwise"])),
            "spearman": float(np.nanmean(agg[method]["spearman"])),
            "n_queries": len(agg[method]["ndcg"]),
        }
        summary[method] = row
        print(
            f'{method:12s} {row["ndcg5"]:8.4f} {row["mrr"]:8.4f} '
            f'{row["pairwise_accuracy"]:9.2%} {row["spearman"]:9.4f}'
        )

    metrics_path = os.path.join(save_dir, "metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"\n[Expert Benchmark] 结果已保存: {metrics_path}")
    return summary


if __name__ == "__main__":
    run()
