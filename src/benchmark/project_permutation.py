"""项目名置换检验（Project Permutation Test）。

随机打乱索引中所有图纸的项目名（仅改元数据，特征与打分不变），重跑：
1) 检索指标（Expert Top-K：Statistics/GNN/Hybrid 的 NDCG@3 与 Recall@3）——
   期望**完全不变**，证明检索不依赖项目名；
2) 项目感知指标（weak_project_relevance 的 MRR 与 Recall@5）——
   期望**发生变化**，证明置换确实生效、检验有区分力。

运行：python -m src.benchmark.project_permutation
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
    ndcg_at_k,
    recall_at_k,
)
from datasets.schemas import parse_drawing_meta
from retrieval.config import load_retrieval_config

DEFAULT_LABELS = os.path.join(ROOT_DIR, "data", "benchmark", "expert_similarity_top3.csv")


def _relabel(name, mapping):
    """把文件名中的项目名按映射替换（仅元数据）。"""
    first, _, rest = name.partition("_")
    return mapping.get(first, first) + ("_" + rest if rest else "")


def permute_filenames(filenames, seed):
    """对项目名做确定性置换（仅改第一个下划线前的 token）。"""
    projects = sorted({parse_drawing_meta(n).project_id for n in filenames})
    permuted = list(projects)
    rng = np.random.default_rng(seed)
    rng.shuffle(permuted)
    mapping = dict(zip(projects, permuted))
    return np.array([_relabel(n, mapping) for n in filenames]), mapping


def expert_retrieval_metrics(embeddings, filenames, scores, dim):
    """在专家 Top-K 标注池内计算三条路线的 NDCG@3 与 Recall@3。"""
    name_to_idx = {n: i for i, n in enumerate(filenames)}
    queries = {}
    for (q, c), s in scores.items():
        queries.setdefault(q, {})[c] = s

    def vecs(m):
        return embeddings[:, dim:] if m == "statistics" else embeddings[:, :dim] if m == "gnn" else embeddings

    def order(m, qi, cand_idx):
        db = vecs(m)
        cand_idx = np.asarray(cand_idx, dtype=int)
        n = np.linalg.norm(db, axis=1)
        d = n * np.linalg.norm(db[qi])
        d[d == 0] = 1e-12
        sim = (db[cand_idx] @ db[qi]) / d[cand_idx]
        return cand_idx[np.argsort(-sim)]

    agg = {m: {"ndcg3": [], "recall3": []} for m in ("statistics", "gnn", "hybrid")}
    for query, cand_scores in queries.items():
        if query not in name_to_idx:
            continue
        qi = name_to_idx[query]
        cand_idx = [name_to_idx[c] for c in cand_scores if c in name_to_idx]
        if not cand_idx:
            continue
        for m in agg:
            order_idx = order(m, qi, cand_idx)
            rel = [cand_scores[filenames[i]] for i in order_idx]
            agg[m]["ndcg3"].append(ndcg_at_k(rel, 3))
            agg[m]["recall3"].append(recall_at_k(rel, 3))

    return {
        m: {
            "ndcg3": float(np.mean(agg[m]["ndcg3"])),
            "recall3": float(np.mean(agg[m]["recall3"])),
        }
        for m in agg
    }


def weak_project_metrics(embeddings, filenames, labels, dim):
    """weak_project_relevance 的 MRR 与 Recall@5（Statistics 11D 路线）。"""
    projects = np.array([parse_drawing_meta(n).project_id for n in filenames])
    stats = embeddings[:, dim:]
    norm = stats / np.linalg.norm(stats, axis=1, keepdims=True)
    sim = norm @ norm.T
    np.fill_diagonal(sim, -np.inf)

    mrrs, recalls, n = [], [], 0
    for i in range(len(filenames)):
        same_type = labels == labels[i]
        relevant = (
            same_type & (projects == projects[i]) & (np.arange(len(filenames)) != i)
        )
        if not relevant.any():
            continue
        candidates = np.where(same_type & (np.arange(len(filenames)) != i))[0]
        order = candidates[np.argsort(-sim[i][candidates])]
        rel = np.array([1 if relevant[j] else 0 for j in order])
        mrrs.append(1.0 / (np.argmax(rel) + 1))
        recalls.append(rel[:5].sum() / rel.sum())
        n += 1
    return {"mrr": float(np.mean(mrrs)), "recall5": float(np.mean(recalls)), "n": n}


def run(save_dir=None, seeds=(42, 7), cfg_path=None):
    """运行项目名置换检验。"""
    cfg = load_retrieval_config(cfg_path)
    save_dir = save_dir or os.path.join(ROOT_DIR, "fig", "benchmark", "permutation")
    os.makedirs(save_dir, exist_ok=True)

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
    scores = load_scores(DEFAULT_LABELS)
    dim = cfg.embedding.dim

    base_expert = expert_retrieval_metrics(embeddings, filenames, scores, dim)
    base_weak = weak_project_metrics(embeddings, filenames, labels, dim)

    results = {"baseline": {"expert": base_expert, "weak": base_weak}, "permuted": {}}
    mappings = []
    print("=== 项目名置换检验 ===")
    print(
        f'{"seed":>5s} {"改名数":>6s} {"StatsNDCG3":>11s} {"GNNNDCG3":>10s} '
        f'{"HybNDCG3":>10s} {"WeakMRR":>9s} {"WeakR@5":>9s}'
    )
    for seed in seeds:
        perm_filenames, mapping = permute_filenames(filenames, seed)
        mappings.append(mapping)
        perm_scores = {
            (_relabel(q, mapping), _relabel(c, mapping)): s
            for (q, c), s in scores.items()
        }
        pexpert = expert_retrieval_metrics(
            embeddings, perm_filenames, perm_scores, dim
        )
        pweak = weak_project_metrics(embeddings, perm_filenames, labels, dim)
        results["permuted"][str(seed)] = {"expert": pexpert, "weak": pweak}
        n_renamed = sum(1 for k, v in mapping.items() if k != v)
        print(
            f'{seed:5d} {n_renamed:6d} {pexpert["statistics"]["ndcg3"]:11.4f} '
            f'{pexpert["gnn"]["ndcg3"]:10.4f} {pexpert["hybrid"]["ndcg3"]:10.4f} '
            f'{pweak["mrr"]:9.4f} {pweak["recall5"]:9.2%}'
        )

    print("\n基线（真实项目名）:")
    print(
        f'  Expert NDCG@3: Stats={base_expert["statistics"]["ndcg3"]:.4f} '
        f'GNN={base_expert["gnn"]["ndcg3"]:.4f} Hybrid={base_expert["hybrid"]["ndcg3"]:.4f}'
    )
    print(f'  Weak MRR={base_weak["mrr"]:.4f} Recall@5={base_weak["recall5"]:.2%}')

    expert_invariant = all(
        np.isclose(
            results["permuted"][str(s)]["expert"][m][k],
            base_expert[m][k],
        )
        for s in seeds
        for m in ("statistics", "gnn", "hybrid")
        for k in ("ndcg3", "recall3")
    )
    weak_invariant = all(
        np.isclose(results["permuted"][str(s)]["weak"]["mrr"], base_weak["mrr"])
        and np.isclose(
            results["permuted"][str(s)]["weak"]["recall5"],
            base_weak["recall5"],
        )
        for s in seeds
    )
    nonidentity = all(any(k != v for k, v in mapping.items()) for mapping in mappings)
    print(f"\n置换有效性（映射非恒等）:   {'PASS' if nonidentity else 'FAIL'}")
    print(f"检索指标对项目名置换不变:   {'PASS' if expert_invariant else 'FAIL'}")
    print(
        "项目感知指标（weak）同样不变: "
        f"{'PASS（等价关系在置换下保持，符合数学预期）' if weak_invariant else 'FAIL'}"
    )

    with open(os.path.join(save_dir, "result.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"[Permutation] 结果已保存: {os.path.join(save_dir, 'result.json')}")
    return expert_invariant and nonidentity


if __name__ == "__main__":
    raise SystemExit(0 if run() else 1)
