"""生成 data/benchmark 下的标注文件。

1) weak_project_labels.csv：Benchmark B 的弱相关标签
   - 同工程且同类 -> weak_relevance=1，其余同类候选 -> 0；
   - 仅包含存在 ≥1 个正样本的查询；无正样本的查询归入跨项目基准。

2) expert_similarity.csv：Benchmark C 的专家标注模板
   - 默认 50 个 query（框架 15 / 框剪 18 / 剪力墙 17），优先多项目覆盖；
   - 刻意包含"跨项目"query（该类型下所在项目仅 1 张图，如则安_框剪_1）；
   - 候选 = statistics/gnn/hybrid 三路线 Top-5 在真实类型内的并集；
   - 已有打分保留，新 query 的 score 列留空待专家填写。

运行：python -m src.benchmark.prepare_data [--queries 30|50] [--topk 5]
"""

import argparse
import csv
import os
import sys
from collections import defaultdict

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from benchmark.expert_retrieval import _detect_encoding
from classification.classifier import StructureTypeClassifier
from datasets.structure_dataset import StructureDataset
from datasets.schemas import parse_drawing_meta
from retrieval.config import load_retrieval_config
from retrieval.search_engine import SearchEngine

BENCH_DIR = os.path.join(ROOT_DIR, "data", "benchmark")

# 类型 id -> query 数量（默认 50：剪力墙 17 / 框剪 18 / 框架 15）
PLAN_50 = {0: 17, 1: 18, 2: 15}
PLAN_30 = {0: 10, 1: 10, 2: 10}

# Top-K 盲评第一轮：15 个 query（每类 5 个，各来自 5 个不同项目）
TOP3_QUERIES = [
    # 框架（5 个项目）
    "仁和上盖_框架_12.dxf",
    "湘湖里_框架_H3.dxf",
    "勾庄好运来_框架_16.dxf",
    "文成_框架_DT.dxf",      # 跨项目（该类型仅此一张）
    "苏州华贸_框架_6C.dxf",
    # 框剪（5 个项目）
    "景芳三堡_框剪_B.dxf",
    "望江府_框剪_1.dxf",
    "高福里_框剪_1.dxf",
    "茅安前_框剪_5.dxf",
    "西大门_框剪_2.dxf",      # 跨项目（该类型仅此一张）
    # 剪力墙（5 个项目）
    "茅安前_剪力墙_10.dxf",
    "高福里_剪力墙_11.dxf",
    "苏州华贸_剪力墙_10A.dxf",
    "日照钢铁_剪力墙_13.dxf",
    "勾庄好运来_剪力墙_1.dxf",
]


def _dataset(cfg):
    return StructureDataset(
        root=cfg.data_root,
        k=cfg.graph.k,
        connect_mode=cfg.graph.connect_mode,
    )


def build_query_plan(dataset, plan):
    """按计划挑选 query：跨项目优先，同项目按项目轮转保证多项目覆盖。

    Returns:
        queries: List[str]（文件名）
        meta: {type_id: {"target", "cross", "same", "n_projects"}}
    """
    proj_type = defaultdict(lambda: defaultdict(list))
    for i in range(len(dataset)):
        name = dataset[i].filename
        proj = parse_drawing_meta(name).project_id
        t = int(dataset[i].y.item())
        proj_type[proj][t].append(name)

    queries = []
    meta = {}
    for t in sorted(plan):
        n = plan[t]
        cross, same_groups = [], []
        for proj in sorted(proj_type):
            if t not in proj_type[proj]:
                continue
            files = sorted(proj_type[proj][t])
            if len(files) == 1:
                cross.extend(files)
            else:
                same_groups.append([proj, files])

        n_cross = min(len(cross), max(0, round(n * 0.3)))
        chosen = list(cross[:n_cross])
        remaining = n - len(chosen)

        groups = [g[1] for g in sorted(same_groups, key=lambda x: x[0])]
        while remaining > 0 and any(groups):
            for g in groups:
                if remaining == 0:
                    break
                if g:
                    chosen.append(g.pop(0))
                    remaining -= 1

        queries.extend(chosen)
        meta[t] = {
            "target": n,
            "cross": n_cross,
            "same": n - n_cross,
            "n_projects": len({parse_drawing_meta(q).project_id for q in chosen}),
        }
    return queries, meta


def write_weak_labels(cfg_path=None, out_path=None):
    """生成 Benchmark B 的弱相关标签 CSV。"""
    cfg = load_retrieval_config(cfg_path)
    dataset = _dataset(cfg)
    names = [dataset[i].filename for i in range(len(dataset))]
    labels = [int(dataset[i].y.item()) for i in range(len(dataset))]
    projects = [parse_drawing_meta(name).project_id for name in names]

    rows = []
    for i in range(len(names)):
        positives = [
            j
            for j in range(len(names))
            if j != i and labels[j] == labels[i] and projects[j] == projects[i]
        ]
        if not positives:
            continue
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


def _load_existing_rows(path):
    """读取已有标注 CSV（若有），返回 {(query_id, candidate_id): (score, comment)}。"""
    if not os.path.exists(path):
        return {}
    existing = {}
    with open(path, "r", encoding=_detect_encoding(path), newline="") as f:
        for row in csv.DictReader(f):
            q = (row.get("query_id") or "").strip()
            c = (row.get("candidate_id") or "").strip()
            if q and c:
                existing[(q, c)] = (row.get("score", "").strip(), row.get("comment", "").strip())
    return existing


def write_expert_template(cfg_path=None, out_path=None, plan=None, topk=5):
    """生成（或增量更新）Benchmark C 专家标注模板。

    已有打分保留；新 query 追加候选行，score 列留空。
    """
    plan = plan or PLAN_50
    cfg = load_retrieval_config(cfg_path)
    dataset = _dataset(cfg)
    true_label = {
        dataset[i].filename: int(dataset[i].y.item())
        for i in range(len(dataset))
    }
    queries, meta = build_query_plan(dataset, plan)

    out_path = out_path or os.path.join(BENCH_DIR, "expert_similarity.csv")
    existing = _load_existing_rows(out_path)

    clf = StructureTypeClassifier(cfg_path)
    engine = SearchEngine(cfg_path)

    rows = []
    seen = set(existing)
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
            key = (query, cand)
            if key in seen:
                continue
            rows.append([query, cand, "", ""])
            seen.add(key)

    # 保留已有行（含已填分数），再追加新行
    kept = sorted(
        [[q, c, score, comment] for (q, c), (score, comment) in existing.items()],
        key=lambda r: (r[0], r[1]),
    )
    rows = kept + sorted(rows, key=lambda r: (r[0], r[1]))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "candidate_id", "score", "comment"])
        writer.writerows(rows)

    filled = sum(1 for r in rows if r[2] != "")
    print(f"[prepare] expert_similarity.csv 更新完成 {len(rows)} 行 "
          f"（已填 {filled}，待填 {len(rows) - filled}）-> {out_path}")
    print("[prepare] query 计划：")
    for t in sorted(meta):
        m = meta[t]
        print(
            f"  type_id={t}: 目标 {m['target']}，同项目 {m['same']}，"
            f"跨项目 {m['cross']}，覆盖项目数 {m['n_projects']}"
        )


def write_expert_top3(
    cfg_path=None,
    registry_path=None,
    labels_path=None,
    queries=None,
    topk=5,
    n_hard=2,
):
    """生成 Top-K 盲评候选：三路线 Top5 并集 + Hard Negative。

    - registry_path: expert_candidates_top3.csv（含各路线排名，程序用）
    - labels_path: expert_similarity_top3.csv（盲评，只含 query/candidate/score，专家打分）
    """
    cfg = load_retrieval_config(cfg_path)
    dataset = _dataset(cfg)
    true_label = {
        dataset[i].filename: int(dataset[i].y.item())
        for i in range(len(dataset))
    }
    queries = queries or TOP3_QUERIES

    registry_path = registry_path or os.path.join(
        BENCH_DIR, "expert_candidates_top3.csv"
    )
    labels_path = labels_path or os.path.join(
        BENCH_DIR, "expert_similarity_top3.csv"
    )

    clf = StructureTypeClassifier(cfg_path)
    engine = SearchEngine(cfg_path)

    registry_rows = []
    label_rows = []
    for query in queries:
        pred = clf.predict_dxf(os.path.join(cfg.data_root, "raw", query))
        if pred is None:
            print(f"[prepare] 解析失败，跳过: {query}")
            continue
        type_id = true_label[query]

        # 每条路线在真实类型内的全量排序（topk 取大，得到完整池）
        route_results = {}
        for method in ("statistics", "gnn", "hybrid"):
            result = engine.search_by_type(
                pred["data"],
                type_id=type_id,
                topk=1000,
                method=method,
            )
            route_results[method] = result["results"]

        top5 = {
            method: [item["filename"] for item in results[:topk]]
            for method, results in route_results.items()
        }
        union_cands = set(top5["statistics"]) | set(top5["gnn"]) | set(top5["hybrid"])

        # 每个候选在三条路线下的相似度（用于 Hard Negative 选取）
        sim_by_route = {
            method: {item["filename"]: item["score"] for item in results}
            for method, results in route_results.items()
        }
        same_type_all = [item["filename"] for item in route_results["statistics"]]

        def max_sim(name):
            return max(
                sim_by_route["statistics"].get(name, -1.0),
                sim_by_route["gnn"].get(name, -1.0),
                sim_by_route["hybrid"].get(name, -1.0),
            )

        hard_negatives = sorted(
            (f for f in same_type_all if f not in union_cands),
            key=max_sim,
        )[:n_hard]

        pool = sorted(union_cands | set(hard_negatives))
        for cand in pool:
            ranks = []
            for method in ("statistics", "gnn", "hybrid"):
                top_list = top5[method]
                ranks.append(
                    str(top_list.index(cand) + 1) if cand in top_list else ""
                )
            registry_rows.append([
                query, cand, ranks[0], ranks[1], ranks[2],
                "hard_negative" if cand in hard_negatives else "route",
            ])
            label_rows.append([query, cand, "", ""])

    os.makedirs(os.path.dirname(registry_path), exist_ok=True)
    with open(registry_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "query_id", "candidate_id",
            "statistics_rank", "gnn_rank", "hybrid_rank", "source",
        ])
        writer.writerows(registry_rows)

    with open(labels_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "candidate_id", "expert_score", "comment"])
        writer.writerows(label_rows)

    print(
        f"[prepare] expert_candidates_top3.csv 已生成 {len(registry_rows)} 行 "
        f"（{len({r[0] for r in registry_rows})} query）-> {registry_path}"
    )
    print(
        f"[prepare] expert_similarity_top3.csv（盲评模板）已生成 "
        f"{len(label_rows)} 行 -> {labels_path}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description="生成 Benchmark 标注文件")
    parser.add_argument("--queries", type=int, choices=[30, 50], default=50,
                        help="query 目标规模（默认 50）")
    parser.add_argument("--top3", action="store_true",
                        help="额外生成 Top-K 盲评模板（15 query）")
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--config", default=None)
    args = parser.parse_args(argv)

    plan = PLAN_50 if args.queries == 50 else PLAN_30
    write_weak_labels(args.config)
    write_expert_template(args.config, plan=plan, topk=args.topk)
    if args.top3:
        write_expert_top3(args.config, topk=args.topk)


if __name__ == "__main__":
    main()
