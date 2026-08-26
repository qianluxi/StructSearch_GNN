"""检索命令行入口：建库 / 检索 / 基准评估。"""

import argparse
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from retrieval.index_builder import build_index
from retrieval.search_engine import SearchEngine
from utils.logging_setup import setup_logging
from utils.run_output import fig_dir, run_output


def _save_build_figures(run_dir):
    """建库完成后，把嵌入分布与分类结果图保存到本次运行目录。"""
    from tools.visualize.run_all_visualization import main as save_all

    print("\n[Run Output] 生成可视化图（t-SNE / PCA / 距离矩阵 / 混淆矩阵）...")
    try:
        save_all(save_dir=os.path.join(run_dir, "figs"))
    except Exception as exc:  # noqa: BLE001 - 可视化失败不应阻断建库
        print(f"[Run Output] 可视化生成失败（可忽略）: {exc}")


def _run_search(engine, dxf_path, topk, run_dir):
    """执行单图检索，并把检索结果图保存到本次运行目录。"""
    from analysis.retrieval_visualization import save_single_query_figure

    result = engine.search(dxf_path, topk=topk)
    if result is None:
        return

    _write_result_summary(run_dir, dxf_path, result)

    try:
        save_single_query_figure(
            engine.dataset,
            dxf_path,
            result,
            fig_dir(run_dir, "retrieval"),
            k=engine.cfg.graph.k,
            connect_mode=engine.cfg.graph.connect_mode,
            query_data=result.get("data"),
        )
    except Exception as exc:  # noqa: BLE001 - 绘图失败不应阻断检索
        print(f"[Run Output] 检索结果图生成失败（可忽略）: {exc}")


def _run_benchmark(run_dir):
    """执行检索质量基准评估，并把指标/图/失败案例归档到本次运行目录。"""
    from analysis.retrieval_benchmark.run_benchmark import run as run_benchmark

    print("\n[Benchmark] 开始检索质量基准评估...")
    run_benchmark(save_dir=run_dir)


def _run_intra_benchmark(run_dir):
    """执行 Stage 2 类内相似检索基准（三路线对比）。"""
    from analysis.intra_type_benchmark import run as run_intra

    print("\n[Intra-Type Benchmark] 开始类内相似检索基准评估...")
    run_intra(save_dir=run_dir)


def _run_expert_benchmark(run_dir, exclude_queries=None):
    """执行 Benchmark C 专家检索基准（需先填写 expert_similarity.csv）。"""
    from benchmark.expert_retrieval import run as run_expert

    print("\n[Expert Benchmark] 开始专家结构相似性检索评估...")
    run_expert(save_dir=run_dir, exclude_queries=exclude_queries)


def _run_topk_benchmark(run_dir):
    """执行 Benchmark C v2：Top-K 候选集评估（需先填写 expert_similarity_top3.csv）。"""
    from benchmark.topk_retrieval import run as run_topk

    print("\n[TopK Benchmark] 开始 Top-K 候选集评估...")
    run_topk(save_dir=run_dir)


def _run_feature_ablation(run_dir):
    """执行论文实验一：Statistics 特征分组消融。"""
    from benchmark.feature_ablation import run as run_ablation

    print("\n[Feature Ablation] 开始特征分组消融...")
    run_ablation(save_dir=run_dir)


def _run_baselines(run_dir):
    """执行论文实验三：传统 Baseline 对比。"""
    from benchmark.baselines import run as run_baselines

    print("\n[Baselines] 开始传统 Baseline 对比...")
    run_baselines(save_dir=run_dir)


def _write_intra_result_summary(run_dir, dxf_path, result):
    """把两阶段检索结果写入 result.txt。"""
    lines = [
        f"查询文件: {dxf_path}",
        "",
        f"【结构类型】: {result['type']} "
        f"(type_id={result['type_id']}, 置信度 {result['confidence']:.2%})",
        "",
        f"【检索方法】: {result['method']}",
        "",
        f"【类内最相似案例 Top {len(result['results'])}】",
    ]
    for item in result["results"]:
        lines.append(f"{item['rank']}. {item['filename']}")
        lines.append(f"   - 相似度: {item['score']:.4f}")
        lines.append("-" * 30)

    summary_path = os.path.join(run_dir, "result.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[Run Output] 两阶段检索结果摘要已保存: {summary_path}")


def _write_result_summary(run_dir, dxf_path, result):
    """把分类与 Top-K 检索结果单独写入 result.txt，便于直接查看。"""
    lines = [
        f"查询文件: {dxf_path}",
        "",
        f"【分类结果】: {result['pred_class']} "
        f"(置信度: {result['probability']:.2%})",
        "",
        f"【最相似的历史案例 Top {len(result['results'])}】",
    ]
    for item in result["results"]:
        lines.append(f"{item['rank']}. {item['filename']}")
        lines.append(f"   - 相似度: {item['score']:.4f}")
        lines.append(f"   - 类型: {item['label']}")
        lines.append("-" * 30)

    summary_path = os.path.join(run_dir, "result.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"[Run Output] 检索结果摘要已保存: {summary_path}")


def main(argv=None):
    setup_logging()

    parser = argparse.ArgumentParser(
        description="StructSearch_v2 检索工具：建库 / 类型识别 / 检索 / 基准评估"
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="构建特征索引（embeddings.npy 等）"
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="运行检索质量基准评估（Top-1/3/5、聚类指标、混淆矩阵、失败案例）"
    )
    parser.add_argument(
        "--intra-benchmark",
        action="store_true",
        help="运行 Stage 2 类内相似检索基准（Statistics/GNN/Hybrid 三路线对比）"
    )
    parser.add_argument(
        "--expert-benchmark",
        action="store_true",
        help="运行 Benchmark C 专家检索基准（需先填写 data/benchmark/expert_similarity.csv）"
    )
    parser.add_argument(
        "--topk-benchmark",
        action="store_true",
        help="运行 Benchmark C v2：Top-K 候选集评估（需先填写 expert_similarity_top3.csv）"
    )
    parser.add_argument(
        "--ablation",
        action="store_true",
        help="运行论文实验一：Statistics 特征分组消融"
    )
    parser.add_argument(
        "--baselines",
        action="store_true",
        help="运行论文实验三：传统 Baseline 对比（Random/原始统计/距离变体/GNN/Hybrid）"
    )
    parser.add_argument(
        "--exclude-query",
        action="append",
        default=None,
        help="从专家基准评估中排除的查询文件名（可多次指定）"
    )
    parser.add_argument(
        "--predict",
        type=str,
        help="对指定 DXF 做结构类型识别（仅 11D 统计特征，无 GNN）"
    )
    parser.add_argument(
        "--search",
        type=str,
        help="对指定 DXF 文件检索相似案例"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--topk", type=int, default=None)
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["statistics", "gnn", "hybrid"],
        help="类内检索方法（配合 --intra-type 使用）"
    )
    parser.add_argument(
        "--intra-type",
        action="store_true",
        help="使用两阶段流水线：先判结构类型，再在类型内检索"
    )

    args = parser.parse_args(argv)

    if args.build:
        with run_output("build") as run_dir:
            build_index(args.config)
            _save_build_figures(run_dir)
    elif args.benchmark:
        with run_output("benchmark") as run_dir:
            _run_benchmark(run_dir)
    elif args.intra_benchmark:
        with run_output("intra_benchmark") as run_dir:
            _run_intra_benchmark(run_dir)
    elif args.expert_benchmark:
        with run_output("expert_benchmark") as run_dir:
            _run_expert_benchmark(run_dir, exclude_queries=args.exclude_query)
    elif args.topk_benchmark:
        with run_output("topk_benchmark") as run_dir:
            _run_topk_benchmark(run_dir)
    elif args.ablation:
        with run_output("ablation") as run_dir:
            _run_feature_ablation(run_dir)
    elif args.baselines:
        with run_output("baselines") as run_dir:
            _run_baselines(run_dir)
    elif args.predict:
        with run_output("predict") as run_dir:
            from classification.classifier import StructureTypeClassifier

            result = StructureTypeClassifier(args.config).predict_dxf(args.predict)
            if result is None:
                print("解析失败或未提取到构件节点")
            else:
                print(f"结构类型: {result['type']} (type_id={result['type_id']})")
                print(f"置信度  : {result['confidence']:.2%}")
    elif args.search:
        with run_output("search") as run_dir:
            if args.intra_type:
                from pipeline.search_pipeline import SearchPipeline

                pipe = SearchPipeline(args.config)
                result = pipe.search(
                    args.search,
                    topk=args.topk,
                    method=args.method or "hybrid",
                )
                if result is not None:
                    _write_intra_result_summary(run_dir, args.search, result)
            else:
                engine = SearchEngine(args.config)
                _run_search(engine, args.search, args.topk, run_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
