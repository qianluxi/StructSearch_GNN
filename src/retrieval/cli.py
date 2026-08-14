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
        description="StructSearch_v2 检索工具：构建索引、检索单张 DXF 或运行基准评估"
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
        "--search",
        type=str,
        help="对指定 DXF 文件进行分类 + Top-K 检索"
    )
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--topk", type=int, default=None)

    args = parser.parse_args(argv)

    if args.build:
        with run_output("build") as run_dir:
            build_index(args.config)
            _save_build_figures(run_dir)
    elif args.benchmark:
        with run_output("benchmark") as run_dir:
            _run_benchmark(run_dir)
    elif args.search:
        with run_output("search") as run_dir:
            engine = SearchEngine(args.config)
            _run_search(engine, args.search, args.topk, run_dir)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
