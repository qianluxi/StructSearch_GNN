"""检索质量基准评估：一键输出检索准确率、聚类指标、混淆矩阵与失败案例。

直接运行：:

    python -m src.analysis.retrieval_benchmark.run_benchmark

推荐通过 CLI 运行（产物归档到 experiments/runs/<时间戳>_benchmark/）：:

    python -m src.retrieval.cli --benchmark
"""

import datetime
import json
import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np

from analysis.embedding_metrics import evaluate_embedding, print_metrics
from analysis.retrieval_benchmark.centroid_distance import plot_centroid_distance
from analysis.retrieval_benchmark.confusion_matrix import (
    CLASS_NAMES,
    build_confusion_matrix,
    collect_failure_cases,
    plot_confusion_matrix,
    save_confusion_matrix_csv,
    write_failure_cases,
)
from analysis.retrieval_benchmark.embedding_tsne import plot_tsne
from analysis.retrieval_benchmark.embedding_variance import plot_variance
from analysis.retrieval_benchmark.retrieval_cases import visualize_cases
from analysis.retrieval_benchmark.retrieval_metrics import (
    compute_topk_accuracy,
    plot_accuracy_curve,
)
from analysis.retrieval_benchmark.similarity_heatmap import plot_similarity_heatmap
from analysis.retrieval_benchmark.utils import ensure_dir


DEFAULT_SAVE_DIR = os.path.join(ROOT_DIR, "fig", "benchmark")


def load_index_embeddings(cfg_path=None):
    """从检索配置的 embedding_dir 加载索引向量（与线上检索同一份数据）。

    Args:
        cfg_path: 检索配置文件路径，默认 configs/retrieval.yaml

    Returns:
        (embeddings, filenames, labels) 三元组。
    """
    from retrieval.config import load_retrieval_config

    cfg = load_retrieval_config(cfg_path)

    def _load(name):
        path = os.path.join(cfg.embedding_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"索引文件缺失: {path}，请先运行 python -m src.retrieval.cli --build"
            )
        return np.load(path, allow_pickle=True)

    embeddings = _load("embeddings.npy")
    filenames = _load("filenames.npy")
    labels = _load("labels.npy")
    return embeddings, filenames, labels


def _write_metrics(save_dir, metrics, cm):
    """把指标写为 metrics.txt（人类可读）与 metrics.json（机器可读）。"""
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    text_lines = [
        "Benchmark Report",
        "================",
        f"时间: {now}",
        f"样本数: {metrics['n_samples']}",
        "",
        "【检索准确率】",
        f"Top-1: {metrics['top1_accuracy'] * 100:.2f}%",
        f"Top-3: {metrics['top3_accuracy'] * 100:.2f}%",
        f"Top-5: {metrics['top5_accuracy'] * 100:.2f}%",
        "",
        "【嵌入聚类指标】",
        f"Silhouette: {metrics['silhouette_score']:.4f}",
        f"Davies-Bouldin: {metrics['davies_bouldin_index']:.4f}",
        f"Calinski-Harabasz: {metrics['calinski_harabasz_score']:.4f}",
        "",
        "【混淆矩阵（行=真实，列=Top-1 预测）】",
        "".join(f"{name:>12}" for name in ["类别"] + CLASS_NAMES),
    ]
    for name, row in zip(CLASS_NAMES, cm):
        text_lines.append(
            f"{name:>8}" + "".join(f"{int(v):>12}" for v in row)
        )

    metrics_path = os.path.join(save_dir, "metrics.txt")
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_lines) + "\n")

    json_path = os.path.join(save_dir, "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(f"[Benchmark] 指标已保存: {metrics_path}")
    print(f"[Benchmark] 指标已保存: {json_path}")


def run(save_dir=None, cfg_path=None, embeddings=None, labels=None, filenames=None):
    """运行检索质量基准评估。

    Args:
        save_dir: 产物输出目录（默认 fig/benchmark）
        cfg_path: 检索配置文件路径
        embeddings / labels / filenames: 直接传入时跳过索引加载

    Returns:
        metrics: dict，含 top1/3/5_accuracy 与 silhouette / DB / CH。
    """
    save_dir = save_dir or DEFAULT_SAVE_DIR
    ensure_dir(save_dir)

    # -----------------------------
    # 1. 数据：优先使用线上索引（GNN 嵌入 + 图级统计特征）
    # -----------------------------
    if embeddings is None:
        embeddings, filenames, labels = load_index_embeddings(cfg_path)

    embeddings = np.asarray(embeddings, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    filenames = list(filenames) if filenames is not None else [
        f"sample_{i}" for i in range(len(labels))
    ]

    n = len(labels)
    if n == 0:
        raise ValueError("索引为空，无法运行基准评估")

    print(f"\n[Benchmark] 样本数: {n}，索引向量维度: {embeddings.shape[1]}")

    # -----------------------------
    # 2. 检索准确率 + 聚类指标
    # -----------------------------
    metrics = evaluate_embedding(embeddings, labels)
    metrics["n_samples"] = int(n)
    print_metrics(metrics)

    # -----------------------------
    # 3. 混淆矩阵 + 失败案例归档
    # -----------------------------
    cm, _pred = build_confusion_matrix(
        embeddings,
        labels,
        class_names=CLASS_NAMES
    )
    plot_confusion_matrix(
        cm,
        os.path.join(save_dir, "confusion_matrix.png"),
        class_names=CLASS_NAMES
    )
    save_confusion_matrix_csv(
        cm,
        os.path.join(save_dir, "confusion_matrix.csv"),
        class_names=CLASS_NAMES
    )

    failures = collect_failure_cases(
        embeddings,
        labels,
        filenames,
        topk=5,
        class_names=CLASS_NAMES
    )
    write_failure_cases(
        failures,
        os.path.join(save_dir, "failures.txt")
    )
    print(f"[Benchmark] Top-1 失败案例数: {len(failures)}")

    # -----------------------------
    # 4. 可视化
    # -----------------------------
    np.random.seed(42)  # 检索案例抽样可复现
    case_dir = os.path.join(save_dir, "retrieval_cases")
    ensure_dir(case_dir)

    plot_tsne(
        embeddings,
        labels,
        os.path.join(save_dir, "embedding_tsne.png")
    )
    plot_similarity_heatmap(
        embeddings,
        os.path.join(save_dir, "similarity_heatmap.png")
    )
    plot_centroid_distance(
        embeddings,
        labels,
        os.path.join(save_dir, "centroid_distance.png")
    )
    plot_variance(
        embeddings,
        os.path.join(save_dir, "embedding_variance.png")
    )
    acc = compute_topk_accuracy(embeddings, labels)
    plot_accuracy_curve(
        acc,
        os.path.join(save_dir, "retrieval_accuracy_curve.png")
    )
    visualize_cases(embeddings, labels, case_dir)

    # -----------------------------
    # 5. 汇总文件
    # -----------------------------
    _write_metrics(save_dir, metrics, cm)

    print(f"\n[Benchmark] 评估完成，全部产物已保存到: {save_dir}")
    return metrics


if __name__ == "__main__":
    run()
