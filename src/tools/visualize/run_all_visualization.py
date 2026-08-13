"""一键生成全部可视化：嵌入分布 + 分类混淆矩阵。"""

import os
import sys

# 兼容两种运行方式：直接执行本脚本，或作为模块被其他代码导入。
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPT_DIR not in sys.path:
    sys.path.insert(0, _SCRIPT_DIR)

from embedding_extractor import extract_embeddings  # noqa: E402
from tsne_plot import plot_tsne  # noqa: E402
from pca_plot import plot_pca  # noqa: E402
from distance_matrix import plot_distance_matrix  # noqa: E402
from confusion_matrix_plot import plot_confusion_matrix  # noqa: E402
from analysis.retrieval_benchmark.similarity_heatmap import plot_similarity_heatmap  # noqa: E402
from analysis.retrieval_benchmark.embedding_variance import plot_variance  # noqa: E402
from analysis.retrieval_benchmark.centroid_distance import plot_centroid_distance  # noqa: E402
from analysis.retrieval_benchmark.retrieval_metrics import (  # noqa: E402
    compute_topk_accuracy,
    plot_accuracy_curve,
)


def main(save_dir=None):
    """生成全部可视化结果。

    Args:
        save_dir: 输出根目录。为 None 时按原行为保存到 fig/<子类>；
            否则保存到 save_dir/embeddings 与 save_dir/classification。
    """
    embeddings, labels, preds = extract_embeddings()

    if save_dir is None:
        plot_tsne(embeddings, labels)
        plot_pca(embeddings, labels)
        plot_distance_matrix(embeddings)
        plot_confusion_matrix(labels, preds)
        return

    embeddings_dir = os.path.join(save_dir, "embeddings")
    classification_dir = os.path.join(save_dir, "classification")
    retrieval_dir = os.path.join(save_dir, "retrieval")
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(classification_dir, exist_ok=True)
    os.makedirs(retrieval_dir, exist_ok=True)

    plot_tsne(embeddings, labels, save_dir=embeddings_dir)
    plot_pca(embeddings, labels, save_dir=embeddings_dir)
    plot_distance_matrix(embeddings, save_dir=embeddings_dir)
    plot_confusion_matrix(labels, preds, save_dir=classification_dir)

    plot_similarity_heatmap(
        embeddings,
        os.path.join(embeddings_dir, "similarity_heatmap.png"),
    )
    plot_variance(
        embeddings,
        os.path.join(embeddings_dir, "embedding_variance.png"),
    )
    plot_centroid_distance(
        embeddings,
        labels,
        os.path.join(classification_dir, "centroid_distance.png"),
    )
    acc = compute_topk_accuracy(embeddings, labels, max_k=10)
    plot_accuracy_curve(
        acc,
        os.path.join(retrieval_dir, "retrieval_accuracy_curve.png"),
    )

    print(f"[Visualize] 相似度热力图: {embeddings_dir}\\similarity_heatmap.png")
    print(f"[Visualize] 嵌入方差分布: {embeddings_dir}\\embedding_variance.png")
    print(f"[Visualize] 类质心距离: {classification_dir}\\centroid_distance.png")
    print(
        f"[Visualize] Top-K 检索准确率: {retrieval_dir}\\retrieval_accuracy_curve.png"
    )


if __name__ == "__main__":
    main()
