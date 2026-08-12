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
    os.makedirs(embeddings_dir, exist_ok=True)
    os.makedirs(classification_dir, exist_ok=True)

    plot_tsne(embeddings, labels, save_dir=embeddings_dir)
    plot_pca(embeddings, labels, save_dir=embeddings_dir)
    plot_distance_matrix(embeddings, save_dir=embeddings_dir)
    plot_confusion_matrix(labels, preds, save_dir=classification_dir)


if __name__ == "__main__":
    main()
