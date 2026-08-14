"""检索 Top-1 混淆矩阵与失败案例：评估"检索即分类"的逐类混淆情况。

预测类别定义为：与查询样本余弦相似度最高的库内邻居的标签（排除自身）。
"""

import csv
import os

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

from .utils import cosine_similarity_matrix

# 与 src/retrieval/config.py CLASS_NAMES 保持一致
CLASS_NAMES = ["剪力墙", "框剪", "框架"]


def top1_predictions(embeddings, labels):
    """返回每个样本的 Top-1 检索邻居标签（排除自身）。

    Args:
        embeddings: (N, D) 索引向量
        labels: (N,) 真实标签

    Returns:
        pred: (N,) int64 数组
    """
    sim = cosine_similarity_matrix(embeddings)
    n = len(labels)
    pred = np.empty(n, dtype=np.int64)

    for i in range(n):
        sim_i = sim[i].copy()
        sim_i[i] = -np.inf  # 排除查询自身
        pred[i] = labels[np.argmax(sim_i)]

    return pred


def build_confusion_matrix(embeddings, labels, class_names=None):
    """构建检索 Top-1 混淆矩阵。

    Args:
        embeddings: (N, D) 索引向量
        labels: (N,) 真实标签
        class_names: 类别名列表，默认使用三结构体系中文名

    Returns:
        cm: (K, K) int64 数组，行=真实类别，列=Top-1 预测类别
        pred: (N,) Top-1 预测标签
    """
    class_names = class_names or CLASS_NAMES
    pred = top1_predictions(embeddings, labels)
    cm = confusion_matrix(
        labels,
        pred,
        labels=list(range(len(class_names)))
    )
    return cm, pred


def plot_confusion_matrix(cm, save_path, class_names=None):
    """保存混淆矩阵图（含数值标注与颜色条）。"""
    class_names = class_names or CLASS_NAMES

    # 中文字体：Windows 优先使用微软雅黑/黑体，避免标签显示为方框
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Top-1 检索预测类别")
    ax.set_ylabel("真实类别")
    ax.set_title("Retrieval Top-1 Confusion Matrix")

    thresh = cm.max() / 2 if cm.max() > 0 else 0.5
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_confusion_matrix_csv(cm, save_path, class_names=None):
    """把混淆矩阵保存为 CSV（行=真实，列=Top-1 预测）。"""
    class_names = class_names or CLASS_NAMES
    with open(save_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["真实类别"] + [f"预测_{name}" for name in class_names])
        for name, row in zip(class_names, cm):
            writer.writerow([name] + [int(v) for v in row])


def collect_failure_cases(embeddings, labels, filenames, topk=5, class_names=None):
    """收集 Top-1 检索错误的样本明细（含 Top-K 邻居）。

    Args:
        embeddings: (N, D) 索引向量
        labels: (N,) 真实标签
        filenames: (N,) 文件名
        topk: 每个失败案例记录的邻居条数
        class_names: 类别名列表

    Returns:
        list[dict]，每个元素含 query / true_label / predicted_label / neighbors。
    """
    class_names = class_names or CLASS_NAMES
    sim = cosine_similarity_matrix(embeddings)
    pred = top1_predictions(embeddings, labels)
    n = len(labels)

    failures = []
    for i in range(n):
        if pred[i] == labels[i]:
            continue

        sim_i = sim[i].copy()
        sim_i[i] = -np.inf
        neighbor_idx = np.argsort(-sim_i)[:topk]

        neighbors = []
        for rank, idx in enumerate(neighbor_idx, start=1):
            neighbors.append({
                "rank": rank,
                "filename": str(filenames[idx]),
                "score": float(sim_i[idx]),
                "label": class_names[labels[idx]],
            })

        failures.append({
            "query": str(filenames[i]),
            "true_label": class_names[labels[i]],
            "predicted_label": class_names[pred[i]],
            "neighbors": neighbors,
        })

    return failures


def write_failure_cases(failures, save_path):
    """把失败案例写入文本文件，便于人工排查。"""
    if not failures:
        with open(save_path, "w", encoding="utf-8") as f:
            f.write("无失败案例：所有查询的 Top-1 检索邻居类别均正确。\n")
        return

    lines = [
        f"Top-1 检索失败案例（共 {len(failures)} 条）",
        "=" * 60,
        "",
    ]
    for i, case in enumerate(failures, start=1):
        lines.append(f"[{i}] 查询: {case['query']}")
        lines.append(
            f"    真实类别: {case['true_label']}  "
            f"Top-1 预测类别: {case['predicted_label']}"
        )
        lines.append("    相似邻居:")
        for nb in case["neighbors"]:
            lines.append(
                f"      {nb['rank']}. {nb['filename']}  "
                f"({nb['label']}, 相似度 {nb['score']:.4f})"
            )
        lines.append("-" * 60)

    with open(save_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def ensure_dir(path):
    """确保目录存在。"""
    os.makedirs(path, exist_ok=True)
