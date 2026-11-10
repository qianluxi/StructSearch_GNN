# -*- coding: utf-8 -*-
"""Regenerate 图2 / 图3 as vertical stacks for A4 portrait layout."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

HERE = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(HERE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

BLUE = "#3B82F6"
GREEN = "#10B981"
ORANGE = "#F59E0B"
PURPLE = "#8B5CF6"
GRAY = "#6B7280"


def fig2_methods_portrait():
    methods = ["Random", "未标准化统计", "GNN", "Hybrid",
               "统计+欧氏", "统计+余弦", "统计+曼哈顿"]
    ndcg3 = [0.6214, 0.5394, 0.5947, 0.7135, 0.7182, 0.7214, 0.7458]
    recall3 = [15.56, 17.78, 11.11, 15.56, 28.89, 26.67, 35.56]
    mean_score3 = [0.667, 0.578, 0.778, 0.800, 1.000, 0.889, 1.111]

    x = np.arange(len(methods))
    fig, axes = plt.subplots(3, 1, figsize=(6.5, 8.2))
    panels = [
        (ndcg3, "NDCG@3", BLUE, "{:.3f}", 0.85),
        (recall3, "Recall@3 / %", GREEN, "{:.1f}%", 42),
        (mean_score3, "MeanScore@3", ORANGE, "{:.3f}", 1.3),
    ]
    for ax, (vals, title, color, fmt, ylim) in zip(axes, panels):
        bars = ax.bar(x, vals, 0.62, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=8, rotation=25, ha="right")
        ax.set_ylim(0, ylim)
        ax.set_title(title, fontsize=10)
        ax.grid(axis="y", ls="--", alpha=0.4)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + ylim * 0.015,
                    fmt.format(v), ha="center", fontsize=7.2)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(os.path.join(FIG_DIR, "fig2_methods_portrait.png"), dpi=300)
    plt.close(fig)


def fig3_ablation_portrait():
    fig, axes = plt.subplots(2, 1, figsize=(6.5, 7.0))

    groups = ["类型占比", "面积占比", "面积分布", "空间统计"]
    top1 = [86.79, 80.19, 58.49, 37.74]
    ax = axes[0]
    bars = ax.bar(groups, top1, color=[BLUE, GREEN, PURPLE, GRAY])
    for rect, v in zip(bars, top1):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 1.2,
                f"{v:.1f}", ha="center", fontsize=9)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Top-1 类型检索准确率 / %", fontsize=9.5)
    ax.set_title("(a) 单特征组", fontsize=10)
    ax.tick_params(axis="x", labelsize=9)
    ax.grid(axis="y", ls="--", alpha=0.4)

    combos = ["类型", "+面积占比", "+面积分布", "+空间统计"]
    top1p = [86.79, 88.68, 87.74, 87.74]
    ax = axes[1]
    ax.plot(range(len(combos)), top1p, "o-", color=BLUE, lw=1.5)
    for i, v in enumerate(top1p):
        ax.text(i, v + 1.2, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(combos, fontsize=9)
    ax.set_ylim(80, 100)
    ax.set_title("(b) 逐组累加", fontsize=10)
    ax.grid(axis="y", ls="--", alpha=0.4)

    fig.tight_layout(h_pad=1.5)
    fig.savefig(os.path.join(FIG_DIR, "fig3_ablation_portrait.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig2_methods_portrait()
    fig3_ablation_portrait()
    print("saved portrait figures to", FIG_DIR)
