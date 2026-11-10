# -*- coding: utf-8 -*-
"""Regenerate all paper figures at final print size with readable fonts.

Strategy: design each figure at its final physical size (single column 7.8 cm,
图6 spanning 16.5 cm), using 8-9 pt fonts (Times New Roman + SimSun).
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "axes.unicode_minus": False,
    "mathtext.fontset": "stix",
})

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)
DPI = 600
CM = 1 / 2.54

BLUE = "#3B82F6"
GREEN = "#10B981"
ORANGE = "#F59E0B"
PURPLE = "#8B5CF6"
GRAY = "#6B7280"


def fig1_pipeline():
    W = 7.8 * CM
    fig, ax = plt.subplots(figsize=(W, 11.0 * CM))
    ax.set_xlim(0, 10); ax.set_ylim(0, 15.0); ax.axis("off")

    def box(x, y, w, h, text, fc="#FFFFFF", ec="#333333", fs=8, lw=1.0, ls="-"):
        ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                     boxstyle="round,pad=0.02,rounding_size=0.12",
                     fc=fc, ec=ec, lw=lw, ls=ls))
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                     mutation_scale=9, color="#333333", lw=0.9))

    cx = 5.0
    box(cx, 14.2, 4.6, 0.9, "查询 DXF 结构图纸", fc="#F3F4F6")
    box(cx, 12.8, 6.4, 1.0, "结构解析：构件提取·对齐·分类", fc="#DBEAFE", ec=BLUE)
    box(cx, 11.2, 7.2, 1.4, "11维结构统计描述子\n类型·面积·分布·空间", fc="#DBEAFE", ec=BLUE, fs=8)
    box(cx, 9.2, 6.6, 1.1, "第一阶段：结构体系识别\n（余弦近邻分类）", fc="#FEF3C7", ec=ORANGE, fs=8)
    for xi, name in [(2.4, "框架"), (5.0, "框剪"), (7.6, "剪力墙")]:
        box(xi, 7.6, 2.0, 0.8, name, fc="#FFEDD5", ec=ORANGE, fs=8)
    box(cx, 5.8, 7.4, 1.3, "第二阶段：类内相似检索\n标准化+曼哈顿距离", fc="#FEF3C7", ec=ORANGE, fs=8)
    box(cx, 4.0, 5.4, 0.9, "Top-K 相似结构案例", fc="#F3F4F6")
    box(8.6, 11.2, 2.6, 2.0, "历史案例库\n106张\n11维描述子", fc="#F3F4F6", fs=7.5, ls="--")

    arrow(cx, 13.75, cx, 13.30)
    arrow(cx, 12.30, cx, 11.90)
    arrow(cx, 10.50, cx, 9.75)
    arrow(3.7, 8.65, 2.6, 8.00)
    arrow(cx, 8.65, cx, 8.00)
    arrow(6.3, 8.65, 7.4, 8.00)
    arrow(2.4, 7.20, 4.0, 6.45)
    arrow(cx, 7.20, cx, 6.45)
    arrow(7.6, 7.20, 6.0, 6.45)
    arrow(cx, 5.15, cx, 4.45)
    arrow(7.6, 10.5, 7.4, 9.5)
    arrow(8.2, 10.2, 7.0, 6.4)

    ax.text(0.2, 9.2, "第一阶段", fontsize=7, color=GRAY, rotation=90, va="center")
    ax.text(0.2, 5.8, "第二阶段", fontsize=7, color=GRAY, rotation=90, va="center")

    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.01)
    fig.savefig(os.path.join(FIG, "final_fig1.png"), dpi=DPI)
    plt.close(fig)


def fig2_methods():
    methods = ["Random", "原始统计", "GNN", "Hybrid", "欧氏", "余弦", "曼哈顿"]
    ndcg3 = [0.6214, 0.5394, 0.5947, 0.7135, 0.7182, 0.7214, 0.7458]
    recall3 = [15.56, 17.78, 11.11, 15.56, 28.89, 26.67, 35.56]
    mean_score3 = [0.667, 0.578, 0.778, 0.800, 1.000, 0.889, 1.111]
    x = np.arange(len(methods))
    fig, axes = plt.subplots(3, 1, figsize=(7.8 * CM, 9.2 * CM))
    panels = [(ndcg3, "NDCG@3", BLUE, "{:.3f}", 0.85),
              (recall3, "Recall@3 / %", GREEN, "{:.1f}%", 42),
              (mean_score3, "MeanScore@3", ORANGE, "{:.3f}", 1.3)]
    for ax, (vals, title, color, fmt, ylim) in zip(axes, panels):
        bars = ax.bar(x, vals, 0.62, color=color)
        ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8, rotation=20, ha="right")
        ax.set_ylim(0, ylim); ax.set_title(title, fontsize=9)
        ax.tick_params(axis="y", labelsize=7.5)
        ax.grid(axis="y", ls="--", alpha=0.4)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + ylim * 0.015,
                    fmt.format(v), ha="center", fontsize=7)
    fig.tight_layout(h_pad=1.0)
    fig.savefig(os.path.join(FIG, "final_fig2.png"), dpi=DPI)
    plt.close(fig)


def fig3_ablation():
    fig, axes = plt.subplots(2, 1, figsize=(7.8 * CM, 7.6 * CM))
    groups = ["类型占比", "面积占比", "面积分布", "空间统计"]
    top1 = [86.79, 80.19, 58.49, 37.74]
    ax = axes[0]
    bars = ax.bar(groups, top1, color=[BLUE, GREEN, PURPLE, GRAY])
    for rect, v in zip(bars, top1):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 1.2, f"{v:.1f}", ha="center", fontsize=7.5)
    ax.set_ylim(0, 100); ax.set_ylabel("Top-1 准确率 / %", fontsize=8)
    ax.set_title("(a) 单特征组", fontsize=9); ax.tick_params(labelsize=7.5)
    ax.grid(axis="y", ls="--", alpha=0.4)
    combos = ["类型", "+面积", "+分布", "+空间"]
    top1p = [86.79, 88.68, 87.74, 87.74]
    ax = axes[1]
    ax.plot(range(len(combos)), top1p, "o-", color=BLUE, lw=1.3)
    for i, v in enumerate(top1p):
        ax.text(i, v + 1.2, f"{v:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(range(len(combos))); ax.set_xticklabels(combos, fontsize=8)
    ax.set_ylim(80, 100); ax.set_title("(b) 逐组累加", fontsize=9)
    ax.tick_params(axis="y", labelsize=7.5); ax.grid(axis="y", ls="--", alpha=0.4)
    fig.tight_layout(h_pad=1.2)
    fig.savefig(os.path.join(FIG, "final_fig3.png"), dpi=DPI)
    plt.close(fig)


def fig4_tsne():
    emb = np.load("experiments/embeddings/embeddings.npy").astype(np.float64)
    labels = np.load("experiments/embeddings/labels.npy").astype(int)
    gnn = emb[:, :128]
    from sklearn.manifold import TSNE
    z = TSNE(n_components=2, perplexity=15, random_state=42, init="pca").fit_transform(gnn)
    names = ["剪力墙", "框剪", "框架"]; colors = [BLUE, ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(7.8 * CM, 5.9 * CM))
    for c, name, color in zip([0, 1, 2], names, colors):
        m = labels == c
        ax.scatter(z[m, 0], z[m, 1], s=14, c=color, label=name, edgecolors="white", linewidths=0.3, alpha=0.85)
    ax.set_xlabel("t-SNE 维度 1", fontsize=8); ax.set_ylabel("t-SNE 维度 2", fontsize=8)
    ax.legend(fontsize=7.5, markerscale=1.2); ax.tick_params(labelsize=7); ax.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "final_fig4.png"), dpi=DPI)
    plt.close(fig)


def fig5_cross():
    methods = ["Statistics", "GNN", "Hybrid"]
    same = [0.8670, 0.8092, 0.8582]; cross = [0.5095, 0.3571, 0.4361]
    x = np.arange(len(methods)); fig, ax = plt.subplots(figsize=(7.8 * CM, 4.4 * CM))
    w = 0.35
    b1 = ax.bar(x - w / 2, same, w, label="同项目", color=GREEN)
    b2 = ax.bar(x + w / 2, cross, w, label="跨项目", color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(methods, fontsize=8)
    ax.set_ylabel("NDCG@5", fontsize=8); ax.set_ylim(0, 1.12)
    ax.legend(fontsize=7.5); ax.tick_params(labelsize=7.5); ax.grid(axis="y", ls="--", alpha=0.4)
    for rect, v in zip(list(b1) + list(b2), same + cross):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.02, f"{v:.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG, "final_fig5.png"), dpi=DPI)
    plt.close(fig)


if __name__ == "__main__":
    fig1_pipeline()
    fig2_methods()
    fig3_ablation()
    fig4_tsne()
    fig5_cross()
    print("saved final figures to", FIG)
