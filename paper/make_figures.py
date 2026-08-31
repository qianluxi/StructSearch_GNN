"""论文插图生成脚本：为《建筑结构》期刊格式论文生成图1~图5。"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

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


# ----------------------------------------------------------------
# 图1 两阶段结构相似性检索框架
# ----------------------------------------------------------------
def fig1_pipeline():
    fig, ax = plt.subplots(figsize=(7.2, 8.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12.4)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#FFFFFF", ec="#333333", fs=10, lw=1.2, ls="-"):
        p = FancyBboxPatch(
            (x - w / 2, y - h / 2), w, h,
            boxstyle="round,pad=0.02,rounding_size=0.12",
            fc=fc, ec=ec, lw=lw, ls=ls,
        )
        ax.add_patch(p)
        ax.text(x, y, text, ha="center", va="center", fontsize=fs)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch(
            (x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
            color="#333333", lw=1.1,
        ))

    cx = 5.0
    box(cx, 11.8, 3.6, 0.7, "查询 DXF 结构图纸", fc="#F3F4F6")
    box(cx, 10.7, 5.4, 0.8, "结构解析：构件提取 · 几何对齐 · 构件分类", fc="#DBEAFE", ec=BLUE)
    box(cx, 9.45, 6.2, 1.0,
        "11 维结构统计描述子\n类型占比 · 面积占比 · 面积分布 · 空间统计",
        fc="#DBEAFE", ec=BLUE, fs=9.5)

    # Stage 1
    box(cx, 7.9, 5.6, 0.85,
        "第一阶段：结构体系识别（余弦近邻分类）",
        fc="#FEF3C7", ec=ORANGE, fs=10)
    for xi, name in [(2.4, "框架"), (5.0, "框剪"), (7.6, "剪力墙")]:
        box(xi, 6.7, 1.7, 0.6, name, fc="#FFEDD5", ec=ORANGE, fs=9.5)

    # Stage 2
    box(cx, 5.35, 6.6, 0.95,
        "第二阶段：类型约束下类内相似性检索\n标准化描述子 + Manhattan 距离排序",
        fc="#FEF3C7", ec=ORANGE, fs=9.5)
    box(cx, 3.95, 4.6, 0.7, "Top-K 相似结构案例", fc="#F3F4F6")

    # 案例库
    box(8.55, 9.45, 2.5, 1.35,
        "历史案例库\n106 张图纸\n11 维标准化描述子",
        fc="#F3F4F6", fs=8.5, ls="--")

    arrow(cx, 11.45, cx, 11.12)
    arrow(cx, 10.3, cx, 9.97)
    arrow(cx, 8.93, cx, 8.35)
    arrow(3.6, 7.5, 2.4, 7.02)
    arrow(cx, 7.47, cx, 7.02)
    arrow(6.4, 7.5, 7.6, 7.02)
    arrow(2.4, 6.38, 4.0, 5.85)
    arrow(5.0, 6.38, cx, 5.85)
    arrow(7.6, 6.38, 6.0, 5.85)
    arrow(cx, 4.86, cx, 4.32)
    # 案例库 -> 阶段1/阶段2
    arrow(7.55, 8.95, 6.7, 8.25)
    arrow(8.0, 8.78, 7.1, 5.7)

    ax.text(0.25, 7.9, "第一阶段", fontsize=9, color=GRAY, rotation=90, va="center")
    ax.text(0.25, 5.35, "第二阶段", fontsize=9, color=GRAY, rotation=90, va="center")

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_pipeline.png"), dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------
# 图2 各检索方法性能对比（NDCG@3、Recall@3、MeanScore@3）
# ----------------------------------------------------------------
def fig2_methods():
    methods = ["Random", "未标准化统计", "GNN", "Hybrid",
               "统计+欧氏", "统计+余弦", "统计+曼哈顿"]
    ndcg3 = [0.6214, 0.5394, 0.5947, 0.7135, 0.7182, 0.7214, 0.7458]
    recall3 = [15.56, 17.78, 11.11, 15.56, 28.89, 26.67, 35.56]
    mean_score3 = [0.667, 0.578, 0.778, 0.800, 1.000, 0.889, 1.111]

    x = np.arange(len(methods))
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 3.0))
    panels = [
        (ndcg3, "NDCG@3", BLUE, "{:.3f}", 0.85),
        (recall3, "Recall@3 / %", GREEN, "{:.1f}%", 42),
        (mean_score3, "MeanScore@3", ORANGE, "{:.3f}", 1.3),
    ]
    for ax, (vals, title, color, fmt, ylim) in zip(axes, panels):
        bars = ax.bar(x, vals, 0.62, color=color)
        ax.set_xticks(x)
        ax.set_xticklabels(methods, fontsize=7.5, rotation=30, ha="right")
        ax.set_ylim(0, ylim)
        ax.set_title(title, fontsize=9.5)
        ax.grid(axis="y", ls="--", alpha=0.4)
        for rect, v in zip(bars, vals):
            ax.text(rect.get_x() + rect.get_width() / 2, v + ylim * 0.015,
                    fmt.format(v), ha="center", fontsize=6.8)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_methods.png"), dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------
# 图3 特征组消融实验
# ----------------------------------------------------------------
def fig3_ablation():
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3))

    groups = ["类型占比", "面积占比", "面积分布", "空间统计"]
    top1 = [86.79, 80.19, 58.49, 37.74]
    ax = axes[0]
    bars = ax.bar(groups, top1, color=[BLUE, GREEN, PURPLE, GRAY])
    for rect, v in zip(bars, top1):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 1.2,
                f"{v:.1f}", ha="center", fontsize=8.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Top-1 类型检索准确率 / %", fontsize=9)
    ax.set_title("(a) 单特征组", fontsize=9.5)
    ax.tick_params(axis="x", labelsize=8.5)
    ax.grid(axis="y", ls="--", alpha=0.4)

    combos = ["类型", "+面积占比", "+面积分布", "+空间统计"]
    top1p = [86.79, 88.68, 87.74, 87.74]
    ax = axes[1]
    ax.plot(range(len(combos)), top1p, "o-", color=BLUE, lw=1.5)
    for i, v in enumerate(top1p):
        ax.text(i, v + 1.2, f"{v:.2f}", ha="center", fontsize=8.5)
    ax.set_xticks(range(len(combos)))
    ax.set_xticklabels(combos, fontsize=8.5)
    ax.set_ylim(80, 100)
    ax.set_title("(b) 逐组累加", fontsize=9.5)
    ax.grid(axis="y", ls="--", alpha=0.4)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_ablation.png"), dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------
# 图4 同项目与跨项目检索性能对比
# ----------------------------------------------------------------
def fig4_cross_project():
    methods = ["Statistics", "GNN", "Hybrid"]
    same = [0.9607, 0.9537, 0.9607]
    cross = [0.5007, 0.2729, 0.4555]

    x = np.arange(len(methods))
    fig, ax = plt.subplots(figsize=(6.0, 3.4))
    w = 0.35
    b1 = ax.bar(x - w / 2, same, w, label="同项目子集", color=GREEN)
    b2 = ax.bar(x + w / 2, cross, w, label="跨项目子集", color=ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels(methods, fontsize=10)
    ax.set_ylabel("NDCG@5", fontsize=10)
    ax.set_ylim(0, 1.12)
    ax.legend(fontsize=9)
    ax.grid(axis="y", ls="--", alpha=0.4)
    for rect, v in zip(list(b1) + list(b2), same + cross):
        ax.text(rect.get_x() + rect.get_width() / 2, v + 0.02,
                f"{v:.3f}", ha="center", fontsize=8.5)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_cross_project.png"), dpi=300)
    plt.close(fig)


# ----------------------------------------------------------------
# 图5 GNN 嵌入 t-SNE 分布（含图例）
# ----------------------------------------------------------------
def fig5_tsne():
    emb = np.load("experiments/embeddings/embeddings.npy").astype(np.float64)
    labels = np.load("experiments/embeddings/labels.npy").astype(int)
    gnn = emb[:, :128]

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="pca")
    z = tsne.fit_transform(gnn)

    names = ["剪力墙", "框剪", "框架"]
    colors = [BLUE, ORANGE, GREEN]
    fig, ax = plt.subplots(figsize=(5.6, 4.2))
    for c, name, color in zip([0, 1, 2], names, colors):
        m = labels == c
        ax.scatter(z[m, 0], z[m, 1], s=34, c=color, label=name,
                   edgecolors="white", linewidths=0.5, alpha=0.85)
    ax.set_xlabel("t-SNE 维度 1", fontsize=10)
    ax.set_ylabel("t-SNE 维度 2", fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(ls="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig5_tsne.png"), dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    fig1_pipeline()
    fig2_methods()
    fig3_ablation()
    fig4_cross_project()
    fig5_tsne()
    print("figures saved to", FIG_DIR)
