# -*- coding: utf-8 -*-
"""Regenerate 图1 (pipeline) tightly, export PNG + SVG (vector)."""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "axes.unicode_minus": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, "figures")
os.makedirs(FIG, exist_ok=True)

BLUE = "#3B82F6"
GREEN = "#10B981"
ORANGE = "#F59E0B"
GRAY = "#6B7280"


def main():
    W = 7.8 / 2.54
    fig, ax = plt.subplots(figsize=(W, 10.2 / 2.54))
    ax.set_xlim(0, 10)
    ax.set_ylim(3.0, 15.2)
    ax.axis("off")

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
    box(cx, 11.2, 7.2, 1.4, "11维结构统计描述子\n类型·面积·分布·空间", fc="#DBEAFE", ec=BLUE)
    box(cx, 9.2, 6.6, 1.1, "第一阶段：结构体系识别\n（余弦近邻分类）", fc="#FEF3C7", ec=ORANGE)
    for xi, name in [(2.4, "框架"), (5.0, "框剪"), (7.6, "剪力墙")]:
        box(xi, 7.6, 2.0, 0.8, name, fc="#FFEDD5", ec=ORANGE)
    box(cx, 5.8, 7.4, 1.3, "第二阶段：类内相似检索\n标准化+曼哈顿距离", fc="#FEF3C7", ec=ORANGE)
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

    fig.savefig(os.path.join(FIG, "final_fig1.png"), dpi=600,
                bbox_inches="tight", pad_inches=0.04)
    fig.savefig(os.path.join(FIG, "final_fig1.svg"),
                bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    print("saved final_fig1.png / .svg")


if __name__ == "__main__":
    main()
