"""论文图6：典型检索案例的结构平面构成对比图。

复用 dxf_parser 的解析管线（与检索管线完全一致），保留对齐后的构件多边形，
按构件类型着色绘制真实构件几何，构成 3 行 × 4 列的案例对比图：

行(a) 高一致案例：湘湖里_框架_H3 及其高分候选
行(b) 跨项目相似与弱标签失效：景芳三堡_框剪_B 及其跨/同项目候选
行(c) 易误判案例：苏州华贸_剪力墙_10A 及其低分近邻候选

全部查询/候选、检索排名与专家评分均来自冻结的专家基准数据。
"""

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))

# 环境中装有同名第三方包 `datasets`，此处按文件路径直接加载项目解析模块
import importlib.util  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "structsearch_dxf_parser",
    os.path.join(ROOT, "src", "datasets", "dxf_parser.py"),
)
_dxf_parser = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dxf_parser)

align_polygon = _dxf_parser.align_polygon
classify_node = _dxf_parser.classify_node
compute_r_from_long_leg = _dxf_parser.compute_r_from_long_leg
compute_spans = _dxf_parser.compute_spans
get_longest_edge_direction = _dxf_parser.get_longest_edge_direction
load_polygons_from_dxf = _dxf_parser.load_polygons_from_dxf

from shapely.geometry import Polygon  # noqa: E402
from shapely.affinity import rotate  # noqa: E402

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False

RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(HERE, "figures", "fig6_cases.png")

# 构件着色：贴合结构平面表达习惯（剪力墙填黑、短肢墙灰、柱描边）
TYPE_STYLE = {
    0: {"fc": "white", "ec": "black", "lw": 0.9},   # 柱
    1: {"fc": "#9CA3AF", "ec": "#4B5563", "lw": 0.5},  # 短肢剪力墙
    2: {"fc": "black", "ec": "black", "lw": 0.3},   # 剪力墙
}


def load_case(name):
    """解析 DXF，返回 [(aligned_polygon, type_id), ...]，与 process_dxf_file 同逻辑。"""
    path = os.path.join(RAW_DIR, name)
    polygons = load_polygons_from_dxf(path)
    items = []
    for polygon in polygons:
        if polygon.area < 1e-4:
            continue
        aligned = align_polygon(polygon)
        Lx, Ly = compute_spans(aligned)
        r, thickness = compute_r_from_long_leg(aligned, Lx, Ly)
        if r is None:
            continue
        if thickness <= 1e-6:
            angle = get_longest_edge_direction(polygon)
            aligned2 = rotate(polygon, -angle, origin="centroid", use_radians=False)
            Lx2, Ly2 = compute_spans(aligned2)
            r2, thickness2 = compute_r_from_long_leg(aligned2, Lx2, Ly2)
            if thickness2 > 1e-6:
                aligned, Lx, Ly, r, thickness = aligned2, Lx2, Ly2, r2, thickness2
        type_str = classify_node(r, thickness)
        items.append((aligned, {"Column": 0, "ShortWall": 1, "Wall": 2}[type_str]))
    return items


def draw_case(ax, items):
    for geom, tid in items:
        style = TYPE_STYLE[tid]
        polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
        for poly in polys:
            coords = list(poly.exterior.coords)
            ax.add_patch(MplPolygon(coords, closed=True, **style))
    ax.set_aspect("equal")
    ax.autoscale_view()
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)


# (行标签, [(文件名, 面板说明), ...])
ROWS = [
    ("(a) 高一致案例：检索前列与专家判断一致", [
        ("湘湖里_框架_H3.dxf", "查询：湘湖里_框架_H3"),
        ("湘湖里_框架_H8.dxf", "湘湖里_框架_H8（统计路线Top-1，3分）"),
        ("湘湖里_框架_H9.dxf", "湘湖里_框架_H9（统计路线Top-2，3分）"),
        ("湘湖里_框架_H4.dxf", "湘湖里_框架_H4（GNN路线Top-1，4分）"),
    ]),
    ("(b) 跨项目相似与弱标签失效", [
        ("景芳三堡_框剪_B.dxf", "查询：景芳三堡_框剪_B"),
        ("勾庄好运来_框剪_20.dxf", "勾庄好运来_框剪_20（跨项目，3分）"),
        ("景芳三堡_框剪_C.dxf", "景芳三堡_框剪_C（同项目，仅1分）"),
        ("西大门_框剪_2.dxf", "西大门_框剪_2（统计路线Top-1，0分）"),
    ]),
    ("(c) 易误判案例：描述子近邻≠结构相似", [
        ("苏州华贸_剪力墙_10A.dxf", "查询：苏州华贸_剪力墙_10A"),
        ("勾庄好运来_剪力墙_8.dxf", "勾庄好运来_剪力墙_8（统计路线Top-1，1分）"),
        ("苏州华贸_剪力墙_6A.dxf", "苏州华贸_剪力墙_6A（GNN路线Top-1，2分）"),
        ("苏州华贸_剪力墙_6B.dxf", "苏州华贸_剪力墙_6B（统计路线Top-5，3分）"),
    ]),
]


def main():
    fig, axes = plt.subplots(3, 4, figsize=(7.2, 5.6))
    for r, (row_label, panels) in enumerate(ROWS):
        for c, (fname, cap) in enumerate(panels):
            ax = axes[r][c]
            items = load_case(fname)
            draw_case(ax, items)
            ax.set_title(cap, fontsize=6.6, pad=2)
        axes[r][0].text(
            -0.22, 0.5, row_label, transform=axes[r][0].transAxes,
            rotation=90, va="center", ha="center", fontsize=7.5,
        )

    # 类型图例
    handles = [
        MplPolygon([(0, 0)], closed=True, **TYPE_STYLE[2], label="剪力墙"),
        MplPolygon([(0, 0)], closed=True, **TYPE_STYLE[1], label="短肢剪力墙"),
        MplPolygon([(0, 0)], closed=True, **TYPE_STYLE[0], label="柱"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.015))

    fig.tight_layout(rect=[0.03, 0.02, 1, 1])
    fig.savefig(OUT, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
