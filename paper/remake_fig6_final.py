# -*- coding: utf-8 -*-
"""Regenerate 图6 at final spanning size (16.5 cm) with readable fonts."""
import os
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MplPolygon
from shapely.geometry import Polygon
from shapely.affinity import rotate

plt.rcParams.update({
    "font.family": ["Times New Roman", "SimSun"],
    "axes.unicode_minus": False,
})

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
RAW_DIR = os.path.join(ROOT, "data", "raw")
OUT = os.path.join(HERE, "figures", "final_fig6.png")

_spec = importlib.util.spec_from_file_location(
    "structsearch_dxf_parser", os.path.join(ROOT, "src", "datasets", "dxf_parser.py"))
_dxf = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_dxf)

TYPE_STYLE = {
    0: {"fc": "white", "ec": "black", "lw": 0.5},
    1: {"fc": "#9CA3AF", "ec": "#4B5563", "lw": 0.3},
    2: {"fc": "black", "ec": "black", "lw": 0.2},
}


def load_case(name):
    polygons = _dxf.load_polygons_from_dxf(os.path.join(RAW_DIR, name))
    items = []
    for polygon in polygons:
        if polygon.area < 1e-4:
            continue
        aligned = _dxf.align_polygon(polygon)
        Lx, Ly = _dxf.compute_spans(aligned)
        r, thickness = _dxf.compute_r_from_long_leg(aligned, Lx, Ly)
        if r is None:
            continue
        if thickness <= 1e-6:
            angle = _dxf.get_longest_edge_direction(polygon)
            a2 = rotate(polygon, -angle, origin="centroid", use_radians=False)
            Lx2, Ly2 = _dxf.compute_spans(a2)
            r2, t2 = _dxf.compute_r_from_long_leg(a2, Lx2, Ly2)
            if t2 > 1e-6:
                aligned, Lx, Ly, r, thickness = a2, Lx2, Ly2, r2, t2
        type_str = _dxf.classify_node(r, thickness)
        items.append((aligned, {"Column": 0, "ShortWall": 1, "Wall": 2}[type_str]))
    return items


def draw_case(ax, items):
    for geom, tid in items:
        style = TYPE_STYLE[tid]
        polys = [geom] if isinstance(geom, Polygon) else list(geom.geoms)
        for poly in polys:
            ax.add_patch(MplPolygon(list(poly.exterior.coords), closed=True, **style))
    ax.set_aspect("equal"); ax.autoscale_view(); ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_linewidth(0.4)


ROWS = [
    ("(a) 高一致案例", [
        ("湘湖里_框架_H3.dxf", "H3 查询"),
        ("湘湖里_框架_H8.dxf", "H8 统计Top-1\n(3分)"),
        ("湘湖里_框架_H9.dxf", "H9 统计Top-2\n(3分)"),
        ("湘湖里_框架_H4.dxf", "H4 GNN Top-1\n(4分)"),
    ]),
    ("(b) 跨项目相似与弱标签失效", [
        ("景芳三堡_框剪_B.dxf", "B 查询"),
        ("勾庄好运来_框剪_20.dxf", "好运来20 跨项\n(3分)"),
        ("景芳三堡_框剪_C.dxf", "C 同项目\n(1分)"),
        ("西大门_框剪_2.dxf", "西大门2 统计\nTop-1 (0分)"),
    ]),
    ("(c) 易误判案例", [
        ("苏州华贸_剪力墙_10A.dxf", "10A 查询"),
        ("勾庄好运来_剪力墙_8.dxf", "好运来8 统计\nTop-1 (1分)"),
        ("苏州华贸_剪力墙_6A.dxf", "6A GNN Top-1\n(2分)"),
        ("苏州华贸_剪力墙_6B.dxf", "6B 统计Top-5\n(3分)"),
    ]),
]


def main():
    fig, axes = plt.subplots(3, 4, figsize=(16.5 / 2.54, 13.0 / 2.54))
    for r, (row_label, panels) in enumerate(ROWS):
        for c, (fname, cap) in enumerate(panels):
            ax = axes[r][c]
            draw_case(ax, load_case(fname))
            ax.set_title(cap, fontsize=7, pad=1.5, linespacing=1.0)
        axes[r][0].text(-0.18, 0.5, row_label, transform=axes[r][0].transAxes,
                        rotation=90, va="center", ha="center", fontsize=7.5)
    handles = [
        MplPolygon([(0, 0)], closed=True, **TYPE_STYLE[2], label="剪力墙"),
        MplPolygon([(0, 0)], closed=True, **TYPE_STYLE[1], label="短肢剪力墙"),
        MplPolygon([(0, 0)], closed=True, **TYPE_STYLE[0], label="柱"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7.5,
               frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=[0.02, 0.03, 1, 1])
    fig.savefig(OUT, dpi=600)
    plt.close(fig)
    print("saved:", OUT)


if __name__ == "__main__":
    main()
