import os
import sys
import numpy as np
import torch

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT_DIR, "src"))

from datasets.dxf_parser import process_dxf_file
from datasets.feature_engineering import FeatureEngineering
from datasets.graph_builder import GraphBuilder
from datasets.graph_statistics import GraphStatistics


def rotate_points(points, angle_deg):
    theta = np.deg2rad(angle_deg)
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return points @ R.T


def main():

    # 选一个文件（动态取 data/raw 下第一个 DXF，避免依赖特定文件名）
    raw_dir = os.path.join(ROOT_DIR, "data", "raw")
    dxf_files = sorted(
        f for f in os.listdir(raw_dir) if f.endswith(".dxf")
    )
    if not dxf_files:
        raise FileNotFoundError(f"data/raw 下没有 DXF 文件: {raw_dir}")

    dxf_path = os.path.join(raw_dir, dxf_files[0])

    nodes = process_dxf_file(dxf_path)

    fe = FeatureEngineering()
    gb = GraphBuilder(k=6)

    # -----------------------------------
    # 原始图
    # -----------------------------------
    nf1, pos1 = fe.build(nodes)
    g1 = gb.build(nf1, pos1)
    gf1 = GraphStatistics(nodes, aligned_pos=pos1).build_feature_vector()

    # -----------------------------------
    # 旋转节点
    # -----------------------------------
    rotated_nodes = []

    for n in nodes:
        new_n = n.copy()
        rotated_centroid = rotate_points(
            np.array([n["centroid"]]),
            30
        )[0]
        new_n["centroid"] = tuple(rotated_centroid)
        rotated_nodes.append(new_n)

    nf2, pos2 = fe.build(rotated_nodes)
    g2 = gb.build(nf2, pos2)
    gf2 = GraphStatistics(rotated_nodes, aligned_pos=pos2).build_feature_vector()

    # -----------------------------------
    # 对比
    # -----------------------------------
    print("Node feature difference:",
          np.abs(nf1 - nf2).max())

    print("Graph feature difference:",
          np.abs(gf1 - gf2).max())

    print("Edge attr difference:",
          torch.abs(g1.edge_attr - g2.edge_attr).max().item())

    print("Pos difference:",
          np.abs(pos1 - pos2).max())


if __name__ == "__main__":
    main()
