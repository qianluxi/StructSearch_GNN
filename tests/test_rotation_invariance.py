"""旋转不变性测试：整体旋转图纸后，特征与构图保持不变。"""

import os

import numpy as np

from datasets.dxf_parser import process_dxf_file
from datasets.feature_engineering import FeatureEngineering
from datasets.graph_builder import GraphBuilder
from datasets.graph_statistics import GraphStatistics

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DIR = os.path.join(ROOT_DIR, "data", "raw")


def _rotate_points(points, angle_deg):
    theta = np.deg2rad(angle_deg)
    rotation = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)],
    ])
    return points @ rotation.T


def test_rotation_invariance():
    dxf_files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".dxf"))
    assert dxf_files, "data/raw 下没有 DXF 文件"

    nodes = process_dxf_file(os.path.join(RAW_DIR, dxf_files[0]))
    assert nodes

    fe = FeatureEngineering()
    gb = GraphBuilder(k=6)

    node_feat_1, pos_1 = fe.build(nodes)
    graph_1 = gb.build(node_feat_1, pos_1)
    graph_stats_1 = GraphStatistics(
        nodes,
        aligned_pos=pos_1,
    ).build_feature_vector()

    rotated_nodes = []
    for node in nodes:
        new_node = dict(node)
        new_node["centroid"] = tuple(
            _rotate_points(np.array([node["centroid"]]), 30)[0]
        )
        rotated_nodes.append(new_node)

    node_feat_2, pos_2 = fe.build(rotated_nodes)
    graph_2 = gb.build(node_feat_2, pos_2)
    graph_stats_2 = GraphStatistics(
        rotated_nodes,
        aligned_pos=pos_2,
    ).build_feature_vector()

    # 节点特征与图级统计特征不依赖原始坐标，应完全一致
    np.testing.assert_allclose(node_feat_1, node_feat_2, atol=1e-5)
    np.testing.assert_allclose(graph_stats_1, graph_stats_2, atol=1e-5)

    # PCA 对齐后的坐标与 KNN 距离边属性只依赖相对位置，旋转后应一致
    np.testing.assert_allclose(pos_1, pos_2, atol=1e-3)
    np.testing.assert_allclose(
        graph_1.edge_attr.numpy(),
        graph_2.edge_attr.numpy(),
        atol=1e-5,
    )
