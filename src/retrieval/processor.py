"""单张 DXF -> 模型可用的 PyG Data。

与 StructureDataset.process() 的构图逻辑保持一致：
DXF -> process_dxf_file -> FeatureEngineering -> GraphBuilder -> GraphStatistics。
"""

import os
import sys
import logging

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import torch
from torch_geometric.data import Data

from datasets.dxf_parser import process_dxf_file
from datasets.feature_engineering import FeatureEngineering
from datasets.graph_builder import GraphBuilder
from datasets.graph_statistics import GraphStatistics


logger = logging.getLogger(__name__)


def process_single_dxf(filepath, k=6, connect_mode="knn", radius=None):
    """将单个 DXF 文件转换为模型可用的 PyG Data 对象。

    Args:
        filepath: DXF 文件路径
        k / connect_mode / radius: 构图参数，须与建库时的参数一致

    Returns:
        Data 对象，包含原始（未标准化）字段：
        x_raw, x, edge_index, edge_attr, pos, graph_feature_raw, filename。
        解析失败或构件不足时返回 None。

    Note:
        模型训练用的是标准化特征。请先用
        standardize_data(data, node_mean, node_std, graph_mean, graph_std)
        对返回结果做标准化后再送入模型。
    """

    try:
        nodes = process_dxf_file(filepath)
    except Exception as exc:  # noqa: BLE001 - 单文件解析失败不应中断检索
        logger.warning("[DXF ERROR] %s: %s", os.path.basename(filepath), exc)
        return None

    if not nodes:
        logger.warning("[DXF WARN] %s: 未解析到构件节点", os.path.basename(filepath))
        return None

    node_features, aligned_pos = FeatureEngineering().build(nodes)
    if node_features is None or aligned_pos is None:
        return None

    graph_data = GraphBuilder(
        connect_mode=connect_mode,
        k=k,
        radius=radius
    ).build(node_features, aligned_pos)

    if graph_data is None:
        return None

    graph_feat_raw = GraphStatistics(
        nodes,
        aligned_pos=aligned_pos
    ).build_feature_vector()

    data = Data(
        x_raw=graph_data.x.clone(),
        x=graph_data.x.clone(),
        edge_index=graph_data.edge_index,
        edge_attr=graph_data.edge_attr,
        pos=graph_data.pos,
        graph_feature_raw=torch.tensor(graph_feat_raw, dtype=torch.float),
    )
    data.filename = os.path.basename(filepath)
    return data


def standardize_data(data, node_mean, node_std, graph_mean, graph_std):
    """按 StructureDataset 保存的统计量标准化节点/图级特征（原地修改）。

    Args:
        data: process_single_dxf 返回的 Data 对象
        node_mean / node_std: 节点特征标准化统计量 (9,)
        graph_mean / graph_std: 图级特征标准化统计量 (11,)

    Returns:
        填充了 data.x 与 data.graph_feature 的同一个 Data 对象。
    """

    device = data.x.device

    node_mean = torch.as_tensor(node_mean, dtype=torch.float32, device=device)
    node_std = torch.as_tensor(node_std, dtype=torch.float32, device=device)
    graph_mean = torch.as_tensor(graph_mean, dtype=torch.float32, device=device)
    graph_std = torch.as_tensor(graph_std, dtype=torch.float32, device=device)

    data.x = (data.x_raw.to(device) - node_mean) / node_std
    data.graph_feature = (
        data.graph_feature_raw.to(device) - graph_mean
    ) / graph_std

    return data
