"""图嵌入提取：供可视化/检索评估脚本复用。

修复要点：
- 使用当前 GraphEncoder 契约（forward 返回 (embedding, logits)）
- 从 checkpoints/model.pt 加载 checkpoint（含 state_dict 元信息）
- 支持任意 DXF 列表：走与 SearchEngine 相同的标准化管线
- GPU 张量先 .cpu() 再 .numpy()
"""

import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
SRC_DIR = os.path.join(ROOT_DIR, "src")
for _path in (SRC_DIR, ROOT_DIR):
    if _path not in sys.path:
        sys.path.insert(0, _path)

import numpy as np
import torch
from torch_geometric.loader import DataLoader

from datasets.structure_dataset import StructureDataset
from models.model import GraphEncoder
from retrieval.processor import process_single_dxf, standardize_data


def load_model(model_path=None, device=None):
    """加载 GraphEncoder checkpoint。

    Args:
        model_path: checkpoint 路径，默认 checkpoints/model.pt
        device: 目标设备，默认自动选择

    Returns:
        处于 eval 模式的 GraphEncoder
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model_path = model_path or os.path.join(ROOT_DIR, "checkpoints", "model.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    model = GraphEncoder(
        hidden_dim=checkpoint.get("hidden_dim", 64),
        emb_dim=checkpoint["emb_dim"],
        num_classes=checkpoint["num_classes"],
    )
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()
    return model


def extract_embeddings(dxf_list=None, batch_size=8, device=None):
    """提取一批 DXF 的图嵌入。

    Args:
        dxf_list: DXF 路径列表；为 None 时使用整个数据集（data/raw）
        batch_size: DataLoader 批大小
        device: 目标设备，默认自动选择

    Returns:
        (N, emb_dim) numpy 数组
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    # 数据集提供标准化统计量
    dataset = StructureDataset(root=os.path.join(ROOT_DIR, "data"))
    model = load_model(device=device)

    if dxf_list is None:
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    else:
        data_list = []
        for f in dxf_list:
            data = process_single_dxf(f)
            if data is None:
                continue
            data = standardize_data(
                data,
                dataset.node_feat_mean,
                dataset.node_feat_std,
                dataset.graph_feat_mean,
                dataset.graph_feat_std
            )
            data_list.append(data)

        if not data_list:
            raise ValueError("没有成功解析任何 DXF 文件")

        loader = DataLoader(data_list, batch_size=batch_size, shuffle=False)

    embeddings = []

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            emb, _ = model(batch.x, batch.edge_index, batch.batch)
            embeddings.append(emb.cpu().numpy())

    return np.vstack(embeddings)
