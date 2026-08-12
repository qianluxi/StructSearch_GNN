"""图嵌入提取工具。

基于训练好的 GraphEncoder 提取 graph-level embedding，输出 numpy 数组。
接口与 legacy src/retrieval/R/embedder.py 保持一致：
extract_one / extract_batch / extract，以及别名 get_embedding / embed。
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
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class Embedder:
    """结构图 embedding 提取工具类。"""

    def __init__(self, model, device=None):
        """Args:
            model: 已加载权重的 GraphEncoder（处于 eval 模式）
            device: 'cpu' 或 'cuda'，默认自动选择
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def extract_one(self, data):
        """提取单个图的 embedding。

        Args:
            data: PyG Data 对象（需已标准化）

        Returns:
            embedding: (emb_dim,) numpy 数组
        """
        data = data.to(self.device)
        batch = torch.zeros(
            data.num_nodes,
            dtype=torch.long,
            device=self.device
        )
        embedding, _ = self.model(data.x, data.edge_index, batch)
        return embedding.squeeze(0).cpu().numpy()

    @torch.no_grad()
    def extract_batch(self, dataset, batch_size=32):
        """批量提取 embedding。

        Args:
            dataset: List[Data]、InMemoryDataset 或 DataLoader
            batch_size: 当 dataset 不是 DataLoader 时使用

        Returns:
            embeddings: (N, emb_dim) numpy 数组
        """
        if isinstance(dataset, DataLoader):
            loader = dataset
        else:
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False
            )

        all_embeddings = []

        for batch in loader:
            batch = batch.to(self.device)
            embedding, _ = self.model(
                batch.x,
                batch.edge_index,
                batch.batch
            )
            all_embeddings.append(embedding.cpu().numpy())

        if not all_embeddings:
            raise ValueError("输入数据为空，无法提取 embedding")

        return np.vstack(all_embeddings)

    def extract(self, data, batch_size=32):
        """自动判断单图或批量并提取 embedding。"""

        if isinstance(data, Data):
            return self.extract_one(data)
        return self.extract_batch(data, batch_size=batch_size)

    # 兼容旧接口
    get_embedding = extract
    embed = extract


def build_index_vector(embedding, graph_feature):
    """拼接 GNN 嵌入与图级统计特征，构成最终检索向量。

    Args:
        embedding: GNN graph embedding (emb_dim,)
        graph_feature: 标准化的图级统计特征 (11,)

    Returns:
        (emb_dim + 11,) float32 numpy 数组
    """
    emb = np.asarray(embedding, dtype=np.float32).ravel()
    stats = np.asarray(graph_feature, dtype=np.float32).ravel()
    return np.concatenate([emb, stats])
