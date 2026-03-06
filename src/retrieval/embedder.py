"""
Embedder Module v2

功能：
- 使用已训练 StructModel 提取 graph-level embedding
- 支持单图、批量图、DataLoader
- 输出 numpy 数组
- 兼容老接口 get_embedding() 和 embed()
"""

import torch
import numpy as np
from typing import List, Union
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader


class Embedder:
    """
    结构图 embedding 提取工具类
    """

    def __init__(self, model: torch.nn.Module, device: str = None):
        """
        Args:
            model: 已加载权重的 StructModel
            device: 'cpu' 或 'cuda'，默认自动选择
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model = model.to(self.device)
        self.model.eval()

    # ---------------------------
    # 单图 embedding
    # ---------------------------
    @torch.no_grad()
    def extract_one(self, data: Data) -> np.ndarray:
        """
        提取单个图的 embedding

        Args:
            data: PyG Data 对象

        Returns:
            embedding: (hidden_dim,) numpy array
        """
        data = data.to(self.device)
        embedding = self.model.encode(data)[1]  # 返回 logits, embedding
        embedding = embedding.squeeze(0).cpu().numpy()
        return embedding

    # ---------------------------
    # 批量 embedding
    # ---------------------------
    @torch.no_grad()
    def extract_batch(
        self,
        dataset: Union[List[Data], DataLoader],
        batch_size: int = 32,
        num_workers: int = 0
    ) -> np.ndarray:
        """
        批量提取 embedding

        Args:
            dataset: list[Data] 或 DataLoader
            batch_size: 当 dataset 为 list 时使用
            num_workers: DataLoader worker 数量

        Returns:
            embeddings: (N, hidden_dim)
        """
        if isinstance(dataset, list):
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers
            )
        else:
            loader = dataset

        all_embeddings = []

        for batch in loader:
            batch = batch.to(self.device)
            embedding = self.model.encode(batch)[1]  # (B, hidden_dim)
            all_embeddings.append(embedding.cpu().numpy())

        return np.vstack(all_embeddings)

    # ---------------------------
    # 通用接口
    # ---------------------------
    def extract(
        self,
        data: Union[Data, List[Data], DataLoader],
        batch_size: int = 32
    ) -> np.ndarray:
        """
        自动判断单图或批量

        Returns:
            单图 -> (hidden_dim,)
            批量 -> (N, hidden_dim)
        """
        if isinstance(data, Data):
            return self.extract_one(data)
        return self.extract_batch(data, batch_size=batch_size)

    # ---------------------------
    # 兼容旧接口
    # ---------------------------
    get_embedding = extract  # 老代码调用 get_embedding(data)
    embed = extract          # 兼容测试程序直接调用 embed(data)
