import torch
import numpy as np
from torch_geometric.data import Data
from sklearn.neighbors import NearestNeighbors


class GraphBuilder:

    def __init__(self, connect_mode="knn", k=6, radius=None):
        """
        connect_mode: "knn" 或 "radius"
        k: knn邻居数
        radius: 半径连接阈值
        """
        self.connect_mode = connect_mode
        self.k = k
        self.radius = radius

    # =========================================================
    # 主接口
    # =========================================================
    def build(self, node_features, pos):
        """
        node_features: numpy array (N, F)
        pos: numpy array (N, 2)   # 已经做过平移中心化 + PCA对齐
        """

        if pos is None or len(pos) == 0:
            return None

        N = len(pos)

        # 单节点图，直接返回空边
        if N == 1:
            data = Data(
                x=torch.tensor(node_features, dtype=torch.float),
                edge_index=torch.empty((2, 0), dtype=torch.long),
                edge_attr=torch.empty((0, 1), dtype=torch.float),
                pos=torch.tensor(pos, dtype=torch.float)
            )
            return data

        edge_index, edge_attr = self._build_edges(pos)

        data = Data(
            x=torch.tensor(node_features, dtype=torch.float),
            edge_index=edge_index,
            edge_attr=edge_attr,
            pos=torch.tensor(pos, dtype=torch.float)
        )

        return data

    # =========================================================
    # 构建边
    # =========================================================
    def _build_edges(self, pos):

        if self.connect_mode == "knn":
            return self._build_knn_edges(pos)

        elif self.connect_mode == "radius":
            return self._build_radius_edges(pos)

        else:
            raise ValueError("Unsupported connect_mode")

    # =========================================================
    # KNN 图
    # =========================================================
    def _build_knn_edges(self, pos):

        N = len(pos)

        # 防止 N < k 报错
        k_eff = min(self.k + 1, N)

        nbrs = NearestNeighbors(n_neighbors=k_eff)
        nbrs.fit(pos)

        distances, indices = nbrs.kneighbors(pos)

        edge_list = []
        edge_attr = []

        for i in range(N):
            # 跳过自己（索引0必然是自己）
            for j_idx in range(1, len(indices[i])):
                j = indices[i][j_idx]
                dist = distances[i][j_idx]

                edge_list.append([i, j])
                edge_list.append([j, i])

                edge_attr.append([dist])
                edge_attr.append([dist])

        if len(edge_list) == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return edge_index, edge_attr

    # =========================================================
    # 半径图
    # =========================================================
    def _build_radius_edges(self, pos):

        if self.radius is None:
            raise ValueError("radius must be specified")

        N = len(pos)

        edge_list = []
        edge_attr = []

        for i in range(N):
            for j in range(N):
                if i == j:
                    continue

                dist = np.linalg.norm(pos[i] - pos[j])

                if dist <= self.radius:
                    edge_list.append([i, j])
                    edge_attr.append([dist])

        if len(edge_list) == 0:
            return torch.empty((2, 0), dtype=torch.long), torch.empty((0, 1), dtype=torch.float)

        edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_attr, dtype=torch.float)

        return edge_index, edge_attr
