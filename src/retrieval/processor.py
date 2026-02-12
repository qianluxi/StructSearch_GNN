import numpy as np
import torch
from torch_geometric.data import Data
from sklearn.neighbors import kneighbors_graph

from src.utils.dxf_utils import parse_dxf_file


def process_single_dxf(filepath):
    """
    将单个DXF文件转化为模型可接受的 Data 对象
    （逻辑与 v1 search.py 完全一致）
    """

    try:
        polygons, centroids, raw_features = parse_dxf_file(filepath)
    except Exception as e:
        print(f"解析错误: {e}")
        return None

    if len(polygons) < 2:
        print("构件过少")
        return None

    # ---------- 归一化 ----------
    centroid_x = centroids[:, 0]
    centroid_y = centroids[:, 1]

    min_x, max_x = np.min(centroid_x), np.max(centroid_x)
    min_y, max_y = np.min(centroid_y), np.max(centroid_y)

    width = max_x - min_x
    height = max_y - min_y

    scale = max(width, height) if max(width, height) > 0 else 1.0

    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2

    norm_pos_x = (centroid_x - center_x) / scale
    norm_pos_y = (centroid_y - center_y) / scale

    bbox_area = width * height if width * height > 0 else 1.0

    norm_area = raw_features[:, 0] / bbox_area
    aspect_ratio = raw_features[:, 1]
    log_aspect_ratio = np.log(aspect_ratio + 1e-6)

    # ---------- 构建特征 ----------
    x_features = np.stack(
        [norm_area, log_aspect_ratio, norm_pos_x, norm_pos_y],
        axis=1
    )

    x = torch.tensor(x_features, dtype=torch.float)

    # ---------- 构图 ----------
    num_nodes = len(polygons)
    k = 5 if num_nodes > 5 else num_nodes - 1

    if k > 0:
        adj_matrix = kneighbors_graph(
            centroids,
            n_neighbors=k,
            mode='connectivity',
            include_self=False
        )
        edge_index = torch.tensor(adj_matrix.nonzero(), dtype=torch.long)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    data = Data(x=x, edge_index=edge_index)
    return data
