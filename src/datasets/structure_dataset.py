import os
import logging
import torch
import numpy as np
from torch_geometric.data import Data, InMemoryDataset

from datasets.dxf_parser import process_dxf_file
from datasets.feature_engineering import FeatureEngineering
from datasets.graph_builder import GraphBuilder
from datasets.graph_statistics import GraphStatistics


logger = logging.getLogger(__name__)

_PIPELINE_FILES = (
    "dxf_parser.py",
    "feature_engineering.py",
    "graph_builder.py",
    "graph_statistics.py",
    "structure_dataset.py",
)


class StructureDataset(InMemoryDataset):

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        k=6,
        connect_mode="knn",
        radius=None
    ):
        self.k = k
        self.connect_mode = connect_mode
        self.radius = radius

        self.fe = FeatureEngineering()

        self.gb = GraphBuilder(
            connect_mode=connect_mode,
            k=k,
            radius=radius
        )

        super().__init__(root, transform, pre_transform)

        if os.path.exists(self.processed_paths[0]):

            loaded = torch.load(self.processed_paths[0])

            if not self._cache_up_to_date(loaded):
                logger.info(
                    "[StructureDataset] raw 目录已变化，自动重建 processed 缓存 ..."
                )
                self.process()
                loaded = torch.load(self.processed_paths[0])

            self.data = loaded["data"]
            self.slices = loaded["slices"]

            self.graph_feat_mean = loaded["graph_mean"]
            self.graph_feat_std = loaded["graph_std"]

            self.node_feat_mean = loaded["node_mean"]
            self.node_feat_std = loaded["node_std"]

        else:
            self.process()

    # -------------------------------------------------------
    # 目录
    # -------------------------------------------------------
    @property
    def raw_dir(self):
        return os.path.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        return [
            f for f in os.listdir(self.raw_dir)
            if f.endswith('.dxf')
        ]

    @property
    def processed_file_names(self):

        name = f'structure_graph_k{self.k}_{self.connect_mode}'

        if self.connect_mode == "radius" and self.radius is not None:
            name += f'_r{self.radius}'

        return [name + '.pt']

    # -------------------------------------------------------
    # 缓存失效检查
    # -------------------------------------------------------
    def _raw_manifest(self):
        """返回当前 raw 目录文件清单（名称、大小、修改时间），用于缓存失效判断。"""
        manifest = []
        for name in self.raw_file_names:
            path = os.path.join(self.raw_dir, name)
            try:
                stat = os.stat(path)
                manifest.append((name, stat.st_size, stat.st_mtime_ns))
            except OSError:
                manifest.append((name, -1, -1))
        return sorted(manifest)

    def _cache_up_to_date(self, loaded):
        """判断已保存的缓存是否仍有效。

        Args:
            loaded: 从 processed 文件加载的 dict

        Returns:
            raw 文件清单（名称/大小/时间）或数据管线源码指纹任一变化，
            都视为缓存过期；旧缓存缺少对应字段时返回 False。
        """
        return (
            loaded.get("raw_manifest") == self._raw_manifest()
            and loaded.get("pipeline_fingerprint") == self._pipeline_fingerprint()
        )

    def _pipeline_fingerprint(self):
        """对数据管线源码做哈希，管线代码变化时缓存自动失效。"""
        import hashlib

        digest = hashlib.md5()
        base = os.path.dirname(os.path.abspath(__file__))
        for name in _PIPELINE_FILES:
            with open(os.path.join(base, name), "rb") as f:
                digest.update(f.read())
        return digest.hexdigest()

    # -------------------------------------------------------
    # 图级特征反标准化
    # -------------------------------------------------------
    def inverse_transform(self, normalized_tensor):

        mean = torch.tensor(
            self.graph_feat_mean,
            dtype=torch.float,
            device=normalized_tensor.device
        )

        std = torch.tensor(
            self.graph_feat_std,
            dtype=torch.float,
            device=normalized_tensor.device
        )

        return normalized_tensor * std + mean

    # -------------------------------------------------------
    # 节点特征反标准化
    # -------------------------------------------------------
    def inverse_node_transform(self, normalized_tensor):

        mean = torch.tensor(
            self.node_feat_mean,
            dtype=torch.float,
            device=normalized_tensor.device
        )

        std = torch.tensor(
            self.node_feat_std,
            dtype=torch.float,
            device=normalized_tensor.device
        )

        return normalized_tensor * std + mean

    # -------------------------------------------------------
    # 主处理流程
    # -------------------------------------------------------
    def process(self):
        # ============================
        # 随机种子
        # ============================
        import random

        seed = 42
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        logger.info("[Dataset Seed] seed=%s", seed)

        data_list = []
        graph_features_list = []
        node_feature_list = []

        class_map = {
            '剪力墙': 0,
            '框剪': 1,
            '框架': 2,
            'WALL': 0,
            'FRAME_SHEAR': 1,
            'FRAME': 2
        }

        for filename in self.raw_file_names:

            label = None

            for key, val in class_map.items():
                if key in filename:
                    label = val
                    break

            if label is None:
                continue

            file_path = os.path.join(self.raw_dir, filename)

            try:
                nodes = process_dxf_file(file_path)
            except Exception as e:
                logger.warning("[DXF ERROR] %s : %s", filename, e)
                continue

            if len(nodes) == 0:
                continue

            node_features, aligned_pos = self.fe.build(nodes)

            if node_features is None or aligned_pos is None:
                continue

            node_feature_list.append(node_features)

            graph_data = self.gb.build(node_features, aligned_pos)

            if graph_data is None:
                continue

            # -------- 图级统计特征 --------
            graph_feat_raw = GraphStatistics(
                nodes,
                aligned_pos=aligned_pos
            ).build_feature_vector()

            graph_features_list.append(graph_feat_raw)

            data = Data(
                x_raw=graph_data.x.clone(),
                x=graph_data.x.clone(),
                edge_index=graph_data.edge_index,
                edge_attr=graph_data.edge_attr,
                y=torch.tensor([label], dtype=torch.long),
                graph_feature_raw=torch.tensor(graph_feat_raw, dtype=torch.float),
                pos=graph_data.pos
            )

            data.filename = filename

            data_list.append(data)

        if len(data_list) == 0:
            raise RuntimeError("No valid graph data generated.")

        # -------------------------------------------------------
        # ⭐ Node Feature 标准化
        # -------------------------------------------------------

        all_nodes = np.vstack(node_feature_list)

        node_mean = all_nodes.mean(axis=0)

        node_std = all_nodes.std(axis=0)

        node_std[node_std < 1e-8] = 1.0

        self.node_feat_mean = node_mean
        self.node_feat_std = node_std

        for data in data_list:

            x = data.x.numpy()

            x_norm = (x - node_mean) / node_std

            data.x = torch.tensor(
                x_norm,
                dtype=torch.float
            )

        logger.info("[Node Feature Normalization] Mean=%s Std=%s",
                    np.round(node_mean, 4), np.round(node_std, 4))

        # -------------------------------------------------------
        # ⭐ Graph Feature 标准化
        # -------------------------------------------------------

        all_features = np.vstack(graph_features_list)

        mean = all_features.mean(axis=0)

        std = all_features.std(axis=0)

        std[std < 1e-8] = 1.0

        self.graph_feat_mean = mean
        self.graph_feat_std = std

        for i, data in enumerate(data_list):

            normalized = (graph_features_list[i] - mean) / std

            data.graph_feature = torch.tensor(
                normalized,
                dtype=torch.float
            )

        logger.info("[Graph Feature Normalization] Mean=%s Std=%s",
                    np.round(mean, 4), np.round(std, 4))

        # -------------------------------------------------------

        data, slices = self.collate(data_list)

        torch.save(
            {
                "data": data,
                "slices": slices,

                "graph_mean": mean,
                "graph_std": std,

                "node_mean": node_mean,
                "node_std": node_std,

                "raw_manifest": self._raw_manifest(),
                "pipeline_fingerprint": self._pipeline_fingerprint(),
            },
            self.processed_paths[0]
        )
