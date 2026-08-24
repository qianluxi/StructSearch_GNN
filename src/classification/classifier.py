"""Stage 1：结构类型识别——只用 11 维统计特征，不依赖 GNN。

实现方式：以库内所有样本的 11D 标准化统计向量为参照，做余弦最近邻
（检索即分类）。置信度取 Top-3 邻居相似度的 softmax 中预测类的概率。
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

from datasets.structure_dataset import StructureDataset
from retrieval.config import CLASS_NAMES, load_retrieval_config
from retrieval.processor import process_single_dxf, standardize_data


def _cosine_similarity(query_vec, db_vectors):
    """计算 query 与库中所有向量的余弦相似度（纯 numpy）。"""
    db_norm = np.linalg.norm(db_vectors, axis=1)
    query_norm = np.linalg.norm(query_vec)
    denom = db_norm * query_norm
    denom[denom == 0] = 1e-12
    return (db_vectors @ query_vec) / denom


class StructureTypeClassifier:
    """基于 11 维图统计特征的最近邻结构类型分类器（无学习、无 GNN）。"""

    STATS_START = 128  # 索引前 128 维为 GNN 嵌入，后 11 维为统计特征

    def __init__(self, cfg_path=None):
        self.cfg = load_retrieval_config(cfg_path)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # 数据集：提供标准化统计量（供 query 标准化使用）
        self.dataset = StructureDataset(
            root=self.cfg.data_root,
            k=self.cfg.graph.k,
            connect_mode=self.cfg.graph.connect_mode,
        )

        # 库内标准化 11 维统计特征、标签、文件名
        embeddings = np.load(
            os.path.join(self.cfg.embedding_dir, "embeddings.npy"),
            allow_pickle=True,
        ).astype(np.float64)
        self.db_stats = embeddings[:, self.STATS_START:]
        self.db_labels = np.load(
            os.path.join(self.cfg.embedding_dir, "labels.npy"),
            allow_pickle=True,
        ).astype(np.int64)
        self.db_filenames = np.asarray(
            np.load(
                os.path.join(self.cfg.embedding_dir, "filenames.npy"),
                allow_pickle=True,
            )
        )

    # -------------------------------------------------------
    def predict(self, data):
        """对已标准化的 Data 对象做结构类型识别。

        Args:
            data: process_single_dxf + standardize_data 后的 Data

        Returns:
            dict: {type, type_id, confidence, score, stats, data}
        """
        if data is None:
            return None

        stats_vec = data.graph_feature.cpu().numpy().ravel().astype(np.float64)
        sims = _cosine_similarity(stats_vec, self.db_stats)

        # 查询本身在库中时排除自身
        query_name = getattr(data, "filename", None)
        if query_name is not None:
            sims[self.db_filenames == query_name] = -1.0

        top3 = np.argsort(sims)[::-1][:3]
        top_sim = sims[top3]

        # softmax(Top-3 相似度) -> 概率
        exp_sim = np.exp(top_sim - top_sim.max())
        probs = exp_sim / exp_sim.sum()

        best = top3[0]
        type_id = int(self.db_labels[best])
        confidence = float(probs[self.db_labels[top3] == type_id].sum())

        return {
            "type": CLASS_NAMES.get(type_id, "未知"),
            "type_id": type_id,
            "confidence": confidence,
            "score": float(top_sim[0]),
            "stats": stats_vec,
            "data": data,
        }

    # -------------------------------------------------------
    def predict_dxf(self, filepath, k=None, connect_mode=None):
        """直接对 DXF 文件做结构类型识别。

        Returns:
            与 predict 相同的 dict；解析失败时返回 None。
        """
        data = process_single_dxf(
            filepath,
            k=k or self.cfg.graph.k,
            connect_mode=connect_mode or self.cfg.graph.connect_mode,
        )
        if data is None:
            return None

        data = standardize_data(
            data,
            self.dataset.node_feat_mean,
            self.dataset.node_feat_std,
            self.dataset.graph_feat_mean,
            self.dataset.graph_feat_std,
        ).to(self.device)
        return self.predict(data)
