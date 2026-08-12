"""相似案例检索：分类 + Top-K 检索。"""

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
from models.model import GraphEncoder
from retrieval.config import CLASS_NAMES, load_retrieval_config
from retrieval.embedder import Embedder, build_index_vector
from retrieval.processor import process_single_dxf, standardize_data


def _cosine_similarity(query_vec, db_vectors):
    """计算 query 与数据库中所有向量的余弦相似度（纯 numpy 实现）。

    Args:
        query_vec: (D,) numpy 数组
        db_vectors: (N, D) numpy 数组

    Returns:
        (N,) 相似度数组
    """
    db_norm = np.linalg.norm(db_vectors, axis=1)
    query_norm = np.linalg.norm(query_vec)

    denom = db_norm * query_norm
    denom[denom == 0] = 1e-12

    return (db_vectors @ query_vec) / denom


class SearchEngine:
    """基于已构建索引的分类 + 相似案例检索入口。"""

    def __init__(self, cfg_path=None):
        self.cfg = load_retrieval_config(cfg_path)
        self.device = (
            self.cfg.device if torch.cuda.is_available() else "cpu"
        )

        # 数据集：提供标准化统计量（供 query 标准化使用）
        self.dataset = StructureDataset(
            root=self.cfg.data_root,
            k=self.cfg.graph.k,
            connect_mode=self.cfg.graph.connect_mode
        )

        # 模型
        checkpoint_path = os.path.join(
            self.cfg.checkpoint_dir,
            self.cfg.model_name
        )
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"未找到模型文件: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.model = GraphEncoder(
            node_dim=self.cfg.model.num_node_features,
            hidden_dim=self.cfg.model.hidden_dim,
            emb_dim=self.cfg.embedding.dim,
            num_classes=self.cfg.model.num_classes
        )
        self.model.load_state_dict(checkpoint["state_dict"])

        self.embedder = Embedder(self.model, device=self.device)

        # 索引
        self.db_embeddings, self.db_filenames, self.db_labels = (
            self._load_index()
        )

    # -------------------------------------------------------
    def _load_index(self):
        """从 embedding_dir 加载索引文件。"""

        def _path(name):
            return os.path.join(self.cfg.embedding_dir, name)

        for name in ("embeddings.npy", "filenames.npy", "labels.npy"):
            if not os.path.exists(_path(name)):
                raise FileNotFoundError(
                    f"索引文件缺失: {_path(name)}，请先运行 "
                    "python -m src.retrieval.cli --build"
                )

        embeddings = np.load(_path("embeddings.npy"))
        filenames = np.load(_path("filenames.npy"), allow_pickle=True)
        labels = np.load(_path("labels.npy"), allow_pickle=True)

        return embeddings, filenames, labels

    # -------------------------------------------------------
    @torch.no_grad()
    def search(self, dxf_path, topk=None):
        """对单个 DXF 文件进行结构体系分类并检索最相似的历史案例。

        Args:
            dxf_path: DXF 文件路径
            topk: 返回的相似案例数，默认取配置中的 retrieval.topk

        Returns:
            dict：query / pred_class / probability / results / data
            （解析失败时返回 None）
        """
        topk = topk or getattr(self.cfg.retrieval, "topk", 5)
        query_name = os.path.basename(dxf_path)

        print(f"\n正在分析文件: {dxf_path}")

        # ---------- DXF -> graph ----------
        data = process_single_dxf(
            dxf_path,
            k=self.cfg.graph.k,
            connect_mode=self.cfg.graph.connect_mode
        )
        if data is None:
            return None

        data = standardize_data(
            data,
            self.dataset.node_feat_mean,
            self.dataset.node_feat_std,
            self.dataset.graph_feat_mean,
            self.dataset.graph_feat_std
        ).to(self.device)

        # ---------- 分类 ----------
        batch = torch.zeros(
            data.num_nodes,
            dtype=torch.long,
            device=self.device
        )
        embedding, logits = self.model(data.x, data.edge_index, batch)

        pred_idx = logits.argmax(dim=1).item()
        pred_class = CLASS_NAMES.get(pred_idx, "未知")
        prob = torch.softmax(logits, dim=1)[0, pred_idx].item()

        print("\n" + "=" * 40)
        print(f"【分类结果】: {pred_class} (置信度: {prob:.2%})")
        print("=" * 40)

        # ---------- 检索 ----------
        query_vec = build_index_vector(
            embedding.squeeze(0).cpu().numpy(),
            data.graph_feature.cpu().numpy()
        )

        sims = _cosine_similarity(query_vec, self.db_embeddings)
        sorted_indices = np.argsort(sims)[::-1]

        print(f"\n【最相似的历史案例 Top {topk}】")

        results = []
        for idx in sorted_indices:
            if len(results) >= topk:
                break
            if self.db_filenames[idx] == query_name:
                continue

            item = {
                "rank": len(results) + 1,
                "filename": self.db_filenames[idx],
                "score": float(sims[idx]),
                "label": CLASS_NAMES.get(
                    int(self.db_labels[idx]),
                    "未知"
                ),
            }
            results.append(item)

            print(f"{item['rank']}. {item['filename']}")
            print(f"   - 相似度: {item['score']:.4f}")
            print(f"   - 类型: {item['label']}")
            print("-" * 30)

        return {
            "query": query_name,
            "pred_class": pred_class,
            "probability": prob,
            "results": results,
            # 已解析的查询图数据（含 x_raw/pos），供结果图直接使用，
            # 避免绘图时再次解析 DXF 并重复打印解析日志。
            "data": data,
        }
