"""构建检索索引：GNN 嵌入 + 图级统计特征 -> embeddings.npy 等。"""

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
from retrieval.config import load_retrieval_config
from retrieval.embedder import Embedder, build_index_vector


def build_index(cfg_path=None):
    """构建特征索引并保存到配置的 embedding_dir。

    Args:
        cfg_path: 检索配置文件路径，默认 configs/retrieval.yaml

    Returns:
        None；在 embedding_dir 下生成
        embeddings.npy / filenames.npy / labels.npy。
    """

    cfg = load_retrieval_config(cfg_path)
    device = cfg.device if torch.cuda.is_available() else "cpu"

    print(f"[IndexBuilder] 使用设备: {device}")

    # -----------------------------
    # 1. 数据集（加载已缓存的标准图数据）
    # -----------------------------
    dataset = StructureDataset(
        root=cfg.data_root,
        k=cfg.graph.k,
        connect_mode=cfg.graph.connect_mode
    )
    print(f"[IndexBuilder] 数据集样本数: {len(dataset)}")

    # -----------------------------
    # 2. 模型
    # -----------------------------
    model = GraphEncoder(
        node_dim=cfg.model.num_node_features,
        hidden_dim=cfg.model.hidden_dim,
        emb_dim=cfg.embedding.dim,
        num_classes=cfg.model.num_classes
    )

    checkpoint_path = os.path.join(cfg.checkpoint_dir, cfg.model_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"未找到模型文件: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    model.eval()

    embedder = Embedder(model, device=device)

    # -----------------------------
    # 3. 提取特征向量
    # -----------------------------
    db_embeddings = []
    db_filenames = []
    db_labels = []

    print("[IndexBuilder] 开始提取特征向量...")

    for i in range(len(dataset)):
        data = dataset[i]

        # extract_one 会把 data 移到 device 上（原地修改），
        # 因此先取 CPU 上的图级统计特征。
        graph_feature = data.graph_feature.cpu().numpy()

        graph_emb = embedder.extract_one(data)
        final_vec = build_index_vector(
            graph_emb,
            graph_feature
        )

        db_embeddings.append(final_vec)
        db_filenames.append(data.filename)
        db_labels.append(data.y.item())

        if (i + 1) % 10 == 0 or (i + 1) == len(dataset):
            print(f"  已处理 {i + 1}/{len(dataset)}")

    # -----------------------------
    # 4. 保存索引
    # -----------------------------
    save_dir = cfg.embedding_dir
    os.makedirs(save_dir, exist_ok=True)

    # round 到 6 位小数：消除 CUDA 浮点抖动，保证索引字节级可复现
    embeddings_arr = np.round(
        np.asarray(db_embeddings, dtype=np.float32),
        6
    )

    np.save(
        os.path.join(save_dir, "embeddings.npy"),
        embeddings_arr
    )
    np.save(
        os.path.join(save_dir, "filenames.npy"),
        np.asarray(db_filenames)
    )
    np.save(
        os.path.join(save_dir, "labels.npy"),
        np.asarray(db_labels, dtype=np.int64)
    )

    print(f"[IndexBuilder] 建库完成！索引已保存至: {save_dir}")
