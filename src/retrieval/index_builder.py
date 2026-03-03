# index_builder.py
import os
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from src.datasets.structure_dataset import StructureDataset
from src.models.struct_model import StructModel
from src.retrieval.embedder import Embedder
from src.datasets.statistics_features import extract_stats
from src.utils.config_loader import load_config


def build_index(cfg_path="configs/retrieval.yaml"):
    """
    构建 GNN + 统计特征索引
    """
    # -----------------------------
    # 1. 加载配置
    # -----------------------------
    cfg = load_config(cfg_path)
    device = cfg.device

    print(f"[IndexBuilder] 使用设备: {device}")
    print("[IndexBuilder] 正在加载数据集和模型...")

    # -----------------------------
    # 2. 加载数据集
    # -----------------------------
    dataset = StructureDataset(root=cfg.data_root)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # -----------------------------
    # 3. 初始化模型
    # -----------------------------
    model = StructModel(cfg).to(device)

    model_path = os.path.join(cfg.checkpoint_dir, cfg.model_name)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"未找到模型文件: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # -----------------------------
    # 4. 初始化 Embedder
    # -----------------------------
    embedder = Embedder(model, device=device)

    # -----------------------------
    # 5. 构建索引
    # -----------------------------
    db_embeddings = []
    db_filenames = []
    db_labels = []

    print("[IndexBuilder] 开始提取特征向量...")
    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(device)

            # ---- 5.1 GNN embedding ----
            graph_emb = embedder.get_embedding(data)  # shape: [1, hidden_dim]

            # ---- 5.2 统计特征 ----
            dxf_path = os.path.join(cfg.data_root, "raw_dxf", data.filename[0])
            stats_vec = extract_stats(dxf_path)  # shape: [num_stats_features]

            # ---- 5.3 拼接最终向量 ----
            final_vec = np.concatenate([graph_emb, stats_vec], axis=0)

            # ---- 5.4 保存
            db_embeddings.append(final_vec)
            db_filenames.append(data.filename[0])
            db_labels.append(data.y.item())

            if (i + 1) % 10 == 0:
                print(f"  已处理 {i+1}/{len(dataset)}")

    # -----------------------------
    # 6. 保存索引
    # -----------------------------
    save_dir = cfg.embedding_dir
    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "embeddings.npy"), np.array(db_embeddings))
    np.save(os.path.join(save_dir, "filenames.npy"), np.array(db_filenames))
    np.save(os.path.join(save_dir, "labels.npy"), np.array(db_labels))

    print(f"[IndexBuilder] 建库完成！索引已保存至: {save_dir}")
