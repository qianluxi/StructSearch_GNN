import os
import torch
import numpy as np
from torch_geometric.loader import DataLoader

from src.datasets.structure_dataset import StructureDataset
from src.models.gnn_model import StructureGNN
import src.config as cfg


def build_index():
    print("正在加载数据集和模型...")

    # 1. 加载数据集
    dataset = StructureDataset(root=cfg.DATA_ROOT)

    # 不 shuffle，保持文件顺序
    loader = DataLoader(dataset, batch_size=1, shuffle=False)

    # 2. 初始化模型
    model = StructureGNN(
        num_node_features=dataset.num_node_features,
        hidden_channels=cfg.HIDDEN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ).to(cfg.DEVICE)

    # 3. 加载权重
    model_path = os.path.join(cfg.CHECKPOINT_DIR, cfg.MODEL_NAME)

    if not os.path.exists(model_path):
        print("错误: 未找到模型文件，请先完成训练。")
        return

    model.load_state_dict(torch.load(model_path, map_location=cfg.DEVICE))
    model.eval()

    database_embeddings = []
    database_filenames = []
    database_labels = []

    print("开始提取特征向量...")

    with torch.no_grad():
        for i, data in enumerate(loader):
            data = data.to(cfg.DEVICE)

            # forward
            _, embedding = model(data.x, data.edge_index, data.batch)

            vec = embedding.cpu().numpy()[0]

            filename = data.filename[0]
            label = data.y.item()

            database_embeddings.append(vec)
            database_filenames.append(filename)
            database_labels.append(label)

            if (i + 1) % 10 == 0:
                print(f"  已处理 {i+1}/{len(dataset)}")

    # 4. 保存到 experiments/embeddings
    save_dir = cfg.EMBEDDING_DIR

    os.makedirs(save_dir, exist_ok=True)

    np.save(os.path.join(save_dir, "embeddings.npy"),
            np.array(database_embeddings))

    np.save(os.path.join(save_dir, "filenames.npy"),
            np.array(database_filenames))

    np.save(os.path.join(save_dir, "labels.npy"),
            np.array(database_labels))

    print(f"\n建库完成！索引已保存至: {save_dir}")
