import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(ROOT_DIR, "src"))

import torch
from torch_geometric.loader import DataLoader

from datasets.structure_dataset import StructureDataset
from models.model import GraphEncoder
from retrieval.config import load_retrieval_config
from training.split_dataset import split_dataset
from utils.logging_setup import setup_logging


def train(cfg_path=None):
    """训练 GraphEncoder：超参与路径全部来自 configs/retrieval.yaml。

    Args:
        cfg_path: 配置文件路径；默认 configs/retrieval.yaml
    """
    setup_logging()

    cfg = load_retrieval_config(cfg_path)

    # -------------------------
    # 加载完整数据集
    # -------------------------
    dataset = StructureDataset(
        root=cfg.data_root,
        k=cfg.graph.k,
        connect_mode=cfg.graph.connect_mode,
    )
    print(f"Dataset size: {len(dataset)}")

    # -------------------------
    # 切分数据集
    # -------------------------
    train_idx, val_idx = split_dataset(
        len(dataset),
        train_ratio=1.0 - cfg.training.val_ratio,
    )
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]

    print(f"Train graphs: {len(train_dataset)}")
    print(f"Val graphs: {len(val_dataset)}")

    batch_size = cfg.training.batch_size
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)

    # -------------------------
    # 模型定义（参数来自配置）
    # -------------------------
    num_classes = cfg.model.num_classes  # 0=剪力墙 1=框剪 2=框架
    model = GraphEncoder(
        node_dim=cfg.model.num_node_features,
        hidden_dim=cfg.model.hidden_dim,
        emb_dim=cfg.embedding.dim,
        num_classes=num_classes,
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.learning_rate,
    )
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -------------------------
    # 训练循环
    # -------------------------
    for epoch in range(cfg.training.num_epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            batch = batch.to(device)

            # forward
            emb, logits = model(batch.x, batch.edge_index, batch.batch)

            # 分类 loss
            loss = criterion(logits, batch.y.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        # -------------------------
        # 打印调试信息
        # -------------------------
        if epoch == 0:
            print("\nNode feature sample (normalized):")
            print(batch.x[:5])

            if hasattr(batch, 'x_raw'):
                print("\nNode feature sample (raw):")
                print(batch.x_raw[:5])

            print("\nGraph embedding sample:")
            print(emb[:5].detach().cpu())

            print("\nLogits sample:")
            print(logits[:5].detach().cpu())

        # -------------------------
        # 验证集准确率
        # -------------------------
        val_correct = 0
        val_total = 0
        model.eval()
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, logits = model(batch.x, batch.edge_index, batch.batch)
                pred = logits.argmax(dim=1)
                val_correct += (pred == batch.y.view(-1)).sum().item()
                val_total += batch.y.numel()
        model.train()

        val_acc = val_correct / val_total if val_total > 0 else 0.0
        print(f"Epoch {epoch} loss {total_loss:.6f} val_acc {val_acc:.4f}")

    # -------------------------
    # 保存模型（路径来自配置，已解析为绝对路径）
    # -------------------------
    save_dir = cfg.checkpoint_dir
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, cfg.training.model_name)

    torch.save(
        {
            'state_dict': model.state_dict(),
            'emb_dim': model.lin.out_features,
            'num_classes': num_classes
        },
        save_path
    )
    print(f"Model saved to '{save_path}'")


if __name__ == "__main__":
    train()
