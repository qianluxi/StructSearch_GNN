import os
import sys

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(ROOT_DIR, "src"))

import torch
from torch_geometric.loader import DataLoader
from datasets.structure_dataset import StructureDataset
from models.model import GraphEncoder
from training.split_dataset import split_dataset


def train():
    # -------------------------
    # 加载完整数据集
    # -------------------------
    dataset = StructureDataset(root="data")
    print(f"Dataset size: {len(dataset)}")

    # -------------------------
    # 切分数据集
    # -------------------------
    train_idx, val_idx = split_dataset(len(dataset))
    train_dataset = [dataset[i] for i in train_idx]
    val_dataset = [dataset[i] for i in val_idx]

    print(f"Train graphs: {len(train_dataset)}")
    print(f"Val graphs: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)

    # -------------------------
    # 模型定义
    # -------------------------
    num_classes = 3  # 剪力墙/框剪/框架
    model = GraphEncoder(num_classes=num_classes)  # GraphEncoder 带分类头
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = torch.nn.CrossEntropyLoss()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # -------------------------
    # 训练循环
    # -------------------------
    for epoch in range(50):
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

        print(f"Epoch {epoch} loss {total_loss:.6f}")

    # -------------------------
    # 保存模型到 checkpoints 文件夹
    # -------------------------
    # 1. 定义保存路径
    save_dir = os.path.join(ROOT_DIR, "checkpoints")  # 项目根目录下的 checkpoints
    # 2. 如果文件夹不存在，自动创建
    os.makedirs(save_dir, exist_ok=True)
    # 3. 完整路径
    save_path = os.path.join(save_dir, "model.pt")

    # 保存
    torch.save(
        {
            'state_dict': model.state_dict(),
            'emb_dim': model.lin.out_features,
            'num_classes': num_classes
        },
        save_path  # 这里改了
    )
    print("Model saved as 'model.pt'")


if __name__ == "__main__":
    train()
