import os
import torch
from torch_geometric.loader import DataLoader
from src.dataset import StructureDataset
from src.model import StructureGNN
import src.config as cfg

def train():
    # 1. 准备目录
    if not os.path.exists(cfg.CHECKPOINT_DIR):
        os.makedirs(cfg.CHECKPOINT_DIR)

    # 2. 加载数据
    print(f"Loading dataset from {cfg.DATA_ROOT}...")
    dataset = StructureDataset(root=cfg.DATA_ROOT)
    
    # 简单的随机划分：80% 训练，20% 测试
    dataset = dataset.shuffle()
    train_size = int(len(dataset) * 0.8)
    test_size = len(dataset) - train_size
    train_dataset = dataset[:train_size]
    test_dataset = dataset[train_size:]

    print(f"Train size: {train_size}, Test size: {test_size}")

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True)
    # test_loader batch_size 可以设大一点，不影响梯度
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False)

    # 3. 初始化模型
    model = StructureGNN(
        num_node_features=dataset.num_node_features, # 应该是 4
        hidden_channels=cfg.HIDDEN_CHANNELS,
        num_classes=cfg.NUM_CLASSES
    ).to(cfg.DEVICE)

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.LEARNING_RATE)
    criterion = torch.nn.CrossEntropyLoss() # 多分类损失函数

    # 4. 训练循环
    best_acc = 0.0
    
    for epoch in range(cfg.NUM_EPOCHS):
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0
        
        for data in train_loader:
            data = data.to(cfg.DEVICE)
            optimizer.zero_grad()
            
            # Forward pass
            out, _ = model(data.x, data.edge_index, data.batch) # 忽略 embedding
            
            # Compute loss
            loss = criterion(out, data.y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            # 计算训练集准确率
            pred = out.argmax(dim=1)
            correct_train += int((pred == data.y).sum())
            total_train += data.num_graphs

        avg_loss = total_loss / len(train_loader)
        train_acc = correct_train / total_train

        # --- 验证/测试 ---
        model.eval()
        correct_test = 0
        total_test = 0
        
        with torch.no_grad():
            for data in test_loader:
                data = data.to(cfg.DEVICE)
                out, _ = model(data.x, data.edge_index, data.batch)
                pred = out.argmax(dim=1)
                correct_test += int((pred == data.y).sum())
                total_test += data.num_graphs
        
        test_acc = correct_test / total_test if total_test > 0 else 0.0

        print(f'Epoch {epoch+1:03d} | Loss: {avg_loss:.4f} | Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}')

        # 保存最佳模型
        if test_acc >= best_acc:
            best_acc = test_acc
            torch.save(model.state_dict(), cfg.MODEL_SAVE_PATH)
            # print(f"  -> Model saved with accuracy: {best_acc:.4f}")

    print(f"Training finished. Best Test Accuracy: {best_acc:.4f}")
    print(f"Model saved to {cfg.MODEL_SAVE_PATH}")

if __name__ == '__main__':
    train()