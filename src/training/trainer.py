import os
import torch
from torch_geometric.loader import DataLoader

from src.datasets.structure_dataset import StructureDataset
from src.models.struct_model import StructModel
from src.training.evaluator import Evaluator
from src.utils.config_loader import load_config


class Trainer:
    def __init__(self, cfg_path="configs/retrieval.yaml"):
        # 1. 加载配置
        self.cfg = load_config(cfg_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Trainer] Using device: {self.device}")

        # 2. 创建 checkpoint 目录
        os.makedirs(self.cfg.training.checkpoint_dir, exist_ok=True)

        # 3. 加载数据集
        print(f"[Trainer] Loading dataset from {self.cfg.data.root}...")
        dataset = StructureDataset(root=self.cfg.data.root)

        # 简单划分训练/测试集
        dataset = dataset.shuffle()
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size

        self.train_dataset = dataset[:train_size]
        self.test_dataset = dataset[train_size:]

        print(f"[Trainer] Train size: {train_size}, Test size: {test_size}")

        # 4. DataLoader
        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=True
        )
        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=self.cfg.training.batch_size,
            shuffle=False
        )

        # 5. 初始化模型
        self.model = StructModel(self.cfg).to(self.device)

        # 6. 优化器和损失
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.cfg.training.learning_rate
        )
        self.criterion = torch.nn.CrossEntropyLoss()

        # 7. Evaluator（只计算准确率）
        self.evaluator = Evaluator(self.model, self.device)

    def train(self):
        best_acc = 0.0
        checkpoint_path = os.path.join(
            self.cfg.training.checkpoint_dir,
            self.cfg.training.model_name
        )

        for epoch in range(self.cfg.training.num_epochs):
            self.model.train()
            total_loss = 0
            correct_train = 0
            total_train = 0

            for data in self.train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()

                # -----------------------------
                # 前向传播：返回 logits & embedding
                # -----------------------------
                logits, embedding = self.model.encode(data)
                loss = self.criterion(logits, data.y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()
                pred = logits.argmax(dim=1)
                correct_train += int((pred == data.y).sum())
                total_train += data.num_graphs

            avg_loss = total_loss / len(self.train_loader)
            train_acc = correct_train / total_train

            # 使用 Evaluator 测试
            test_acc = self.evaluator.evaluate(self.test_loader)

            print(
                f"[Epoch {epoch+1:03d}] "
                f"Loss: {avg_loss:.4f} | "
                f"Train Acc: {train_acc:.4f} | "
                f"Test Acc: {test_acc:.4f}"
            )

            # 保存最佳模型
            if test_acc >= best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), checkpoint_path)
                print(f"  -> Model saved at epoch {epoch+1} with Test Acc: {best_acc:.4f}")

        print(f"[Trainer] Training finished. Best Test Accuracy: {best_acc:.4f}")
        print(f"[Trainer] Model saved to {checkpoint_path}")
