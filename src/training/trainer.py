import os
import torch
from torch_geometric.loader import DataLoader

from src.datasets.structure_dataset import StructureDataset
from src.models.gnn_model import StructureGNN
from src.training.evaluator import Evaluator
import src.config as cfg


class Trainer:

    def __init__(self):
        self.device = cfg.DEVICE

        # 1. 准备目录
        if not os.path.exists(cfg.CHECKPOINT_DIR):
            os.makedirs(cfg.CHECKPOINT_DIR)

        # 2. 加载数据
        print(f"Loading dataset from {cfg.DATA_ROOT}...")
        dataset = StructureDataset(root=cfg.DATA_ROOT)

        dataset = dataset.shuffle()
        train_size = int(len(dataset) * 0.8)
        test_size = len(dataset) - train_size

        self.train_dataset = dataset[:train_size]
        self.test_dataset = dataset[train_size:]

        print(f"Train size: {train_size}, Test size: {test_size}")

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=True
        )

        self.test_loader = DataLoader(
            self.test_dataset,
            batch_size=cfg.BATCH_SIZE,
            shuffle=False
        )

        # 3. 初始化模型
        self.model = StructureGNN(
            num_node_features=dataset.num_node_features,
            hidden_channels=cfg.HIDDEN_CHANNELS,
            num_classes=cfg.NUM_CLASSES
        ).to(self.device)

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=cfg.LEARNING_RATE
        )

        self.criterion = torch.nn.CrossEntropyLoss()

        self.evaluator = Evaluator(self.model, self.device)

    def train(self):

        best_acc = 0.0

        for epoch in range(cfg.NUM_EPOCHS):

            self.model.train()
            total_loss = 0
            correct_train = 0
            total_train = 0

            for data in self.train_loader:
                data = data.to(self.device)
                self.optimizer.zero_grad()

                out, _ = self.model(data.x, data.edge_index, data.batch)

                loss = self.criterion(out, data.y)
                loss.backward()
                self.optimizer.step()

                total_loss += loss.item()

                pred = out.argmax(dim=1)
                correct_train += int((pred == data.y).sum())
                total_train += data.num_graphs

            avg_loss = total_loss / len(self.train_loader)
            train_acc = correct_train / total_train

            # 使用 evaluator
            test_acc = self.evaluator.evaluate(self.test_loader)

            print(
                f'Epoch {epoch+1:03d} | '
                f'Loss: {avg_loss:.4f} | '
                f'Train Acc: {train_acc:.4f} | '
                f'Test Acc: {test_acc:.4f}'
            )

            if test_acc >= best_acc:
                best_acc = test_acc
                torch.save(self.model.state_dict(), cfg.MODEL_SAVE_PATH)

        print(f"Training finished. Best Test Accuracy: {best_acc:.4f}")
        print(f"Model saved to {cfg.MODEL_SAVE_PATH}")
