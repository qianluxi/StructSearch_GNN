import os
import sys
import torch
import numpy as np
import json

# ===================== 路径修正 =====================
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))

from torch_geometric.loader import DataLoader
from datasets.structure_dataset import StructureDataset
from models.model import GraphEncoder

# 新增：embedding评价
from analysis.embedding_metrics import evaluate_embedding, print_metrics


def extract_embeddings():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ===================== 数据集 =====================
    dataset = StructureDataset(root=os.path.join(ROOT_DIR, "data"))
    loader = DataLoader(dataset, batch_size=8, shuffle=False)

    # ===================== 模型 =====================
    checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", "model.pt")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = GraphEncoder(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])

    model.to(device)
    model.eval()

    embeddings = []
    labels = []
    preds = []

    # ===================== 提取embedding =====================
    with torch.no_grad():
        for batch in loader:

            batch = batch.to(device)

            emb, logits = model(batch.x, batch.edge_index, batch.batch)

            pred = torch.argmax(logits, dim=1)

            embeddings.append(emb.cpu().numpy())
            labels.append(batch.y.cpu().numpy())
            preds.append(pred.cpu().numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)
    preds = np.concatenate(preds)

    print("\n✅ Embedding extraction done!")
    print("Embedding shape:", embeddings.shape)

    return embeddings, labels, preds


def evaluate_and_save(embeddings, labels):

    metrics = evaluate_embedding(embeddings, labels)

    print_metrics(metrics)

    # ===================== 保存metrics =====================

    save_dir = os.path.join(ROOT_DIR, "fig", "metrics")
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, "embedding_metrics.json")

    with open(save_path, "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"\n📊 Metrics saved to {save_path}")


if __name__ == "__main__":

    embeddings, labels, preds = extract_embeddings()

    evaluate_and_save(embeddings, labels)
