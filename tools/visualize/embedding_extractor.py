import os
import sys
import torch
import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(os.path.join(ROOT_DIR, "src"))

from torch_geometric.loader import DataLoader
from datasets.structure_dataset import StructureDataset
from models.model import GraphEncoder


def extract_embeddings():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = StructureDataset(root="data")
    loader = DataLoader(dataset, batch_size=8)

    checkpoint = torch.load("model.pt")

    model = GraphEncoder(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])

    model.to(device)
    model.eval()

    embeddings = []
    labels = []
    preds = []

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

    print("Embedding shape:", embeddings.shape)

    return embeddings, labels, preds
