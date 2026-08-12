import os
import sys
import torch
import numpy as np
from torch_geometric.loader import DataLoader

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, "src"))

from datasets.structure_dataset import StructureDataset
from models.model import GraphEncoder


def extract_embeddings(batch_size=8):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = StructureDataset(root=os.path.join(ROOT_DIR, "data"))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    checkpoint_path = os.path.join(ROOT_DIR, "checkpoints", "model.pt")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = GraphEncoder(num_classes=checkpoint["num_classes"])
    model.load_state_dict(checkpoint["state_dict"])

    model.to(device)
    model.eval()

    embeddings = []
    labels = []

    with torch.no_grad():

        for batch in loader:

            batch = batch.to(device)

            emb, logits = model(batch.x, batch.edge_index, batch.batch)

            embeddings.append(emb.cpu().numpy())
            labels.append(batch.y.cpu().numpy())

    embeddings = np.concatenate(embeddings)
    labels = np.concatenate(labels)

    print("Embedding shape:", embeddings.shape)

    return embeddings, labels
