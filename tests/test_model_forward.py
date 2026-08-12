import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))

from datasets.structure_dataset import StructureDataset
from models.model import GraphEncoder


def main():
    dataset = StructureDataset(root="data")

    data = dataset[0]

    model = GraphEncoder(node_dim=data.x.shape[1])

    batch = torch.zeros(data.x.shape[0], dtype=torch.long)

    emb, logits = model(data.x, data.edge_index, batch)

    print("Embedding shape:", tuple(emb.shape))
    print("Logits shape:", tuple(logits.shape))

    assert emb.shape == (1, 128)
    assert logits.shape == (1, 3)
    print("Model forward OK")


if __name__ == "__main__":
    main()
