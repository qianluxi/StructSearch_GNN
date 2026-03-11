import os
import sys
import torch

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))

from datasets.structure_dataset import StructureDataset
from models.model import GraphEncoder

dataset = StructureDataset(root="data")

data = dataset[0]

model = GraphEncoder(in_dim=data.x.shape[1])

batch = torch.zeros(data.x.shape[0], dtype=torch.long)

emb = model(data.x, data.edge_index, batch)

print("Embedding shape:", emb.shape)
print("Embedding:", emb)
