import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))

from torch_geometric.loader import DataLoader
from datasets.structure_dataset import StructureDataset

dataset = StructureDataset(root="data")

loader = DataLoader(dataset, batch_size=4)

for batch in loader:

    print("Batch nodes:", batch.x.shape)
    print("Batch edges:", batch.edge_index.shape)
    print("Batch graphs:", batch.batch.max().item()+1)

    break
