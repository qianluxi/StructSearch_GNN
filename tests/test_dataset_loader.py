import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(ROOT, "src"))

from datasets.structure_dataset import StructureDataset

dataset = StructureDataset(root="data")

print("Dataset size:", len(dataset))

data = dataset[0]

print("Nodes:", data.x.shape)
print("Edges:", data.edge_index.shape)
print("Edge attr:", data.edge_attr.shape)
print("Graph feature:", data.graph_feature.shape)
print("Position:", data.pos.shape)
