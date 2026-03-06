import torch
import numpy as np

from torch_geometric.loader import DataLoader

from datasets.structure_dataset import StructureDataset
from models.model import GraphEncoder


def extract_embeddings(dxf_list):

    dataset = []

    for f in dxf_list:

        ds = StructureDataset(f)

        dataset.append(ds.data)

    loader = DataLoader(dataset, batch_size=8)

    model = GraphEncoder()

    model.load_state_dict(torch.load("model.pt"))

    model.eval()

    embeddings = []

    with torch.no_grad():

        for batch in loader:

            emb = model(batch.x, batch.edge_index, batch.batch)

            embeddings.append(emb.numpy())

    embeddings = np.vstack(embeddings)

    return embeddings
