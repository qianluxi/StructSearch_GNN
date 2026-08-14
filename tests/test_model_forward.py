"""GraphEncoder 前向契约测试：返回 (embedding, logits)。"""

import torch

from models.model import GraphEncoder


def test_single_graph_forward_shapes(dataset):
    data = dataset[0]
    model = GraphEncoder(node_dim=data.x.shape[1])

    batch = torch.zeros(data.x.shape[0], dtype=torch.long)
    emb, logits = model(data.x, data.edge_index, batch)

    assert emb.shape == (1, 128)
    assert logits.shape == (1, 3)


def test_batched_forward_shapes(dataset):
    from torch_geometric.loader import DataLoader

    loader = DataLoader([dataset[0], dataset[1]], batch_size=2)
    batch = next(iter(loader))

    model = GraphEncoder(node_dim=batch.x.shape[1])
    emb, logits = model(batch.x, batch.edge_index, batch.batch)

    assert emb.shape == (2, 128)
    assert logits.shape == (2, 3)
