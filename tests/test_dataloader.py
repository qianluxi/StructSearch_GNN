"""DataLoader 批处理正确性测试。"""

from torch_geometric.loader import DataLoader


def test_dataloader_batch_shapes(dataset):
    loader = DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))

    assert batch.num_graphs <= 4
    assert batch.x.ndim == 2
    assert batch.x.shape[1] == 9
    assert batch.edge_index.shape[0] == 2
    assert batch.batch.max().item() + 1 == batch.num_graphs


def test_dataloader_batch_attrs_present(dataset):
    loader = DataLoader(dataset, batch_size=4)
    batch = next(iter(loader))

    assert batch.edge_attr.shape == (batch.edge_index.shape[1], 1)
    assert batch.pos.shape[0] == batch.x.shape[0]
    # 图级特征按行拼接，批内为 (num_graphs * 11,)
    assert batch.graph_feature.numel() == batch.num_graphs * 11
