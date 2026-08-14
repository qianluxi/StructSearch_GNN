"""StructureDataset 加载与字段维度测试。"""


def test_dataset_nonempty(dataset):
    assert len(dataset) > 0


def test_data_fields_shapes(dataset):
    data = dataset[0]

    n = data.x.shape[0]
    assert data.x.shape == (n, 9)
    assert data.pos.shape == (n, 2)
    assert data.edge_index.shape[0] == 2
    assert data.edge_attr.shape == (data.edge_index.shape[1], 1)
    assert data.graph_feature.shape == (11,)
    assert data.graph_feature_raw.shape == (11,)
    assert data.y.numel() == 1
    assert data.y.item() in (0, 1, 2)
    assert data.filename.endswith(".dxf")


def test_every_sample_valid(dataset):
    for i in range(len(dataset)):
        data = dataset[i]
        assert data.x.shape[0] == data.pos.shape[0] > 0
        assert data.y.item() in (0, 1, 2)
