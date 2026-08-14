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


def test_cache_manifest_matches_raw(dataset):
    """processed 缓存应记录 raw 清单与管线指纹，且与当前状态一致。"""
    import torch

    loaded = torch.load(dataset.processed_paths[0])
    assert "raw_manifest" in loaded
    assert loaded["raw_manifest"] == dataset._raw_manifest()
    assert "pipeline_fingerprint" in loaded
    assert loaded["pipeline_fingerprint"] == dataset._pipeline_fingerprint()
