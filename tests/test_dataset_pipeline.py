"""数据集管线一致性测试。"""


def test_pipeline_consistency(dataset):
    for i in range(len(dataset)):
        data = dataset[i]
        n = data.x.shape[0]
        assert data.pos.shape[0] == n
        assert data.graph_feature.shape == (11,)
        assert data.y.item() in (0, 1, 2)
        assert data.filename


def test_filenames_unique(dataset):
    names = [dataset[i].filename for i in range(len(dataset))]
    assert len(names) == len(set(names))
