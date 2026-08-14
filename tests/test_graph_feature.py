"""图级/节点特征标准化与反标准化测试。"""

import numpy as np


def test_inverse_transform_recovers_raw_graph_feature(dataset):
    data = dataset[0]
    recovered = dataset.inverse_transform(data.graph_feature)
    np.testing.assert_allclose(
        recovered.numpy(),
        data.graph_feature_raw.numpy(),
        atol=1e-5,
    )


def test_inverse_node_transform_recovers_raw_node_feature(dataset):
    data = dataset[0]
    recovered = dataset.inverse_node_transform(data.x)
    # 节点特征为 float32，且 r 维存在极端离群值（std 可达 1e8），
    # 反标准化误差按“float32 精度 × 最大特征标准差”给出容差。
    max_std = float(np.asarray(dataset.node_feat_std).max())
    np.testing.assert_allclose(
        recovered.numpy(),
        data.x_raw.numpy(),
        atol=2e-6 * max_std + 1e-5,
        rtol=0,
    )


def test_type_ratio_sums_to_one(dataset):
    data = dataset[0]
    raw = dataset.inverse_transform(data.graph_feature).numpy()
    np.testing.assert_allclose(raw[0] + raw[1] + raw[2], 1.0, atol=1e-5)


def test_r_feature_log1p_transform():
    from datasets.feature_engineering import FeatureEngineering

    nodes = [{
        "centroid": (0.0, 0.0),
        "area": 1000.0,
        "Lx": 10.0,
        "Ly": 5.0,
        "long_leg_length": 10.0,
        "long_leg_thickness": 5.0,
        "r": 1e8,
        "type": 2,
    }]
    features, _ = FeatureEngineering().build(nodes)
    # 第 8 维应为 log1p(r)，而不是原始 r
    np.testing.assert_allclose(features[0, 8], float(np.log1p(1e8)), rtol=1e-6)
