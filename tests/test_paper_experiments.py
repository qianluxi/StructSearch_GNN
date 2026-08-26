"""论文实验模块的基础正确性测试。"""

import numpy as np

from benchmark.baselines import _rank_pool
from benchmark.feature_ablation import FEATURE_GROUPS


def test_feature_groups_cover_all_dims():
    cols = sorted(c for cols in FEATURE_GROUPS.values() for c in cols)
    assert cols == list(range(11))


def test_rank_pool_distance_orders_ascending():
    q = np.array([0.0, 0.0])
    pool = np.array([[1.0, 0.0], [3.0, 0.0], [0.0, 0.5]])
    rng = np.random.default_rng(0)
    idx = _rank_pool("statistics_euclidean", q, pool, rng)
    assert list(idx) == [2, 0, 1]  # 距离 0.5 < 1.0 < 3.0


def test_rank_pool_cosine_orders_descending():
    q = np.array([1.0, 0.0])
    pool = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    rng = np.random.default_rng(0)
    idx = _rank_pool("statistics_cosine", q, pool, rng)
    assert list(idx) == [0, 2, 1]


def test_rank_pool_random_deterministic():
    q = np.zeros(2)
    pool = np.random.default_rng(1).normal(size=(5, 2))
    rng1 = np.random.default_rng(42)
    rng2 = np.random.default_rng(42)
    assert np.array_equal(
        _rank_pool("random", q, pool, rng1),
        _rank_pool("random", q, pool, rng2),
    )
