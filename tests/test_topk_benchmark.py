"""Benchmark C v2（Top-K）指标函数测试。"""

import pytest

from benchmark.topk_retrieval import (
    hit_at_k,
    incremental_novel,
    mean_score_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    union_top3,
)


def test_ndcg_perfect_is_one():
    assert ndcg_at_k([5, 4, 3, 1, 0], k=5) == 1.0


def test_recall_precision_hit_mean():
    rel = [5, 2, 4, 0, 1]  # 相关(>=3): 5, 4 -> total 2
    assert recall_at_k(rel, 3, threshold=3.0) == 1.0   # 前3命中2个/共2个
    assert precision_at_k(rel, 3, threshold=3.0) == 2 / 3
    assert hit_at_k(rel, 3, threshold=4.0) == 1.0       # 5 和 4 都在前3
    assert mean_score_at_k(rel, 3) == pytest.approx((5 + 2 + 4) / 3)


def test_union_top3_rerank_by_hybrid():
    stats = ["A", "B", "D"]
    gnn = ["B", "C", "E"]
    hybrid_sim = {"A": 0.9, "B": 0.8, "C": 0.7, "D": 0.6, "E": 0.5}
    assert union_top3(stats, gnn, hybrid_sim) == ["A", "B", "C"]


def test_incremental_novel():
    scores = {"A": 5, "B": 4, "C": 5, "D": 1, "E": 0}
    inc = incremental_novel(["B", "C", "E"], ["A", "B", "D"], scores)
    assert inc["n_novel"] == 2
    assert inc["novel_ge3"] == 1  # C
    assert inc["novel_ge4"] == 1  # C
    assert inc["novel_eq5"] == 1  # C
