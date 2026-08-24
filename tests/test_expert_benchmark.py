"""Benchmark C 指标函数与模板格式测试。"""

import csv
import os

import pytest

from benchmark.expert_retrieval import ndcg_at_k, pairwise_accuracy, spearman


def test_ndcg_perfect_ranking_is_one():
    assert ndcg_at_k([3, 2, 1, 0], k=5) == 1.0


def test_ndcg_worse_ranking_lower():
    good = ndcg_at_k([3, 2, 1, 0], k=5)
    bad = ndcg_at_k([0, 1, 2, 3], k=5)
    assert good > bad


def test_pairwise_accuracy():
    scores = {"a": 3, "b": 1, "c": 0}
    assert pairwise_accuracy(["a", "b", "c"], scores) == 1.0
    assert pairwise_accuracy(["c", "b", "a"], scores) == 0.0


def test_spearman_perfect_positive():
    assert spearman([1, 2, 3, 4], [2, 4, 6, 8]) == pytest.approx(1.0)


def test_expert_template_format():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "data",
        "benchmark",
        "expert_similarity.csv",
    )
    with open(path, encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert rows
    assert set(rows[0]) == {"query_id", "candidate_id", "score", "comment"}
    assert all(row["score"] == "" for row in rows)
