import math

import pytest

from recall.eval.metrics import (
    false_confident_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_at_k():
    assert precision_at_k(["a", "b", "c"], ["a", "c"], 3) == pytest.approx(2 / 3)
    assert precision_at_k(["a", "b", "c"], ["b"], 1) == 0.0  # top-1 is 'a'
    assert precision_at_k(["a"], ["a"], 0) == 0.0


def test_recall_at_k():
    assert recall_at_k(["a", "b", "c"], ["a", "x"], 3) == pytest.approx(0.5)
    assert recall_at_k(["a", "b"], ["a"], 2) == 1.0
    assert recall_at_k(["a"], [], 2) == 0.0


def test_mrr():
    assert mrr(["a", "b", "c"], ["b"]) == pytest.approx(0.5)
    assert mrr(["a", "b"], ["a"]) == 1.0
    assert mrr(["a", "b"], ["c"]) == 0.0


def test_ndcg_at_k():
    assert ndcg_at_k(["a", "b"], ["a"], 2) == pytest.approx(1.0)
    # relevant item at rank 2: dcg = 1/log2(3), idcg = 1/log2(2) = 1.0
    assert ndcg_at_k(["a", "b"], ["b"], 2) == pytest.approx(1.0 / math.log2(3))
    assert ndcg_at_k(["a", "b"], ["z"], 2) == 0.0


def test_false_confident_rate():
    # 2 of 4 unanswerable queries had gap_warning=False (wrongly confident)
    assert false_confident_rate([True, True, False, False]) == pytest.approx(0.5)
    assert false_confident_rate([True, True]) == 0.0  # guard fired on all -> good
    assert false_confident_rate([]) == 0.0
