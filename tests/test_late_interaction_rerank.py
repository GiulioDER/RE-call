import numpy as np
import pytest

from recall.rerank import maxsim


def test_maxsim_matches_hand_computed_value():
    # q0 . d0 = 1.0, q0 . d1 = 0.6  -> max 1.0
    # q1 . d0 = 0.0, q1 . d1 = 0.8  -> max 0.8
    # sum = 1.8
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    doc = np.array([[1.0, 0.0], [0.6, 0.8]])
    assert maxsim(query, doc) == pytest.approx(1.8)


def test_maxsim_is_max_not_mean():
    """The mutation check in G5 relies on these differing. If they ever coincide the gate is
    vacuous, so the difference is pinned here rather than assumed."""
    query = np.array([[1.0, 0.0], [0.0, 1.0]])
    doc = np.array([[1.0, 0.0], [0.6, 0.8]])
    mean_version = float((query @ doc.T).mean(axis=1).sum())
    assert mean_version == pytest.approx(1.2)
    assert maxsim(query, doc) != pytest.approx(mean_version)


def test_maxsim_refuses_empty_document():
    """A document with no tokens cannot be scored. Returning 0.0 would place it mid-ranking,
    which is the same silent-corruption shape `rerank_order` refuses a missing score for."""
    query = np.array([[1.0, 0.0]])
    with pytest.raises(ValueError, match="no tokens"):
        maxsim(query, np.zeros((0, 2)))


def test_maxsim_refuses_empty_query():
    with pytest.raises(ValueError, match="no tokens"):
        maxsim(np.zeros((0, 2)), np.array([[1.0, 0.0]]))


def test_maxsim_refuses_dimension_mismatch():
    with pytest.raises(ValueError, match="dimension"):
        maxsim(np.array([[1.0, 0.0]]), np.array([[1.0, 0.0, 0.0]]))
