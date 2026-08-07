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


from recall.rerank import (
    DEFAULT_LATE_INTERACTION_MODEL,
    LATE_INTERACTION_MODELS,
    PERMISSIVE_LICENCES,
    late_interaction_licence,
)


def test_mit_is_permissive():
    """The load-bearing correction to `sparse.py:195`, which gates on `!= "apache-2.0"` and would
    therefore refuse the MIT primary arm under its own guard."""
    assert "mit" in PERMISSIVE_LICENCES
    assert late_interaction_licence("colbert-ir/colbertv2.0") == "mit"


def test_apache_is_permissive():
    assert late_interaction_licence("answerdotai/answerai-colbert-small-v1") == "apache-2.0"


def test_default_model_is_permissive():
    assert LATE_INTERACTION_MODELS[DEFAULT_LATE_INTERACTION_MODEL] in PERMISSIVE_LICENCES


def test_noncommercial_refused_without_optin():
    with pytest.raises(ValueError, match="cc-by-nc-4.0"):
        late_interaction_licence("jinaai/jina-colbert-v2")


def test_noncommercial_allowed_with_optin():
    assert late_interaction_licence(
        "jinaai/jina-colbert-v2", accept_noncommercial_license=True
    ) == "cc-by-nc-4.0"


def test_unknown_checkpoint_refused():
    """An unrecorded licence is exactly what this check exists to prevent, so an unknown model
    raises even though it might be perfectly permissive."""
    with pytest.raises(ValueError, match="unknown late-interaction model"):
        late_interaction_licence("some/unrecorded-colbert")


def test_unknown_checkpoint_refused_even_with_optin():
    """The opt-in waives the LICENCE check, not the REGISTRY check."""
    with pytest.raises(ValueError, match="unknown late-interaction model"):
        late_interaction_licence("some/unrecorded-colbert", accept_noncommercial_license=True)
