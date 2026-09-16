from __future__ import annotations

import numpy as np
import pytest

from scripts.run_atomic_fact_context4_pilot import AtomicView, rank_atomic
from scripts.run_atomic_fact_current_generation_shadow import _normalize_rows, rank_atomic_matrix


def test_matrix_rank_matches_the_frozen_scalar_reference() -> None:
    """The optimized shadow preserves scalar ranking and parent deduplication.

    Red proof receipt ``atomic-shadow-matrix-01``: before the implementation existed,
    ``pytest -q tests/test_atomic_fact_current_generation_shadow.py`` failed during collection
    with ``ModuleNotFoundError: scripts.run_atomic_fact_current_generation_shadow``.
    """

    views = [
        AtomicView("recall/a.md", 0, 0, "a zero", "parent a"),
        AtomicView("recall/a.md", 0, 1, "a one", "parent a"),
        AtomicView("recall/b.md", 0, 0, "b zero", "parent b"),
        AtomicView("recall/c.md", 2, 0, "c zero", "parent c"),
    ]
    vectors = [
        [1.0, 0.0, 0.0],
        [0.9, 0.1, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    query = [0.8, 0.6, 0.0]

    matrix = _normalize_rows(np.asarray(vectors, dtype=np.float32))
    expected = rank_atomic(views, matrix.tolist(), query, cutoff=3)
    actual = rank_atomic_matrix(
        views,
        matrix,
        np.asarray(query, dtype=np.float32),
        cutoff=3,
    )

    assert [item.identity for item in actual] == [item.identity for item in expected]
    assert [item.score for item in actual] == pytest.approx(
        [item.score for item in expected], abs=1e-6
    )
