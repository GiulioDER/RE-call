from __future__ import annotations

import numpy as np

from scripts.run_atomic_fact_context4_pilot import AtomicView
from scripts.run_atomic_fact_exact_rescue_shortlist import (
    build_parent_codes,
    select_exact_masked_max,
    select_full_sort_reference,
)


def test_masked_max_matches_reference_with_exclusion_duplicates_and_ties() -> None:
    """Mask dense parents and preserve deterministic source, parent, and view tie ordering.

    Red proof receipt ``atomic-exact-shortlist-01``: the plausible production mutation changed
    `select_exact_masked_max` from `codes == code` to `codes != code`. This exact node failed at
    the intended identity assertion because it returned `dense.md` instead of `a.md`.
    """

    views = [
        AtomicView("dense.md", 0, 0, "dense zero", "dense parent"),
        AtomicView("dense.md", 0, 1, "dense one", "dense parent"),
        AtomicView("z.md", 0, 0, "z zero", "z parent"),
        AtomicView("a.md", 2, 1, "a one", "a parent"),
        AtomicView("a.md", 2, 0, "a zero", "a parent"),
    ]
    vectors: np.ndarray = np.asarray(
        [
            [1.0, 0.0],
            [1.0, 0.0],
            [0.8, 0.6],
            [0.8, 0.6],
            [0.8, 0.6],
        ],
        dtype=np.float32,
    )
    query: np.ndarray = np.asarray([1.0, 0.0], dtype=np.float32)
    excluded = [("dense.md", 0)]
    parent_codes, code_by_identity = build_parent_codes(views)

    reference = select_full_sort_reference(views, vectors, query, excluded)
    candidate = select_exact_masked_max(
        views,
        vectors,
        query,
        excluded,
        parent_codes,
        code_by_identity,
    )

    assert reference.identity == ("a.md", 2)
    assert candidate.identity == reference.identity
    assert candidate.score == reference.score
