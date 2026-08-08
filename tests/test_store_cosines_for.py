"""`cosines_for` puts every returned hit's score on one basis.

`search_fused` retrieves with two different query embeddings. Without a way to re-score the
returned hits against the QUERY's embedding, hits surfaced only by the history variant would
carry a cosine measured against the history, and `trust.py` would push that through
`cal.confidence()` as though it meant the same thing.
"""

from __future__ import annotations

import pytest

from recall.store import STORE_QUERY_LEGS, TIMED_PUBLIC_METHODS

DIM = 8


def test_cosines_for_is_registered_in_the_timing_surface() -> None:
    """A public store method missing from these tuples drops its timing silently.

    `GenerationStore` made this mistake twice, and it broke CI again on 2026-08-06. The guard is
    a tuple rather than a docstring precisely because docstrings do not fail builds.
    """
    assert "cosines_for" in TIMED_PUBLIC_METHODS
    assert "rescore" in STORE_QUERY_LEGS


def test_cosines_for_returns_the_cosine_against_the_given_vector(make_store) -> None:
    """The value must match what `query_dense` reports for the same chunk and vector."""
    from recall.types import Chunk

    store = make_store(DIM)
    chunks = [Chunk(id="c1", source="s", text="alpha", metadata={})]
    vec = [1.0] + [0.0] * (DIM - 1)
    store.upsert(chunks, [vec])

    dense = store.query_dense(vec, k=1)
    rescored = store.cosines_for(["c1"], vec)

    assert rescored["c1"] == pytest.approx(dense[0].score, abs=1e-6)


def test_cosines_for_omits_ids_that_do_not_exist(make_store) -> None:
    """An absent id is not a zero. Zero is a real cosine and would look like a poor match."""
    store = make_store(DIM)
    vec = [1.0] + [0.0] * (DIM - 1)

    assert store.cosines_for(["nope"], vec) == {}


def test_cosines_for_returns_empty_for_no_ids(make_store) -> None:
    """No round trip for an empty request."""
    store = make_store(DIM)
    vec = [1.0] + [0.0] * (DIM - 1)

    assert store.cosines_for([], vec) == {}
