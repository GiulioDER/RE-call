"""Retained NUM001/FIX004 public positive-scaling properties for PR894.

The original CCA Hypothesis property is retained in the private proof receipts.
Its four declared cases scale stored or query e1 independently by 1e20 or 1e-30,
all finite nonzero float32 values accepted by the public dimension-8 API.
The installed numeric gate found stored scaling changed cosine from 1 to 0.
This native equivalent keeps the relation and domains without depending on CCA
in the project's test suite, and covers all four public cosine consumers.
Baseline 6aa7ff5bdb4230ad620849eb1211e119859b04e4, measured 2026-10-13:
every case fails the named scaling assertion, then passes unchanged after repair.
Targets: LiteStore._load and _query_vector norm accumulation. Synthetic data only.
"""

from __future__ import annotations

from contextlib import closing
from pathlib import Path
from tempfile import TemporaryDirectory

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from recall.lite import LiteStore
from recall.types import Chunk


def _cosine(method: str, stored_scale: float, query_scale: float) -> float:
    with TemporaryDirectory() as directory, closing(LiteStore(Path(directory) / "m.db", dim=8)) as store:
        store.upsert([Chunk(id="a", source="a.md", text="alpha", metadata={})], [[stored_scale] + [0.0] * 7])
        vector = [query_scale] + [0.0] * 7
        if method == "dense":
            return store.query_dense(vector, 1)[0].score
        if method == "sparse":
            return store.query_sparse("alpha", 1, vec=vector)[0].score
        if method == "top":
            return store.top_cosine(vector)
        return store.cosines_for(["a"], vector)["a"]


@pytest.mark.parametrize("method", ["dense", "sparse", "top", "ids"])
@pytest.mark.parametrize("factor", [1e20, 1e-30])
@pytest.mark.parametrize("index", [0, 1])
@settings(max_examples=10, deadline=None)
@given(stored_scale=st.just(1.0), query_scale=st.just(1.0))
def test_positive_vector_scaling_preserves_public_cosine(
    method: str, factor: float, index: int, stored_scale: float, query_scale: float,
) -> None:
    """FIX004: positive finite scaling of either vector preserves cosine one."""
    args = [stored_scale, query_scale]
    before = _cosine(method, *args)
    assert before == pytest.approx(1.0, abs=1e-6)
    args[index] *= factor
    assert _cosine(method, *args) == pytest.approx(before, abs=1e-6), "positive vector scaling changed cosine"
