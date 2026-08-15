"""The eval calibration fleet: see docs/EVAL_CALIBRATION_FLEET_DESIGN.md."""
from __future__ import annotations

import pytest

from recall.eval.harness import _score_config
from tests.fleet.scripted import QueryKeyedStore, ScriptedEmbedder


def test_scripted_store_returns_the_scripted_order_through_the_real_retriever():
    """The whole fleet rests on this: with fusion="dense" the retrieved order IS the script.

    If RRF or the sparse leg ever reordered these, every closed form in members.py would be
    wrong while still looking plausible, so this is checked directly rather than assumed.
    """
    embedder = ScriptedEmbedder()
    script = {"q": [("a.md:0", 0.8), ("b.md:0", 0.7), ("gold.md:0", 0.6)]}
    store = QueryKeyedStore(embedder, script)
    queries = [{"id": "q1", "query": "q", "relevant_ids": ["gold.md:0"], "answerable": True}]

    result = _score_config(store, embedder, queries, "dense", None)

    # gold is at 1-based rank 3 of 3: MRR = 1/3, R@5 = 1.0 (it is inside the top 5).
    assert result.mrr == pytest.approx(1 / 3)
    assert result.r_at_5 == pytest.approx(1.0)


def test_scripted_store_refuses_an_unscripted_query():
    """An unscripted query must raise, not return zero rows.

    Zero rows scores as a total retrieval failure, which is a FIXTURE bug that reads exactly
    like a defect the fleet detected. `arm_check.EmptySampleError` sets the same precedent.
    """
    embedder = ScriptedEmbedder()
    store = QueryKeyedStore(embedder, {"scripted": [("a.md:0", 0.8)]})
    queries = [{"id": "q1", "query": "unscripted", "relevant_ids": ["a.md:0"], "answerable": True}]

    with pytest.raises(KeyError, match="no script for query"):
        _score_config(store, embedder, queries, "dense", None)


def test_a_member_without_a_declared_blind_spot_is_refused():
    """`does_not_catch` is required, not optional.

    An optional field would be empty on every member within a month, and a fleet that does not
    state what it misses invites being read as covering more than it does.
    """
    from tests.fleet.members import FleetMember

    with pytest.raises(ValueError, match="does_not_catch"):
        FleetMember(
            name="blank",
            defect="none",
            build=lambda: None,
            expected={},
            does_not_catch="   ",
        )
