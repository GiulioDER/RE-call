from __future__ import annotations

from datetime import datetime, timezone

import pytest

from recall.eval.legdiag import _assert_hit_agrees, build_report, classify_gold
from recall.retriever import LegProbe
from recall.types import Chunk, ScoredChunk


def _hit(dia: str, score: float = 0.5) -> ScoredChunk:
    """A hit whose dia id resolves to `dia`.

    `locomo._filename_to_dia_id` is `Path(name).stem.replace("_", ":", 1)` and it reads
    `chunk.metadata["file"]` — there is NO `dia_id` metadata key. Build the filename the way
    the harness does or the classifier silently matches nothing.
    """
    fname = dia.replace(":", "_", 1) + ".md"
    return ScoredChunk(
        chunk=Chunk(id=fname, source=fname, text=dia, metadata={"file": fname}),
        score=score,
        indexed_at=datetime.now(timezone.utc),
    )


def _probe(dense_ids, sparse_ids, fused_ids) -> LegProbe:
    return LegProbe(
        query="q",
        dense=[_hit(c) for c in dense_ids],
        sparse=[_hit(c) for c in sparse_ids],
        sparse_ranks=[1.0 / (i + 1) for i in range(len(sparse_ids))],
        fused=[_hit(c) for c in fused_ids],
    )


def test_classify_hit_when_gold_is_inside_top_k():
    p = _probe(["D1:1", "D1:2"], ["D1:1"], ["D1:1", "D1:2"])
    assert classify_gold(p, ["D1:1"], k=5) == "hit"


def test_classify_a_when_gold_is_in_the_pool_but_below_k():
    fused = [f"D1:{i}" for i in range(1, 9)]        # gold D1:8 sits at rank 8
    p = _probe(fused, [], fused)
    assert classify_gold(p, ["D1:8"], k=5) == "a_misranked"


def test_classify_b_when_gold_is_in_neither_leg():
    p = _probe(["D1:1", "D1:2"], ["D1:3"], ["D1:1", "D1:2", "D1:3"])
    assert classify_gold(p, ["D1:99"], k=5) == "b_unretrieved"


def test_classify_c_when_there_is_no_gold_at_all():
    p = _probe(["D1:1"], [], ["D1:1"])
    assert classify_gold(p, [], k=5) == "c_absent"


def test_report_splits_hit_rate_by_trigger_and_reports_firing_rate():
    records = [
        {"trigger": True, "hit": False, "bucket": "b_unretrieved", "category": 3, "n_sparse": 20},
        {"trigger": True, "hit": False, "bucket": "a_misranked", "category": 3, "n_sparse": 20},
        {"trigger": True, "hit": True, "bucket": "hit", "category": 1, "n_sparse": 20},
        {"trigger": False, "hit": True, "bucket": "hit", "category": 1, "n_sparse": 20},
        {"trigger": False, "hit": True, "bucket": "hit", "category": 2, "n_sparse": 20},
    ]
    r = build_report(records)

    assert r["q2_firing_rate"]["rate"] == 0.6
    assert r["q2_firing_rate"]["n"] == 5
    assert r["q1_hit_at_k"]["firing"]["rate"] == pytest.approx(1 / 3)
    assert r["q1_hit_at_k"]["not_firing"]["rate"] == 1.0
    # approx, not ==: the computed delta is (1/3 - 1.0) == -0.6666666666666667, while the
    # literal -2/3 is -0.6666666666666666. Exact equality here fails on float representation.
    assert r["q1_hit_at_k"]["delta"] == pytest.approx(-2 / 3)
    assert r["q3_buckets"]["a_misranked"] == 1
    assert r["q3_buckets"]["b_unretrieved"] == 1
    # every published rate carries an interval
    assert len(r["q2_firing_rate"]["ci"]) == 2


def test_report_handles_an_empty_firing_group():
    records = [{"trigger": False, "hit": True, "bucket": "hit", "category": 1, "n_sparse": 20}]
    r = build_report(records)
    assert r["q2_firing_rate"]["rate"] == 0.0
    assert r["q1_hit_at_k"]["firing"]["n"] == 0
    assert r["q1_hit_at_k"]["delta"] is None      # undefined, not zero


def test_report_stratifies_q1_by_sparse_depth():
    """The confound control. `more_decisive` leaves a residual sample-size bias (on iid noise a
    5-vs-20 comparison fires 35.1% rather than 50%), so a pooled Q1 effect could be sparse-leg
    depth rather than leg disagreement. Each bin must carry its own firing/not-firing split."""
    records = [
        {"trigger": True, "hit": False, "bucket": "b_unretrieved", "category": 3, "n_sparse": 2},
        {"trigger": False, "hit": True, "bucket": "hit", "category": 1, "n_sparse": 3},
        {"trigger": True, "hit": True, "bucket": "hit", "category": 1, "n_sparse": 20},
        {"trigger": False, "hit": True, "bucket": "hit", "category": 2, "n_sparse": 20},
    ]
    strata = build_report(records)["q1_stratified_by_sparse_depth"]

    assert set(strata) == {"n_sparse_0-4", "n_sparse_20+"}
    assert strata["n_sparse_0-4"]["n"] == 2
    assert strata["n_sparse_0-4"]["firing"]["rate"] == 0.0
    assert strata["n_sparse_0-4"]["not_firing"]["rate"] == 1.0
    assert strata["n_sparse_0-4"]["delta"] == -1.0
    # both hits in the deep bin, so the trigger separates nothing there
    assert strata["n_sparse_20+"]["delta"] == 0.0


def test_assert_hit_agrees_is_silent_when_bucket_and_harness_agree():
    # bucket == "hit" and harness_hit == True: agree.
    _assert_hit_agrees("s1", "q", "hit", True, ["D1:1"], ["D1:1"])
    # bucket != "hit" and harness_hit == False: also agree.
    _assert_hit_agrees("s1", "q", "a_misranked", False, ["D1:8"], ["D1:1"])


def test_assert_hit_agrees_raises_when_classify_gold_says_hit_but_harness_says_miss():
    with pytest.raises(RuntimeError, match="classify_gold/harness disagree"):
        _assert_hit_agrees("s1", "q", "hit", False, ["D1:1"], ["D1:1"])


def test_assert_hit_agrees_raises_when_harness_says_hit_but_classify_gold_says_miss():
    with pytest.raises(RuntimeError, match="classify_gold/harness disagree"):
        _assert_hit_agrees("s1", "q", "b_unretrieved", True, ["D1:99"], ["D1:1"])
