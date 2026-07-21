"""Pure trust-layer tests: verdicts, precedence, cycles, abstention — no DB, no clock reads."""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from recall.calibration import Calibration
from recall.trust import evaluate, resolve_successor
from recall.types import Chunk, RetrievalResult, ScoredChunk, StalenessReport

NOW = datetime(2026, 7, 17, 12, 0, tzinfo=timezone.utc)
FRESH = StalenessReport(stale=False, newest_indexed_at=NOW, age=timedelta(0), max_age=timedelta(days=2))
CAL = Calibration(embedder="test", threshold=0.5, scale=0.05)


def _hit(cid: str, file: str, score: float, **meta) -> ScoredChunk:
    chunk = Chunk(id=cid, source=file, text=f"text of {file}", metadata={"file": file, "ord": 0, **meta})
    return ScoredChunk(chunk=chunk, score=score, indexed_at=NOW)


def _result(hits: list[ScoredChunk], gap: bool = False) -> RetrievalResult:
    return RetrievalResult(query="q", hits=hits, gap_warning=gap, staleness=FRESH)


def test_resolve_successor_transitive_chain():
    sup = {"a.md": "b.md", "b.md": "c.md"}
    assert resolve_successor("a.md", sup) == "c.md"
    assert resolve_successor("b.md", sup) == "c.md"
    assert resolve_successor("c.md", sup) is None


def test_resolve_successor_cycle_does_not_hang():
    sup = {"a.md": "b.md", "b.md": "a.md"}
    assert resolve_successor("a.md", sup) == "b.md"  # cycle member: direct successor, no loop


def test_ok_verdict_above_threshold():
    res = evaluate(_result([_hit("x", "doc.md", 0.8)]), {}, CAL, NOW)
    assert res.hits[0].verdict == "ok"
    assert res.abstained is False
    assert res.reason == ""
    assert res.calibrated is True
    assert res.hits[0].confidence > 0.5


def test_low_confidence_verdict_below_threshold_and_abstains():
    res = evaluate(_result([_hit("x", "doc.md", 0.3)], gap=True), {}, CAL, NOW)
    assert res.hits[0].verdict == "low_confidence"
    assert res.abstained is True
    assert "threshold" in res.reason


def test_superseded_loses_even_with_high_score():
    hits = [_hit("old", "v1.md", 0.95), _hit("new", "v2.md", 0.70, supersedes="v1.md")]
    res = evaluate(_result(hits), {"v1.md": "v2.md"}, CAL, NOW)
    by_file = {h.provenance.file: h for h in res.hits}
    assert by_file["v1.md"].verdict == "superseded"
    assert by_file["v1.md"].validity.superseded_by == "v2.md"
    assert by_file["v2.md"].verdict == "ok"
    # successor redirect: the valid successor outranks the semantically-closer stale hit
    assert res.hits[0].provenance.file == "v2.md"
    assert res.abstained is False


def test_superseded_only_hits_abstain_with_reason():
    res = evaluate(_result([_hit("old", "v1.md", 0.95)]), {"v1.md": "v2.md"}, CAL, NOW)
    assert res.abstained is True
    assert "v2.md" in res.reason  # points the agent at the successor


def test_expired_verdict_and_window_boundaries():
    ok = evaluate(
        _result([_hit("x", "d.md", 0.9, valid_until="2026-07-17")]), {}, CAL, NOW
    )
    assert ok.hits[0].verdict == "ok"  # NOW is inside the last valid day (inclusive)
    expired = evaluate(
        _result([_hit("x", "d.md", 0.9, valid_until="2026-07-16")]), {}, CAL, NOW
    )
    assert expired.hits[0].verdict == "expired"
    assert expired.abstained is True


def test_not_yet_valid_verdict():
    res = evaluate(_result([_hit("x", "d.md", 0.9, valid_from="2027-01-01")]), {}, CAL, NOW)
    assert res.hits[0].verdict == "not_yet_valid"
    assert res.abstained is True


def test_precedence_superseded_beats_expired_beats_low_confidence():
    # superseded + expired + low score -> superseded wins
    h = _hit("x", "v1.md", 0.2, valid_until="2020-01-01")
    res = evaluate(_result([h]), {"v1.md": "v2.md"}, CAL, NOW)
    assert res.hits[0].verdict == "superseded"
    # expired + low score -> expired wins
    h2 = _hit("y", "d.md", 0.2, valid_until="2020-01-01")
    res2 = evaluate(_result([h2]), {}, CAL, NOW)
    assert res2.hits[0].verdict == "expired"


def test_uncalibrated_fallback_uses_default_threshold_and_flags_it():
    res = evaluate(_result([_hit("x", "doc.md", 0.8)]), {}, None, NOW)
    assert res.calibrated is False
    assert res.hits[0].verdict == "ok"  # 0.8 >= DEFAULT_GAP_THRESHOLD (0.50)
    assert 0.0 < res.hits[0].confidence < 1.0  # still computed, just uncalibrated


def test_provenance_and_validity_populated():
    res = evaluate(
        _result([_hit("x", "d.md", 0.9, valid_from="2026-01-01", valid_until="2099-01-01")]),
        {},
        CAL,
        NOW,
    )
    h = res.hits[0]
    assert h.provenance.file == "d.md" and h.provenance.ord == 0
    assert h.provenance.indexed_at == NOW
    assert h.validity.valid_from is not None and h.validity.valid_until is not None
    assert h.cosine == 0.9


def test_successor_promotion_when_stale_hit_was_confident():
    # stale v1 matches strongly; its successor is retrieved but worded differently (low cosine)
    hits = [_hit("old", "v1.md", 0.95), _hit("new", "v2.md", 0.30, supersedes="v1.md")]
    res = evaluate(_result(hits), {"v1.md": "v2.md"}, CAL, NOW)
    by_file = {h.provenance.file: h for h in res.hits}
    assert by_file["v2.md"].verdict == "ok"  # promoted: the supersession edge transfers relevance
    assert by_file["v2.md"].confidence < 0.5  # confidence stays honest (low)
    assert res.hits[0].provenance.file == "v2.md"
    assert res.abstained is False


def test_no_successor_promotion_when_stale_hit_was_weak():
    # neither hit clears the threshold: an unrelated query must NOT ride the supersession edge
    hits = [_hit("old", "v1.md", 0.20), _hit("new", "v2.md", 0.10, supersedes="v1.md")]
    res = evaluate(_result(hits, gap=True), {"v1.md": "v2.md"}, CAL, NOW)
    by_file = {h.provenance.file: h for h in res.hits}
    assert by_file["v2.md"].verdict == "low_confidence"
    assert res.abstained is True


def test_invalid_metadata_verdict_instead_of_crash():
    # a malformed date (reachable via direct store.upsert) must not crash retrieval,
    # and must fail CLOSED: the hit is not trustworthy
    res = evaluate(_result([_hit("x", "d.md", 0.9, valid_until="June 2026")]), {}, CAL, NOW)
    assert res.hits[0].verdict == "invalid_metadata"
    assert res.abstained is True
    assert "malformed" in res.reason


def test_self_supersession_is_ignored():
    # `supersedes:` the file's own name is an authoring mistake, not a real edge
    assert resolve_successor("a.md", {"a.md": "a.md"}) is None
    res = evaluate(_result([_hit("x", "a.md", 0.9)]), {"a.md": "a.md"}, CAL, NOW)
    assert res.hits[0].verdict == "ok"


def test_naive_now_is_interpreted_as_utc():
    naive = NOW.replace(tzinfo=None)
    res = evaluate(
        _result([_hit("x", "d.md", 0.9, valid_until="2026-07-16")]), {}, CAL, naive
    )
    assert res.hits[0].verdict == "expired"  # no TypeError on naive-vs-aware comparison


def test_ambiguous_supersession_endpoint_is_not_served_as_ok():
    # two files share the basename the edge names: which one is superseded is unknowable,
    # so the hit must not be presented as trustworthy (fail closed, not silent mismap)
    res = evaluate(
        _result([_hit("x", "notes.md", 0.9)]), {}, CAL, NOW, unresolved=frozenset({"notes.md"})
    )
    assert res.hits[0].verdict == "ambiguous_supersession"
    assert res.hits[0].validity.superseded_by is None
    assert res.abstained is True
    assert "basename" in res.reason


def test_unresolved_does_not_touch_other_files():
    res = evaluate(
        _result([_hit("x", "other.md", 0.9)]), {}, CAL, NOW, unresolved=frozenset({"notes.md"})
    )
    assert res.hits[0].verdict == "ok"
    assert res.abstained is False


def test_ambiguity_outranks_a_declared_edge_for_the_same_file():
    # if the store still carried an edge for an unresolved basename, ambiguity wins
    res = evaluate(
        _result([_hit("x", "notes.md", 0.9)]),
        {"notes.md": "new.md"},
        CAL,
        NOW,
        unresolved=frozenset({"notes.md"}),
    )
    assert res.hits[0].verdict == "ambiguous_supersession"
