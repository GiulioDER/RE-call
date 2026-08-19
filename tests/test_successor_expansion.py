"""Successor directed expansion: fetch the declared successor the candidate pool did not hold.

Pre-registered in `docs/preregistrations/2026-08-19-successor-directed-expansion.md`. These are the
INVARIANT tests from that record, not the quality measurement: they pin the mechanism and the
safety properties, and say nothing about the rate, which needs a real pool over a real corpus.

The failure being closed: `_verdict` resolves the successor filename and `evaluate` promotes that
file only if it already contributed a chunk to the pool. Nothing fetched it, so a superseded hit
whose successor ranked outside the pool abstained while naming a successor that was in the index.
"""

from datetime import datetime, timedelta, timezone

import pytest

from recall.calibration import Calibration
from recall.retriever import SuccessorExpansionPolicy, expand_retrieval_by_successor
from recall.trust import evaluate, order_promoted, trusted_search
from recall.trust_policy import TrustPolicy
from recall.types import (
    Chunk,
    RetrievalDiagnostics,
    RetrievalResult,
    ScoredChunk,
    StalenessReport,
)

NOW = datetime(2026, 8, 19, tzinfo=timezone.utc)
CALIBRATION = Calibration(embedder="fixture", threshold=0.70)


def _scored(cid: str, file: str, score: float, ordinal: int = 0) -> ScoredChunk:
    return ScoredChunk(
        chunk=Chunk(cid, f"/corpus/{file}", f"text of {file}", {"file": file, "ord": ordinal}),
        score=score,
        indexed_at=NOW,
    )


def _retrieval(*hits: ScoredChunk) -> RetrievalResult:
    return RetrievalResult(
        query="what is the current setting",
        hits=list(hits),
        gap_warning=False,
        staleness=StalenessReport(False, NOW, timedelta(0), timedelta(days=1)),
        diagnostics=RetrievalDiagnostics(stage_ms={"dense_retrieval": 1.0}),
    )


def _corpus(scoped, calls):
    """A scoped `search` that records every source it was asked for."""

    def search(query: str, k: int, source: str | None = None) -> RetrievalResult:
        calls.append(source or "")
        return _retrieval(*scoped.get(source or "", ())[:k])

    return search


def _resolver(supersession):
    def resolve(file: str):
        return supersession.get(file)

    return resolve


ENABLED = SuccessorExpansionPolicy(enabled=True, max_sources=2, chunks_per_source=3)


def test_baseline_abstains_when_the_successor_is_outside_the_pool() -> None:
    """The failure this change exists to close, pinned so it cannot be quietly reintroduced."""
    result = _retrieval(_scored("c1", "ttl_v1.md", 0.91))
    trusted = evaluate(result, {"ttl_v1.md": "ttl_v2.md"}, CALIBRATION, NOW)

    assert trusted.abstained
    assert "ttl_v2.md" in trusted.reason
    # The successor is NAMED in the reason while never having been retrieved. That gap is the bug.
    assert [h.provenance.file for h in trusted.hits] == ["ttl_v1.md"]


def test_absent_successor_is_fetched_and_becomes_the_answer() -> None:
    calls = []
    result = _retrieval(_scored("c1", "ttl_v1.md", 0.91))
    expanded = expand_retrieval_by_successor(
        result,
        _corpus({"ttl_v2.md": [_scored("c2", "ttl_v2.md", 0.64)]}, calls),
        _resolver({"ttl_v1.md": "ttl_v2.md"}),
        ENABLED,
    )
    trusted = evaluate(expanded, {"ttl_v1.md": "ttl_v2.md"}, CALIBRATION, NOW)

    assert calls == ["ttl_v2.md"]
    assert not trusted.abstained
    # Promoted despite scoring 0.64, BELOW the 0.70 threshold: the declared edge transfers the
    # relevance the stale memory proved. Valid-first ordering puts it above the stale memory.
    assert trusted.hits[0].provenance.file == "ttl_v2.md"
    assert trusted.hits[0].verdict == "ok"


def test_the_superseded_hit_is_never_promoted_by_the_fetch() -> None:
    """`str_trust` cannot rise. Fetching more material must not make a stale memory servable."""
    result = _retrieval(_scored("c1", "ttl_v1.md", 0.91))
    expanded = expand_retrieval_by_successor(
        result,
        _corpus({"ttl_v2.md": [_scored("c2", "ttl_v2.md", 0.64)]}, []),
        _resolver({"ttl_v1.md": "ttl_v2.md"}),
        ENABLED,
    )
    trusted = evaluate(expanded, {"ttl_v1.md": "ttl_v2.md"}, CALIBRATION, NOW)

    stale = [h for h in trusted.hits if h.provenance.file == "ttl_v1.md"]
    assert [h.verdict for h in stale] == ["superseded"]


def test_a_successor_already_in_the_pool_is_not_refetched() -> None:
    """Stratum A must be untouched: no scoped search, and the same hits back."""
    calls = []
    result = _retrieval(
        _scored("c1", "ttl_v1.md", 0.91),
        _scored("c2", "ttl_v2.md", 0.88),
    )
    expanded = expand_retrieval_by_successor(
        result, _corpus({}, calls), _resolver({"ttl_v1.md": "ttl_v2.md"}), ENABLED
    )

    assert calls == []
    assert expanded.hits == result.hits


def test_disabled_policy_returns_the_result_unchanged() -> None:
    calls = []
    result = _retrieval(_scored("c1", "ttl_v1.md", 0.91))
    expanded = expand_retrieval_by_successor(
        result,
        _corpus({"ttl_v2.md": [_scored("c2", "ttl_v2.md", 0.64)]}, calls),
        _resolver({"ttl_v1.md": "ttl_v2.md"}),
        SuccessorExpansionPolicy(enabled=False),
    )

    assert calls == []
    assert expanded is result


def test_a_hit_with_no_successor_triggers_nothing() -> None:
    calls = []
    result = _retrieval(_scored("c1", "stable.md", 0.91))
    expanded = expand_retrieval_by_successor(result, _corpus({}, calls), _resolver({}), ENABLED)

    assert calls == []
    assert expanded is result


def test_the_fetch_is_bounded_by_max_sources() -> None:
    calls = []
    result = _retrieval(
        _scored("a1", "a_v1.md", 0.91),
        _scored("b1", "b_v1.md", 0.90),
        _scored("d1", "d_v1.md", 0.89),
    )
    expand_retrieval_by_successor(
        result,
        _corpus(
            {
                "a_v2.md": [_scored("a2", "a_v2.md", 0.6)],
                "b_v2.md": [_scored("b2", "b_v2.md", 0.6)],
                "d_v2.md": [_scored("d2", "d_v2.md", 0.6)],
            },
            calls,
        ),
        _resolver({"a_v1.md": "a_v2.md", "b_v1.md": "b_v2.md", "d_v1.md": "d_v2.md"}),
        SuccessorExpansionPolicy(enabled=True, max_sources=2, chunks_per_source=3),
    )

    assert calls == ["a_v2.md", "b_v2.md"]


def test_two_stale_hits_sharing_one_successor_fetch_it_once() -> None:
    calls = []
    result = _retrieval(
        _scored("c1", "ttl_v1.md", 0.91, ordinal=0),
        _scored("c2", "ttl_v1.md", 0.85, ordinal=1),
    )
    expand_retrieval_by_successor(
        result,
        _corpus({"ttl_v2.md": [_scored("c3", "ttl_v2.md", 0.64)]}, calls),
        _resolver({"ttl_v1.md": "ttl_v2.md"}),
        ENABLED,
    )

    assert calls == ["ttl_v2.md"]


def test_a_chunk_already_in_the_pool_is_not_duplicated() -> None:
    """The scoped search may legitimately return a chunk the pool already holds."""
    shared = _scored("c1", "ttl_v1.md", 0.91)
    result = _retrieval(shared)
    expanded = expand_retrieval_by_successor(
        result,
        _corpus({"ttl_v2.md": [shared, _scored("c2", "ttl_v2.md", 0.64)]}, []),
        _resolver({"ttl_v1.md": "ttl_v2.md"}),
        ENABLED,
    )

    ids = [hit.chunk.id for hit in expanded.hits]
    assert ids == ["c1", "c2"]
    assert len(ids) == len(set(ids))


def test_a_supersession_cycle_terminates() -> None:
    """`resolve_successor` breaks cycles; the expander must not reintroduce one."""
    calls = []
    result = _retrieval(_scored("c1", "a.md", 0.91))
    expand_retrieval_by_successor(
        result,
        _corpus({"b.md": [_scored("c2", "b.md", 0.64)]}, calls),
        # A resolver that would loop forever if the expander walked the chain itself.
        _resolver({"a.md": "b.md", "b.md": "a.md"}),
        ENABLED,
    )

    # One fetch, for the resolved successor of the hit that was actually retrieved. The expander
    # does not walk the chain: resolution belongs to the resolver, which is already cycle safe.
    assert calls == ["b.md"]


def test_expansion_is_recorded_in_the_diagnostics() -> None:
    result = _retrieval(_scored("c1", "ttl_v1.md", 0.91))
    expanded = expand_retrieval_by_successor(
        result,
        _corpus({"ttl_v2.md": [_scored("c2", "ttl_v2.md", 0.64)]}, []),
        _resolver({"ttl_v1.md": "ttl_v2.md"}),
        ENABLED,
    )

    stage = expanded.diagnostics.stage_ms
    assert "successor_expansion" in stage
    assert stage["successor_expansion_sources"] == 1.0
    # The pre-existing timings survive the merge.
    assert stage["dense_retrieval"] == 1.0


def test_a_hit_without_file_metadata_is_skipped_rather_than_crashing() -> None:
    calls = []
    bare = ScoredChunk(
        chunk=Chunk("c1", "/corpus/legacy.md", "legacy row", {}),
        score=0.91,
        indexed_at=NOW,
    )
    expanded = expand_retrieval_by_successor(
        _retrieval(bare), _corpus({}, calls), _resolver({"legacy.md": "new.md"}), ENABLED
    )

    # `_verdict` keys supersession on the `file` metadata and so does this. A row without it
    # cannot be superseded either, so fetching would add material the trust layer will not use.
    assert calls == []
    assert expanded.hits == [bare]


# --- wiring, through `trusted_search` -----------------------------------------------------------
#
# The tests above exercise the pure function with an injected resolver. These two check that
# `trusted_search` builds that resolver from the store edges and places the call where both facts
# are available, which no unit test of the expander can see.


class _Store:
    """The read surface `trusted_search` touches, with one declared supersession edge."""

    tenant = "tenant-1"

    def generation_binding(self) -> dict[str, str]:
        return {
            "tenant_id": self.tenant,
            "generation_id": "gen-1",
            "pipeline_fingerprint": "p" * 64,
            "corpus_fingerprint": "c" * 64,
        }

    def supersession(self) -> tuple[dict[str, str], frozenset[str]]:
        return {"ttl_v1.md": "ttl_v2.md"}, frozenset()


class _Retriever:
    """The successor exists but is reachable ONLY through a source-scoped search."""

    def __init__(self, *_args, **_kwargs) -> None:
        pass

    def search(self, query: str, k: int = 5, source: str | None = None) -> RetrievalResult:
        if source is None:
            return _retrieval(_scored("c1", "ttl_v1.md", 0.91))
        assert source == "ttl_v2.md"
        return _retrieval(_scored("c2", "ttl_v2.md", 0.64))


def _search(monkeypatch, **kwargs):
    monkeypatch.setattr("recall.trust.HybridRetriever", _Retriever)
    return trusted_search(
        _Store(),
        object(),
        "what is the current cache ttl",
        k=5,
        calibration=CALIBRATION,
        policy=TrustPolicy.development(),
        now=NOW,
        **kwargs,
    )


def test_trusted_search_without_the_policy_still_abstains(monkeypatch) -> None:
    """The default path is unchanged, so enabling the feature is the only thing that moves."""
    result = _search(monkeypatch)

    assert result.abstained
    assert "ttl_v2.md" in result.reason
    assert "successor_expansion_sources" not in result.diagnostics.stage_ms


def test_enabling_it_on_a_corpus_with_no_edges_costs_no_extra_query(monkeypatch) -> None:
    """The common case. A corpus that declares no supersession must pay nothing for the feature."""
    calls: list[str | None] = []

    class _NoEdges(_Store):
        def supersession(self) -> tuple[dict[str, str], frozenset[str]]:
            return {}, frozenset()

    class _Counting(_Retriever):
        def search(self, query: str, k: int = 5, source: str | None = None) -> RetrievalResult:
            calls.append(source)
            return _retrieval(_scored("c1", "ttl_v1.md", 0.91))

    monkeypatch.setattr("recall.trust.HybridRetriever", _Counting)
    result = trusted_search(
        _NoEdges(),
        object(),
        "what is the current cache ttl",
        k=5,
        calibration=CALIBRATION,
        policy=TrustPolicy.development(),
        now=NOW,
        successor_expansion=SuccessorExpansionPolicy(enabled=True),
    )

    assert calls == [None]
    assert not result.abstained
    assert "successor_expansion" not in result.diagnostics.stage_ms


# --- ordering of promoted successors ------------------------------------------------------------
#
# Registered in docs/preregistrations/2026-08-20-successor-ordering-regression.md. Fetching alone is
# not enough: measured, the successor is promoted and then lands at rank 5 behind distractors,
# because `evaluate` preserves pool position and the fetched chunk is appended last.


def _ordering_case(ordering: str, predecessor_rank: int) -> list[str]:
    """One `ok` distractor, a superseded predecessor, and its fetched successor appended last.

    `predecessor_rank` is the predecessor's pool position. At 0 it outranks the distractor; at 2 it
    does not, which is the case that separates the two orderings.
    """
    distractor = _scored("d1", "distractor.md", 0.88)
    stale = _scored("s1", "ttl_v1.md", 0.86)
    successor = _scored("c2", "ttl_v2.md", 0.64)
    pool = [stale, distractor, successor] if predecessor_rank == 0 else [distractor, stale, successor]
    trusted = evaluate(_retrieval(*pool), {"ttl_v1.md": "ttl_v2.md"}, CALIBRATION, NOW)
    ordered = order_promoted(
        trusted, {hit.chunk.id: i for i, hit in enumerate(pool)}, ordering
    )
    return [h.provenance.file or "" for h in ordered.hits if h.verdict == "ok"]


def test_pool_ordering_leaves_the_successor_behind_the_distractor() -> None:
    """The shipped behaviour, and the reason this record exists."""
    assert _ordering_case("pool", predecessor_rank=0) == ["distractor.md", "ttl_v2.md"]


def test_promoted_first_puts_the_successor_ahead_even_when_its_predecessor_did_not_lead() -> None:
    """The unconditional ordering. Note it wins rank 1 from a predecessor that was only second."""
    assert _ordering_case("promoted_first", predecessor_rank=2) == ["ttl_v2.md", "distractor.md"]


def test_inherit_gives_the_successor_the_rank_its_predecessor_actually_held() -> None:
    """The distinguishing case, and the whole argument for `inherit`.

    The predecessor was SECOND for this query, so it is no evidence that its successor is first.
    `promoted_first` asserts that anyway and displaces the distractor; `inherit` does not.
    """
    assert _ordering_case("inherit", predecessor_rank=2) == ["distractor.md", "ttl_v2.md"]


def test_inherit_does_promote_when_the_predecessor_led() -> None:
    """`inherit` is not a no-op: where the stale memory WAS best, its successor becomes best."""
    assert _ordering_case("inherit", predecessor_rank=0) == ["ttl_v2.md", "distractor.md"]


def test_ordering_never_touches_the_demoted_hits() -> None:
    """`rest` is demoted material and its order is not a claim about anything."""
    pool = [
        _scored("d1", "distractor.md", 0.88),
        _scored("s1", "ttl_v1.md", 0.86),
        _scored("c2", "ttl_v2.md", 0.64),
        _scored("w1", "weak.md", 0.10),
    ]
    trusted = evaluate(_retrieval(*pool), {"ttl_v1.md": "ttl_v2.md"}, CALIBRATION, NOW)
    index = {hit.chunk.id: i for i, hit in enumerate(pool)}
    before = [h.provenance.file for h in trusted.hits if h.verdict != "ok"]
    after = [
        h.provenance.file
        for h in order_promoted(trusted, index, "promoted_first").hits
        if h.verdict != "ok"
    ]

    assert before == after


def test_an_unknown_ordering_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="ordering must be one of"):
        SuccessorExpansionPolicy(enabled=True, ordering="best")  # type: ignore[arg-type]


def test_trusted_search_fetches_the_successor_and_stops_abstaining(monkeypatch) -> None:
    result = _search(
        monkeypatch,
        successor_expansion=SuccessorExpansionPolicy(enabled=True, max_sources=1),
    )

    assert not result.abstained
    assert result.hits[0].provenance.file == "ttl_v2.md"
    assert result.hits[0].verdict == "ok"
    assert result.diagnostics.stage_ms["successor_expansion_sources"] == 1.0
    # The stale memory is still in the result and still refused. Recovering an answer must not be
    # bought by serving the memory the corpus declared superseded.
    stale = [h for h in result.hits if h.provenance.file == "ttl_v1.md"]
    assert [h.verdict for h in stale] == ["superseded"]
