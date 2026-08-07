"""`search_fused` refuses rather than degrades.

The measured gain is CONDITIONAL on reranking: raw, this arm is worse by 0.0447 nDCG@5 and
tripped three preregistered ranking vetoes. RE-call ships with the reranker OFF by default, so an
operator enabling fusion without one would silently get a worse system than `search()`.
"""

from __future__ import annotations

import pytest

from recall.retriever import HybridRetriever
from tests.fakes import FakeEmbedder, FakeStore


class _StubReranker:
    def rerank(self, query, hits):
        return list(hits)


def _retriever(reranker=None):
    return HybridRetriever(
        FakeStore(dense=[("a", 0.9)], sparse=[("b", 0.4)]),
        FakeEmbedder(),
        reranker=reranker,
        sparse_backend="lexical",
    )


def test_fused_search_without_a_reranker_is_refused() -> None:
    """The gain does not exist without reranking, so serving it without one is a worse system."""
    with pytest.raises(ValueError, match="reranker"):
        _retriever().search_fused("q", ["earlier turn"])


def test_empty_history_is_refused_rather_than_silently_becoming_a_single_query() -> None:
    """A caller wanting single-query behaviour should call `search`, not get it by accident.

    Matched on "non-empty history", text unique to this guard's message. A later guard's message
    also contains the word "history" (it fires on a history that stripped to no usable text), so
    matching on that word alone would still pass if this guard were deleted and execution fell
    through to the later one.
    """
    with pytest.raises(ValueError, match="non-empty history"):
        _retriever(_StubReranker()).search_fused("q", [])


def test_an_over_budget_history_is_refused_and_names_both_lengths() -> None:
    """Refused, never truncated.

    A truncated history is a configuration the benchmark never tested, served under the measured
    configuration's name. Same principle as `resolve_reranker` refusing an unparseable flag rather
    than reading it as "off".
    """
    with pytest.raises(ValueError, match="4096"):
        _retriever(_StubReranker()).search_fused("q", ["x" * 5000])


def test_k_below_one_is_refused() -> None:
    """The message must name the actual value, not just the expected one, so a caller passing

    `k=-3` and a caller passing `k=0` get distinguishable text rather than identical, ungreppable
    ones.
    """
    with pytest.raises(ValueError, match="k must be >= 1") as exc_info:
        _retriever(_StubReranker()).search_fused("q", ["earlier"], k=0)
    assert "got 0" in str(exc_info.value)


def test_a_bare_string_history_is_refused_rather_than_iterated_character_by_character() -> None:
    """A bare `str` satisfies `Sequence[str]` at runtime, so the type annotation alone does not

    stop a caller from passing one turn as a string instead of a one-element list. Without a
    guard, `build_history_query` would iterate it character by character and join each character
    with a newline, and retrieval would silently proceed on that garbage.
    """
    with pytest.raises(ValueError, match="list|sequence"):
        _retriever(_StubReranker()).search_fused("q", "an earlier turn")


def test_every_returned_score_is_the_cosine_against_the_query() -> None:
    """The reason `cosines_for` exists.

    A hit surfaced only by the HISTORY variant carries a cosine against the history. `trust.py`
    feeds `hit.score` to `cal.confidence()`, a calibration fitted on cosines against the QUERY, so
    a mixed basis produces a confidence that silently means something else.
    """
    store = FakeStore(dense=[("a", 0.9)], sparse=[("b", 0.4)])
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_StubReranker(), sparse_backend="lexical"
    )

    result = retriever.search_fused("q", ["earlier turn"], k=5)

    assert result.hits, "expected hits"
    assert all(hit.score == 0.77 for hit in result.hits)  # FakeStore.cosines_for returns 0.77
    assert store.cosines_calls, "cosines_for was never called"
    ids, vec = store.cosines_calls[-1]
    assert set(ids) == {hit.chunk.id for hit in result.hits}
    assert list(vec) == retriever._retrieve_legs("q", source=None).qvec


def test_the_pool_is_capped_before_reranking() -> None:
    """Reranking 547 candidates lost 0.0513 R@100 where 200 gained 0.0226.

    Fusion roughly doubles the pool, so an absent cap is a measured regression, not an
    inefficiency.
    """
    seen: dict[str, int] = {}

    class _CountingReranker:
        def rerank(self, query, hits):
            seen["count"] = len(hits)
            return list(hits)

    rows = [(f"d{i}", 0.5) for i in range(150)]
    store = FakeStore(dense=rows, sparse=[(f"s{i}", 0.4) for i in range(150)])
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_CountingReranker(),
        candidate_k=150, sparse_backend="lexical",
    )

    retriever.search_fused("q", ["earlier"], k=5)

    assert seen["count"] == 100


def test_gap_warning_uses_the_query_not_the_history() -> None:
    """The honesty guard must answer "does the corpus hold an answer to what the USER asked".

    Letting a strong match on stale earlier context suppress the warning is the guard failing in
    the dangerous direction.
    """
    store = FakeStore(dense=[("a", 0.01)], sparse=[])
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_StubReranker(), sparse_backend="lexical",
        gap_threshold=0.5,
    )

    assert retriever.search_fused("q", ["earlier"], k=5).gap_warning is True


def test_the_result_carries_the_query_not_the_concatenation() -> None:
    store = FakeStore(dense=[("a", 0.9)], sparse=[])
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_StubReranker(), sparse_backend="lexical"
    )

    assert retriever.search_fused("the question", ["earlier"], k=5).query == "the question"


def test_diagnostics_report_the_realised_pool_and_the_fusion_stages() -> None:
    """`candidate_pool_size` is the REALISED fused pool, not the configured candidate_k.

    The benchmark learned this distinction the hard way: its `pool_bound()` overstated the
    realised pool by 3x on 307 of 777 queries and no score in the output revealed it.
    """
    store = FakeStore(dense=[("a", 0.9), ("b", 0.8)], sparse=[("c", 0.4)])
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_StubReranker(), candidate_k=50,
        sparse_backend="lexical",
    )

    diagnostics = retriever.search_fused("q", ["earlier"], k=5).diagnostics

    assert diagnostics.reranking_ran is True
    assert diagnostics.candidate_pool_size == 3  # a, b, c: realised, not 50
    assert "history_retrieval" in diagnostics.stage_ms
    assert "outer_fusion" in diagnostics.stage_ms


class _StubSparseProfile:
    profile_id = "stub-splade"


class _StubSparseEncoder:
    """A minimal learned-sparse encoder: fixed non-empty weights, no model to load.

    Exists to cover the SECOND `report_vec` substitution site (`query_learned_sparse`), which
    `test_every_returned_score_is_the_cosine_against_the_query` above does not exercise because it
    runs with `sparse_backend="lexical"`, the brief's default for every test. The measured arm
    uses `sparse_backend="splade"`, so the learned-sparse leg needs its own coverage that the
    QUERY's vector, not the history's, is what it reports its cosine against.
    """

    profile = _StubSparseProfile()

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        return [{1: 1.0} for _ in texts]


def test_splade_backend_reports_the_learned_leg_against_the_query_not_the_history() -> None:
    """The `report_vec` substitution happens at two call sites; this covers the second.

    If `search_fused` ever regressed to letting the HISTORY variant's own call report its own
    vector (rather than the QUERY's, passed in as `report_vec`), the learned leg's reported basis
    would silently become the history's cosine, the same hazard `cosines_for` exists to close for
    the returned hits, but here in the leg's own reporting.
    """
    store = FakeStore(dense=[("a", 0.9)], sparse=[], learned=[("c", 0.6)])
    retriever = HybridRetriever(
        store,
        FakeEmbedder(),
        reranker=_StubReranker(),
        sparse_backend="splade",
        sparse_encoder=_StubSparseEncoder(),
    )

    retriever.search_fused("q", ["earlier turn"], k=5)

    assert store.learned_vec_used == retriever._retrieve_legs("q", source=None).qvec


class _VanishingStore(FakeStore):
    """Like `FakeStore`, but `cosines_for` omits one id, simulating a chunk deleted between
    retrieval and the final rescore.

    `cosines_for`'s own docstring (`recall/store.py`) says ids that no longer exist are OMITTED
    rather than reported as 0.0 cosine, "so the caller can tell". This fake makes that omission
    reproducible without a real database race.
    """

    def __init__(self, *args, missing_id: str, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._missing_id = missing_id

    def cosines_for(self, ids, vec):
        self.cosines_calls.append((tuple(ids), tuple(vec)))
        return {cid: 0.77 for cid in ids if cid != self._missing_id}


def test_a_hit_missing_from_the_final_rescore_is_dropped_not_served_stale() -> None:
    """`cosines_for` omits ids for chunks gone by rescore time so the caller can tell and act.

    Falling back to the hit's pre-rescore score on a miss would silently defeat that: the
    pre-rescore score can still be a cosine against the HISTORY, since the dense leg (unlike the
    sparse legs) has no `report_vec` override and always reports against its own leg's vector. A
    hit whose id vanished between retrieval and rescore must be dropped, not served with a score
    that might be on the wrong basis.
    """
    store = _VanishingStore(dense=[("a", 0.9)], sparse=[("b", 0.4)], missing_id="b")
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_StubReranker(), sparse_backend="lexical"
    )

    result = retriever.search_fused("q", ["earlier turn"], k=5)

    ids = {hit.chunk.id for hit in result.hits}
    assert "b" not in ids
    assert "a" in ids


def test_history_identical_to_the_query_matches_plain_search_below_the_cap() -> None:
    """Nested RRF over two copies of one ranking is order-preserving.

    The same structural property the benchmark's 102 turn-1 queries rested on: identical text
    gives identical leg rankings, and fusing a ranking with itself cannot reorder it.
    """
    rows = [(f"d{i}", 0.9 - i * 0.01) for i in range(10)]
    store = FakeStore(dense=rows, sparse=[(f"s{i}", 0.4) for i in range(10)])
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_StubReranker(), candidate_k=20,
        sparse_backend="lexical",
    )

    plain = [h.chunk.id for h in retriever.search("q", k=10).hits]
    fused = [h.chunk.id for h in retriever.search_fused("q", ["q"], k=10).hits]

    assert fused == plain


def test_above_the_cap_the_two_legitimately_diverge() -> None:
    """Asserted rather than hidden.

    `search` reranks its whole pool; `search_fused` caps at 100 because reranking a wider pool was
    measured to LOSE coverage. Above the cap the invariant above does not hold, and pretending it
    did would be an untrue claim in a passing test.
    """
    from recall.retriever import FUSED_RERANK_POOL_CAP

    rows = [(f"d{i}", 0.9) for i in range(FUSED_RERANK_POOL_CAP + 40)]
    store = FakeStore(dense=rows, sparse=[])
    retriever = HybridRetriever(
        store, FakeEmbedder(), reranker=_StubReranker(),
        candidate_k=FUSED_RERANK_POOL_CAP + 40, sparse_backend="lexical",
    )

    plain = retriever.search("q", k=1000).hits
    fused = retriever.search_fused("q", ["q"], k=1000).hits

    assert len(plain) > FUSED_RERANK_POOL_CAP
    assert len(fused) == FUSED_RERANK_POOL_CAP
