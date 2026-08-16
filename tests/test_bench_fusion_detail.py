"""The fusion detail a benchmark can capture must be the fusion `search()` actually performed.

`search()` throws away the two quantities the triage mechanism probe needs: the RRF score that
ordered the list, and which legs found each chunk. `benchmarks.fusion_detail` recovers them from
the shared `_Legs` seam, which means it necessarily restates the few lines of `search()` that turn
legs into a ranked list.

⚠️ That restatement is the whole risk. Two copies of a fusion pipeline drift, and the drift is
invisible because both keep returning plausible hits — the reason `_Legs` was extracted in the
first place. So the first test here is not about the new fields at all: it pins the helper's
ranked output to `search()`'s, hit for hit and score for score. If `search()` changes how it
fuses, this goes red before a seven-hour retrieval run bakes the old ordering into a fixture.
"""

from __future__ import annotations

import pytest

from benchmarks.fusion_detail import LEG_NAMES, dense_gap_warning, fuse
from recall.retriever import HybridRetriever
from recall.sparse import SparseProfile
from tests.fakes import FakeEmbedder, FakeStore

PROFILE_ID = "test/keyword@sha256:test"


class KeywordSparseEncoder:
    """A real, deterministic encoder for the learned-sparse leg. One term id per known word."""

    def __init__(self, vocabulary: dict[str, int]) -> None:
        self._vocabulary = vocabulary
        self.profile = SparseProfile(
            profile_id=PROFILE_ID, model_name="test/keyword",
            artifact_digest="sha256:test", dimension=30522, top_k=1000,
        )

    def encode(self, texts: list[str]) -> list[dict[int, float]]:
        return [
            {self._vocabulary[w]: 1.0 for w in text.lower().split() if w in self._vocabulary}
            for text in texts
        ]


def _retriever(store: FakeStore) -> HybridRetriever:
    return HybridRetriever(
        store,
        FakeEmbedder(),
        candidate_k=200,
        sparse_backend="both",
        sparse_encoder=KeywordSparseEncoder({"alpha": 1, "beta": 2}),
    )


def _store() -> FakeStore:
    """Three legs that disagree, which is the only configuration where fusion order is testable.

    `c` is in all three legs, `a` leads dense only, `d` is learned-only — so the ranked order can
    distinguish "sorted by dense score" from "sorted by RRF", and a helper that accidentally did
    the former would fail rather than coincide.
    """
    return FakeStore(
        dense=[("a", 0.91), ("b", 0.80), ("c", 0.72)],
        sparse=[("c", 0.50), ("b", 0.40)],
        learned=[("d", 0.30), ("c", 0.20)],
    )


def test_ranked_order_and_scores_match_what_search_returns() -> None:
    """🔑 The anti-drift guard. The helper must reproduce `search()` exactly, unreranked.

    Mutation-checked: sorting by dense score instead of the fused score, recording the position
    instead of the RRF value, and writing 0 for an absent leg all go red here or in the tests
    below. One mutant survives and is EQUIVALENT rather than uncaught: dropping the `_rescored`
    call changes nothing, because `by_id` already prefers the dense hit and the fallback is that
    hit's own score, so the rescore assigns a value the hit already carries. That is true of
    `search()` itself, not just of this mirror.
    """
    store = _store()
    retriever = _retriever(store)
    query = "alpha beta"

    served = retriever.search(query, k=4)
    captured = fuse(retriever._retrieve_legs(query, source=None))[:4]

    assert [h.chunk.id for h in served.hits] == [f.hit.chunk.id for f in captured]
    assert [h.score for h in served.hits] == [f.hit.score for f in captured]


def test_fused_score_is_the_rrf_definition_summed_over_the_legs_that_found_it() -> None:
    """A5 in miniature: the captured number must satisfy sum(1 / (60 + rank + 1)), not merely
    order the list plausibly. A capture that recorded the POSITION would pass the order test
    above and still be the wrong quantity."""
    captured = {f.hit.chunk.id: f for f in fuse(_retriever(_store())._retrieve_legs("alpha", source=None))}

    # `c` is rank 2 in dense, rank 0 in sparse, rank 1 in learned.
    expected_c = 1 / (60 + 3) + 1 / (60 + 1) + 1 / (60 + 2)
    assert captured["c"].fused_score == pytest.approx(expected_c, abs=1e-12)
    # `a` is dense-only, at rank 0.
    assert captured["a"].fused_score == pytest.approx(1 / (60 + 1), abs=1e-12)


def test_per_leg_ranks_are_zero_based_and_none_where_the_leg_missed_the_chunk() -> None:
    """The mechanism hypothesis is about leg AGREEMENT, so an absent leg must be distinguishable
    from a leg that ranked the chunk first. `None` and `0` are not interchangeable here."""
    captured = {f.hit.chunk.id: f for f in fuse(_retriever(_store())._retrieve_legs("alpha", source=None))}

    assert LEG_NAMES == ("dense", "lexical", "learned")
    assert captured["c"].ranks == (2, 0, 1)
    assert captured["a"].ranks == (0, None, None)
    assert captured["d"].ranks == (None, None, 0)


def test_legs_hit_counts_the_legs_that_returned_the_chunk() -> None:
    captured = {f.hit.chunk.id: f for f in fuse(_retriever(_store())._retrieve_legs("alpha", source=None))}

    assert captured["c"].legs_hit == 3
    assert captured["b"].legs_hit == 2
    assert captured["d"].legs_hit == 1


def test_gap_warning_matches_search_even_when_the_dense_leg_repeats_an_id() -> None:
    """`search()` reads the gap off a {chunk_id: score} dict, so a repeated id keeps the LAST
    score. Computing it from the raw leg keeps the highest and silently reports no gap. The
    duplicate is chosen so the two answers differ: deduped {a: 0.1} is a gap, raw [0.9, 0.1]
    is not."""
    store = FakeStore(dense=[("a", 0.9), ("a", 0.1)], sparse=[("b", 0.4)])
    retriever = _retriever(store)
    query = "alpha"

    served = retriever.search(query, k=4)

    assert dense_gap_warning(retriever._retrieve_legs(query, source=None)) is served.gap_warning
    assert served.gap_warning is True  # the scenario is only meaningful if the dedup wins


def test_an_empty_pool_fuses_to_an_empty_list_rather_than_raising() -> None:
    """A query whose every leg comes back empty is legitimate (a pure-stopword query encodes to
    no sparse terms). It must produce no rows, not an IndexError inside the capture."""
    retriever = _retriever(FakeStore())

    assert fuse(retriever._retrieve_legs("zzz", source=None)) == []
