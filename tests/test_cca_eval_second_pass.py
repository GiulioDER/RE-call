"""Regression tests for the CCA DEEP second-pass fixes (EVAL-001, SEC-001, CODE-001, PERF-001).

Pure-unit (no database) so they run in any environment; the DB-backed eval paths keep their
existing @requires_db coverage.
"""
from __future__ import annotations

from datetime import timedelta

import pytest

from recall.types import Chunk, RetrievalResult, ScoredChunk, StalenessReport


class _StubRetriever:
    """Returns hits in RRF (fusion) order — NOT sorted by cosine — so hits[0].score != max."""

    def __init__(self, scores: list[float]) -> None:
        self._scores = scores

    def search(self, query: str, k: int) -> RetrievalResult:
        hits = [
            ScoredChunk(chunk=Chunk(id=str(i), source="s", text="t"), score=s)
            for i, s in enumerate(self._scores)
        ]
        return RetrievalResult(
            query=query,
            hits=hits,
            gap_warning=False,
            staleness=StalenessReport(
                stale=False, newest_indexed_at=None, age=None, max_age=timedelta(days=2)
            ),
        )


def test_top_cosine_uses_max_over_hits_not_fused_rank0() -> None:  # EVAL-001
    from recall.eval.locomo_abstention import _top_cosine

    # rank-0 (fused winner) carries a LOWER cosine than a lower-ranked hit — the exact case the
    # abstention decision (max-over-hits >= threshold) turns on. Pre-fix returned 0.40 (hits[0]).
    assert _top_cosine(_StubRetriever([0.40, 0.91, 0.55]), "q", k=3) == 0.91


def test_top_cosine_none_when_no_hits() -> None:
    from recall.eval.locomo_abstention import _top_cosine

    assert _top_cosine(_StubRetriever([]), "q", k=3) is None


def test_dia_id_traversal_is_rejected() -> None:  # SEC-001
    from recall.eval.locomo import _dia_id_to_filename

    for evil in ["../../evil", "/etc/cron.d/x", "a/b", "..", "a\\b"]:
        with pytest.raises(ValueError):
            _dia_id_to_filename(evil)


def test_dia_id_legitimate_value_still_encodes() -> None:
    from recall.eval.locomo import _dia_id_to_filename

    assert _dia_id_to_filename("D1:3") == "D1_3.md"


def test_check_corpus_name_guards_sample_id() -> None:
    from recall.eval.locomo import _check_corpus_name

    assert _check_corpus_name("conv12", "sample_id") == "conv12"
    with pytest.raises(ValueError):
        _check_corpus_name("../../../etc", "sample_id")


def test_entailment_sweep_reuses_locomo_rate() -> None:  # CODE-001
    from recall.eval import locomo, locomo_entailment_sweep

    assert locomo_entailment_sweep._rate is locomo._rate


class _FakeStore:
    def __init__(self, texts: list[str]) -> None:
        self._texts = texts

    def iter_chunks(self):
        for i, t in enumerate(self._texts):
            yield Chunk(id=str(i), source="s", text=t)


def test_bm25_scores_only_documents_containing_the_term() -> None:  # PERF-001
    from recall.eval.bm25 import BM25Retriever

    scores = BM25Retriever(
        _FakeStore(["alpha beta", "alpha alpha gamma", "delta epsilon"])
    ).score("alpha")
    assert scores[0] > 0.0 and scores[1] > 0.0  # both contain "alpha"
    assert scores[2] == 0.0  # "delta epsilon" does not
    assert scores[1] > scores[0]  # more occurrences -> higher score


def test_bm25_empty_and_unknown_terms_score_zero() -> None:
    from recall.eval.bm25 import BM25Retriever

    r = BM25Retriever(_FakeStore(["alpha beta", "gamma"]))
    assert r.score("") == [0.0, 0.0]
    assert r.score("zzzz") == [0.0, 0.0]


def test_bm25_search_ranks_and_respects_k() -> None:
    from recall.eval.bm25 import BM25Retriever

    res = BM25Retriever(_FakeStore(["alpha beta", "alpha alpha gamma", "delta"])).search(
        "alpha", k=2
    )
    assert [h.chunk.id for h in res.hits] == ["1", "0"]  # doc1 (2x alpha) ranks above doc0
    assert len(res.hits) == 2
