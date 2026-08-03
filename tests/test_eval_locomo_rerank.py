"""The rerank arm for LOCOMO.

§9a measures hit@5 0.671 against hit@20 0.855: for 85.5% of questions the right memory is already
retrieved and merely ranked below position 5. Closing that gap is what a cross-encoder does, and
this harness has never been run with one.

These tests pin the wiring, not the benchmark. A reranker that is accepted and then dropped would
produce a plausible "no effect" result — which is exactly the finding the experiment is trying to
measure, so the wiring has to be proven separately from the outcome.
"""
from __future__ import annotations

from pathlib import Path

import psycopg
import pytest

from recall.embeddings import HashingEmbedder
from recall.eval.locomo import run_conversation
from recall.store import PgVectorStore
from recall.types import ScoredChunk
from tests.conftest import TEST_DSN, requires_db


class RecordingReranker:
    """Passes hits through untouched, and records what it was asked to rank."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, int]] = []

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        self.calls.append((query, len(hits)))
        return hits


def _fresh(table: str) -> PgVectorStore:
    """A store on an empty table.

    Tests share a database and `run_conversation` now refuses to index over existing rows — rows a
    previous run of this same test left behind included. Dropping first keeps these assertions
    about reranking rather than about test order.
    """
    with psycopg.connect(TEST_DSN, autocommit=True) as conn:
        conn.execute(f"DROP TABLE IF EXISTS {table} CASCADE")
        conn.execute(
            "DELETE FROM recall_schema_migrations WHERE target_table = %s", (table,)
        )
    store = PgVectorStore(TEST_DSN, dim=64, table=table)
    store.ensure_schema()
    return store


def _conversation() -> dict:
    turns = [
        {"speaker": "Caroline", "dia_id": f"D1:{i}", "text": t}
        for i, t in enumerate(
            [
                "I adopted a rescue greyhound called Pepper.",
                "The vet said her teeth needed cleaning.",
                "We walked six kilometres along the canal.",
                "I started a pottery class on Thursdays.",
                "The kiln at the studio broke last week.",
                "My sister moved to Lisbon in March.",
                "She sent photos of the tiled facades.",
                "I am learning Portuguese with an app.",
                "The apartment has a balcony full of herbs.",
                "I cooked cataplana for the first time.",
            ],
            start=1,
        )
    ]
    return {
        "speaker_a": "Caroline",
        "speaker_b": "Mel",
        "session_1_date_time": "1 Jan 2023",
        "session_1": turns,
    }


def _qa() -> list[dict]:
    return [{"question": "What is the name of the rescue dog?", "category": 1, "evidence": ["D1:1"]}]


@requires_db
def test_run_conversation_without_a_reranker_leaves_the_ranking_untouched(
    tmp_path: Path, monkeypatch: "pytest.MonkeyPatch"
) -> None:
    """The baseline arm must not silently acquire a reranker.

    Asserted against what `run_conversation` actually passes DOWN the retrieval path. The
    previous version of this test constructed a `RecordingReranker`, never handed it to
    anything, and then asserted its call list was empty — which is true of any object given to
    nobody, whatever the code under test does. It was the only guard on this property and it
    could not fail.
    """
    import recall.eval.locomo as locomo_module

    # Intercept the object the retriever is actually CONSTRUCTED with — that is where a reranker
    # would leak into the baseline arm, and it is reached on every question rather than only on
    # the abstention path.
    seen: list[object] = []
    original = locomo_module.HybridRetriever

    def _spy(*args: object, **kwargs: object):
        seen.append(kwargs.get("reranker"))
        return original(*args, **kwargs)

    monkeypatch.setattr(locomo_module, "HybridRetriever", _spy)
    with _fresh("lab_rr_none") as store:
        run_conversation(
            _conversation(), _qa(), store=store, embedder=HashingEmbedder(dim=64),
            k=5, corpus_dir=tmp_path / "corpus", ks=[1, 5], candidate_k=20,
        )
    assert seen, "the retrieval path was never reached — this test would prove nothing"
    assert all(r is None for r in seen), f"baseline arm acquired a reranker: {seen}"


@requires_db
def test_run_conversation_passes_the_reranker_into_the_retrieval_path(tmp_path: Path) -> None:
    """A reranker accepted and then dropped would report 'no effect' — the very outcome under
    test. So the wiring is pinned by counting calls, not by watching the metric move."""
    recorder = RecordingReranker()
    with _fresh("lab_rr_wired") as store:
        run_conversation(
            _conversation(), _qa(), store=store, embedder=HashingEmbedder(dim=64),
            k=5, corpus_dir=tmp_path / "corpus", ks=[1, 5], candidate_k=20,
            reranker=recorder,
        )
    assert len(recorder.calls) >= 1, "reranker was never called — it is not in the retrieval path"


@requires_db
def test_the_reranker_sees_the_whole_pool_not_the_truncated_top_k(tmp_path: Path) -> None:
    """The load-bearing property. Reranking a list already cut to k cannot raise hit@k — it only
    reorders documents that already counted as hits. The gain this experiment is looking for comes
    entirely from documents ranked BELOW k, so the reranker must receive more than k candidates."""
    recorder = RecordingReranker()
    with _fresh("lab_rr_pool") as store:
        run_conversation(
            _conversation(), _qa(), store=store, embedder=HashingEmbedder(dim=64),
            k=1, corpus_dir=tmp_path / "corpus", ks=[1], candidate_k=20,
            reranker=recorder,
        )
    assert recorder.calls, "reranker was never called"
    assert max(n for _, n in recorder.calls) > 1, (
        "the reranker only ever saw k candidates — it is being applied after truncation, "
        "which cannot move hit@k"
    )
