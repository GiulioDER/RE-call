"""`run_conversation` must verify the corpus it just built, not just the one it found.

The shipped guard at recall/eval/locomo.py:274 is a PRE-condition: it reads store.count()
before indexing and refuses if rows are already there. That catches a sequential re-run.

It does not catch what actually happened on 2026-07-27: two CONCURRENT launchers. The second
one passes the pre-check (the table was empty when it looked) and writes while the first is
still indexing. Both finish, every tenant holds its corpus twice, nothing errors, and every
depth of the curve comes in ~0.05 low — plausible, self-consistent and wrong.

The post-condition is exact and cheap: Indexer.index_path returns IndexStats.chunks, which is
how many chunks THIS call wrote. On a fresh tenant the tenant-scoped row count must equal it.
Anything else means someone else wrote here.
"""
from __future__ import annotations

import pytest

from recall.embeddings import HashingEmbedder
from recall.eval.locomo import run_conversation
from recall.types import Chunk

from tests.conftest import requires_db

DIM = 64

_CONVERSATION = {
    "speaker_a": "Caroline",
    "speaker_b": "Melanie",
    "session_1_date_time": "1:00 pm on 8 May, 2023",
    "session_1": [
        {"speaker": "Caroline", "dia_id": "D1:1", "text": "I finally adopted a greyhound."},
        {"speaker": "Melanie", "dia_id": "D1:2", "text": "I signed up for a pottery class."},
    ],
}
_QA = [{"question": "What did Caroline adopt?", "category": 1, "evidence": ["D1:1"]}]


@requires_db
def test_run_conversation_reports_the_rows_it_indexed(make_store, tmp_path):
    """The happy path: the count is reported, and it is the count that was written."""
    store = make_store(DIM)
    res = run_conversation(
        _CONVERSATION, _QA,
        store=store, embedder=HashingEmbedder(dim=DIM), k=5, corpus_dir=tmp_path / "corpus",
    )
    assert res["corpus_rows"] > 0, "the run reported no corpus at all"
    assert res["corpus_rows"] == store.count(), (
        "corpus_rows must be the tenant's actual row count, not a number carried from elsewhere"
    )


@requires_db
def test_a_concurrent_writer_fails_the_run_instead_of_depressing_it(
    make_store, tmp_path, monkeypatch
):
    """A second writer landing DURING indexing must fail the run, not skew it.

    Simulated by writing extra rows into the same tenant from inside index_path, which is
    exactly the window the pre-condition cannot see.
    """
    store = make_store(DIM)
    embedder = HashingEmbedder(dim=DIM)

    from recall.index import Indexer

    real_index_path = Indexer.index_path

    def racing_index_path(self, *args, **kwargs):
        stats = real_index_path(self, *args, **kwargs)
        # The "other launcher", arriving after the pre-check and before the post-check.
        store.upsert(
            [Chunk(id="intruder", source="intruder.md", text="another run wrote this", metadata={})],
            [[0.1] * DIM],
        )
        return stats

    monkeypatch.setattr(Indexer, "index_path", racing_index_path)

    with pytest.raises(RuntimeError, match="CONCURRENTLY"):
        run_conversation(
            _CONVERSATION, _QA,
            store=store, embedder=embedder, k=5, corpus_dir=tmp_path / "corpus",
        )
