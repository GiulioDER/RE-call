"""`recall.eval.labelled --table`: a named index that survives the run.

Without it, `evaluate` builds into `lab_<uuid>` and drops it in `finally` — correct for a small
private corpus, ruinous for LongMemEval, where the merged-S index alone costs ~6h39m to embed
(FINDINGS §10). A named table lets one build serve the merged `labelled` arm, the
`longmemeval_perq --master` arm, and any future re-score: `Indexer` skips by content hash, so a
second run over a populated table is a no-op instead of a re-embed.
"""
from __future__ import annotations

import uuid

from recall.embeddings import HashingEmbedder
from recall.eval.labelled import evaluate
from recall.store import PgVectorStore

from .conftest import TEST_DSN, requires_db


def _corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a.md").write_text("the deploy target is staging", encoding="utf-8")
    (corpus / "b.md").write_text("the database backup runs nightly", encoding="utf-8")
    (corpus / "c.md").write_text("the oncall rotation is weekly", encoding="utf-8")
    (corpus / "d.md").write_text("the incident review is on friday", encoding="utf-8")
    return corpus


_QUESTIONS = [
    {"id": "q1", "query": "deploy target", "relevant_files": ["a.md"], "answerable": True},
    {"id": "q2", "query": "database backup", "relevant_files": ["b.md"], "answerable": True},
    {"id": "q3", "query": "oncall rotation", "relevant_files": ["c.md"], "answerable": True},
    {"id": "q4", "query": "incident review", "relevant_files": ["d.md"], "answerable": True},
    {"id": "q5", "query": "airspeed of a swallow", "relevant_files": [], "answerable": False},
    {"id": "q6", "query": "unrelated nonsense xyzzy", "relevant_files": [], "answerable": False},
]


@requires_db
def test_named_table_survives_the_run_and_a_rerun_reuses_it(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    emb = HashingEmbedder(dim=64)
    table = "lab_keep_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=emb.dim, table=table)
    try:
        first = evaluate(TEST_DSN, corpus, _QUESTIONS, emb, k=3, table=table)
        assert first["corpus"]["table"] == table
        # The whole point: the index is still there after evaluate returns.
        assert store.count() == first["corpus"]["chunks"] > 0

        # A second run over the same table must reuse rows (content-hash skip), not duplicate
        # them — equality of the chunk count is what pins that.
        second = evaluate(TEST_DSN, corpus, _QUESTIONS, emb, k=3, table=table)
        assert second["corpus"]["chunks"] == first["corpus"]["chunks"]
        assert store.count() == first["corpus"]["chunks"]
        # Scores must not depend on which run built the index.
        assert second["arms"]["hybrid"]["hit_at_3"] == first["arms"]["hybrid"]["hit_at_3"]
    finally:
        store.drop_table()
        store.close()


def _lab_tables(store: PgVectorStore) -> set[str]:
    with store._connect() as conn:  # noqa: SLF001 - test-only probe
        rows = conn.execute(
            "SELECT tablename FROM pg_tables WHERE tablename LIKE 'lab_%'"
        ).fetchall()
    return {r[0] for r in rows}


@requires_db
def test_default_run_still_drops_its_table(tmp_path) -> None:
    corpus = _corpus(tmp_path)
    probe = PgVectorStore(TEST_DSN, dim=64, table="lab_probe_" + uuid.uuid4().hex[:8])
    try:
        before = _lab_tables(probe)
        rep = evaluate(TEST_DSN, corpus, _QUESTIONS, HashingEmbedder(dim=64), k=3)
        # The anonymous path must keep dropping: no new lab_* table survives, none is reported.
        assert "table" not in rep["corpus"]
        assert _lab_tables(probe) == before
    finally:
        probe.close()
