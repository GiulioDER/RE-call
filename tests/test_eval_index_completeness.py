"""A kept index must prove it is complete before anything scores against it.

`--table` made an index outlive its run, which is what makes a long benchmark affordable. It also
created a failure mode that did not exist while every table was dropped: Postgres is crash-safe,
so a reset mid-build never yields corrupt rows — it yields **fewer** rows. The committed part
survives, the build stops, and a later run against that table returns real-but-low numbers with
nothing anywhere reporting an error.

`labelled` repairs that by itself (a source absent from the table has no stored content hash, so
the next run re-indexes it). `longmemeval_perq` cannot: it never indexes, it only copies rows out
of `--master`. A partial master silently shrinks the per-question haystack, which changes the
score in whichever direction the missing sessions happened to matter. That is the path these
tests pin.
"""
from __future__ import annotations

import uuid

import pytest

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.store import PgVectorStore

from .conftest import TEST_DSN, requires_db


def _corpus(tmp_path, names):
    corpus = tmp_path / "corpus"
    corpus.mkdir(exist_ok=True)
    for i, n in enumerate(names):
        (corpus / n).write_text(f"session {n} discusses topic number {i} at length", encoding="utf-8")
    return corpus


@requires_db
def test_perq_refuses_a_master_missing_haystack_sessions(tmp_path) -> None:
    from recall.eval.longmemeval_perq import evaluate as perq_evaluate

    # The master is built from only two of the three sessions the questions reference — exactly
    # the shape a crash mid-index leaves behind.
    built = _corpus(tmp_path, ["s1.md", "s2.md"])
    emb = HashingEmbedder(dim=64)
    master = "m_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=emb.dim, table=master)
    try:
        store.ensure_schema()
        Indexer(store, emb).index_path(built)

        questions = [
            {"id": "q1", "query": "topic number 0", "haystack_files": ["s1.md", "s3.md"],
             "relevant_files": ["s1.md"], "answerable": True, "question_type": "single"},
            {"id": "q2", "query": "topic number 1", "haystack_files": ["s2.md", "s3.md"],
             "relevant_files": ["s2.md"], "answerable": True, "question_type": "single"},
        ]
        with pytest.raises(ValueError) as exc:
            perq_evaluate(TEST_DSN, master, questions, emb, k=3)
        # The message has to name the table and the shortfall, or an operator cannot act on it.
        assert "s3.md" in str(exc.value)
        assert master in str(exc.value)
    finally:
        store.drop_table()
        store.close()


@requires_db
def test_perq_accepts_a_complete_master(tmp_path) -> None:
    from recall.eval.longmemeval_perq import evaluate as perq_evaluate

    built = _corpus(tmp_path, ["s1.md", "s2.md", "s3.md"])
    emb = HashingEmbedder(dim=64)
    master = "m_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=emb.dim, table=master)
    try:
        store.ensure_schema()
        Indexer(store, emb).index_path(built)
        questions = [
            {"id": "q1", "query": "topic number 0", "haystack_files": ["s1.md", "s3.md"],
             "relevant_files": ["s1.md"], "answerable": True, "question_type": "single"},
            {"id": "q2", "query": "topic number 1", "haystack_files": ["s2.md", "s3.md"],
             "relevant_files": ["s2.md"], "answerable": True, "question_type": "single"},
        ]
        rep = perq_evaluate(TEST_DSN, master, questions, emb, k=3)
        # Coverage is recorded in the report, so a reviewer sees the check ran rather than
        # trusting that it did.
        assert rep["master_coverage"]["sessions_required"] == 3
        assert rep["master_coverage"]["sessions_present"] == 3
    finally:
        store.drop_table()
        store.close()


@requires_db
def test_labelled_reports_index_completeness(tmp_path) -> None:
    from recall.eval.labelled import evaluate

    corpus = _corpus(tmp_path, ["a.md", "b.md", "c.md", "d.md"])
    questions = [
        {"id": "q1", "query": "topic number 0", "relevant_files": ["a.md"], "answerable": True},
        {"id": "q2", "query": "topic number 1", "relevant_files": ["b.md"], "answerable": True},
        {"id": "q3", "query": "topic number 2", "relevant_files": ["c.md"], "answerable": True},
        {"id": "q4", "query": "topic number 3", "relevant_files": ["d.md"], "answerable": True},
    ]
    table = "lab_cmp_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=64, table=table)
    try:
        rep = evaluate(TEST_DSN, corpus, questions, HashingEmbedder(dim=64), k=3, table=table)
        # expected/actual as a recorded pair, not a bare boolean: a reviewer reading the results
        # JSON can see what was compared.
        assert rep["corpus"]["sources_expected"] == 4
        assert rep["corpus"]["sources_indexed"] == 4
    finally:
        store.drop_table()
        store.close()
