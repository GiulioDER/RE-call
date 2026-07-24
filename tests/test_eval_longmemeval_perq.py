"""Per-question haystack scoring — the protocol LongMemEval actually publishes.

These tests hit the real pgvector container (`conftest.TEST_DSN`), because the thing under test
is a SQL copy between two tables with a generated column and a vector column in them. A mock
would assert that the code calls the functions it calls.
"""
from __future__ import annotations

import uuid

import pytest

from recall.eval.longmemeval_perq import populate_haystack
from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.store import PgVectorStore

from .conftest import TEST_DSN


@pytest.fixture()
def master(tmp_path):
    """A 3-session master index, built once per test."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "s1.md").write_text("the deploy target is staging", encoding="utf-8")
    (corpus / "s2.md").write_text("the database backup runs nightly", encoding="utf-8")
    (corpus / "s3.md").write_text("the oncall rotation is weekly", encoding="utf-8")

    emb = HashingEmbedder(dim=64)
    table = "m_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=emb.dim, table=table)
    store.ensure_schema()
    Indexer(store, emb).index_path(corpus)
    yield store, emb, table
    store.drop_table()
    store.close()


def test_the_scratch_table_holds_only_the_requested_sessions(master):
    store, emb, table = master
    scratch = "sc_" + uuid.uuid4().hex[:8]

    sub = populate_haystack(TEST_DSN, emb.dim, table, scratch, ["s1.md", "s3.md"])
    try:
        sources = {r["source"] for r in sub.all_rows()} if hasattr(sub, "all_rows") else None
        assert sub.count() == 2
        assert sources is None or all("s2.md" not in s for s in sources)
    finally:
        sub.drop_table()
        sub.close()


def test_the_copied_rows_are_searchable_without_re_embedding(master):
    # The whole point: embeddings come from the master index, so a per-question run costs a
    # table copy rather than an embed. If the vector did not survive the copy, dense search
    # returns nothing and this fails.
    store, emb, table = master
    scratch = "sc_" + uuid.uuid4().hex[:8]

    sub = populate_haystack(TEST_DSN, emb.dim, table, scratch, ["s1.md", "s2.md"])
    try:
        hits = sub.query_dense(emb.embed(["deploy target"])[0], k=5)
        assert hits, "dense search returned nothing — the embedding did not survive the copy"
    finally:
        sub.drop_table()
        sub.close()


def test_the_generated_tsvector_is_rebuilt_so_sparse_search_still_works(master):
    # `tsv` is GENERATED ALWAYS ... STORED. It cannot be inserted into; it has to regenerate
    # from the copied text. A copy that lost it would silently disable the sparse leg of the
    # hybrid retriever — and the run would still produce a number.
    store, emb, table = master
    scratch = "sc_" + uuid.uuid4().hex[:8]

    sub = populate_haystack(TEST_DSN, emb.dim, table, scratch, ["s1.md", "s2.md"])
    try:
        hits = sub.query_sparse("deploy target", k=5)
        assert hits, "sparse search returned nothing — the tsvector did not regenerate"
    finally:
        sub.drop_table()
        sub.close()


def test_reusing_the_scratch_table_replaces_the_previous_haystack(master):
    # 500 questions reuse one scratch table. A populate that appended instead of replacing
    # would grow the haystack monotonically and quietly make every later question easier.
    store, emb, table = master
    scratch = "sc_" + uuid.uuid4().hex[:8]

    sub = populate_haystack(TEST_DSN, emb.dim, table, scratch, ["s1.md"])
    sub.close()
    sub = populate_haystack(TEST_DSN, emb.dim, table, scratch, ["s2.md", "s3.md"])
    try:
        assert sub.count() == 2
    finally:
        sub.drop_table()
        sub.close()


def test_an_underscore_in_a_session_id_is_not_treated_as_a_wildcard(tmp_path):
    # LongMemEval session ids are full of underscores ("answer_c63c0458"). Under LIKE, `_`
    # matches any single character, so a suffix-match pull of one session also drags in every
    # session whose name differs only at that position — silently enlarging the haystack the
    # question is scored against. Caught in the real run: 49 requested, 50 copied.
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    (corpus / "a_b.md").write_text("the wanted session about deploys", encoding="utf-8")
    (corpus / "aXb.md").write_text("an unrelated session about lunch", encoding="utf-8")

    emb = HashingEmbedder(dim=64)
    table = "m_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=emb.dim, table=table)
    store.ensure_schema()
    Indexer(store, emb).index_path(corpus)
    scratch = "sc_" + uuid.uuid4().hex[:8]
    try:
        sub = populate_haystack(TEST_DSN, emb.dim, table, scratch, ["a_b.md"])
        try:
            with sub._connect() as conn:  # noqa: SLF001
                got = {r[0] for r in conn.execute(f"SELECT DISTINCT source FROM {scratch}")}
            assert len(got) == 1, f"underscore matched more than the requested session: {got}"
        finally:
            sub.drop_table()
            sub.close()
    finally:
        store.drop_table()
        store.close()


def test_a_haystack_naming_an_absent_session_copies_what_exists(master):
    # The converter already refuses gold outside the haystack; a haystack naming a session the
    # master index does not hold means the index is incomplete, and the count is how the runner
    # notices rather than scoring a silently smaller haystack.
    store, emb, table = master
    scratch = "sc_" + uuid.uuid4().hex[:8]

    sub = populate_haystack(TEST_DSN, emb.dim, table, scratch, ["s1.md", "nope.md"])
    try:
        assert sub.count() == 1
    finally:
        sub.drop_table()
        sub.close()
