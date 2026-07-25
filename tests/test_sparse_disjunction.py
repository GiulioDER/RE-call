"""Regression tests for #81 — the sparse leg must survive a natural-language question.

`websearch_to_tsquery` ANDs every content term, so a chunk had to contain EVERY word of the
question to match. Measured on 150 FinanceBench questions, the sparse leg returned rows for
**0 of them**, and `HybridRetriever` silently degraded to dense-only. These tests fail against
that behaviour and pass against a disjunctive query.
"""
from __future__ import annotations

import pytest

from recall.types import Chunk

from tests.conftest import requires_db


@requires_db
def test_sparse_survives_a_multi_term_question(make_store):
    """The bug, minimally: no chunk contains every term, so AND matched nothing."""
    store = make_store(3)
    store.upsert(
        [
            Chunk("a", "f.md", "Total revenue for the fiscal year was 34,229 million dollars."),
            Chunk("b", "f.md", "The board of directors met to discuss governance matters."),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    # 'total' and 'revenue' are in chunk a; '3m' and 'fy2022' are in neither.
    hits = store.query_sparse("What was 3M's total revenue in FY2022?", k=5)
    assert [h.chunk.id for h in hits] == ["a"]


@requires_db
def test_sparse_ranks_more_matched_terms_higher(make_store):
    """Dropping the AND must not drop the ranking: matching more query terms still wins."""
    store = make_store(3)
    store.upsert(
        [
            Chunk("many", "f.md", "operating cash flow and capital expenditure for the segment"),
            Chunk("few", "f.md", "capital city travel notes"),
        ],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    hits = store.query_sparse("What was operating cash flow and capital expenditure?", k=5)
    assert hits, "disjunctive query must return the partially-matching chunks"
    assert hits[0].chunk.id == "many"


@requires_db
def test_sparse_single_keyword_still_exact(make_store):
    """The original behaviour on a one-word query is unchanged — no false positives."""
    store = make_store(3)
    store.upsert(
        [Chunk("a", "f.md", "the caching layer decision"), Chunk("b", "f.md", "unrelated text")],
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
    )
    assert [h.chunk.id for h in store.query_sparse("caching", k=5)] == ["a"]


@requires_db
@pytest.mark.parametrize("query", ["", "   ", "the of and to", "???"])
def test_sparse_empty_or_stopword_only_query_returns_nothing(make_store, query):
    """A query that normalises to no lexemes must return no rows, not raise and not match all.

    The disjunction is built by aggregating lexemes; with none, the aggregate is NULL. This
    pins that the NULL tsquery filters everything out rather than erroring or matching.
    """
    store = make_store(3)
    store.upsert([Chunk("a", "f.md", "some indexed content")], [[1.0, 0.0, 0.0]])
    assert store.query_sparse(query, k=5) == []


@requires_db
def test_sparse_quotes_are_not_an_injection_vector(make_store):
    """Lexemes are quoted into the tsquery, so a quote in the question must not break it."""
    store = make_store(3)
    store.upsert([Chunk("a", "f.md", "quarterly dividend declared")], [[1.0, 0.0, 0.0]])
    hits = store.query_sparse("what's the quarterly dividend' | 'x", k=5)
    assert [h.chunk.id for h in hits] == ["a"]


@requires_db
def test_sparse_respects_the_source_filter(make_store):
    """The source filter still applies once the query is a join against the tsquery CTE."""
    store = make_store(3)
    store.upsert(
        [Chunk("a", "one.md", "total revenue"), Chunk("b", "two.md", "total revenue")],
        [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
    )
    hits = store.query_sparse("total revenue figures", k=5, source="two.md")
    assert [h.chunk.id for h in hits] == ["b"]


@requires_db
def test_sparse_with_vec_still_returns_true_cosine(make_store):
    """The `vec` path must keep reporting the dense cosine, not the ts_rank."""
    store = make_store(3)
    store.upsert([Chunk("a", "f.md", "total revenue for the year")], [[1.0, 0.0, 0.0]])
    qvec = [0.6, 0.8, 0.0]
    dense_score = store.query_dense(qvec, k=1)[0].score
    sparse_hit = store.query_sparse("what was total revenue", k=1, vec=qvec)[0]
    assert abs(sparse_hit.score - dense_score) < 1e-6
