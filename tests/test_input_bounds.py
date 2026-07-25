"""Client-supplied input must be bounded in the dimensions that cost the server work.

`k` was already clamped (`MAX_SEARCH_K`), and SECURITY.md concluded from that "query length is
unrelated". It is not. `PgVectorStore.query_sparse` builds a DISJUNCTIVE tsquery from every
distinct lexeme of the query — `to_tsvector('english', %(q)s)` normalised server-side, then
`string_agg(quote_literal(lexeme), ' | ')::tsquery` — so cost grows with the query, while the
rate limiter debits exactly one token no matter how large it is.

That asymmetry is the whole attack. At the default `read = 120/min`, `POOL_SIZE = 8` and
`statement_timeout = 15s`, a tenant sending multi-megabyte queries keeps eight connections busy
on 15-second scans indefinitely — against the ONE Postgres every tenant shares, so the damage
does not stay inside the tenant that caused it.

Refused rather than truncated: silently searching a prefix of the question answers a question
the caller did not ask, and the caller cannot tell. `recall_forget` has the same unbounded shape
in its `sources` list.
"""
from __future__ import annotations

import pytest

from recall_mcp.service import (
    MAX_FORGET_SOURCES,
    MAX_QUERY_CHARS,
    forget_memory,
    search_memory,
)


class _ConstantEmbedder:
    dim = 2
    name = "constant"

    def embed(self, texts):
        return [[1.0, 0.0] for _ in texts]


class _ExplodingStore:
    """Any database access at all is a test failure — the refusal must come first."""

    def __getattr__(self, name):
        raise AssertionError(f"store.{name} was reached; the request should have been refused")


def test_an_oversized_query_is_refused_before_the_database_is_touched():
    with pytest.raises(ValueError) as exc:
        search_memory(_ExplodingStore(), _ConstantEmbedder(), "x" * (MAX_QUERY_CHARS + 1))

    assert str(MAX_QUERY_CHARS) in str(exc.value), "the refusal must state the limit"


def test_a_query_at_the_limit_is_still_served():
    """The bound must sit far above any real question, so it never refuses honest use."""
    assert MAX_QUERY_CHARS >= 1000, "a natural-language question must never hit this"

    with pytest.raises(AssertionError, match="store."):
        # Reaching the store proves the guard let it through — this is the pass condition.
        search_memory(_ExplodingStore(), _ConstantEmbedder(), "x" * MAX_QUERY_CHARS)


def test_an_oversized_forget_list_is_refused():
    with pytest.raises(ValueError) as exc:
        forget_memory(_ExplodingStore(), ["s"] * (MAX_FORGET_SOURCES + 1))

    assert str(MAX_FORGET_SOURCES) in str(exc.value)


def test_an_empty_query_is_still_refused_the_way_it_always_was():
    """Pins that the new bound did not replace the existing lower-end handling."""
    with pytest.raises(AssertionError, match="store."):
        search_memory(_ExplodingStore(), _ConstantEmbedder(), "")
