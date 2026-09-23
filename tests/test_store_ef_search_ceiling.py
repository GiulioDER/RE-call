"""The derived unfiltered `hnsw.ef_search` must respect pgvector's 1..1000 range.

The filtered path has validated its configured `ef_search` since it was written. The unfiltered
path computes the SAME quantity — `k * RECALL_HNSW_EF_SEARCH_MULTIPLIER` — and checked nothing,
so a large candidate pool reached Postgres as `ef_search = 2400` and failed the query with
`InvalidParameterValue`, naming a knob the caller never set instead of the `k` they did. One
derivation, a guard on one of its two paths.

The cap is the right behaviour rather than a refusal: `k * multiplier` is over-fetch MARGIN, and
correctness needs only `ef_search >= k`. A 600-candidate pool is served correctly at ef_search
1000. Only `k` itself above 1000 is unserviceable, because then the scan cannot reach k rows and
would truncate silently — the failure #84 exists to prevent — so that one raises.
"""
from __future__ import annotations

import tempfile
import warnings
from pathlib import Path

import pytest

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.store import _HNSW_EF_SEARCH_MAX, _ef_search_multiplier

from .conftest import requires_db


def test_the_max_matches_pgvectors_documented_range() -> None:
    assert _HNSW_EF_SEARCH_MAX == 1000


def test_default_multiplier_puts_the_ceiling_at_250_candidates() -> None:
    """The number the BEAM `--candidate-k` help text quotes; pinned so they cannot drift."""
    assert _HNSW_EF_SEARCH_MAX // _ef_search_multiplier() == 250


@pytest.mark.parametrize(
    ("k", "multiplier", "expected"),
    [
        (45, 4, 180),      # ordinary: margin applies in full
        (250, 4, 1000),    # exactly at the ceiling
        (600, 4, 1000),    # past it: capped, still >= k, so k rows are reachable
        (1000, 4, 1000),   # k == ceiling: no margin left, still serviceable
        (600, 1, 600),     # a lower multiplier keeps the request under the cap
    ],
)
def test_derived_ef_search_is_capped_not_overflowed(
    k: int, multiplier: int, expected: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RECALL_HNSW_EF_SEARCH_MULTIPLIER", str(multiplier))
    derived = min(k * _ef_search_multiplier(), _HNSW_EF_SEARCH_MAX)
    assert derived == expected
    assert derived <= _HNSW_EF_SEARCH_MAX, "would be rejected by pgvector"
    assert derived >= k or k > _HNSW_EF_SEARCH_MAX, "must cover k or the scan truncates"


def test_a_capped_scan_still_covers_k() -> None:
    """The property the cap has to preserve: margin may shrink, coverage may not."""
    for k in (45, 250, 600, 1000):
        assert min(k * 4, _HNSW_EF_SEARCH_MAX) >= k


# --- the tests above assert arithmetic on constants and would pass against the unpatched
# --- module. These two run the query that actually failed.


# Only the two tests below need a database; the four above derive the cap arithmetically and must
# keep running without a container, so the guard goes per-test rather than on the module.
@pytest.fixture
def populated(make_store):
    store = make_store(64)
    with tempfile.TemporaryDirectory() as d:
        for i in range(30):
            Path(d, f"doc{i}.md").write_text(f"memory {i} about topic {i % 7}\n", encoding="utf-8")
        Indexer(store, HashingEmbedder(dim=64)).index_path(Path(d))
    return store


@requires_db
def test_a_pool_past_the_ceiling_queries_successfully_and_warns(populated) -> None:
    """Before the cap this raised `InvalidParameterValue: 2400 is outside the valid range`."""
    vector = HashingEmbedder(dim=64).embed(["topic 3"])[0]
    with pytest.warns(RuntimeWarning, match="ef_search capped at 1000"):
        hits = populated.query_dense(vector, k=600)
    assert hits, "the query must still return rows, not merely avoid raising"


@requires_db
def test_a_pool_under_the_ceiling_does_not_warn(populated) -> None:
    """Guards the guard: without this the warning above could fire on every query."""
    vector = HashingEmbedder(dim=64).embed(["topic 3"])[0]
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert populated.query_dense(vector, k=45)


def test_the_generation_store_widens_a_filtered_scan_for_a_large_k(monkeypatch) -> None:
    """The production store widens `hnsw.ef_search` for k exactly as the legacy store does.

    Invariant: a filtered dense query for `k` rows runs with `ef_search >= k * multiplier`
    (capped at pgvector's maximum), because a filtered HNSW walk that stops at the default width
    of 200 returns fewer than k rows while reporting nothing. `PgVectorStore._query_dense`
    passes `k` to `_hnsw_filtered_tuning`; `GenerationStore._query_dense`, which is the class
    `RECALL_ENV=production` serves, called it with no `k`, so it never widened and never refused
    a `k` above 1000.

    Red proof, recorded 2026-09-23 against `origin/master` at `3cc57b81`, whose
    `GenerationStore._query_dense` called `self._hnsw_filtered_tuning()`: this test failed at
    the final assertion with the scan run at `SET LOCAL hnsw.ef_search = 200` instead of the
    widened value. Passing `k` turns it green. (P3 then moved the tuning into one bound
    `set_config` statement, so the assertion reads the bound value instead of the SQL text; the
    same mutation, dropping `k`, fails it with `'200' == '600'`.)
    """
    from contextlib import nullcontext
    from contextvars import ContextVar

    from recall.generation_store import GenerationStore

    monkeypatch.delenv("RECALL_HNSW_EF_SEARCH_FILTERED", raising=False)
    monkeypatch.delenv("RECALL_HNSW_EF_SEARCH_MULTIPLIER", raising=False)
    monkeypatch.delenv("RECALL_HNSW_ITERATIVE_SCAN_FILTERED", raising=False)
    statements: list[tuple[str, object]] = []

    class _Rows:
        def fetchall(self):
            return [("chunk-1", "a.md", "text", {"file": "a.md"}, None, 0.9)]

    class _Connection:
        def transaction(self):
            return nullcontext()

        def execute(self, sql, params=None):
            statements.append((str(sql), params))
            return _Rows()

    store = object.__new__(GenerationStore)
    store._tenant = "acme"
    store._pinned_generation = ContextVar("pinned_generation", default="gen-1")
    store._pinned_corpus = ContextVar("pinned_corpus", default=None)
    store._fixed_generation = None
    store._with_retry = lambda op: op(_Connection())

    k = 150
    store._query_dense([1.0] * 4, k)

    expected = min(k * _ef_search_multiplier(), _HNSW_EF_SEARCH_MAX)
    assert expected > 200  # the case is one the default width would truncate
    from recall.store import _HNSW_FILTERED_TUNING_SQL

    tuning = [params for sql, params in statements if sql == _HNSW_FILTERED_TUNING_SQL]
    assert tuning and tuning[0][0] == str(expected)  # the value that reaches hnsw.ef_search
