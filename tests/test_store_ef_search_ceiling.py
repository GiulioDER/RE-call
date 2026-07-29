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
