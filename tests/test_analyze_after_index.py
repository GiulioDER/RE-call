"""A bulk index run leaves the planner able to see the table it just built.

The defect, measured against the real container: on a freshly built, never-analyzed table
PostgreSQL reports `reltuples = -1`, `relpages = 0` and has no `pg_stats` row for `source`, so
the planner estimates ONE matching row for `query_dense`'s source-filtered arm and picks an
exact plan (Bitmap Heap Scan + Sort, cost ~15) over `Index Scan using <table>_emb_idx`. Results
stay correct — an exact scan is an exact search — but the HNSW index is not consulted at all,
which means the `hnsw.ef_search` / `hnsw.iterative_scan` tuning in `query_dense` is inert until
autovacuum's analyze lands (naptime 60s by default). On a large corpus that window is a full
scan plus a sort of every matching row, per query.

Measured on a 20,000-row / dim-64 corpus with a 10%-selective filter, holding the HNSW graph
constant and varying ONLY whether statistics existed:

    no statistics (reltuples = -1) ... untuned recall@10 1.0000, 0/40 truncated (exact plan)
    after ANALYZE .................... untuned recall@10 0.3700, 40/40 truncated (HNSW plan)

The first row is the planner refusing to use the index at all. See
`tests/test_hnsw_filtered_recall.py`, whose fixture depends on the index actually being used.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from recall.embeddings import HashingEmbedder
from recall.index import Indexer
from recall.store import PgVectorStore

from tests.conftest import requires_db

pytestmark = requires_db

DIM = 64


def _stats(store: PgVectorStore) -> tuple[float, int]:
    """`(reltuples, number of pg_stats rows for the `source` column)` for this store's table.

    ⚠️ The second element is meaningful only when `_pg_stats_visible` is true — see `_analyzed`.
    """
    return store._with_retry(
        lambda conn: (
            conn.execute(
                "SELECT reltuples FROM pg_class WHERE oid = %s::regclass", (store.table,)
            ).fetchone()[0],
            conn.execute(
                "SELECT count(*) FROM pg_stats WHERE tablename = %s AND attname = 'source'",
                (store.table,),
            ).fetchone()[0],
        )
    )


def _analyzed(store: PgVectorStore) -> bool:
    """Whether an ANALYZE has been recorded for this table, read where RLS cannot hide it.

    `pg_stats` is the natural place to look and the wrong thing to depend on. `ensure_schema`
    issues `ALTER TABLE ... FORCE ROW LEVEL SECURITY`, and PostgreSQL suppresses `pg_stats` rows
    whenever row-level security is active for the querying role — which FORCE extends to the
    table's own owner. Only a superuser bypasses it. A suite that asserts on `pg_stats` alone
    therefore passes under `docker compose up -d` (which connects as `postgres`) and fails
    against any ordinary owner role, reporting "no statistics" for statistics that were in fact
    refreshed and are merely invisible. Measured, not assumed: `SET row_security = off` does not
    lift it either — `row_security_active()` stays true for the owner under FORCE. See #156.

    `pg_stat_user_tables` is not filtered that way, so this is the signal the invariant hangs on.
    `last_autoanalyze` is folded in for completeness; the `store` fixture disables autovacuum, so
    in these tests only `last_analyze` can fire.
    """
    return store._with_retry(
        lambda conn: conn.execute(
            "SELECT last_analyze IS NOT NULL OR last_autoanalyze IS NOT NULL "
            "FROM pg_stat_user_tables WHERE relid = %s::regclass",
            (store.table,),
        ).fetchone()[0]
    )


def _pg_stats_visible(store: PgVectorStore) -> bool:
    """Whether this role can see `pg_stats` rows for the table at all (see `_analyzed`)."""
    return not store._with_retry(
        lambda conn: conn.execute(
            "SELECT row_security_active(%s::regclass)", (store.table,)
        ).fetchone()[0]
    )


def _corpus(tmp_path: Path, n: int) -> Path:
    root = tmp_path / "corpus"
    root.mkdir(parents=True)
    for i in range(n):
        (root / f"memo-{i:03d}.md").write_text(f"# Memo {i}\n\nDecision {i} about caching.\n")
    return root


def _no_autovacuum(store: PgVectorStore) -> None:
    """Pin the never-analyzed state so these tests measure the code, not a race.

    Without this the autovacuum launcher (60s naptime) may analyze the table mid-test, and the
    assertions below would pass or fail on timing. Two probes of the same 20,000-row build
    landed on opposite sides of exactly that race.
    """
    store._with_retry(
        lambda conn: conn.execute(f"ALTER TABLE {store.table} SET (autovacuum_enabled = false)")
    )


@pytest.fixture
def store(make_store):
    s = make_store(DIM)
    _no_autovacuum(s)
    return s


def test_a_fresh_table_has_no_statistics_before_an_index_run(store):
    """Guards the guard: without this, the assertion below could pass vacuously.

    Note the `source_stats == 0` half was ALSO vacuous for any non-superuser role, which sees
    zero rows in `pg_stats` whatever the table's state (#156). The `_analyzed` check is the one
    that means something for every role.
    """
    reltuples, source_stats = _stats(store)
    assert reltuples == -1.0
    assert not _analyzed(store), "the table was already analyzed before the index run"
    if _pg_stats_visible(store):
        assert source_stats == 0


def test_index_run_leaves_the_planner_with_statistics(store, tmp_path):
    Indexer(store, HashingEmbedder(dim=DIM)).index_path(_corpus(tmp_path, 60))
    reltuples, _ = _stats(store)
    assert reltuples >= 0, "the table is still never-analyzed after a bulk index run"
    assert _analyzed(store), "no ANALYZE recorded for the table after a bulk index run"


def test_index_run_records_column_statistics_for_source(store, tmp_path):
    """The column-level half of the invariant, split out because only a superuser can see it.

    Kept as its own test rather than folded into an `if` above so that a run which cannot check
    it says so — `pytest -rs` names the skip and its cause, instead of the coverage quietly
    disappearing inside a passing test.
    """
    if not _pg_stats_visible(store):
        pytest.skip(
            "pg_stats is filtered by FORCE ROW LEVEL SECURITY for this non-superuser role; "
            "the statistics exist but cannot be read here (#156)"
        )
    Indexer(store, HashingEmbedder(dim=DIM)).index_path(_corpus(tmp_path, 60))
    _, source_stats = _stats(store)
    assert source_stats == 1, "no pg_stats row for `source` — the filtered arm's selectivity " \
                              "estimate is still a guess"


def test_an_unchanged_reindex_does_not_analyze_again(store, tmp_path):
    """The steady-state cost of the feature is zero.

    A re-index that re-embeds nothing writes nothing, so there is no statistics change to make.
    Asserted by watching for the statement rather than by inspecting catalog values, which a
    concurrent autovacuum could move underneath the test.
    """
    corpus = _corpus(tmp_path, 60)
    embedder = HashingEmbedder(dim=DIM)
    Indexer(store, embedder).index_path(corpus)

    issued: list[str] = []
    real = store.analyze_if_stale

    def spy(modified: int) -> bool:
        issued.append(f"analyze_if_stale({modified})")
        return real(modified)

    store.analyze_if_stale = spy  # type: ignore[method-assign]
    stats = Indexer(store, embedder).index_path(corpus)
    assert stats.skipped == 60 and stats.chunks == 0
    assert issued == [], "an unchanged re-index asked for a statistics refresh"


def test_a_second_bulk_load_into_a_tiny_analyzed_table_is_analyzed(store, tmp_path):
    """The case a bare never-analyzed check would miss.

    First run: 2 files, so the table is analyzed while it is tiny and `reltuples` stops being
    -1. Second run: a bulk load whose size dwarfs those statistics. Autovacuum would analyze
    that; this makes it happen before the next query instead of up to a naptime after it.
    """
    embedder = HashingEmbedder(dim=DIM)
    small = _corpus(tmp_path / "a", 2)
    Indexer(store, embedder).index_path(small)
    reltuples_after_small, _ = _stats(store)
    assert 0 <= reltuples_after_small < 10  # analyzed, and describing a tiny table

    big = _corpus(tmp_path / "b", 300)
    Indexer(store, embedder).index_path(big)
    reltuples_after_big, _ = _stats(store)
    assert reltuples_after_big >= 300, (
        f"statistics still describe {reltuples_after_big} rows after a 300-file bulk load"
    )
