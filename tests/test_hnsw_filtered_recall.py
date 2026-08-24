"""Regression tests for issue #11's third checkbox: HNSW post-filtering.

`query_dense()` (`recall/store.py`) applies tenant and optional source predicates alongside an HNSW
`ORDER BY embedding <=> ...`. The index walk is filter-blind: it finds the globally nearest
neighbours and only THEN discards the ones that fail the filter, so a selective filter can
silently return fewer than `k` rows, or omit true nearest neighbours the table certainly
contains. The fix tunes `hnsw.ef_search` + `hnsw.iterative_scan` (see
`DEFAULT_HNSW_EF_SEARCH_FILTERED` / `DEFAULT_HNSW_ITERATIVE_SCAN_FILTERED` in `recall/store.py`),
applied to every tenant-scoped query and the optional `source`-filtered branch.

These run against the REAL pgvector container (`@requires_db`) — the pathology is a genuine
planner/executor behaviour under an approximate index, not something a fake connection can
reproduce.

A note on the corpus construction, because one part of it IS load-bearing and the other is not.

Load-bearing: the ANALYZE at the end of `_build_corpus`. Everything measured here is a property
of the plan Postgres picks for the filtered query, and an unanalyzed table gets `Seq Scan + Sort`
— an exact search that never consults the HNSW index and therefore reports recall 1.0000 under
any `hnsw.ef_search` at all. `filtered_corpus` asserts the index is genuinely walked before any
test runs.

Not load-bearing: the batching. An earlier version of this docstring claimed the collapse
reproduces only when the rows arrive in several separate upserts (batches of 1,000, as a real
multi-file `recall index` run does) and attributed that to pgvector building a less well-connected
graph across several committed transactions. That was wrong. Measured both ways with the ANALYZE
in place, a single 20,000-row single-transaction upsert reproduces the pathology just as hard
(recall 0.3625, 40/40 truncated). What the batching actually did was commit rows in the middle of
the build, which let an autovacuum worker analyze the table before the tests ran — i.e. it was
winning the statistics race, not shaping the graph. The batching is kept because it stays
representative of a real index run, not because the measurement needs it.
"""
from __future__ import annotations

import random

import pytest
from pgvector import Vector

from recall.store import (
    DEFAULT_HNSW_EF_SEARCH_FILTERED,
    DEFAULT_HNSW_ITERATIVE_SCAN_FILTERED,
    PgVectorStore,
)
from recall.types import Chunk

from tests.conftest import TEST_DSN, requires_db

# The module-scoped `filtered_corpus` fixture upserts 20,000 rows and builds an HNSW index over
# them, and that cost is charged to whichever test happens to trigger it first — which, under
# random ordering, is any of the four. Measured at ~106 s on an idle machine, i.e. 88% of the
# global `timeout = 120` in pyproject.toml, so the run dies on a loaded CI box while passing
# locally. Raise the ceiling for this module only, per the escape hatch that setting documents;
# the point of the global timeout is to fail a HUNG chunker fast, and it keeps doing that here.
pytestmark = [requires_db, pytest.mark.timeout(300)]

DIM = 64
N_ROWS = 20_000
N_TARGET = 2_000  # 10% selectivity -- the measured pathology's "middle band" (see store.py's
                  # DEFAULT_HNSW_EF_SEARCH_FILTERED comment for the ~1%/~50% extremes, where the
                  # planner already gets recall 1.000 without any of this)
BATCH = 1_000
N_OTHER_SOURCES = 50
K = 10
N_QUERIES = 40
QUERY_SEED = 8
_ENV_EF = "RECALL_HNSW_EF_SEARCH_FILTERED"
_ENV_SCAN = "RECALL_HNSW_ITERATIVE_SCAN_FILTERED"

#: Recall threshold for the FIXED (tuned) path. Measured across independent corpus builds with
#: this exact shape, all analyzed: tuned 0.8825-0.9400, untuned 0.3625-0.4250. 0.75 leaves real
#: margin below every observed tuned value and above every observed untuned value -- HNSW's own
#: graph construction is not seeded by anything this test controls, so the exact figure moves a
#: little build to build; this margin is what keeps the assertion honest without being flaky.
#: (The earlier 0.92-0.93 quoted here for the tuned path was measured over too few builds; the
#: floor is unchanged, but the real spread reaches lower and a reader should know that.)
TUNED_RECALL_FLOOR = 0.75


def _random_vector(rng: random.Random) -> list[float]:
    return [rng.gauss(0, 1) for _ in range(DIM)]


def _build_corpus(seed: int) -> PgVectorStore:
    table = "hnsw_recall_" + "".join(random.choices("0123456789abcdef", k=8))
    store = PgVectorStore(TEST_DSN, dim=DIM, table=table)
    store.ensure_schema()

    rng = random.Random(seed)
    chunks: list[Chunk] = []
    vectors: list[list[float]] = []
    for i in range(N_ROWS):
        source = "target" if i < N_TARGET else f"other-{i % N_OTHER_SOURCES}"
        chunks.append(Chunk(f"c{i}", source, f"row {i}"))
        vectors.append(_random_vector(rng))
    # Shuffle so `target` rows are not contiguous in insertion order -- a source filter should be
    # exercised against a target scattered through the table, not conveniently clustered near the
    # start of the HNSW build.
    order = list(range(N_ROWS))
    rng.shuffle(order)
    chunks = [chunks[i] for i in order]
    vectors = [vectors[i] for i in order]

    # Batched, not one upsert() call -- see the module docstring.
    for start in range(0, N_ROWS, BATCH):
        store.upsert(chunks[start : start + BATCH], vectors[start : start + BATCH])
    # Statistics, or the planner will not use the HNSW index at all and this whole module
    # measures nothing. A freshly built table reports `reltuples = -1` and carries no `pg_stats`
    # row for `source`, so the planner estimates ONE matching row for the filtered query and
    # takes an exact plan (Bitmap Heap Scan + Sort) instead of `Index Scan using <t>_emb_idx`.
    # An exact plan cannot truncate and cannot miss a neighbour, so the measurement comes back
    # recall 1.0000 / 0 truncated -- indistinguishable, from the outside, from "this build's
    # HNSW graph happened to come out well-connected". See the `filtered_corpus` docstring.
    #
    # These rows go in via `store.upsert` rather than through `Indexer.index_path`, so the
    # ANALYZE that a real index run now issues (see `Indexer.index_path`) does not happen here
    # on its own.
    store.analyze()
    return store


def _filtered_dense_plan(store: PgVectorStore, *, ef_search: str, iterative_scan: str) -> str:
    """The plan Postgres picks for `query_dense`'s filtered arm, flattened to one string.

    Mirrors that query exactly where it matters -- same WHERE, same ORDER BY, same LIMIT, same
    two `SET LOCAL`s inside one transaction. Only the SELECT list is trimmed, which changes the
    estimated row width and nothing about which plan wins. The duplication is unavoidable:
    `EXPLAIN` has to prefix the statement, and `query_dense` does not expose one.
    """

    def _op(conn: "object") -> list[tuple]:
        with conn.transaction():  # type: ignore[attr-defined]
            conn.execute(f"SET LOCAL hnsw.ef_search = {ef_search}")  # type: ignore[attr-defined]
            conn.execute(  # type: ignore[attr-defined]
                f"SET LOCAL hnsw.iterative_scan = {iterative_scan}"
            )
            return conn.execute(  # type: ignore[attr-defined]
                f"""
                EXPLAIN SELECT id FROM {store.table}
                WHERE tenant_id = %(tenant)s AND source = %(source)s
                ORDER BY embedding <=> %(vec)s
                LIMIT %(k)s
                """,
                {
                    "tenant": store._tenant,
                    "source": "target",
                    "vec": Vector(_random_vector(random.Random(QUERY_SEED))),
                    "k": K,
                },
            ).fetchall()

    return " | ".join(r[0].strip() for r in store._with_retry(_op))


@pytest.fixture(scope="module")
def filtered_corpus():
    """A 20,000-row / 10%-selective corpus, built once and shared by every test in this module.

    Rebuilding this per test would multiply a ~45s build across every assertion; the tests below
    only ever query it, so sharing it read-only is safe. Not using the `make_store` fixture from
    conftest because that one is function-scoped (a fresh table per test) -- exactly what this
    module deliberately avoids.

    Verifies before yielding that the filtered dense query actually WALKS the HNSW index, under
    both the tuned and the untuned settings the tests below measure. That check is the whole
    precondition of this module: on a plan that does not consult the index, `query_dense` is an
    exact search, and every assertion here passes or fails for a reason that has nothing to do
    with `hnsw.ef_search` -- the untuned test reads recall 1.0000 and fails, while the two tuned
    tests pass VACUOUSLY, reporting a fix that was never exercised. `_build_corpus`'s ANALYZE is
    what pins the plan (see the comment there); this asserts the pin actually held, so that an
    environment where it does not says so in one line instead of surfacing as a flake.

    ONE build, no retry loop, and no "did this build reproduce the pathology?" recall gate in
    front of the tests. All three existed because the outcome looked random per build -- a build
    either showed the pathology hard (recall 0.34-0.42, 39-40/40 truncated) or not at all (recall
    exactly 1.0000, 0 truncated), with nothing in between, and the same data seed produced both.
    That was read as pgvector's HNSW graph construction being unseeded, and two commits (73888b0,
    50cc57a) sized a retry margin against it. It was not graph construction: the two outcomes are
    two PLANS, and which one Postgres picks was racing the build. Retrying resampled the corpus,
    and the corpus was never the variable -- which is why raising the cap twice left master red on
    roughly 1 in 4 runs. With the plan pinned the pathology reproduces on every build measured, so
    one build is enough, and a failure now means the pathology itself changed. That is a finding
    to read, not a flake to re-run.
    """
    store = _build_corpus(seed=1000)
    for label, ef_search, iterative_scan in (
        ("untuned", "40", "off"),
        (
            "tuned",
            str(DEFAULT_HNSW_EF_SEARCH_FILTERED),
            DEFAULT_HNSW_ITERATIVE_SCAN_FILTERED,
        ),
    ):
        plan = _filtered_dense_plan(store, ef_search=ef_search, iterative_scan=iterative_scan)
        if f"{store.table}_emb_idx" not in plan:
            # Drop before failing: `pytest.fail` raises, so the teardown below never runs and the
            # 20,000-row table would leak -- invisible in ephemeral CI, cumulative locally.
            store.drop_table()
            store.close()
            pytest.fail(
                f"the planner declined the HNSW index for the {label} filtered query, so this "
                f"module cannot measure what it exists to measure (a plan without the index is "
                f"an exact search: recall 1.0000, 0 truncated, whatever hnsw.ef_search says). "
                f"Statistics are the usual cause -- see `_build_corpus`'s ANALYZE. Plan was:\n"
                f"  {plan}"
            )

    yield store

    store.drop_table()
    store.close()


def _exact_topk_ids(store: PgVectorStore, vector: list[float], k: int, source: str) -> list[str]:
    """The TRUE top-k under `source`, forcing a Seq Scan + Sort so the HNSW index cannot be used
    at all -- the ground truth `query_dense`'s recall is measured against.

    `enable_indexscan`/`enable_bitmapscan` are themselves `SET LOCAL`, for the same reason
    `query_dense`'s own tuning is: without the transaction they would not apply, and every plan
    would silently fall back to whatever the session already had -- the exact wrong-scope failure
    this whole fix exists to avoid.
    """

    def _op(conn: "object") -> list[tuple]:
        with conn.transaction():  # type: ignore[attr-defined]
            conn.execute("SET LOCAL enable_indexscan = off")  # type: ignore[attr-defined]
            conn.execute("SET LOCAL enable_bitmapscan = off")  # type: ignore[attr-defined]
            return conn.execute(  # type: ignore[attr-defined]
                f"""
                SELECT id FROM {store.table}
                WHERE tenant_id = %(tenant)s AND source = %(source)s
                ORDER BY embedding <=> %(vec)s
                LIMIT %(k)s
                """,
                {"tenant": store._tenant, "source": source, "vec": Vector(vector), "k": k},
            ).fetchall()

    rows = store._with_retry(_op)
    return [r[0] for r in rows]


def _recall_at_k(got_ids: list[str], exact_ids: list[str], k: int) -> float:
    rel = set(exact_ids)
    if not rel:
        return 0.0
    return len(set(got_ids[:k]) & rel) / len(rel)


def _run_queries(store: PgVectorStore) -> tuple[float, int]:
    """Mean recall@k and the truncated-query count, over `N_QUERIES` fixed random queries, under
    whatever `RECALL_HNSW_*_FILTERED` env is already in effect when this is called."""
    qrng = random.Random(QUERY_SEED)
    recalls: list[float] = []
    truncated = 0
    for _ in range(N_QUERIES):
        q = _random_vector(qrng)
        exact_ids = _exact_topk_ids(store, q, K, "target")
        got = store.query_dense(q, k=K, source="target")
        got_ids = [h.chunk.id for h in got]
        if len(got_ids) < min(K, len(exact_ids)):
            truncated += 1
        recalls.append(_recall_at_k(got_ids, exact_ids, K))
    return sum(recalls) / len(recalls), truncated


def _measure(store: PgVectorStore, *, tuned: bool, monkeypatch) -> tuple[float, int]:
    """`_run_queries`, optionally forcing the untuned (pre-fix-equivalent) env first.

    `tuned=False` forces `RECALL_HNSW_EF_SEARCH_FILTERED`/`RECALL_HNSW_ITERATIVE_SCAN_FILTERED` to
    pgvector's own defaults (40 / off) -- numerically identical to the pre-fix code path, which
    never touched these GUCs at all and so ran under whatever the session default was.
    """
    if not tuned:
        monkeypatch.setenv(_ENV_EF, "40")
        monkeypatch.setenv(_ENV_SCAN, "off")
    return _run_queries(store)


def test_filtered_recall_regression_pinned_above_threshold(filtered_corpus, monkeypatch):
    """The fix's core guarantee: recall@10 under a 10%-selective filter stays high."""
    recall, truncated = _measure(filtered_corpus, tuned=True, monkeypatch=monkeypatch)
    assert recall >= TUNED_RECALL_FLOOR, (
        f"recall@{K} was {recall:.4f} under a 10%-selective filter, below the "
        f"{TUNED_RECALL_FLOOR} floor -- the HNSW post-filtering fix may have regressed"
    )
    assert truncated == 0, f"{truncated}/{N_QUERIES} filtered queries returned fewer than k={K}"


def test_filtered_query_returns_full_k_when_k_rows_exist(filtered_corpus, monkeypatch):
    """The sharpest signal from the original measurement: 40/40 queries returned FEWER than `k`
    before the fix (every one of them had >= k=10 matching-source rows to find). Asserted
    independently of the recall computation above -- a truncated result set is wrong regardless
    of whether the rows it DID return happen to be the right ones."""
    _, truncated = _measure(filtered_corpus, tuned=True, monkeypatch=monkeypatch)
    assert truncated == 0


def test_filtered_recall_collapses_without_the_tuning(filtered_corpus, monkeypatch):
    """Before/after, on the SAME corpus -- proves the tuning is doing the work, not the corpus
    shape. Forces `RECALL_HNSW_EF_SEARCH_FILTERED=40` / `RECALL_HNSW_ITERATIVE_SCAN_FILTERED=off`,
    reproducing the pre-fix code path's effective behaviour exactly (those ARE pgvector's own
    session defaults -- see `_measure`)."""
    recall, truncated = _measure(filtered_corpus, tuned=False, monkeypatch=monkeypatch)
    assert recall < TUNED_RECALL_FLOOR, (
        f"expected the untuned defaults to collapse recall@{K} well below "
        f"{TUNED_RECALL_FLOOR}, got {recall:.4f} -- the pathology this fix addresses has "
        f"weakened or gone. Note what this is NOT: `filtered_corpus` has already verified that "
        f"this query walks the HNSW index, so this is not the planner quietly running an exact "
        f"scan (which is what a reading of exactly 1.0000 used to mean). Re-running will not "
        f"change it"
    )
    assert truncated > 0, "expected the untuned defaults to truncate at least one query"


@requires_db
def test_filtered_query_sets_hnsw_guc_only_inside_its_own_transaction(make_store, monkeypatch):
    """The `SET LOCAL` scoping this fix depends on, made observable directly.

    The author of this fix first measured against an autocommit connection with no explicit
    transaction, and every configuration looked identical (0.385 recall) because the GUC never
    actually applied -- `SET LOCAL` outside a transaction block is silently a no-op. This test
    would fail exactly that way: it asserts the `SET LOCAL` statements are actually SENT for both
    the tenant-scoped and source-filtered queries, and that a plain `SHOW` afterwards proves they
    did not leak past their own transaction into the store's long-lived session.
    """
    monkeypatch.setenv("RECALL_HNSW_EF_SEARCH_FILTERED", "321")
    monkeypatch.setenv("RECALL_HNSW_ITERATIVE_SCAN_FILTERED", "strict_order")
    store = make_store(3)
    store.upsert([Chunk("a", "src", "hello")], [[0.1, 0.2, 0.3]])

    calls: list[str] = []
    real_execute = store._conn.execute

    def _spy(sql, *a, **kw):
        calls.append(" ".join(str(sql).split()))
        return real_execute(sql, *a, **kw)

    monkeypatch.setattr(store._conn, "execute", _spy)

    store.query_dense([0.1, 0.2, 0.3], k=1)
    assert any("SET LOCAL hnsw.ef_search = 321" in c for c in calls), calls
    assert any("SET LOCAL hnsw.iterative_scan = strict_order" in c for c in calls), calls

    calls.clear()
    store.query_dense([0.1, 0.2, 0.3], k=1, source="src")
    assert any("SET LOCAL hnsw.ef_search = 321" in c for c in calls), calls
    assert any("SET LOCAL hnsw.iterative_scan = strict_order" in c for c in calls), calls

    # Not leaked: a wrong-scope bug (a plain `SET`, or `SET LOCAL` issued outside a transaction)
    # would either make the assertions above pass vacuously (no-op -> no observable effect) or
    # leave this GUC changed for the rest of the store's session. `SHOW` outside any transaction
    # of ours proves neither happened.
    after = store._with_retry(lambda c: c.execute("SHOW hnsw.ef_search").fetchone()[0])
    assert after == "40"  # pgvector's own default -- confirms the transaction actually closed
