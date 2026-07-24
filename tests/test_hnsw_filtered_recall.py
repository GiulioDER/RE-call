"""Regression tests for issue #11's third checkbox: HNSW + `source` filter = post-filtering.

`query_dense()` (`recall/store.py`) applies `WHERE source = ...` alongside an HNSW
`ORDER BY embedding <=> ...`. The index walk is filter-blind: it finds the globally nearest
neighbours and only THEN discards the ones that fail the filter, so a selective filter can
silently return fewer than `k` rows, or omit true nearest neighbours the table certainly
contains. The fix tunes `hnsw.ef_search` + `hnsw.iterative_scan` (see
`DEFAULT_HNSW_EF_SEARCH_FILTERED` / `DEFAULT_HNSW_ITERATIVE_SCAN_FILTERED` in `recall/store.py`),
applied only to the `source`-filtered branch of `query_dense`.

These run against the REAL pgvector container (`@requires_db`) — the pathology is a genuine
planner/executor behaviour under an approximate index, not something a fake connection can
reproduce.

A note on the corpus construction, because it is NOT incidental: the same 20,000-row / dim-64 /
10%-selective shape reliably shows the collapse ONLY when the rows are upserted in several
separate calls (batches of 1,000 — as a real `recall index` run naturally does, file by file),
not as one giant single-transaction upsert. That is itself a real (if separate) characteristic of
pgvector's incremental HNSW build: a graph built across several committed transactions comes out
measurably less well-connected under this exact filter/selectivity combination than one built in
a single transaction, for reasons this fix does not attempt to explain further. Batching the
corpus build below is therefore what makes this test representative of a REAL multi-file index
run, not merely a way to force a failure.
"""
from __future__ import annotations

import os
import random

import pytest
from pgvector import Vector

from recall.store import PgVectorStore
from recall.types import Chunk

from tests.conftest import TEST_DSN, requires_db

pytestmark = requires_db

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
#: Up to this many independent corpus builds, before giving up (see `filtered_corpus`'s docstring
#: for why a build can need a retry at all).
#:
#: Measured, not guessed: over 14 builds against pgvector 0.8.4 the pathology reproduced on 11,
#: so p ~= 0.79 per build and the spurious-failure rate is (1 - p) ** ATTEMPTS. At 4 that is ~2e-3,
#: which is not rare enough for a suite that runs on every push — it turned master red on
#: cc292df with no code change behind it. At 8 it is ~4e-6.
#:
#: Raising the cap is close to free because the loop stops at the FIRST build that reproduces:
#: the expected cost stays ~1/p ~= 1.3 builds. A higher cap only lengthens the rare bad tail, it
#: does not slow the common path.
#:
#: The measurement also settled which knob to turn. Outcomes are bimodal — a build either shows
#: the pathology hard (recall 0.34-0.42, 39-40/40 truncated) or not at all (recall exactly 1.0000,
#: 0 truncated), with nothing in between — so loosening TUNED_RECALL_FLOOR would buy nothing and
#: would blunt a real regression. And the same data seed produced both outcomes across repeated
#: builds (1003 -> pathology, pathology, none), which confirms the variance is pgvector's HNSW
#: graph construction and not the corpus: re-seeding differently would not help either.
MAX_CORPUS_BUILD_ATTEMPTS = 8
_ENV_EF = "RECALL_HNSW_EF_SEARCH_FILTERED"
_ENV_SCAN = "RECALL_HNSW_ITERATIVE_SCAN_FILTERED"

#: Recall threshold for the FIXED (tuned) path. Measured across several independent corpus builds
#: with this exact shape: 0.92-0.93. 0.75 leaves real margin below every observed tuned value and
#: real margin above every observed untuned value (0.36-0.41) -- HNSW's own graph construction is
#: not seeded by anything this test controls, so the exact figure moves a little build to build;
#: this margin is what keeps the assertion honest without being flaky.
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
    return store


def _measure_untuned(store: PgVectorStore) -> tuple[float, int]:
    """Measure this build under the untuned defaults: `(mean recall@k, truncated count)`.

    The caller decides whether that clears the bar; returning the numbers rather than a bool is
    what lets the fixture say *what it saw* when it gives up, instead of only that it did.

    Runs the EXACT same `_run_queries` the real tests call, under the EXACT same forced-untuned
    env `_measure(tuned=False, ...)` applies -- not a cheaper proxy. An earlier version of this
    gate used a small 8-query truncation-only sample as a stand-in for speed, and that proxy could
    accept a build the real 40-query recall-vs-exact measurement then failed on: the cheaper check
    and the real assertion were not actually testing the same thing. There is no substitute for
    asking the real question.
    """
    old_ef, old_scan = os.environ.get(_ENV_EF), os.environ.get(_ENV_SCAN)
    os.environ[_ENV_EF] = "40"
    os.environ[_ENV_SCAN] = "off"
    try:
        recall, truncated = _run_queries(store)
    finally:
        for key, old in ((_ENV_EF, old_ef), (_ENV_SCAN, old_scan)):
            if old is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old
    return recall, truncated


@pytest.fixture(scope="module")
def filtered_corpus():
    """A 20,000-row / 10%-selective corpus, built once and shared by every test in this module.

    Rebuilding this per test would multiply a ~45s build across every assertion; the tests below
    only ever query it, so sharing it read-only is safe. Not using the `make_store` fixture from
    conftest because that one is function-scoped (a fresh table per test) -- exactly what this
    module deliberately avoids.

    Retries the build (fresh seed, fresh table) up to `MAX_CORPUS_BUILD_ATTEMPTS` times: the
    pathology this module exists to test is real and reproduces on most builds, but pgvector's
    HNSW graph construction carries its own internal randomness (graph-level assignment) that
    nothing here controls, so an otherwise-identical build occasionally comes out well-connected
    enough that even the untuned defaults do not collapse recall. Retrying a fresh build is a
    truthful fix for that -- loosening the threshold instead would just as easily paper over a
    genuine regression that weakens the pathology rather than removes it.
    """
    store: PgVectorStore | None = None
    observed: list[str] = []
    for attempt in range(MAX_CORPUS_BUILD_ATTEMPTS):
        if store is not None:
            store.drop_table()
            store.close()
        store = _build_corpus(seed=1000 + attempt)
        recall, truncated = _measure_untuned(store)
        observed.append(f"seed {1000 + attempt}: recall={recall:.4f} truncated={truncated}")
        if recall < TUNED_RECALL_FLOOR and truncated > 0:
            break
    else:
        # Drop the last corpus before failing. `pytest.fail` raises, so without this the teardown
        # below never runs and every give-up leaks a 20,000-row table into the test database --
        # invisible in ephemeral CI, cumulative for anyone running the suite locally.
        if store is not None:
            store.drop_table()
            store.close()
        # Report what was actually measured. The outcome is bimodal, so these numbers separate the
        # two causes on sight: values near recall=1.0000 / truncated=0 mean the pathology did not
        # occur at all (bad luck, or pgvector no longer exhibits it), whereas values just above
        # the floor would mean it has genuinely weakened. Without them a reader cannot tell which,
        # and the only recourse is to re-run and hope.
        pytest.fail(
            f"could not build a corpus reproducing the untuned HNSW recall pathology in "
            f"{MAX_CORPUS_BUILD_ATTEMPTS} attempts -- either the environment's pgvector build "
            f"differs materially from the one this test was written against, or the pathology "
            f"itself has changed. Wanted recall < {TUNED_RECALL_FLOOR} and truncated > 0; "
            f"measured per attempt:\n  " + "\n  ".join(observed)
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
        f"{TUNED_RECALL_FLOOR}, got {recall:.4f} -- the corpus may no longer reproduce the "
        f"pathology this fix addresses"
    )
    assert truncated > 0, "expected the untuned defaults to truncate at least one query"


@requires_db
def test_filtered_query_sets_hnsw_guc_only_inside_its_own_transaction(make_store, monkeypatch):
    """The `SET LOCAL` scoping this fix depends on, made observable directly.

    The author of this fix first measured against an autocommit connection with no explicit
    transaction, and every configuration looked identical (0.385 recall) because the GUC never
    actually applied -- `SET LOCAL` outside a transaction block is silently a no-op. This test
    would fail exactly that way: it asserts the `SET LOCAL` statements are actually SENT for a
    filtered query, that they are NOT sent for an unfiltered one (the tuning must not tax the arm
    that doesn't need it), and that a plain `SHOW` afterwards proves they did not leak past their
    own transaction into the store's long-lived session.
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

    store.query_dense([0.1, 0.2, 0.3], k=1, source="src")
    assert any("SET LOCAL hnsw.ef_search = 321" in c for c in calls), calls
    assert any("SET LOCAL hnsw.iterative_scan = strict_order" in c for c in calls), calls

    calls.clear()
    store.query_dense([0.1, 0.2, 0.3], k=1)  # unfiltered -- must skip the tuning entirely
    assert not any("hnsw" in c for c in calls), calls

    # Not leaked: a wrong-scope bug (a plain `SET`, or `SET LOCAL` issued outside a transaction)
    # would either make the assertions above pass vacuously (no-op -> no observable effect) or
    # leave this GUC changed for the rest of the store's session. `SHOW` outside any transaction
    # of ours proves neither happened.
    after = store._with_retry(lambda c: c.execute("SHOW hnsw.ef_search").fetchone()[0])
    assert after == "40"  # pgvector's own default -- confirms the transaction actually closed
