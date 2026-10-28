"""Measure what the served dense path actually returns, per tenant, against exact search.

    python scripts/check_hnsw_recall.py --out recall.json            # every active tenant
    python scripts/check_hnsw_recall.py --tenants memory --queries 200

For each tenant's ACTIVE generation in `recall_chunks_v1` this samples query vectors from the
generation itself (deterministically, by seed), asks `GenerationStore.query_dense` for the top k on
its HNSW path, and scores it against exact cosine search computed in numpy over the same
generation's vectors. The query's own row is excluded from both lists. Generation-scoped search is
exact by default since #847, so the measurement forces `RECALL_GENERATION_DENSE_SEARCH=hnsw`, the
setting an operator would opt into, and records the path it measured: without that it would
compare exact search with itself and report full recall for an index it never walked.

Why it exists. The HNSW index is built over the whole table, every tenant and every retained
generation, while a query asks for one tenant's active generation: a small slice, which is the case
where a filtered graph walk loses neighbours. `_hnsw_filtered_tuning` and pgvector's iterative scans
are the mitigation; this measures whether they are enough on the real corpus, after a pgvector
upgrade, a re-index, or a generation gc changes the graph under them. Nothing else does.

Recall is tie tolerant: a returned row counts if its exact score reaches the k-th exact score
(within 1e-5), so equal-scoring duplicates cannot make a correct answer look wrong. A walk that
returns nothing is re-run by the store as an exact scan; such a query is counted in
`exact_fallbacks` and scored as the empty result the walk gave, so a failing index cannot read as
perfect recall. Each query is pinned to the measured generation, and a promotion or an erasure
during the measurement (a new generation or a new corpus fingerprint) makes it measure again.

Needs a superuser or BYPASSRLS role: the tables are under FORCE row level security, so any other
role would see no tenant, and the script refuses rather than report a run that measured nothing.
It exits non-zero when nothing was measured, a requested tenant has no active generation, or a
tenant failed (recorded under `failed`, the others still measured and written).

Read-only: SELECT statements and the production query path, nothing written; the listing
connection is read only for its whole session and autocommits, so it holds no lock between
statements; `--statement-timeout-ms`, when given, bounds every statement on both connections. It
prints and writes per-tenant aggregates (recall, counts, timings) and the generation id, never
chunk ids, scores or text. Corpus vectors are not real questions, so this bounds the index's
behaviour on in-distribution points; it is not a retrieval-quality benchmark.

Each tenant's result: `recall_mean` and `recall_min` per k, `queries_below_full_recall` per k (the
queries that lost at least one neighbour), `short_results` (walks that returned fewer than the
largest k), `exact_fallbacks`, `ids_outside_generation` (returned ids that are not in the measured
generation), `share_of_table` (the tenant's rows over the whole table, the selectivity of the
filtered walk), the `ef_search` and `iterative_scan` it ran under, `hnsw_ms_p50` and `hnsw_ms_p95`,
and `remeasured_after_change`.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Any

import numpy as np
import psycopg
from pgvector.psycopg import register_vector

from recall.generation_store import GENERATION_DENSE_SEARCH_ENV, GenerationStore
from recall.observability import percentile

TIE_EPSILON = 1e-5
#: The dense path every measurement runs on, forced for its duration (see the module docstring).
MEASURED_PATH = "hnsw"
#: The variables that tune the measured walk, recorded raw in every result so two results
#: can be compared and tied to the setting an operator is about to opt into.
HNSW_ENV = (
    "RECALL_HNSW_EF_SEARCH_FILTERED",
    "RECALL_HNSW_ITERATIVE_SCAN_FILTERED",
    "RECALL_HNSW_EF_SEARCH_MULTIPLIER",
)
# The column width is fixed per installation at migration (`__RECALL_DIM__`); the store reads it
# only on its schema paths, which this script never calls, and queries go at the vectors' own width.
DIMENSION = 1024


def active_generations(conn: psycopg.Connection) -> list[tuple[str, str, int]]:
    return [
        (str(t), str(g), int(n))
        for t, g, n in conn.execute(
            """SELECT g.tenant_id, g.generation_id, count(c.chunk_id)
               FROM recall_generations g
               LEFT JOIN recall_chunks_v1 c
                 ON c.tenant_id = g.tenant_id AND c.generation_id = g.generation_id
               WHERE g.state = 'active'
               GROUP BY 1, 2 ORDER BY 3 DESC"""
        )
    ]


#: Rows normalised at a time, so the norm's temporaries are a block wide rather than the matrix's
#: size (16 MiB at 1,024 dimensions; the same waste #845 removed from the atomic rescue loader).
NORMALISE_BLOCK_ROWS = 4096


def generation_vectors(
    conn: psycopg.Connection, tenant: str, generation: str
) -> tuple[list[str], np.ndarray]:
    """The generation's ids and unit-normalised vectors, holding about one matrix in memory.

    Rows are streamed (libpq hands them over one at a time) into a matrix sized by a count, then
    normalised in place by blocks. A `fetchall` held every row as Python objects beside the matrix,
    and the whole-matrix norm and division added two more matrix-sized arrays.
    """
    expected = int(
        conn.execute(
            "SELECT count(*) FROM recall_chunks_v1 WHERE tenant_id = %s AND generation_id = %s",
            (tenant, generation),
        ).fetchone()[0]
    )
    ids: list[str] = []
    matrix: np.ndarray | None = None
    with conn.cursor() as cur:
        for chunk_id, embedding in cur.stream(
            "SELECT chunk_id, embedding FROM recall_chunks_v1 WHERE tenant_id = %s AND generation_id = %s",
            (tenant, generation),
        ):
            # pgvector >= 0.4 hands back `Vector` objects rather than arrays; older versions give arrays.
            vector = np.asarray(
                embedding.to_numpy() if hasattr(embedding, "to_numpy") else embedding, dtype=np.float32
            )
            if matrix is None:
                matrix = np.empty((expected, vector.shape[0]), dtype=np.float32)
            if len(ids) == expected:
                raise GenerationMoved(tenant)  # rows were added after the count
            matrix[len(ids)] = vector
            ids.append(str(chunk_id))
    if matrix is None:
        return ids, np.empty((0, 0), dtype=np.float32)
    matrix = matrix[: len(ids)]  # fewer rows than counted: an erasure, which the fingerprint check catches
    for start in range(0, len(ids), NORMALISE_BLOCK_ROWS):
        block = matrix[start : start + NORMALISE_BLOCK_ROWS]
        norms = np.sqrt(np.einsum("ij,ij->i", block, block))[:, None]
        norms[norms == 0] = 1.0
        block /= norms
    return ids, matrix


def sample_indices(ids: list[str], count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    order = sorted(range(len(ids)), key=lambda i: ids[i])  # stable across runs before shuffling
    rng.shuffle(order)
    return order[: min(count, len(ids))]


def rls_bypass_problem(conn: psycopg.Connection) -> str | None:
    """Why this role cannot measure anything, or None when it can.

    `recall_generations` and `recall_chunks_v1` are under FORCE row level security keyed on the
    per-connection tenant setting, which this script's listing connection does not set. A role
    without superuser or BYPASSRLS (the documented production role, and the table owner too) sees
    no rows at all, so the run would list no tenant, measure nothing and look clean. The same
    check `benchmarks/beam/transfer_index.py` makes before it reads every tenant.
    """
    row = conn.execute(
        "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
    ).fetchone()
    if row and row[0]:
        return None
    return (
        "REFUSED: this role is neither superuser nor BYPASSRLS, so row level security hides every "
        "generation from the listing and nothing would be measured. Run it with a role that can "
        "read every tenant."
    )


class GenerationMoved(Exception):
    """The tenant's served corpus changed while it was being measured.

    A promotion changes the active generation; an erasure (`forget`, `withdraw`) deletes rows in
    place and changes only the corpus fingerprint. Either makes the loaded ground truth stale.
    """


def measure_tenant(
    conn: psycopg.Connection, dsn: str, tenant: str, dimension: int, args: argparse.Namespace
) -> dict[str, Any]:
    # The store's own connections get the listing's read only, time-bounded session as well.
    timeout_ms = getattr(args, "statement_timeout_ms", None)
    store_dsn = psycopg.conninfo.make_conninfo(dsn, options=read_only_options(dsn, timeout_ms))
    store = GenerationStore(store_dsn, dimension, tenant=tenant)
    try:
        # The generation comes from the STORE, not from an earlier listing: the hourly refresh can
        # promote one between the two, and measuring one generation's vectors against another's
        # index would report a recall loss that is not there. Every query is pinned to it, so a
        # timed call is the served query alone and cannot drift to a newer generation mid-run.
        state = store._active_generation_state()  # noqa: SLF001, (generation, fingerprint)
        generation = state[0]
        with store.pin_generation(generation):
            ids, matrix = generation_vectors(conn, tenant, generation)
            row = _measure(store, generation, ids, matrix, args)
        if store._active_generation_state() != state:  # noqa: SLF001
            raise GenerationMoved(tenant)
        return row
    finally:
        store.close()


def _measure(
    store: GenerationStore, generation: str, ids: list[str], matrix: np.ndarray, args: argparse.Namespace
) -> dict[str, Any]:
    previous = os.environ.get(GENERATION_DENSE_SEARCH_ENV)
    os.environ[GENERATION_DENSE_SEARCH_ENV] = MEASURED_PATH
    try:
        return _measure_hnsw(store, generation, ids, matrix, args)
    finally:
        if previous is None:
            os.environ.pop(GENERATION_DENSE_SEARCH_ENV, None)
        else:
            os.environ[GENERATION_DENSE_SEARCH_ENV] = previous


def _measure_hnsw(
    store: GenerationStore, generation: str, ids: list[str], matrix: np.ndarray, args: argparse.Namespace
) -> dict[str, Any]:
    tenant = store.tenant
    position = {chunk_id: i for i, chunk_id in enumerate(ids)}
    ks = args.k
    top = max(ks)
    ef_search, iterative_scan = store._hnsw_filtered_tuning(top + 1)  # noqa: SLF001
    recall = {k: [] for k in ks}
    short = 0
    unknown_ids = 0
    exact_fallbacks = 0
    hnsw_ms: list[float] = []

    # An HNSW walk that returns no row is re-run by the store as an exact scan, whose answer would
    # score as perfect recall: the worst index failure would read as the best result. Count those
    # queries and score them as what the walk returned, nothing.
    fell_back = False
    exact_fallback = store._dense_exact_fallback  # noqa: SLF001

    def counting_fallback(sql: str, params: dict) -> list[tuple]:
        nonlocal fell_back
        fell_back = True
        return exact_fallback(sql, params)

    store._dense_exact_fallback = counting_fallback  # type: ignore[method-assign]  # noqa: SLF001
    try:
        for qi in sample_indices(ids, args.queries, args.seed):
            query = matrix[qi]
            scores = matrix @ query
            scores[qi] = -np.inf  # the query's own row is not a neighbour
            exact_order = np.argsort(-scores, kind="stable")[:top]

            fell_back = False
            started = time.perf_counter()
            hits = store.query_dense(query.tolist(), top + 1)
            hnsw_ms.append((time.perf_counter() - started) * 1000.0)
            if fell_back:
                exact_fallbacks += 1
                hits = []
            returned = [h.chunk.id for h in hits if h.chunk.id != ids[qi]][:top]
            if len(returned) < min(top, len(ids) - 1):
                short += 1
            # A returned id outside the generation is a miss at every k; counted once per id.
            unknown_ids += sum(1 for chunk_id in returned if chunk_id not in position)
            for k in ks:
                threshold = scores[exact_order[k - 1]] - TIE_EPSILON
                good = sum(
                    1
                    for chunk_id in returned[:k]
                    if chunk_id in position and scores[position[chunk_id]] >= threshold
                )
                recall[k].append(good / k)
    finally:
        del store._dense_exact_fallback  # noqa: SLF001
    ordered_ms = sorted(hnsw_ms)
    return {
        "tenant": tenant,
        "generation": generation,
        "path": MEASURED_PATH,
        "ef_search": ef_search,
        "iterative_scan": iterative_scan,
        "rows": len(ids),
        "queries": len(hnsw_ms),
        # Unrounded: one lost neighbour in a thousand queries must not read as 1.0.
        "recall_mean": {str(k): statistics.fmean(v) for k, v in recall.items()},
        "recall_min": {str(k): min(v) for k, v in recall.items()},
        "queries_below_full_recall": {str(k): sum(1 for x in v if x < 1.0) for k, v in recall.items()},
        "short_results": short,
        "exact_fallbacks": exact_fallbacks,
        "ids_outside_generation": unknown_ids,
        "hnsw_ms_p50": round(percentile(ordered_ms, 0.50, ndigits=None), 1),
        "hnsw_ms_p95": round(percentile(ordered_ms, 0.95, ndigits=None), 1),
    }


def format_row(row: dict[str, Any], ks: list[int]) -> str:
    """One console line per tenant; it shows every lost neighbour, not only a rounded mean."""
    r = row["recall_mean"]
    below = row["queries_below_full_recall"]
    return (
        f"{row['tenant']:<52} rows={row['rows']:>6} share={row['share_of_table']:.4f} "
        + " ".join(f"r@{k}={r[str(k)]:.4f}({below[str(k)]} below)" for k in ks)
        + f" short={row['short_results']} fallbacks={row['exact_fallbacks']}"
        + f" outside={row['ids_outside_generation']} ef={row['ef_search']}"
        + f" p50={row['hnsw_ms_p50']}ms p95={row['hnsw_ms_p95']}ms"
    )


def measure_all(
    conn: psycopg.Connection,
    generations: list[tuple[str, str, int]],
    table_rows: int,
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Measure every listed tenant; one tenant's failure is recorded and the run goes on."""
    ks = args.k
    results: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for tenant, _listed_generation, n in generations:
        if args.tenants and tenant not in args.tenants:
            continue
        if n < args.min_rows:
            skipped.append({"tenant": tenant, "rows": n})
            continue
        try:
            try:
                row = measure_tenant(conn, args.dsn, tenant, DIMENSION, args)
                row["remeasured_after_change"] = False
            except GenerationMoved:
                print(f"{tenant}: its served corpus changed mid-measurement; measuring it once more")
                row = measure_tenant(conn, args.dsn, tenant, DIMENSION, args)
                row["remeasured_after_change"] = True
        except Exception as exc:  # noqa: BLE001  # BROAD-CATCH: one tenant must not lose the run
            failures.append({"tenant": tenant, "error": f"{type(exc).__name__}: {exc}"})
            print(f"{tenant}: FAILED, {type(exc).__name__}: {exc}", flush=True)
            continue
        row["share_of_table"] = round(row["rows"] / table_rows, 5) if table_rows else 0.0
        results.append(row)
        print(format_row(row, ks), flush=True)
    return results, skipped, failures


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--dsn", default=os.environ.get("RECALL_DSN"),
        help="connection string (default: RECALL_DSN; prefer it, a password here reaches the process table)",
    )
    parser.add_argument("--tenants", nargs="*", help="limit to these tenants (default: every active one)")
    parser.add_argument("--queries", type=int, default=100, help="query vectors sampled per tenant, at least 1")
    parser.add_argument(
        "--k", type=int, nargs="+", default=[1, 5, 20],
        help="the cut-offs recall is scored at; each at least 1 and below --min-rows",
    )
    parser.add_argument("--min-rows", type=int, default=50, help="skip a tenant whose generation has fewer rows")
    parser.add_argument("--seed", type=int, default=20261001, help="seed for the query sample")
    parser.add_argument(
        "--statement-timeout-ms", type=int, default=None,
        help="statement timeout on every connection, in ms (0: no timeout; default: the role's own setting,"
        " which a value here would override)",
    )
    parser.add_argument("--out", help="write the full result as JSON here")
    args = parser.parse_args(argv)
    if not args.dsn:
        parser.error("--dsn or RECALL_DSN is required")
    # A k at or above a generation's row count has no k-th neighbour (the query's own row is
    # excluded), and a k below 1 scores nonsense; --min-rows bounds every measured generation.
    if args.queries < 1 or min(args.k) < 1 or max(args.k) >= args.min_rows:
        parser.error("--queries and every --k must be at least 1, and every --k below --min-rows")
    if args.statement_timeout_ms is not None and args.statement_timeout_ms < 0:
        parser.error("--statement-timeout-ms must not be negative")
    args.k = sorted(set(args.k))

    # Read only for the whole session, and autocommit, so every statement is its own short
    # transaction: a `SET default_transaction_read_only` here would land inside an implicit
    # transaction that already began read-write, and that one transaction would then hold its
    # table locks for the whole run, queueing a migration and every served query behind it.
    with psycopg.connect(args.dsn, autocommit=True, options=read_only_options(args.dsn, args.statement_timeout_ms)) as conn:
        register_vector(conn)
        problem = rls_bypass_problem(conn)
        if problem:
            print(problem, file=sys.stderr)
            return 2
        generations = active_generations(conn)
        table_rows = int(conn.execute("SELECT count(*) FROM recall_chunks_v1").fetchone()[0])
        results, skipped, failures = measure_all(conn, generations, table_rows, args)
    listed = {tenant for tenant, _generation, _n in generations}
    missing = sorted(set(args.tenants or ()) - listed)
    summary = {
        "table_rows": table_rows, "seed": args.seed, "queries_requested_per_tenant": args.queries,
        "k": args.k, "min_rows": args.min_rows, "tie_epsilon": TIE_EPSILON,
        "hnsw_env": {name: os.environ.get(name) for name in HNSW_ENV},
        "tenants": results, "skipped_below_min_rows": skipped, "failed": failures,
        "requested_but_not_active": missing,
    }
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
    print(f"skipped (< {args.min_rows} rows): {[s['tenant'] for s in skipped]}")
    return run_status(results, failures, missing)


def read_only_options(dsn: str, statement_timeout_ms: int | None = None) -> str:
    """The DSN's own `options` (a search_path, say) plus a read only session, and a statement
    timeout when one is given; the keyword would otherwise replace them rather than add to them."""
    existing = psycopg.conninfo.conninfo_to_dict(dsn).get("options") or ""
    timeout = "" if statement_timeout_ms is None else f" -c statement_timeout={statement_timeout_ms}"
    return f"{existing} -c default_transaction_read_only=on{timeout}".strip()


def run_status(results: list[dict[str, Any]], failures: list[dict[str, Any]], missing: list[str]) -> int:
    """0 only when something was measured and nothing failed or went missing."""
    if missing:
        print(f"requested tenants with no active generation: {missing}", file=sys.stderr)
    if not results:
        print("nothing was measured", file=sys.stderr)
        return 1
    return 1 if failures or missing else 0


if __name__ == "__main__":
    sys.exit(main())
