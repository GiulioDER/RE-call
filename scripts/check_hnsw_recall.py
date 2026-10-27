"""Measure what the served dense path actually returns, per tenant, against exact search.

    python scripts/check_hnsw_recall.py --out recall.json            # every active tenant
    python scripts/check_hnsw_recall.py --tenants memory --queries 200

For each tenant's ACTIVE generation in `recall_chunks_v1` this samples query vectors from the
generation itself (deterministically, by seed), asks the production path
`GenerationStore.query_dense` for the top k, and scores it against exact cosine search computed in
numpy over the same generation's vectors. The query's own row is excluded from both lists.

Why it exists. The HNSW index is built over the whole table, every tenant and every retained
generation, while a query asks for one tenant's active generation: a small slice, which is the case
where a filtered graph walk loses neighbours. `_hnsw_filtered_tuning` and pgvector's iterative scans
are the mitigation; this measures whether they are enough on the real corpus, after a pgvector
upgrade, a re-index, or a generation gc changes the graph under them. Nothing else does.

Recall is tie tolerant: a returned row counts if its exact score reaches the k-th exact score
(within 1e-5), so equal-scoring duplicates cannot make a correct answer look wrong.

Read-only: SELECT statements and the production query path, nothing written. It prints ids,
scores and timings, never chunk text. Corpus vectors are not real questions, so this bounds the
index's behaviour on in-distribution points; it is not a retrieval-quality benchmark.
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

from recall.generation_store import GenerationStore

TIE_EPSILON = 1e-5
# `recall_chunks_v1.embedding` is one fixed-width column for every tenant, whatever its embedder.
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


def generation_vectors(
    conn: psycopg.Connection, tenant: str, generation: str
) -> tuple[list[str], np.ndarray]:
    rows = conn.execute(
        "SELECT chunk_id, embedding FROM recall_chunks_v1 WHERE tenant_id = %s AND generation_id = %s",
        (tenant, generation),
    ).fetchall()
    ids = [str(r[0]) for r in rows]
    # pgvector >= 0.4 hands back `Vector` objects rather than arrays; older versions give arrays.
    matrix = np.asarray(
        [np.asarray(r[1].to_numpy() if hasattr(r[1], "to_numpy") else r[1], dtype=np.float32) for r in rows],
        dtype=np.float32,
    )
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return ids, matrix / norms


def sample_indices(ids: list[str], count: int, seed: int) -> list[int]:
    rng = np.random.default_rng(seed)
    order = sorted(range(len(ids)), key=lambda i: ids[i])  # stable across runs before shuffling
    rng.shuffle(order)
    return order[: min(count, len(ids))]


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int(round(q * (len(ordered) - 1))))]


class GenerationMoved(Exception):
    """The tenant's active generation changed while it was being measured (a promotion landed)."""


def measure_tenant(
    conn: psycopg.Connection, dsn: str, tenant: str, dimension: int, args: argparse.Namespace
) -> dict[str, Any]:
    store = GenerationStore(dsn, dimension, tenant=tenant)
    try:
        # The generation comes from the STORE, not from an earlier listing: the hourly refresh can
        # promote one between the two, and measuring one generation's vectors against another's
        # index would report a recall loss that is not there.
        generation = store._generation_id()  # noqa: SLF001, the pin is what is under test
        ids, matrix = generation_vectors(conn, tenant, generation)
        row = _measure(store, generation, ids, matrix, args)
        if store._generation_id() != generation:  # noqa: SLF001
            raise GenerationMoved(tenant)
        return row
    finally:
        store.close()


def _measure(
    store: GenerationStore, generation: str, ids: list[str], matrix: np.ndarray, args: argparse.Namespace
) -> dict[str, Any]:
    tenant = store._tenant  # noqa: SLF001
    position = {chunk_id: i for i, chunk_id in enumerate(ids)}
    ks = sorted(set(args.k))
    top = max(ks)
    recall = {k: [] for k in ks}
    short = 0
    unknown_ids = 0
    hnsw_ms: list[float] = []
    for qi in sample_indices(ids, args.queries, args.seed):
        query = matrix[qi]
        scores = matrix @ query
        scores[qi] = -np.inf  # the query's own row is not a neighbour
        exact_order = np.argsort(-scores, kind="stable")[:top]

        started = time.perf_counter()
        hits = store.query_dense(query.tolist(), top + 1)
        hnsw_ms.append((time.perf_counter() - started) * 1000.0)
        returned = [h.chunk.id for h in hits if h.chunk.id != ids[qi]][:top]
        if len(returned) < min(top, len(ids) - 1):
            short += 1
        for k in ks:
            threshold = scores[exact_order[k - 1]] - TIE_EPSILON
            good = 0
            for chunk_id in returned[:k]:
                index = position.get(chunk_id)
                if index is None:
                    unknown_ids += 1
                    continue
                if scores[index] >= threshold:
                    good += 1
            recall[k].append(good / k)
    return {
        "tenant": tenant,
        "generation": generation,
        "rows": len(ids),
        "queries": len(hnsw_ms),
        "recall_mean": {str(k): round(statistics.fmean(v), 4) for k, v in recall.items()},
        "recall_min": {str(k): round(min(v), 4) for k, v in recall.items()},
        "queries_below_full_recall": {str(k): sum(1 for x in v if x < 1.0) for k, v in recall.items()},
        "short_results": short,
        "ids_outside_generation": unknown_ids,
        "hnsw_ms_p50": round(percentile(hnsw_ms, 0.50), 1),
        "hnsw_ms_p95": round(percentile(hnsw_ms, 0.95), 1),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dsn", default=os.environ.get("RECALL_DSN"))
    parser.add_argument("--tenants", nargs="*", help="limit to these tenants (default: every active one)")
    parser.add_argument("--queries", type=int, default=100)
    parser.add_argument("--k", type=int, nargs="+", default=[1, 5, 20])
    parser.add_argument("--min-rows", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261001)
    parser.add_argument("--out", help="write the full result as JSON here")
    args = parser.parse_args(argv)
    if not args.dsn:
        parser.error("--dsn or RECALL_DSN is required")

    with psycopg.connect(args.dsn) as conn:
        register_vector(conn)
        conn.execute("SET default_transaction_read_only = on")
        generations = active_generations(conn)
        table_rows = int(conn.execute("SELECT count(*) FROM recall_chunks_v1").fetchone()[0])
        results, skipped = [], []
        for tenant, generation, n in generations:
            if args.tenants and tenant not in args.tenants:
                continue
            if n < args.min_rows:
                skipped.append({"tenant": tenant, "rows": n})
                continue
            try:
                row = measure_tenant(conn, args.dsn, tenant, DIMENSION, args)
            except GenerationMoved:
                print(f"{tenant}: a new generation was promoted mid-measurement; measuring it once more")
                row = measure_tenant(conn, args.dsn, tenant, DIMENSION, args)
                row["remeasured_after_promotion"] = True
            row["share_of_table"] = round(row["rows"] / table_rows, 5)
            results.append(row)
            r = row["recall_mean"]
            print(
                f"{tenant:<52} rows={row['rows']:>6} share={row['share_of_table']:.4f} "
                + " ".join(f"r@{k}={r[str(k)]:.3f}" for k in sorted(set(args.k)))
                + f" short={row['short_results']} p50={row['hnsw_ms_p50']}ms p95={row['hnsw_ms_p95']}ms",
                flush=True,
            )
    summary = {"table_rows": table_rows, "seed": args.seed, "queries_per_tenant": args.queries,
               "k": sorted(set(args.k)), "tenants": results, "skipped_below_min_rows": skipped}
    if args.out:
        with open(args.out, "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=1)
    print(f"skipped (< {args.min_rows} rows): {[s['tenant'] for s in skipped]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
