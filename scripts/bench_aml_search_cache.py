"""Time C9's canonical BM25 leg and supersession read, uncached against cached, on a local DB.

Builds a synthetic tenant of real Code4 windows (``build_chunks`` at 160/120 words, content only,
stable window identity), shaped like the 13,097 memory tenant the official run served, in a
throwaway table, then measures per Search:

* ``old``: ``rank_bm25_chunks(list(store.iter_chunks()), query, k=100, stable_ties=True)``, the
  code this change replaced;
* ``cold``: the first ``TenantSearchCache.rank_bm25`` (fingerprint, full read, index build, rank);
* ``warm``: every later one (fingerprint query plus postings scoring);
* the fingerprint query alone, and the supersession scan uncached against a cache hit;
* the snapshot's Python heap, by ``tracemalloc``.

Every timed query is also checked for exact parity with the reference, so a fast wrong answer
cannot pass for a result. No embedding provider is called: vectors are deterministic noise.

    python scripts/bench_aml_search_cache.py --dsn postgresql://recall:recall@127.0.0.1:5608/recall

The DSN has no default on purpose: this creates and drops a table, so it must be pointed at a
database the caller owns (``scripts/session-db.sh up``), never at a shared one.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
import threading
import time
import tracemalloc
import uuid

import psycopg

from recall.schema import LEDGER_TABLE, apply_migrations
from recall.store import PgVectorStore
from recall_aml.code4 import rank_bm25_chunks
from recall_aml.graph import attach_grounded_relations
from recall_aml.models import AddRequest, CodingMemoryRecord, EvidenceSpan, Message
from recall_aml.search_cache import (
    Bm25Snapshot,
    TenantSearchCache,
    _read_all,
    current_fingerprint,
)
from recall_aml.service import build_chunks
from recall_aml.variants import variant


C9 = variant("C9_routed_specialists_grounded_graph_atomic")
_ENGLISH = (
    "the a to of and in is it that for on with as was this be are by at from or an not have "
    "error test fix build run file function class module value return call config path query "
    "cache index retry timeout parser schema migration deploy server client request response "
    "thread lock async await import package version update change remove add check expected"
).split()


def _vocabulary(rng: random.Random, size: int) -> tuple[list[str], list[float]]:
    words = list(_ENGLISH)
    while len(words) < size:
        stem = rng.choice(["get", "set", "load", "parse", "build", "make", "handle", "run", "to"])
        noun = rng.choice(["user", "config", "cache", "item", "node", "path", "row", "file", "job"])
        words.append(f"{stem}_{noun}_{len(words)}" if rng.random() < 0.6 else f"sym{len(words)}")
    weights = [1.0 / (rank + 1) ** 1.05 for rank in range(len(words))]
    return words, weights


def _records(rng: random.Random, request: AddRequest, prior: list[str]) -> list:
    """Up to eight grounded compiled records per session, as the anchored compiler writes them."""
    records = []
    for index in range(rng.randint(1, 8)):
        ordinal = rng.randrange(len(request.messages))
        content = str(request.messages[ordinal].content)
        quote = content[:120]
        records.append(
            CodingMemoryRecord(
                kind=rng.choice(["successful repair", "root cause", "procedure", "constraint"]),
                task_shape=content[120:260],
                problem=content[260:520],
                action=content[520:900],
                outcome=content[900:1100] or "done",
                validation=content[1100:1300],
                entities=content.split()[:12],
                evidence_spans=[
                    EvidenceSpan(message_ordinal=ordinal, start=0, end=len(quote), quote=quote)
                ],
                source_session_id=request.session_id,
                supersedes=prior[-2:] if index == 0 else [],
            )
        )
    return records


def _chunks(target: int, seed: int, graph: list | None = None) -> list:
    rng = random.Random(seed)
    words, weights = _vocabulary(rng, 30_000)
    chunks: list = []
    prior_ids: list[str] = []
    session = 0
    while len(chunks) < target:
        session += 1
        messages = [
            Message(
                role="user" if index % 2 == 0 else "assistant",
                content=" ".join(rng.choices(words, weights, k=rng.randint(40, 400))),
            )
            for index in range(rng.randint(6, 30))
        ]
        request = AddRequest(
            request_id=f"bench-{session}",
            user_id="bench",
            session_id=f"sessions/bench-{session}.jsonl",
            messages=messages,
        )
        built = attach_grounded_relations(
            request,
            build_chunks(
                request,
                _records(rng, request, prior_ids) if graph is not None else [],
                embedding_profile=C9.embedding_profile,
                word_window_size=C9.word_window_size,
                word_window_stride=C9.word_window_stride,
                content_only_windows=C9.content_only_windows,
                stable_window_identity=C9.stable_window_order,
            ),
        )
        chunks.extend(chunk for chunk in built if chunk.metadata.get("record_type") == "raw")
        if graph is not None:
            compiled = [chunk for chunk in built if chunk.metadata.get("record_type") == "compiled"]
            graph.extend(compiled)
            prior_ids = [chunk.id for chunk in compiled]
    return chunks[:target]


def _queries(count: int, seed: int) -> list[str]:
    rng = random.Random(seed)
    words, weights = _vocabulary(random.Random(seed - 1), 30_000)
    return [" ".join(rng.choices(words, weights, k=rng.randint(6, 20))) for _ in range(count)]


def _exact(hits: list) -> list[tuple[object, ...]]:
    return [(hit.chunk, hit.score.hex()) for hit in hits]


def _summary(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    return {
        "n": len(ordered),
        "median_ms": round(statistics.median(ordered), 2),
        "min_ms": round(ordered[0], 2),
        "max_ms": round(ordered[-1], 2),
    }


def _timed(call) -> tuple[float, object]:
    started = time.perf_counter()
    value = call()
    return (time.perf_counter() - started) * 1_000, value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--chunks", type=int, default=13_097)
    parser.add_argument("--dim", type=int, default=1024)
    parser.add_argument("--old-queries", type=int, default=8)
    parser.add_argument("--warm-queries", type=int, default=60)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()

    table = "bench_" + uuid.uuid4().hex[:8]
    apply_migrations(args.dsn, table=table, dim=args.dim)
    store = PgVectorStore(args.dsn, dim=args.dim, table=table, pool_size=args.threads + 2)
    try:
        graph_chunks: list = []
        chunks = _chunks(args.chunks, seed=13_097, graph=graph_chunks)
        noise = random.Random(5)
        graph_store = PgVectorStore(args.dsn, dim=args.dim, table=table, tenant="bench_graph")
        for target, rows in ((store, chunks), (graph_store, graph_chunks)):
            for start in range(0, len(rows), 500):
                batch = rows[start : start + 500]
                target.upsert(
                    batch, [[noise.uniform(-1, 1) for _ in range(args.dim)] for _ in batch]
                )
        with psycopg.connect(args.dsn, autocommit=True) as conn:
            conn.execute(f"ANALYZE {table}")
        queries = _queries(args.warm_queries, seed=99)
        report: dict[str, object] = {"chunks": len(chunks), "dim": args.dim, "table": table}

        old: list[float] = []
        references: dict[str, list[tuple[object, ...]]] = {}
        for query in queries[: args.old_queries]:
            elapsed, ranked = _timed(
                lambda query=query: rank_bm25_chunks(
                    list(store.iter_chunks()), query, k=100, stable_ties=True
                )
            )
            old.append(elapsed)
            references[query] = _exact(ranked)  # type: ignore[arg-type]
        report["old"] = _summary(old)

        cache = TenantSearchCache()
        elapsed, (ranked, status) = _timed(  # type: ignore[misc]
            lambda: cache.rank_bm25(store, queries[0], k=100, stable_ties=True)
        )
        report["cold"] = {"ms": round(elapsed, 2), "status": status}
        mismatches = int(_exact(ranked) != references[queries[0]])

        warm: list[float] = []
        statuses: dict[str, int] = {}
        for query in queries:
            elapsed, (ranked, status) = _timed(  # type: ignore[misc]
                lambda query=query: cache.rank_bm25(store, query, k=100, stable_ties=True)
            )
            warm.append(elapsed)
            statuses[status] = statuses.get(status, 0) + 1
            if query in references:
                mismatches += int(_exact(ranked) != references[query])
        report["warm"] = {**_summary(warm), "statuses": statuses}
        report["parity_checked"] = len(references) + 1
        report["parity_mismatches"] = mismatches

        report["fingerprint"] = _summary(
            [_timed(lambda: current_fingerprint(store))[0] for _ in range(20)]
        )
        uncached = [_timed(store.explicit_superseded_chunk_ids)[0] for _ in range(10)]
        cache.superseded_ids(store)
        cached = [_timed(lambda: cache.superseded_ids(store))[0] for _ in range(10)]
        report["supersession_uncached"] = _summary(uncached)
        report["supersession_cached"] = _summary(cached)
        # The graph tenant: compiled records, large metadata, real `supersedes` arrays.
        graph_ids = graph_store.explicit_superseded_chunk_ids()
        report["graph_rows"] = len(graph_chunks)
        report["graph_superseded_ids"] = len(graph_ids)
        report["graph_parity"] = cache.superseded_ids(graph_store) == graph_ids
        report["graph_fingerprint"] = _summary(
            [_timed(lambda: current_fingerprint(graph_store))[0] for _ in range(20)]
        )
        report["graph_supersession_uncached"] = _summary(
            [_timed(graph_store.explicit_superseded_chunk_ids)[0] for _ in range(10)]
        )
        report["graph_supersession_cached"] = _summary(
            [_timed(lambda: cache.superseded_ids(graph_store))[0] for _ in range(10)]
        )

        # Concurrency: the official run's shape, several Searches on one tenant at once.
        def concurrent(call, count: int) -> list[float]:
            barrier = threading.Barrier(count)
            samples: list[float] = []
            lock = threading.Lock()

            def worker(query: str) -> None:
                barrier.wait()
                elapsed, _ = _timed(lambda: call(query))
                with lock:
                    samples.append(elapsed)

            threads = [
                threading.Thread(target=worker, args=(query,)) for query in queries[:count]
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            return samples

        report["old_concurrent"] = _summary(
            concurrent(
                lambda query: rank_bm25_chunks(
                    list(store.iter_chunks()), query, k=100, stable_ties=True
                ),
                min(4, args.threads),
            )
        )
        report["warm_concurrent"] = _summary(
            concurrent(
                lambda query: cache.rank_bm25(store, query, k=100, stable_ties=True),
                args.threads,
            )
        )

        rows, fingerprint = _read_all(store)
        tracemalloc.start()
        before = tracemalloc.take_snapshot()
        snapshot = Bm25Snapshot.build(rows, fingerprint)
        del rows
        after = tracemalloc.take_snapshot()
        tracemalloc.stop()
        built = sum(stat.size_diff for stat in after.compare_to(before, "filename"))
        report["snapshot_heap_mb_excluding_rows"] = round(built / 2**20, 1)
        tracemalloc.start()
        rows_again, _ = _read_all(store)
        current, _peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        report["rows_heap_mb"] = round(current / 2**20, 1)
        del rows_again, snapshot
        print(json.dumps(report, indent=2))
    finally:
        graph_store.close()
        store.drop_table()
        store.close()
        with psycopg.connect(args.dsn, autocommit=True) as conn:
            conn.execute(f"DELETE FROM {LEDGER_TABLE} WHERE target_table = %s", (table,))


if __name__ == "__main__":
    main()
