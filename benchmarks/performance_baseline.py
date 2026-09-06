"""Run the preregistered retrieval performance baseline.

The protocol is frozen in benchmarks/PREREGISTRATION-performance-baseline.md. This runner uses one
fresh child process per configuration, records raw request and stage observations, and writes a
JSON source of truth plus a compact Markdown view. The default invocation is the complete matrix.
Use --smoke only to validate the harness; smoke output is explicitly marked invalid.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import ctypes
import hashlib
import json
import os
import platform
import subprocess
import sys
import time
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict, cast

from recall.calibration import Calibration
from recall.embeddings import Embedder, FastEmbedEmbedder, HashingEmbedder, embedding_profile_id
from recall.observability import percentile
from recall.profiles import (
    FAST_PROFILE,
    QUALITY_PROFILE,
    RetrievalAdmission,
    RetrievalOverloaded,
    RetrievalProfile,
)
from recall.rerank import CrossEncoderReranker, PINNED_RERANKER_SHA256, Reranker
from recall.store import PgVectorStore
from recall.trust import trusted_search
from recall.trust_policy import TrustPolicy
from recall.types import Chunk, TrustedResult
from recall.schema import apply_migrations

PROTOCOL = Path("benchmarks/PREREGISTRATION-performance-baseline.md")
FIXTURE = Path("benchmarks/fixtures/performance_baseline.json")
STAGES = (
    "query_embedding",
    "dense_retrieval",
    "sparse_retrieval",
    "learned_sparse_retrieval",
    "fusion",
    "reranking",
    "trust_evaluation",
    "evidence_assembly",
    "admission_wait",
)
PROFILES = {"fast": FAST_PROFILE, "quality": QUALITY_PROFILE}


class BenchmarkConfig(TypedDict):
    profile: str
    k: int
    candidate_k: int
    concurrency: int


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rss_bytes() -> int:
    """Return current process RSS without adding a benchmark-only dependency."""
    if os.name == "nt":
        class MemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", ctypes.c_ulong),
                ("page_faults", ctypes.c_ulong),
                ("peak_ws", ctypes.c_size_t),
                ("ws", ctypes.c_size_t),
                ("quota_peak", ctypes.c_size_t),
                ("quota", ctypes.c_size_t),
                ("pool_nonpaged", ctypes.c_size_t),
                ("pool_paged", ctypes.c_size_t),
                ("pagefile", ctypes.c_size_t),
                ("private", ctypes.c_size_t),
                ("fault_count", ctypes.c_ulong),
            ]

        counters = MemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        handle = ctypes.windll.kernel32.GetCurrentProcess()
        ok = ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        )
        return int(counters.ws) if ok else 0
    proc = Path("/proc/self/statm")
    if proc.exists():
        sysconf = cast(Any, getattr(os, "sysconf"))
        return int(proc.read_text().split()[1]) * int(sysconf("SC_PAGE_SIZE"))
    try:
        import resource

        getrusage = cast(Any, getattr(resource, "getrusage"))
        value = getrusage(getattr(resource, "RUSAGE_SELF")).ru_maxrss
        return int(value * (1024 if sys.platform != "darwin" else 1))
    except (ImportError, OSError, ValueError):
        return 0


def _nearest(values: list[float], q: float) -> float | None:
    return percentile(sorted(values), q, ndigits=None) if values else None


def _fixture() -> tuple[list[dict[str, Any]], list[str]]:
    payload = json.loads(FIXTURE.read_text(encoding="utf-8"))
    corpus = payload["corpus"]
    queries = payload["queries"]
    if not isinstance(corpus, list) or not isinstance(queries, list) or not corpus or not queries:
        raise ValueError("performance fixture must contain nonempty corpus and query lists")
    return corpus, [str(query) for query in queries]


def _build_embedder(name: str) -> Embedder:
    if name == "hashing":
        return HashingEmbedder(dim=64)
    return FastEmbedEmbedder(model_name=name)


def _build_reranker(profile: str, path: str | None, digest: str | None) -> Reranker | None:
    if profile == "fast":
        return None
    if not path or digest != PINNED_RERANKER_SHA256:
        raise ValueError(
            "quality requires --reranker-path and the pinned reranker digest"
        )
    return CrossEncoderReranker(
        model=path,
        local_files_only=True,
        artifact_sha256=digest,
        inference_threads=QUALITY_PROFILE.inference_threads,
    )


def _assemble(result: TrustedResult) -> None:
    """Exercise the bounded JSON-shaped evidence fields used by the service surface."""
    json.dumps(
        [
            {"id": hit.chunk.id, "source": hit.provenance.file, "verdict": hit.verdict}
            for hit in result.hits
        ],
        separators=(",", ":"),
    )


def _one_request(
    store: PgVectorStore,
    embedder: Embedder,
    query: str,
    profile: RetrievalProfile,
    candidate_k: int,
    k: int,
    reranker: Reranker | None,
    calibration: Calibration,
    request_id: int,
) -> dict[str, Any]:
    started = time.perf_counter()
    row: dict[str, Any] = {"request_id": request_id, "query": query, "ok": False}
    try:
        admitted_at = time.perf_counter()
        with RetrievalAdmission(profile):
            admission_wait = (time.perf_counter() - admitted_at) * 1000.0
            result = trusted_search(
                store,
                embedder,
                query,
                k=k,
                calibration=calibration,
                reranker=reranker,
                candidate_k=candidate_k,
                retrieval_profile=profile.name,
                index_generation="benchmark",
                policy=TrustPolicy.development(),
            )
            assembly_started = time.perf_counter()
            _assemble(result)
            assembly_ms = (time.perf_counter() - assembly_started) * 1000.0
        row.update(
            {
                "ok": True,
                "stage_ms": dict(result.diagnostics.stage_ms)
                | {
                    "admission_wait": admission_wait,
                    "evidence_assembly": assembly_ms,
                },
                "hit_count": len(result.hits),
            }
        )
    except RetrievalOverloaded as exc:
        row.update({"error_class": type(exc).__name__, "error_reason": exc.reason})
    except Exception as exc:  # BROAD-CATCH: error-translation
        row.update({"error_class": type(exc).__name__, "error_reason": "unexpected"})
    row["elapsed_ms"] = (time.perf_counter() - started) * 1000.0
    row["rss_bytes"] = _rss_bytes()
    return row


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    successes = [row for row in rows if row["ok"]]
    failures = [row for row in rows if not row["ok"]]
    stage: dict[str, dict[str, Any]] = {}
    for name in STAGES:
        values = [
            float(row["stage_ms"][name])
            for row in successes
            if name in row.get("stage_ms", {})
        ]
        stage[name] = {
            "n": len(values),
            "p50_ms": _nearest(values, 0.50),
            "p95_ms": _nearest(values, 0.95),
        }
    errors: dict[str, int] = {}
    for row in failures:
        key = str(row.get("error_reason") or row.get("error_class") or "unknown")
        errors[key] = errors.get(key, 0) + 1
    rss = [int(row["rss_bytes"]) for row in rows if row.get("rss_bytes")]
    elapsed = [float(row["elapsed_ms"]) for row in rows]
    successful_elapsed = [float(row["elapsed_ms"]) for row in successes]
    return {
        "attempted": len(rows),
        "successes": len(successes),
        "failures": len(failures),
        "error_rate": len(failures) / len(rows) if rows else None,
        "errors": errors,
        "latency_ms": {
            "p50": _nearest(elapsed, 0.50),
            "p95": _nearest(elapsed, 0.95),
        },
        "successful_latency_ms": {
            "p50": _nearest(successful_elapsed, 0.50),
            "p95": _nearest(successful_elapsed, 0.95),
        },
        "rss_bytes": {
            "peak": max(rss, default=None),
            "baseline": rss[0] if rss else None,
            "delta_peak": max(rss, default=0) - (rss[0] if rss else 0),
        },
        "stages": stage,
    }


def _worker(
    config: BenchmarkConfig,
    dsn: str,
    table: str,
    embedder_name: str,
    reranker_path: str | None,
    reranker_digest: str | None,
    queries: list[str],
    warmup: int,
    samples: int,
    output: Path,
) -> None:
    profile = PROFILES[config["profile"]]
    started = time.perf_counter()
    embedder = _build_embedder(embedder_name)
    reranker = _build_reranker(config["profile"], reranker_path, reranker_digest)
    pool_size = profile.max_concurrency + profile.queue_capacity
    store = PgVectorStore(
        dsn,
        dim=embedder.dim,
        table=table,
        tenant="benchmark",
        pool_size=pool_size,
        generation_id="benchmark",
    )
    calibration = Calibration(
        embedder=embedding_profile_id(embedder),
        threshold=0.0,
        scale=0.05,
        separability=1.0,
        n_answerable=20,
        n_unanswerable=20,
    )
    construction_ms = (time.perf_counter() - started) * 1000.0
    rows: list[dict[str, Any]] = []
    try:
        cold = _one_request(
            store,
            embedder,
            queries[0],
            profile,
            config["candidate_k"],
            config["k"],
            reranker,
            calibration,
            0,
        )
        cold["phase"] = "cold"
        cold["construction_ms"] = construction_ms
        rows.append(cold)
        for index in range(warmup):
            _one_request(
                store,
                embedder,
                queries[index % len(queries)],
                profile,
                config["candidate_k"],
                config["k"],
                reranker,
                calibration,
                -(index + 1),
            )
        offered = config["concurrency"]
        completed = 0
        while completed < samples:
            batch_size = min(offered, samples - completed)
            query = queries[completed % len(queries)]
            if batch_size == 1:
                batch = [
                    _one_request(
                        store,
                        embedder,
                        query,
                        profile,
                        config["candidate_k"],
                        config["k"],
                        reranker,
                        calibration,
                            completed + 1,
                        )
                    ]
            else:
                with concurrent.futures.ThreadPoolExecutor(max_workers=offered) as executor:
                    futures = [
                        executor.submit(
                            _one_request,
                            store,
                            embedder,
                            queries[(completed + offset) % len(queries)],
                            profile,
                            config["candidate_k"],
                            config["k"],
                            reranker,
                            calibration,
                            completed + offset + 1,
                        )
                        for offset in range(batch_size)
                    ]
                    batch = [future.result() for future in futures]
            for row in batch:
                row["phase"] = "warm"
            rows.extend(batch)
            completed += batch_size
        output.write_text(
            json.dumps(
                {
                    "config": config,
                    "construction_ms": construction_ms,
                    "rows": rows,
                    "summary": {
                        "cold": _summary([cold]),
                        "warm": _summary(
                            [row for row in rows if row["phase"] == "warm"]
                        ),
                    },
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    finally:
        store.close()


def _metadata(dsn: str, embedder: Embedder, table: str) -> dict[str, Any]:
    import psycopg

    with psycopg.connect(dsn, autocommit=True) as conn:
        row = conn.execute("SELECT version()").fetchone()
        postgres = str(row[0] if row else "unknown")
    return {
        "os": platform.platform(),
        "cpu": platform.processor(),
        "cpu_count": os.cpu_count(),
        "python": platform.python_version(),
        "postgres": postgres,
        "embedder": embedding_profile_id(embedder),
        "dimension": embedder.dim,
        "table": table,
    }


def _markdown(payload: dict[str, Any]) -> str:
    lines = [
        "# RE-call retrieval performance baseline",
        "",
        f"Measured at {payload['measured_at']} from commit {payload['git_commit']}.",
        "",
        "The JSON artifact is authoritative. Percentiles use nearest rank and raw rows remain in the artifact.",
        "",
        "| profile | state | k | candidate pool | concurrency | p50 ms | p95 ms | error rate | peak RSS |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for run in payload["runs"]:
        config = run["config"]
        for state in ("cold", "warm"):
            summary = run["summary"][state]
            latency = summary["latency_ms"]
            lines.append(
                f"| {config['profile']} | {state} | {config['k']} | "
                f"{config['candidate_k']} | {config['concurrency']} | "
                f"{latency['p50']} | {latency['p95']} | "
                f"{summary['error_rate']} | {summary['rss_bytes']['peak']} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.performance_baseline"
    )
    parser.add_argument("--dsn", default=os.environ.get("RECALL_TEST_DSN"))
    parser.add_argument("--embedder", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--reranker-path")
    parser.add_argument("--reranker-digest", default=PINNED_RERANKER_SHA256)
    parser.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args(argv)
    if not args.dsn:
        parser.error("--dsn or RECALL_TEST_DSN is required; refusing a shared database")
    if subprocess.run(
        ["git", "diff", "--exit-code", "--", str(PROTOCOL)],
        capture_output=True,
    ).returncode:
        parser.error(f"{PROTOCOL} is dirty; amend the protocol in a new file before measuring")
    corpus, queries = _fixture()
    protocol_commit = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    embedder = _build_embedder(args.embedder)
    table = "perf_baseline_" + uuid.uuid4().hex[:10]
    apply_migrations(args.dsn, dim=embedder.dim)
    apply_migrations(args.dsn, table=table, dim=embedder.dim)
    store = PgVectorStore(
        args.dsn,
        dim=embedder.dim,
        table=table,
        tenant="benchmark",
        generation_id="benchmark",
    )
    store.check_schema()
    chunks = [
        Chunk(item["id"], item["source"], item["text"], {"file": item["source"]})
        for item in corpus
    ]
    store.upsert(chunks, embedder.embed([item.text for item in chunks]))
    store.close()
    args.out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    configs: list[BenchmarkConfig] = [
        {
            "profile": profile,
            "k": k,
            "candidate_k": candidate_k,
            "concurrency": concurrency,
        }
        for profile in PROFILES
        for k in ((1,) if args.smoke else (1, 5, 10))
        for candidate_k in ((20,) if args.smoke else (20, 50, 100))
        for concurrency in ((1,) if args.smoke else (1, 2, 4, 8))
    ]
    runs: list[dict[str, Any]] = []
    try:
        for config in configs:
            if config["candidate_k"] < config["k"]:
                raise ValueError(f"invalid configuration refused: {config}")
            path = args.out / (
                f"{stamp}.{config['profile']}.{config['k']}."
                f"{config['candidate_k']}.{config['concurrency']}.json"
            )
            import multiprocessing as mp

            process = mp.get_context("spawn").Process(
                target=_worker,
                args=(
                    config,
                    args.dsn,
                    table,
                    args.embedder,
                    args.reranker_path,
                    args.reranker_digest,
                    queries,
                    1 if args.smoke else 10,
                    2 if args.smoke else 50,
                    path,
                ),
            )
            process.start()
            process.join()
            if process.exitcode != 0 or not path.exists():
                raise RuntimeError(
                    f"benchmark worker failed for {config} with exit {process.exitcode}"
                )
            runs.append(json.loads(path.read_text(encoding="utf-8")))
            path.unlink()
    finally:
        cleanup = PgVectorStore(
            args.dsn, dim=embedder.dim, table=table, tenant="benchmark",
            generation_id="benchmark",
        )
        cleanup.drop_table()
        cleanup.close()
    payload = {
        "protocol": {
            "path": str(PROTOCOL),
            "commit": protocol_commit,
            "sha256": _sha256(PROTOCOL),
        },
        "valid": not args.smoke,
        "measured_at": datetime.now(UTC).isoformat(),
        "git_commit": protocol_commit,
        "fixture": {
            "path": str(FIXTURE),
            "sha256": _sha256(FIXTURE),
            "corpus_count": len(corpus),
            "query_count": len(queries),
        },
        "metadata": _metadata(args.dsn, embedder, table),
        "runs": runs,
    }
    output = args.out / f"performance_baseline_{stamp}.json"
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    (args.out / f"performance_baseline_{stamp}.md").write_text(
        _markdown(payload), encoding="utf-8"
    )
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
