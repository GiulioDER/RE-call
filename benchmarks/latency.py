"""Isolated latency micro-benchmark: RE-call vs Mem0, memory layer only (no generator/judge).

The accuracy runs cannot prove a speed claim: they record no timing, their wall-clock is dominated
by the shared generator+judge, and the arms use different storage backends. This measures the two
things that actually differ:

- **ingest** (`system.ingest(conv)`): RE-call embeds+stores (no LLM); Mem0 runs an LLM
  fact-extraction call per session. This is the write-path gap the cost edge predicts, measured.
- **retrieve** (`system.retrieve(question)`): the genuinely-open question. RE-call does hybrid
  dense+sparse fusion + the trust layer; Mem0 does a vector search. Costs $0 (all local).

Design for a defensible number:
- **Local embedder** (bge-small) for BOTH arms, so query embedding is local — otherwise a cloud
  embedder's network call dominates `retrieve` and masks the difference we want to see.
- **Sequential, single process, one arm at a time.** Run on an IDLE machine (after the accuracy
  runs finish) or the numbers are polluted by contention.
- **Warm-up passes discarded**, then each query retrieved `--reps` times; report median, p90, p95
  and a bootstrap 95% CI on the median — not one mean.

Honest caveats (printed with the result):
- `RecallSystem.retrieve` opens a fresh Postgres connection per call; a pooled deployment would be
  faster. Mem0 uses an in-process Qdrant. This is an "as-deployed via these adapters" number, not a
  pure algorithm comparison.
- Mem0's ingest spends a little (its extraction LLM calls) — unavoidable, and exactly the cost.

Run from the repo root, on an idle machine:
    python -m benchmarks.latency --conversations 2 --queries-per-conv 40 --reps 5
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from statistics import median
from typing import Any

from benchmarks.systems import Mem0System, MemorySystem, RecallSystem
from recall.eval.locomo import DEFAULT_DSN


def _percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    i = min(len(s) - 1, max(0, int(round(q * (len(s) - 1)))))
    return s[i]


def _boot_median_ci(xs: list[float], reps: int, rng: random.Random) -> tuple[float, float]:
    n = len(xs)
    if n < 2:
        return (float("nan"), float("nan"))
    meds = []
    for _ in range(reps):
        sample = [xs[rng.randrange(n)] for _ in range(n)]
        meds.append(median(sample))
    meds.sort()
    return (_percentile(meds, 0.025), _percentile(meds, 0.975))


def _questions(conv: dict[str, Any], q_per: int, rng: random.Random) -> list[str]:
    qs = [str(qa.get("question") or "").strip() for qa in (conv.get("qa") or [])]
    qs = [q for q in qs if q]
    rng.shuffle(qs)
    return qs[:q_per]


def _build(arm: str, embedder: str, model: str, k: int, dsn: str, table: str, run_id: str) -> MemorySystem:
    if arm == "recall":
        return RecallSystem(dsn, embedder_name=embedder, k=k, table=table)
    key = os.environ["OPENROUTER_API_KEY"]
    hf = embedder[len("fastembed:"):] if embedder.startswith("fastembed:") else "BAAI/bge-small-en-v1.5"
    return Mem0System(key, model, k=k, run_id=run_id, hf_model=hf)


def _run_arm(
    arm: str, convs: list[dict[str, Any]], embedder: str, model: str, k: int, dsn: str,
    table: str, q_per: int, reps: int, warmup: int, rng: random.Random, stamp: str,
) -> tuple[list[float], list[float]]:
    system = _build(arm, embedder, model, k, dsn, f"{table}", f"latency-{arm}-{stamp}")
    ingest_secs: list[float] = []
    retrieve_ms: list[float] = []
    warmed = False
    try:
        for conv in convs:
            t0 = time.perf_counter()
            system.ingest(conv)
            ingest_secs.append(time.perf_counter() - t0)
            qs = _questions(conv, q_per, rng)
            if not warmed:
                for q in qs[:warmup]:
                    system.retrieve(q)  # discard: warm connections / caches
                warmed = True
            for q in qs:
                for _ in range(reps):
                    t = time.perf_counter()
                    system.retrieve(q)
                    retrieve_ms.append((time.perf_counter() - t) * 1000.0)
        return ingest_secs, retrieve_ms
    finally:
        # This function builds an arm and then abandons it, so its handles have to be released
        # here: a `Mem0System` left open holds exclusive Qdrant locks for the rest of the process
        # (see `Mem0System.close`). Note the two arms as they run today do NOT collide on the
        # per-run store — `recall` opens no Qdrant at all and each arm gets its own `run_id` — but
        # mem0's telemetry store lives on one machine-global path, and a lock left to the garbage
        # collector can outlive the process's ability to release it. Duck-typed on purpose:
        # `MemorySystem` is deliberately three members, and `RecallSystem` has nothing to release
        # — it holds a DSN string, not a connection.
        close = getattr(system, "close", None)
        try:
            if callable(close):
                close()
        finally:
            # Releasing handles is not reclaiming memory, and this module is the one place that
            # difference is measurable. Both arms hold an embedder inside a REFERENCE CYCLE, so
            # dropping the last reference frees the wrapper and leaves the weights: measured
            # against the real class, dropping mem0's `HuggingFaceEmbedding` keeps its
            # `SentenceTransformer` (~133MB) alive until a collection runs, and
            # `RecallSystem._embedder` is abandoned the same way. `main` times the arms
            # sequentially in one process, so without this arm 2 is measured on a machine still
            # holding arm 1's model, and the comparison this module exists to produce is between
            # an arm that ran clean and one that did not.
            #
            # `del` BOTH names first, and that is the whole trick rather than a tidy-up. While
            # this frame is alive, `system` and the bound method in `close` are each a strong
            # reference, so the arm is still REACHABLE and a collection here frees nothing.
            # Measured: with both bound, `gc.collect()` returns 0 and the arm stays alive; with
            # only `system` deleted it still returns without freeing the arm, because the bound
            # method pins it independently; with both deleted the cycle goes.
            #
            # This reclaims on the SUCCESS path only, which is the path that has a next arm. When
            # the run raises, the propagating traceback holds the arm's own method frame, whose
            # `self` pins the arm, and no `del` here can reach that. Not worth chasing: `main`
            # catches nothing, so a crashed arm ends the process.
            #
            # HERE and not in `Mem0System.close`: a library teardown should not run a full
            # collection to flatter a benchmark, and `RecallSystem` has no `close` at all yet
            # holds an embedder just the same. Bare `gc.collect()` is a full collection; a
            # generational one would skip a model already promoted out of the young generations.
            # The cost is single-digit milliseconds, once per arm, outside every measured region:
            # `ingest_secs` and `retrieve_ms` are both complete before `finally` runs.
            del close, system
            gc.collect()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m benchmarks.latency")
    p.add_argument("--conversations", type=int, default=2)
    p.add_argument("--queries-per-conv", type=int, default=40)
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--warmup", type=int, default=8)
    p.add_argument("--embedder", default="fastembed", help="LOCAL embedder recommended (default bge-small)")
    p.add_argument("--model", default="openai/gpt-4o-mini", help="Mem0 extraction model (ingest only)")
    p.add_argument("--k", type=int, default=10)
    p.add_argument("--table", default="bench_latency_chunks")
    p.add_argument("--data", type=Path, default=Path("locomo10.json"))
    p.add_argument("--out", type=Path, default=Path("benchmarks/results"))
    p.add_argument("--seed", type=int, default=1234)
    args = p.parse_args(argv)

    if "OPENROUTER_API_KEY" not in os.environ:
        p.error("OPENROUTER_API_KEY must be set (Mem0's ingest extraction needs it)")
    convs = json.loads(args.data.read_text(encoding="utf-8"))[: args.conversations]
    dsn = os.environ.get("RECALL_TEST_DSN") or os.environ.get("RECALL_DSN") or DEFAULT_DSN
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"convs={len(convs)} q/conv={args.queries_per_conv} reps={args.reps} embedder={args.embedder}")
    print("(sequential, memory-layer only; run on an IDLE machine)\n")

    results: dict[str, Any] = {}
    for arm in ("recall", "mem0"):
        rng = random.Random(args.seed)  # same query sample + bootstrap draws per arm
        ing, ret = _run_arm(arm, convs, args.embedder, args.model, args.k, dsn,
                            args.table, args.queries_per_conv, args.reps, args.warmup, rng, stamp)
        ci = _boot_median_ci(ret, 2000, rng)
        results[arm] = {
            "ingest_secs_total": round(sum(ing), 2),
            "ingest_secs_per_conv": [round(x, 2) for x in ing],
            "retrieve_ms": {
                "n": len(ret),
                "median": round(median(ret), 2) if ret else None,
                "median_ci95": [round(ci[0], 2), round(ci[1], 2)],
                "p90": round(_percentile(ret, 0.90), 2) if ret else None,
                "p95": round(_percentile(ret, 0.95), 2) if ret else None,
                "mean": round(sum(ret) / len(ret), 2) if ret else None,
            },
        }
        r = results[arm]
        print(f"[{arm}] ingest {r['ingest_secs_total']}s total {r['ingest_secs_per_conv']} per-conv")
        rm = r["retrieve_ms"]
        print(f"[{arm}] retrieve n={rm['n']}  median={rm['median']}ms "
              f"CI95{rm['median_ci95']}  p95={rm['p95']}ms  mean={rm['mean']}ms\n")

    rr, mm = results["recall"], results["mem0"]
    print("=" * 64)
    print("INGEST  (write path):")
    print(f"  RE-call {rr['ingest_secs_total']}s   Mem0 {mm['ingest_secs_total']}s   "
          f"-> Mem0 {mm['ingest_secs_total'] / max(rr['ingest_secs_total'], 1e-9):.1f}x slower")
    print("RETRIEVE (query path, local embedder):")
    print(f"  RE-call median {rr['retrieve_ms']['median']}ms  vs  Mem0 {mm['retrieve_ms']['median']}ms")
    print("  (overlapping CI95 => no measurable difference; adapter opens a PG conn per RE-call retrieve)")
    print("=" * 64)

    args.out.mkdir(parents=True, exist_ok=True)
    out = args.out / f"latency_{stamp}.json"
    out.write_text(json.dumps({"config": vars(args) | {"dsn": "***"}, "results": results},
                              indent=2, default=str), encoding="utf-8")
    print(f"artifact -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
