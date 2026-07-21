"""`python -m recall.eval.scale` — run the trust evaluation on a generated corpus at scale.

The shipped evaluation answers "does the mechanism work at all?" on 14 documents and 6 validity
queries. This answers the two questions that one cannot:

1. **Is the headline rate real, or is it six coin flips?** The interval is what decides that, and
   the interval is driven by query count. At n=6 a rate of 0.00 carries a 95% Wilson interval of
   [0.00, 0.39]; at n=150 the same 0.00 carries roughly [0.00, 0.02].
2. **Does retrieval hold up under index pressure?** HNSW behaves differently at 100k vectors than
   at 20, and a `source`-filtered query walks a filter-blind graph — the recall-collapse case
   that cannot appear in a corpus small enough to scan exhaustively.

Reported alongside every rate: its Wilson interval, its n, and end-to-end search latency
percentiles (p50/p95/p99 — a mean hides exactly the tail an operator is paged for).
"""
from __future__ import annotations

import argparse
import os
import statistics
import time
from pathlib import Path

from recall.embeddings import Embedder
from recall.eval.harness import (
    _throwaway_store,
    run_trust_eval,
    trust_results_to_markdown,
)
from recall.eval.metrics import recall_at_k, wilson_ci
from recall.eval.synthetic import generate
from recall.retriever import HybridRetriever

DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")


def _make_embedder(name: str) -> Embedder:
    if name == "hashing":
        from recall.embeddings import HashingEmbedder

        return HashingEmbedder(dim=64)
    if name == "fastembed":
        from recall.embeddings import FastEmbedEmbedder

        return FastEmbedEmbedder()
    raise ValueError(f"unknown embedder: {name!r}")


def _percentiles(samples_ms: list[float]) -> dict[str, float]:
    """p50/p95/p99 of a latency sample. A mean is not a latency report: one 3-second query in a
    thousand is invisible in the mean and is the entire user complaint."""
    if not samples_ms:
        return {"p50": float("nan"), "p95": float("nan"), "p99": float("nan")}
    s = sorted(samples_ms)
    return {
        "p50": statistics.median(s),
        "p95": s[min(len(s) - 1, int(0.95 * len(s)))],
        "p99": s[min(len(s) - 1, int(0.99 * len(s)))],
    }


def measure_retrieval(
    store, embedder: Embedder, queries: list[dict], k: int = 5
) -> tuple[dict, dict, dict]:
    """Recall@k unfiltered, recall@k under a `source` filter, and latency percentiles.

    The filtered arm is the point of interest. `query_dense` applies `WHERE source = ...`
    alongside an HNSW `ORDER BY embedding <=> ...`; the index walk cannot see the predicate, so
    a selective filter can return fewer (or no) in-filter neighbours than exist. Filtering to
    the one source that DOES hold the answer makes that visible: recall should be 1.00, and
    anything less is the graph failing to surface a row it certainly contains.
    """
    retr = HybridRetriever(store, embedder)
    answerable = [q for q in queries if not q.get("trust") and q["answerable"]]
    unfiltered_flags: list[bool] = []
    filtered_flags: list[bool] = []
    latencies: list[float] = []

    for q in answerable:
        t0 = time.perf_counter()
        res = retr.search(q["query"], k=k)
        latencies.append((time.perf_counter() - t0) * 1000.0)
        got = [f"{h.chunk.metadata['file']}:{h.chunk.metadata['ord']}" for h in res.hits]
        unfiltered_flags.append(recall_at_k(got, q["relevant_ids"], k) == 1.0)

        # the source column stores the resolved absolute path; recover it from the hit set so
        # the filter is exactly the one Postgres indexed, not a reconstructed guess
        target_rel = q["relevant_ids"][0].rsplit(":", 1)[0]
        src = next(
            (h.chunk.source for h in res.hits if h.chunk.metadata["file"] == target_rel), None
        )
        if src is None:
            filtered_flags.append(False)  # never surfaced unfiltered either
            continue
        fres = retr.search(q["query"], k=k, source=src)
        fgot = [f"{h.chunk.metadata['file']}:{h.chunk.metadata['ord']}" for h in fres.hits]
        filtered_flags.append(recall_at_k(fgot, q["relevant_ids"], k) == 1.0)

    def _rate(flags: list[bool]) -> dict:
        lo, hi = wilson_ci(flags)
        return {
            "rate": (sum(flags) / len(flags)) if flags else float("nan"),
            "ci": (lo, hi),
            "n": len(flags),
        }

    return _rate(unfiltered_flags), _rate(filtered_flags), _percentiles(latencies)


def main() -> None:
    ap = argparse.ArgumentParser(prog="recall.eval.scale")
    ap.add_argument("--out", default="results/scale", help="where to write corpus + report")
    ap.add_argument("--embedder", default="hashing", choices=["hashing", "fastembed"])
    ap.add_argument("--answerable", type=int, default=200)
    ap.add_argument("--unanswerable", type=int, default=100)
    ap.add_argument("--successor", type=int, default=150)
    ap.add_argument("--abstain", type=int, default=100)
    ap.add_argument("--filler", type=int, default=50_000, help="filler chunks (index pressure)")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    emb = _make_embedder(args.embedder)

    print(f"generating corpus under {out / 'corpus'} ...")
    corpus = generate(
        out / "corpus",
        n_answerable=args.answerable,
        n_unanswerable=args.unanswerable,
        n_successor=args.successor,
        n_abstain=args.abstain,
        n_filler_chunks=args.filler,
        seed=args.seed,
    )
    print(f"  {corpus.n_files} files, {corpus.n_chunks} chunks, {len(corpus.queries)} queries")

    t0 = time.perf_counter()
    print(f"indexing + measuring retrieval with {emb.name} ...")
    with _throwaway_store(args.dsn, emb, corpus.root, "scale_") as store:
        index_s = time.perf_counter() - t0
        indexed = store.count()
        unfiltered, filtered, lat = measure_retrieval(store, emb, corpus.queries)
    print(f"  indexed {indexed} chunks in {index_s:.1f}s")

    print("running trust evaluation ...")
    trust = run_trust_eval(
        args.dsn, [emb], corpus_dir=corpus.root, queries_path=corpus.queries_path
    )

    def _ci(d: dict) -> str:
        return f"{d['rate']:.4f} [{d['ci'][0]:.4f}, {d['ci'][1]:.4f}] (n={d['n']})"

    lines = [
        "# recall — evaluation at scale",
        "",
        "Generated corpus (`recall.eval.synthetic`), not the 14-document demo corpus. Reproduce "
        f"with `python -m recall.eval.scale --embedder {args.embedder} --filler {args.filler} "
        f"--seed {args.seed}`.",
        "",
        f"- corpus: **{indexed} chunks** across {corpus.n_files} files",
        f"- queries: **{len(corpus.queries)}** ({args.answerable} answerable, "
        f"{args.unanswerable} unanswerable, {args.successor} successor, {args.abstain} abstain)",
        f"- embedder: `{emb.name}` · index time: {index_s:.1f}s",
        "",
        "## Retrieval under index pressure",
        "",
        "| measurement | value |",
        "|---|---|",
        f"| recall@5, unfiltered | {_ci(unfiltered)} |",
        f"| recall@5, `source`-filtered | {_ci(filtered)} |",
        f"| search latency p50 / p95 / p99 (ms) | {lat['p50']:.1f} / {lat['p95']:.1f} / "
        f"{lat['p99']:.1f} |",
        "",
        "The filtered arm restricts the query to the one source that holds the answer, so recall "
        "of 1.00 is the only correct result. A shortfall is HNSW post-filtering: the graph walk "
        "cannot see the `WHERE` clause, so it can fail to surface a row the table certainly "
        "contains.",
        "",
        "## Trust layer",
        "",
        trust_results_to_markdown(trust),
        "",
    ]
    report = out / "SCALE.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwrote {report}")


if __name__ == "__main__":
    main()
