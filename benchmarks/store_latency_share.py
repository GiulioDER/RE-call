"""Attribute end-to-end search latency to its four legs, in one process on one host.

**Why this exists.** Every latency figure this project publishes is whole-`search()` wall time:
`RESULTS.md` §10 reports 45 ms local and 246 ms cloud, §7a reports 66-90 ms, `SCALE.md` reports a
46.9 ms p50. None of them says how much of that is the STORE. Answering "would a faster vector
backend help?" from those numbers means subtracting figures measured on different machines, and
RESULTS.md itself warns its latency columns are comparable only within one table.

So this measures embed, dense, sparse and rerank together, in the same run, and prints what is
left over. The leftover matters: if the four legs do not account for the total, the attribution is
incomplete and saying "the store is N%" would be a guess dressed as a measurement. `residual_ms`
is therefore reported, never absorbed, and a NEGATIVE residual aborts the run rather than
rounding away — parts that exceed the whole mean the instrumentation double-counts.

**What it does not measure.** The shipped best configuration uses a cloud embedder
(`voyage-4-large`). This runs local embedders only, so the embed leg here is the local cost. To
reason about the cloud case, take the dense/sparse/rerank figures from this run and substitute a
cloud embed cost: that is a COMPOSITION, and must be labelled as one.

**The sparse leg is verified to fire, not assumed to.** Issue #81 found the sparse leg returning
rows for 0 of 150 real questions, so "hybrid" was silently dense-only; #82 fixed it (0/150 ->
150/150). A latency split taken in that state would report a real sparse cost for a leg
contributing nothing, so `sparse_fire_rate` is measured and a rate of 0.0 aborts the run.

Prior work: searched `docs_search(source_type="memory")` for latency attribution / store share /
leg breakdown on 2026-08-04. No prior measurement exists — every hit was about retrieval QUALITY
(best-config, reranker choice, near-miss signals), none about where the milliseconds go. The
#81/#82 sparse-leg history above came out of that same search and is the reason for the guard.

Usage:
    python benchmarks/store_latency_share.py --embedder fastembed --filler 20000 \
        --candidate-k 20 --candidate-k 250 --rerank
"""
from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from recall.embeddings import Embedder, embedding_profile_id
from recall.eval.harness import _throwaway_store
from recall.eval.scale import _make_embedder
from recall.eval.synthetic import generate
from recall.observability import METRICS, percentile
from recall.rerank import CrossEncoderReranker, Reranker
from recall.retriever import HybridRetriever
from recall.store import LEG_DENSE, LEG_SPARSE, STORE_QUERY_METRIC, PgVectorStore
from recall.timing import TimedEmbedder, TimedReranker

DEFAULT_DSN = os.environ.get("RECALL_DSN", "postgresql://recall:recall@localhost:5432/recall")


@dataclass
class LegSplit:
    """One configuration's latency attribution. All figures are ms per query."""

    candidate_k: int
    reranked: bool
    n_queries: int
    n_chunks: int
    embedder: str

    total_mean: float
    total_p50: float
    total_p95: float

    embed_mean: float
    dense_mean: float
    sparse_mean: float
    rerank_mean: float
    residual_mean: float

    store_mean: float
    store_share_of_total: float
    #: Fraction of queries for which the sparse leg returned at least one row. A hybrid split
    #: measured while this is 0.0 is a dense-only split wearing a hybrid label (issue #81).
    sparse_fire_rate: float
    #: The honest ceiling on a backend swap: what the total would become if the store cost ZERO.
    #: Not achievable (a replacement store is not free), which is what makes it an upper bound.
    total_if_store_were_free: float
    best_case_speedup_pct: float

    truncated: bool


def _drain(leg: str) -> tuple[list[float], bool]:
    samples, total = METRICS.drain_histogram(STORE_QUERY_METRIC, leg=leg)
    return samples, total > len(samples)


def _sparse_fire_rate(
    store: PgVectorStore, queries: list[dict], candidate_k: int
) -> float:
    """Fraction of queries the sparse leg answers with at least one row.

    Probed separately and BEFORE the timed loop, then the samples it generates are drained, so
    the probe does not enter the latency figures it exists to qualify.
    """
    fired = sum(1 for q in queries if store.query_sparse(q["query"], k=candidate_k))
    for leg in (LEG_DENSE, LEG_SPARSE):
        METRICS.drain_histogram(STORE_QUERY_METRIC, leg=leg)
    return fired / len(queries) if queries else 0.0


def measure(
    store: PgVectorStore,
    embedder: Embedder,
    queries: list[dict],
    *,
    candidate_k: int,
    reranker: Reranker | None,
    k: int = 5,
) -> LegSplit:
    timed_emb = TimedEmbedder(embedder)
    timed_rr = TimedReranker(reranker) if reranker is not None else None
    retr = HybridRetriever(
        store, timed_emb, reranker=timed_rr, candidate_k=candidate_k
    )

    for leg in (LEG_DENSE, LEG_SPARSE):  # clear anything a previous configuration left
        METRICS.drain_histogram(STORE_QUERY_METRIC, leg=leg)

    fire_rate = _sparse_fire_rate(store, queries, candidate_k)
    if fire_rate == 0.0:
        raise AssertionError(
            "the sparse leg returned rows for 0 queries, so this is a dense-only pipeline and "
            "its sparse latency buys nothing. That is issue #81; do not quote a hybrid split "
            "from this run."
        )

    totals: list[float] = []
    for q in queries:
        t0 = time.perf_counter()
        retr.search(q["query"], k=k)
        totals.append((time.perf_counter() - t0) * 1000.0)

    dense, dense_trunc = _drain(LEG_DENSE)
    sparse, sparse_trunc = _drain(LEG_SPARSE)

    embed_mean = timed_emb.stats.mean_ms
    rerank_mean = timed_rr.stats.mean_ms if timed_rr else 0.0
    dense_mean = mean(dense) if dense else 0.0
    sparse_mean = mean(sparse) if sparse else 0.0
    total_mean = mean(totals)
    store_mean = dense_mean + sparse_mean
    residual = total_mean - (embed_mean + store_mean + rerank_mean)

    if residual < -0.01:
        raise AssertionError(
            f"attributed legs ({embed_mean + store_mean + rerank_mean:.3f} ms) exceed the measured "
            f"total ({total_mean:.3f} ms) by {-residual:.3f} ms. Something is counted twice; the "
            "split is not trustworthy and no share may be quoted from it."
        )

    free = total_mean - store_mean
    return LegSplit(
        candidate_k=candidate_k,
        reranked=reranker is not None,
        n_queries=len(queries),
        n_chunks=store.count(),
        embedder=embedding_profile_id(embedder),
        total_mean=round(total_mean, 3),
        total_p50=percentile(sorted(totals), 0.50),
        total_p95=percentile(sorted(totals), 0.95),
        embed_mean=round(embed_mean, 3),
        dense_mean=round(dense_mean, 3),
        sparse_mean=round(sparse_mean, 3),
        rerank_mean=round(rerank_mean, 3),
        residual_mean=round(residual, 3),
        store_mean=round(store_mean, 3),
        store_share_of_total=round(store_mean / total_mean, 4) if total_mean else 0.0,
        sparse_fire_rate=round(fire_rate, 4),
        total_if_store_were_free=round(free, 3),
        best_case_speedup_pct=round(100.0 * store_mean / total_mean, 2) if total_mean else 0.0,
        truncated=dense_trunc or sparse_trunc,
    )


def to_markdown(splits: list[LegSplit]) -> str:
    rows = [
        "| candidate_k | rerank | total mean | embed | dense | sparse | **store** | rerank | "
        "residual | store share | best-case speedup |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in splits:
        rows.append(
            f"| {s.candidate_k} | {'yes' if s.reranked else 'no'} | {s.total_mean:.1f} ms | "
            f"{s.embed_mean:.1f} | {s.dense_mean:.1f} | {s.sparse_mean:.1f} | "
            f"**{s.store_mean:.1f}** | {s.rerank_mean:.1f} | {s.residual_mean:.1f} | "
            f"{s.store_share_of_total:.1%} | {s.best_case_speedup_pct:.1f}% |"
        )
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(prog="store_latency_share")
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--embedder", default="fastembed", choices=["hashing", "fastembed"])
    ap.add_argument("--filler", type=int, default=0, help="filler chunks (index pressure)")
    ap.add_argument("--candidate-k", type=int, action="append", dest="candidate_ks")
    ap.add_argument("--rerank", action="store_true", help="also measure with the cross-encoder")
    ap.add_argument("--queries", type=int, default=100, help="answerable queries to time")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="results/store_latency")
    args = ap.parse_args()

    candidate_ks = args.candidate_ks or [20, 250]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    emb = _make_embedder(args.embedder)
    print(f"generating corpus (filler={args.filler}) ...")
    corpus = generate(
        out / "corpus",
        n_answerable=args.queries,
        n_unanswerable=0,
        n_successor=0,
        n_abstain=0,
        n_filler_chunks=args.filler,
        seed=args.seed,
    )
    queries = [q for q in corpus.queries if q.get("answerable")][: args.queries]
    print(f"  {corpus.n_files} files, {corpus.n_chunks} chunks, {len(queries)} timed queries")

    reranker: Reranker | None = None
    if args.rerank:
        reranker = CrossEncoderReranker()

    t0 = time.perf_counter()
    print(f"indexing with {embedding_profile_id(emb)} ...")
    splits: list[LegSplit] = []
    with _throwaway_store(args.dsn, emb, corpus.root, "storelat_") as store:
        print(f"  indexed {store.count()} chunks in {time.perf_counter() - t0:.1f}s")
        for ck in candidate_ks:
            for rr in ([None, reranker] if reranker is not None else [None]):
                label = f"candidate_k={ck} rerank={'yes' if rr else 'no'}"
                print(f"  measuring {label} ...")
                splits.append(measure(store, emb, queries, candidate_k=ck, reranker=rr))

    (out / "splits.json").write_text(
        json.dumps([asdict(s) for s in splits], indent=2), encoding="utf-8"
    )
    md = to_markdown(splits)
    (out / "SPLIT.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    if any(s.truncated for s in splits):
        print("\nWARNING: metric ring evicted samples; means are over a suffix, not the run.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
