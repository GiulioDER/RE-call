"""Attribute end-to-end search latency to its stages, to price a store swap.

**The question.** RE-call's store is Postgres + pgvector. Would moving it to another backend
(Redis, say) buy enough to be worth the port? That is decided by the store's SHARE of end-to-end
latency, not by how fast the store is: a backend that is free still cannot remove more than the
share it occupies.

**Where the numbers come from.** `HybridRetriever` already records per-query stage timings and
returns them on every result as `diagnostics.stage_ms` (`query_embedding`, `dense_retrieval`,
`sparse_retrieval`, `fusion`, `reranking`). This benchmark READS that, rather than adding a
parallel instrument. An earlier draft of this file did add one, on the false premise that no
per-stage timing existed — it did, and had shipped; the premise came from reading a checkout 61
commits behind master. The store-internal `recall_store_query_ms` metric is still used here, for
the two things `stage_ms` cannot give:

  1. `newest_indexed_at()`. `search()` calls it once per query for its staleness report and it is
     an uncached `SELECT max(indexed_at)` — a real store round trip that sits OUTSIDE every
     `stage_ms` bracket. Left unattributed it books store cost as Python glue and understates the
     store's share, in the same direction as the "the store is cheap" hypothesis under test.
  2. A CROSS-CHECK. `stage_ms["dense_retrieval"]` brackets the call from outside; the store metric
     times the same work from inside. The second must nest within the first. Two independent
     instruments that agree is evidence the measurement is real; one instrument is an assumption.

**What is asserted rather than hoped.** Per configuration: one metric sample per leg per query
(a leg that silently stops recording must not read as a leg that costs nothing); the store metric
nests inside its stage bracket; the residual is non-negative (parts exceeding the whole means
double counting); the sparse leg actually returns rows (issue #81 had it returning rows for 0 of
150 real questions, so "hybrid" was silently dense-only); and no figure derived from a cross-stage
ratio is emitted when the metric ring truncated, because a mean over a retained suffix and a mean
over the run are different statistics.

**Measurement hygiene.** Every configuration gets a discarded warm-up pass through the FULL
pipeline, so no leg is measured warm against another measured cold, and each configuration is
repeated `--repeats` times. NOTE: repetitions are nested INSIDE a configuration, not interleaved
across them, so a slow drift over the run is still confounded with configuration order; the
warm-up pass is what removes the first-touch component.

**What it does not measure.** The shipped best configuration uses a cloud embedder
(`voyage-4-large`); this runs local embedders only. To reason about the cloud case, substitute a
cloud embed cost into these stage figures — that is a COMPOSITION and must be labelled as one.

**Prior work, and the number that should frame any result from this file.** Commit `9a5165b`
(the #82 fix) already measured the legs on a 72k-chunk corpus: *"the sparse leg goes from
effectively free (it matched nothing) to a median of 496 ms and p95 of 1205 ms, against 9.6 ms
median for the dense leg"*. So the expensive leg at scale is the SPARSE one — Postgres `ts_rank`
over a large match set — not the vector index, and the store's share is strongly corpus-size
dependent. That commit also names the in-Postgres remedies (lexeme capping by document frequency,
or a RUM index) and notes small memory corpora should not see it. Any run of this benchmark on a
small corpus therefore measures the regime where the store is cheap BY CONSTRUCTION, and must not
be quoted as a general result. That measurement lives only in a commit message, on a different
host, with no embed or rerank leg, which is why a committed four-leg artifact is still worth having.

(Searched `docs_search(source_type="memory")` for latency attribution on 2026-08-04: no memo
covers it; every hit was about retrieval QUALITY. The 9a5165b figure came from git, not memory.)

Usage:
    python benchmarks/store_latency_share.py --embedder fastembed --filler 20000 \
        --candidate-k 20 --candidate-k 250 --rerank --repeats 3
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean

from recall.embeddings import Embedder, embed_query, embedding_profile_id
from recall.eval.harness import _throwaway_store
from recall.eval.provenance import generated_at, model_stack
from recall.eval.scale import DEFAULT_DSN, _make_embedder
from recall.eval.synthetic import generate
from recall.observability import HISTOGRAM_CAPACITY, METRICS, percentile
from recall.rerank import CrossEncoderReranker, Reranker
from recall.retriever import DEFAULT_CANDIDATE_K, HybridRetriever
from recall.store import (
    LEG_DENSE,
    LEG_META,
    LEG_SPARSE,
    STORE_QUERY_METRIC,
    PgVectorStore,
    warn_if_insecure_dsn,
)

#: The pressure arm: an order of magnitude above the shipped default, and the pool size the
#: project's best measured configuration uses.
WIDE_CANDIDATE_K = 250
#: Slack for float accumulation across the summed stages. Below this a negative residual is
#: arithmetic noise; at or beyond it, some interval is being counted twice.
_RESIDUAL_TOLERANCE_MS = 0.01
#: Below this sparse fire rate the "hybrid" label is not earned (issue #81 was 0/150).
_MIN_SPARSE_FIRE_RATE = 0.05


@dataclass
class LegSplit:
    """One configuration's latency attribution.

    Only the `*_ms` fields are milliseconds per query. `candidate_k`, `n_queries`, `n_chunks` and
    `repeats` are counts; `store_share`, `sparse_fire_rate` are fractions in [0, 1].
    """

    candidate_k: int
    reranked: bool
    repeats: int
    n_queries: int
    n_chunks: int
    embedder: str

    total_ms_mean: float
    total_ms_p50: float
    total_ms_p95: float

    embed_ms_mean: float
    dense_ms_mean: float
    sparse_ms_mean: float
    fusion_ms_mean: float
    rerank_ms_mean: float
    #: `newest_indexed_at()`, the store round trip outside every `stage_ms` bracket.
    meta_ms_mean: float
    residual_ms_mean: float

    #: dense + sparse + meta. What a store swap could address.
    #:
    #: NOTE the two are measured from different instruments: the mean sums the retriever's
    #: OUTER stage brackets for dense/sparse (plus the inner timer for meta), while the p50 is
    #: the inner store timer for all three. The gap is per-call Python overhead, which is what
    #: `max_nesting_violation_ms` bounds; do not read them as the mean and median of one
    #: series.
    store_ms_mean: float
    store_ms_p50: float
    #: Fraction of end-to-end latency spent in the store, in [0, 1].
    store_share: float
    #: The upper bound on a store swap: what the total becomes if the replacement store is FREE.
    #: Unachievable by construction, which is what makes it a bound rather than a forecast.
    total_ms_if_store_were_free: float

    sparse_fire_rate: float
    #: True when the metric ring evicted samples. When set, no ratio above is emitted.
    truncated: bool
    #: max over queries of (store metric ms - its stage bracket ms). Must be <= 0: the store's
    #: internal timer measures a strict subinterval of the retriever's bracket around the call.
    max_nesting_violation_ms: float = 0.0
    notes: list[str] = field(default_factory=list)


def _drain(leg: str) -> tuple[list[float], int]:
    return METRICS.drain_histogram(STORE_QUERY_METRIC, leg=leg)


def _clear_legs() -> None:
    for leg in (LEG_DENSE, LEG_SPARSE, LEG_META):
        METRICS.drain_histogram(STORE_QUERY_METRIC, leg=leg)


def _sparse_fire_rate(
    store: PgVectorStore, embedder: Embedder, queries: list[dict], candidate_k: int
) -> float:
    """Fraction of queries whose sparse leg returns at least one row.

    Runs the SAME statement the retriever runs. `query_sparse` has two branches, and the one taken
    depends on whether `vec` is passed (`vec` makes each hit carry its true dense cosine); probing
    without it would certify a different statement than the pipeline executes. Samples are drained
    afterwards so the probe cannot enter the latency figures it qualifies.
    """
    fired = 0
    for q in queries:
        qvec = embed_query(embedder, q["query"])
        if store.query_sparse(q["query"], k=candidate_k, vec=qvec):
            fired += 1
    _clear_legs()
    return fired / len(queries) if queries else 0.0


def measure(
    store: PgVectorStore,
    embedder: Embedder,
    queries: list[dict],
    *,
    candidate_k: int,
    reranker: Reranker | None,
    n_chunks: int,
    repeats: int = 1,
    k: int = 5,
) -> LegSplit:
    if not queries:
        raise ValueError("no queries to time")
    retr = HybridRetriever(
        store, embedder, reranker=reranker, candidate_k=candidate_k, use_sparse=True
    )

    fire_rate = _sparse_fire_rate(store, embedder, queries, candidate_k)
    if fire_rate < _MIN_SPARSE_FIRE_RATE:
        raise AssertionError(
            f"sparse leg returned rows for {fire_rate:.1%} of queries (floor "
            f"{_MIN_SPARSE_FIRE_RATE:.0%}). This is a dense-only pipeline wearing a hybrid label "
            "(issue #81); no hybrid split may be quoted from it."
        )

    # Discarded warm-up through the FULL pipeline: warms both legs, the plan cache and the pool
    # symmetrically. Warming only one leg biases the dense/sparse split, which is the quantity
    # being published.
    for q in queries:
        retr.search(q["query"], k=k)
    _clear_legs()

    totals: list[float] = []
    stages: dict[str, list[float]] = {}
    nesting_violation = 0.0
    for _ in range(repeats):
        for q in queries:
            t0 = time.perf_counter()
            res = retr.search(q["query"], k=k)
            totals.append((time.perf_counter() - t0) * 1000.0)
            for name, value in res.diagnostics.stage_ms.items():
                stages.setdefault(name, []).append(value)

    n = len(queries) * repeats
    dense_metric, dense_total = _drain(LEG_DENSE)
    sparse_metric, sparse_total = _drain(LEG_SPARSE)
    meta_metric, meta_total = _drain(LEG_META)

    # One sample per leg per query. Without this, a leg that stops recording reads exactly like a
    # leg that costs nothing, and every other guard here stays green while the share goes to zero.
    for label, samples, observed in (
        (LEG_DENSE, dense_metric, dense_total),
        (LEG_SPARSE, sparse_metric, sparse_total),
        (LEG_META, meta_metric, meta_total),
    ):
        if observed != n:
            raise AssertionError(
                f"leg {label!r} recorded {observed} samples for {n} queries. The per-query "
                "denominator does not hold, so no per-query mean may be quoted."
            )

    truncated = any(t > len(s) for s, t in (
        (dense_metric, dense_total), (sparse_metric, sparse_total), (meta_metric, meta_total)
    ))

    # Nesting cross-check: the store's internal timer measures a subinterval of the retriever's
    # bracket around the same call, so stage >= metric for every query. A violation means the two
    # instruments are not timing what their names say.
    # `strict=True` matters more than it looks. `stages[...]` is a head-ordered list and the metric
    # samples come from a TAIL-retaining ring, so on eviction the two do not merely differ in
    # length, they MISALIGN — sample n-1023 would be compared against bracket 1, producing both
    # false violations and false passes. Unequal lengths must raise, not be silently absorbed.
    if not truncated:
        for stage_key, metric_samples in (
            ("dense_retrieval", dense_metric), ("sparse_retrieval", sparse_metric)
        ):
            for bracket, inner in zip(stages.get(stage_key, []), metric_samples, strict=True):
                nesting_violation = max(nesting_violation, inner - bracket)

    def stage_mean(name: str) -> float:
        return mean(stages[name]) if stages.get(name) else 0.0

    embed_ms = stage_mean("query_embedding")
    dense_ms = stage_mean("dense_retrieval")
    sparse_ms = stage_mean("sparse_retrieval")
    fusion_ms = stage_mean("fusion")
    rerank_ms = stage_mean("reranking")
    meta_ms = mean(meta_metric) if meta_metric else 0.0
    total_ms = mean(totals)
    if total_ms <= 0:
        raise AssertionError("measured total is zero; the run did not time anything")

    store_ms = dense_ms + sparse_ms + meta_ms
    residual = total_ms - (embed_ms + dense_ms + sparse_ms + fusion_ms + rerank_ms + meta_ms)
    if residual < -_RESIDUAL_TOLERANCE_MS:
        raise AssertionError(
            f"attributed stages exceed the measured total by {-residual:.3f} ms. Some interval is "
            "counted twice; no share may be quoted from this split."
        )

    if truncated:
        # FATAL, not suppressed. A partial suppression is where the asymmetry creeps back in: the
        # first version NaN'd `store_share` while still emitting `store_ms_p50` (computed from the
        # same misaligned samples) and `total_ms_if_store_were_free`, so a truncated run still
        # published three figures derived from a basis it had just declared incomparable. And
        # `float("nan")` serialises into `splits.json` as a bare `NaN`, which is not valid JSON.
        raise AssertionError(
            "the metric ring evicted samples, so the store legs are means over a retained suffix "
            "while embed/rerank/total are means over the run. No figure may be derived from both."
        )

    notes: list[str] = []
    if nesting_violation > 0:
        notes.append(f"nesting violated by {nesting_violation:.3f} ms")

    store_p50 = percentile(
        sorted(a + b + c for a, b, c in zip(dense_metric, sparse_metric, meta_metric)), 0.50
    )
    return LegSplit(
        candidate_k=candidate_k,
        reranked=reranker is not None,
        repeats=repeats,
        n_queries=len(queries),
        n_chunks=n_chunks,
        embedder=embedding_profile_id(embedder),
        total_ms_mean=round(total_ms, 3),
        total_ms_p50=percentile(sorted(totals), 0.50),
        total_ms_p95=percentile(sorted(totals), 0.95),
        embed_ms_mean=round(embed_ms, 3),
        dense_ms_mean=round(dense_ms, 3),
        sparse_ms_mean=round(sparse_ms, 3),
        fusion_ms_mean=round(fusion_ms, 3),
        rerank_ms_mean=round(rerank_ms, 3),
        meta_ms_mean=round(meta_ms, 3),
        residual_ms_mean=round(residual, 3),
        store_ms_mean=round(store_ms, 3),
        store_ms_p50=store_p50,
        store_share=round(store_ms / total_ms, 4),
        total_ms_if_store_were_free=round(total_ms - store_ms, 3),
        sparse_fire_rate=round(fire_rate, 4),
        truncated=truncated,
        max_nesting_violation_ms=round(nesting_violation, 3),
        notes=notes,
    )


def to_markdown(splits: list[LegSplit], ctx: str) -> str:
    rows = [
        "| chunks | cand_k | rerank | total | embed | dense | sparse | meta | fusion | rerank | "
        "**store** | resid | **store share** | sparse fire |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in splits:
        share = "truncated" if s.truncated else f"{s.store_share:.1%}"
        rows.append(
            f"| {s.n_chunks} | {s.candidate_k} | {'yes' if s.reranked else 'no'} | "
            f"{s.total_ms_mean:.1f} | {s.embed_ms_mean:.1f} | {s.dense_ms_mean:.1f} | "
            f"{s.sparse_ms_mean:.1f} | {s.meta_ms_mean:.1f} | {s.fusion_ms_mean:.1f} | "
            f"{s.rerank_ms_mean:.1f} | **{s.store_ms_mean:.1f}** | {s.residual_ms_mean:.1f} | "
            f"**{share}** | {s.sparse_fire_rate:.0%} |"
        )
    rows.append("")
    rows.append(
        "All figures are ms/query, means over `repeats x n_queries`, measured warm (each "
        "configuration gets a discarded full-pipeline warm-up pass). `store` = dense + sparse + "
        "meta, where meta is `newest_indexed_at()`, the per-search round trip that sits outside "
        "every `stage_ms` bracket. `store share` is the ceiling on any store swap: a replacement "
        "that cost nothing would remove exactly this fraction."
    )
    rows.append("")
    # The caveats travel WITH the numbers. A reader opens this file, not the module docstring or
    # the commit message, and every sentence below changes how the headline may be read.
    rows.append(
        f"⚠️ **Scope.** Corpus is SYNTHETIC (`recall.eval.synthetic`), {ctx}. Two limits follow. "
        "(1) The sparse leg here is not representative: commit `9a5165b` measured sparse median "
        "**496 ms** on a real 72k-chunk corpus, where this run measures single-digit ms. The cost "
        "is corpus-vocabulary dependent and neither figure generalises. (2) At `candidate_k=250` "
        "the dense leg runs at `hnsw.ef_search = min(k x multiplier, 1000)` against 80 at k=20 — "
        "so the k=250 row prices an OVER-FETCH SETTING inside the store, not Postgres against "
        "another backend. A different engine re-pays that walk rather than removing it."
    )
    flagged = [s for s in splits if s.notes]
    if flagged:
        rows.append("")
        for s in flagged:
            rows.append(f"- ⚠️ candidate_k={s.candidate_k}: {'; '.join(s.notes)}")
    return "\n".join(rows)


def main() -> int:
    ap = argparse.ArgumentParser(prog="store_latency_share")
    ap.add_argument("--dsn", default=DEFAULT_DSN)
    ap.add_argument("--embedder", default="fastembed", choices=["hashing", "fastembed"])
    ap.add_argument("--filler", type=int, default=0, help="filler chunks (index pressure)")
    ap.add_argument("--candidate-k", type=int, action="append", dest="candidate_ks")
    ap.add_argument("--rerank", action="store_true")
    ap.add_argument("--queries", type=int, default=100)
    ap.add_argument("--repeats", type=int, default=3, help="repetitions per config")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--out", default="results/store_latency")
    args = ap.parse_args()

    candidate_ks = args.candidate_ks or [DEFAULT_CANDIDATE_K, WIDE_CANDIDATE_K]
    # Refuse up front rather than warn after indexing: past the ring the leg means become means
    # over a suffix, and the operator would learn it only after paying for the corpus build.
    if args.queries * args.repeats > HISTOGRAM_CAPACITY:
        ap.error(
            f"--queries x --repeats = {args.queries * args.repeats} exceeds the metric ring "
            f"({HISTOGRAM_CAPACITY}); leg means would be over a truncated suffix"
        )
    warn_if_insecure_dsn(args.dsn)

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

    reranker: Reranker | None = CrossEncoderReranker() if args.rerank else None

    t0 = time.perf_counter()
    print(f"indexing with {embedding_profile_id(emb)} ...")
    splits: list[LegSplit] = []
    with _throwaway_store(args.dsn, emb, corpus.root, "storelat_") as store:
        n_chunks = store.count()  # invariant across the sweep; one count, not one per config
        print(f"  indexed {n_chunks} chunks in {time.perf_counter() - t0:.1f}s")
        configs = [(ck, rr) for ck in candidate_ks for rr in ([None, reranker] if reranker else [None])]
        for ck, rr in configs:
            print(f"  measuring candidate_k={ck} rerank={'yes' if rr else 'no'} ...")
            splits.append(
                measure(
                    store, emb, queries, candidate_k=ck, reranker=rr,
                    n_chunks=n_chunks, repeats=args.repeats,
                )
            )

    # Provenance is EMITTED, not stamped on afterwards. Latency is the most host- and
    # stack-dependent quantity this repo publishes — the CHANGELOG attributes a 691.7 -> 2383.0
    # rerank shift to "a slower shared CPU" — so an undated, unstacked latency artifact cannot be
    # reproduced or even compared against itself. Wrapped in an object rather than left as a bare
    # array so the `_provenance` key has somewhere to live; `results/ARTIFACTS.md` indexes it.
    (out / "splits.json").write_text(
        json.dumps(
            {
                "_provenance": {
                    "generation": "post-#81/#84",
                    "status": "current",
                    "superseded_by": None,
                    "backs": ["store latency share — the Redis-port decision"],
                    "note": (
                        "SYNTHETIC corpus (recall.eval.synthetic), so the sparse leg's cost is "
                        "NOT representative: commit 9a5165b measured sparse median 496 ms on a "
                        "real 72k-chunk corpus where this run measures single-digit ms. Do not "
                        "generalise the sparse figure in either direction."
                    ),
                },
                "artifact": "per-leg latency attribution behind the store-share figure",
                "generated_at": generated_at(),
                "embedder": embedding_profile_id(emb),
                "corpus": "synthetic",
                "n_chunks": n_chunks,
                "filler": args.filler,
                "seed": args.seed,
                "queries": len(queries),
                "repeats": args.repeats,
                "stack": model_stack(),
                "splits": [asdict(s) for s in splits],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    md = to_markdown(
        splits,
        f"{n_chunks} chunks, embedder `{embedding_profile_id(emb)}`, seed {args.seed}, "
        f"{len(queries)} queries x {args.repeats} repeats",
    )
    (out / "SPLIT.md").write_text(md + "\n", encoding="utf-8")
    print("\n" + md)
    # `notes` is built from the unrounded violation, so the exit code reads `notes` rather
    # than the rounded field: otherwise a sub-microsecond violation prints a warning and
    # exits 0, and the warning and the exit status disagree about the same run.
    return 1 if any(s.notes for s in splits) else 0


if __name__ == "__main__":
    raise SystemExit(main())
