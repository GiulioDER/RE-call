"""Attribute end-to-end search latency to its stages, to price a store swap.

**The question.** RE-call's store is Postgres + pgvector. Would moving it to another backend
(Redis, say) buy enough to be worth the port? That is decided by the store's SHARE of end-to-end
latency, not by how fast the store is: a backend that is free still cannot remove more than the
share it occupies.

**Where the numbers come from.** `HybridRetriever` already records per-query stage timings and
returns them on every result as `diagnostics.stage_ms` (`query_embedding`, `dense_retrieval`,
`sparse_retrieval`, `learned_sparse_retrieval`, `fusion`, `reranking`). This benchmark READS that,
rather than adding a parallel instrument. An earlier draft of this file did add one, on the false premise that no
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

**What is asserted rather than hoped.** Per configuration: one metric sample per query for every
leg the backend SELECTED, and none at all for a leg it did not (a leg that silently stops
recording must not read as a leg that costs nothing, and a leg running when the split says it is
not would have its cost booked as residual); the store metric nests inside its stage bracket; the
residual is non-negative (parts exceeding the whole means double counting); each selected sparse
leg actually returns rows (issue #81 had ts_rank returning rows for 0 of 150 real questions, so
"hybrid" was silently dense-only); and no figure derived from a cross-stage ratio is emitted when
the metric ring truncated, because a mean over a retained suffix and a mean over the run are
different statistics.

**The learned sparse arm.** `measure(sparse_backend=...)` accepts `splade` / `both`. Each guard
here has a learned-leg counterpart: the per-query denominator, the nesting cross-check, the fire
rate floor, and the attribution itself. The CLI below cannot yet select the arm, because `splade`
needs a `sparse_encoder` and this file has no flag to build one. The one committed artifact under
`results/store_latency/` PREDATES the learned arm and carries none of its fields, so nothing was
asserted about the learned leg when it was produced; a `lexical` run under current code publishes
those fields with the leg asserted idle and its fire rate `null`. Note that `splade` REPLACES the
ts_rank leg rather than adding to it, so under it the LEXICAL leg is the one asserted idle, and
its fire rate reads `n/a` rather than 0%.

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
    LEG_LEARNED_SPARSE,
    LEG_META,
    LEG_SPARSE,
    STORE_QUERY_METRIC,
    PgVectorStore,
    warn_if_insecure_dsn,
)

#: The pressure arm: an order of magnitude above the shipped default, and the pool size the
#: project's best measured configuration uses.
WIDE_CANDIDATE_K = 250
#: Slack for float accumulation when one timed quantity is subtracted from another. Two callers,
#: both refusing on the same shape of fault: a negative residual (some interval counted twice) and
#: an inner store timer exceeding the stage bracket that encloses it (the two instruments are not
#: timing what their names say). Below this the difference is arithmetic noise; at or beyond it,
#: it is a fault. Deliberately far above the ~1e-13 ms that `mean()` accumulation actually costs.
_RESIDUAL_TOLERANCE_MS = 0.01
#: Below this sparse fire rate the "hybrid" label is not earned (issue #81 was 0/150).
_MIN_SPARSE_FIRE_RATE = 0.05


@dataclass
class LegSplit:
    """One configuration's latency attribution.

    Only the `*_ms` fields are milliseconds per query. `candidate_k`, `n_queries`, `n_chunks` and
    `repeats` are counts; `store_share` is a fraction in [0, 1]. The two fire rates are fractions
    in [0, 1] OR `null`, which serialises into `splits.json` as JSON null and means that leg was
    not in the pipeline. `null` must NOT be read as 0: zero is the issue #81 alarm value.
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
    #: The LEARNED sparse (SPLADE) stage bracket, whole. Recorded separately from
    #: `sparse_ms_mean` because `splade` REPLACES the ts_rank leg rather than adding to it —
    #: summing them would book a swap as a sum. For an UNSELECTED leg this is not zero but the
    #: branch-test interval, sub-microsecond, which rounds to 0.0 at 3 dp.
    #:
    #: ⚠️ Unlike `dense_ms_mean` and `sparse_ms_mean`, this bracket is not store-only: it
    #: encloses the SPLADE query encode as well. `learned_sparse_encode_ms_mean` splits that out,
    #: and only the remainder is counted in `store_ms_mean`.
    learned_sparse_ms_mean: float
    #: This leg's share of `learned_sparse_ms_mean` that is NOT a store round trip, measured as
    #: the stage bracket minus the store's own inner timer. Dominated by the SPLADE query encode,
    #: but strictly it is the encode PLUS the per-call overhead that the dense and ts_rank legs
    #: book into `store_ms`; microseconds against a forward pass. `max_nesting_violation_ms`
    #: does not bound that overhead — it measures the OPPOSITE sign, inner minus bracket, and
    #: must stay 0 — it only certifies the nesting the overhead follows from. The dense leg's
    #: counterpart is the separate `query_embedding` stage, which `store_ms` likewise excludes.
    learned_sparse_encode_ms_mean: float
    fusion_ms_mean: float
    rerank_ms_mean: float
    #: `newest_indexed_at()`, the store round trip outside every `stage_ms` bracket.
    meta_ms_mean: float
    residual_ms_mean: float

    #: dense + meta + whichever sparse legs the backend SELECTED, with the learned leg counted
    #: by its store round trip only (its encode is not store cost). What a store swap could
    #: address.
    #:
    #: NOTE the two are measured from different instruments. The mean sums the retriever's OUTER
    #: stage brackets for the dense and ts_rank legs, and the INNER store timer for meta and for
    #: the learned leg (whose bracket is not store-only); the p50 is the inner timer throughout.
    #: So on dense and ts_rank the mean/p50 gap includes per-call Python overhead, whose sign
    #: `max_nesting_violation_ms` certifies (it measures inner minus bracket, and must stay 0,
    #: so it bounds the gap away from zero rather than capping it), while on meta and the
    #: learned leg both come from one
    #: instrument and differ only as a mean differs from a median. Do not read them as the mean
    #: and median of one series. Both cover the SAME legs, which is why a leg that is switched
    #: off contributes to neither, despite still having a stage bracket.
    store_ms_mean: float
    store_ms_p50: float
    #: Fraction of end-to-end latency spent in the store, in [0, 1].
    store_share: float
    #: The upper bound on a store swap: what the total becomes if the replacement store is FREE.
    #: Unachievable by construction, which is what makes it a bound rather than a forecast.
    total_ms_if_store_were_free: float

    #: `None` when that leg was not in the pipeline, NOT 0.0. Zero is the issue #81 alarm value
    #: (a leg present but matching nothing), so collapsing "not measured" onto it would publish
    #: the alarm for every healthy run of the other backend.
    sparse_fire_rate: float | None
    learned_sparse_fire_rate: float | None
    #: True when the metric ring evicted samples. When set, no ratio above is emitted.
    truncated: bool
    #: max over queries of (store metric ms - its stage bracket ms). Must be <= 0: the store's
    #: internal timer measures a strict subinterval of the retriever's bracket around the call.
    max_nesting_violation_ms: float = 0.0
    notes: list[str] = field(default_factory=list)


def _drain(leg: str) -> tuple[list[float], int]:
    return METRICS.drain_histogram(STORE_QUERY_METRIC, leg=leg)


def _clear_legs() -> None:
    for leg in (LEG_DENSE, LEG_SPARSE, LEG_LEARNED_SPARSE, LEG_META):
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


def _learned_fire_rate(
    store: PgVectorStore,
    embedder: Embedder,
    encoder: object,
    queries: list[dict],
    candidate_k: int,
) -> float:
    """Fraction of queries whose LEARNED sparse leg returns at least one row.

    The issue #81 hazard is not specific to `ts_rank`: a leg that matches nothing for every query
    leaves fusion with a single list and the run dense-only under a hybrid label, whichever leg it
    is. `query_learned_sparse` raises `LookupError` on an unencoded corpus, which is a different
    and already-loud failure; this covers the corpus that IS encoded and still matches nothing.

    ⛔ Also the pre-flight for empty query encodings. `retriever.py` legitimately SKIPS the store
    call when a query encodes to no terms (a query of pure stopwords), so such a query records no
    metric sample. That is correct there and fatal here: it silently breaks the one-sample-per-
    query denominator, and the per-query p50 below zips the legs together under `strict=True`. So
    refuse up front, naming the query, rather than raising a length mismatch 100 lines later.
    """
    fired = 0
    for q in queries:
        weights = encoder.encode([q["query"]])[0]  # type: ignore[attr-defined]
        if not weights:
            raise AssertionError(
                f"query {q['query']!r} encodes to an EMPTY sparse vector. The retriever skips the "
                "store call for it, so the learned leg would record one sample fewer than there "
                "are queries and no per-query split could be quoted. Drop it from the query set."
            )
        # `vec` for the same reason `_sparse_fire_rate` passes it: `_query_learned_sparse`
        # branches its SELECT list on it, scoring by dense cosine when given one and by the
        # sparse dot product otherwise. The row set is the same either way, so the fraction
        # would be right regardless, but the probe would certify a statement the run never
        # executes and would warm a cheaper one that never touches the embedding column.
        if store.query_learned_sparse(
            weights,
            k=candidate_k,
            profile_id=encoder.profile.profile_id,  # type: ignore[attr-defined]
            vec=embed_query(embedder, q["query"]),
        ):
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
    sparse_backend: str = "lexical",
    sparse_encoder: object | None = None,
) -> LegSplit:
    if not queries:
        raise ValueError("no queries to time")
    # Up front, like the other refusals: with `repeats=0` the timing loop never runs, every
    # per-query count check passes vacuously against n=0, and the first symptom is an opaque
    # StatisticsError from `mean(totals)` — after the operator has already paid for the corpus
    # build and the index.
    if repeats < 1:
        raise ValueError(f"repeats must be >= 1, got {repeats}; nothing would be measured")
    # Which sparse legs actually run decides which legs MUST record one sample per query below.
    # Deriving that from "did this leg record anything" instead would be circular: recording
    # nothing is exactly the failure the per-query denominator check exists to catch, so a leg
    # that silently stopped would excuse itself from its own guard.
    wants_lexical = sparse_backend in ("lexical", "both")
    wants_learned = sparse_backend in ("splade", "both")
    retr = HybridRetriever(
        store,
        embedder,
        reranker=reranker,
        candidate_k=candidate_k,
        use_sparse=True,
        sparse_backend=sparse_backend,
        sparse_encoder=sparse_encoder,
    )

    # Probes `query_sparse`, i.e. the ts_rank leg specifically. Under `splade` that leg is not in
    # the pipeline at all, so the probe would certify a statement the run never executes and its
    # samples would be the only ts_rank timings in the split. The floor still applies wherever the
    # lexical leg DOES run, which is what issue #81 was about. `None` rather than 0.0 when it does
    # not run: 0.0 is the issue #81 ALARM value, and a healthy splade run must not publish it.
    fire_rate = _sparse_fire_rate(store, embedder, queries, candidate_k) if wants_lexical else None
    if fire_rate is not None and fire_rate < _MIN_SPARSE_FIRE_RATE:
        raise AssertionError(
            f"sparse leg returned rows for {fire_rate:.1%} of queries (floor "
            f"{_MIN_SPARSE_FIRE_RATE:.0%}). This is a dense-only pipeline wearing a hybrid label "
            "(issue #81); no hybrid split may be quoted from it."
        )

    learned_fire_rate = (
        _learned_fire_rate(store, embedder, sparse_encoder, queries, candidate_k)
        if wants_learned
        else None
    )
    if learned_fire_rate is not None and learned_fire_rate < _MIN_SPARSE_FIRE_RATE:
        raise AssertionError(
            f"learned sparse leg returned rows for {learned_fire_rate:.1%} of queries (floor "
            f"{_MIN_SPARSE_FIRE_RATE:.0%}). Issue #81 in the SPLADE arm: fusion sees one list and "
            "the pipeline is dense-only wearing a hybrid label. No split may be quoted from it."
        )

    # Discarded warm-up through the FULL pipeline: warms every selected leg, the plan cache and
    # the pool symmetrically. Warming only one leg biases the split, which is the quantity being
    # published.
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
    learned_metric, learned_total = _drain(LEG_LEARNED_SPARSE)
    meta_metric, meta_total = _drain(LEG_META)

    # Only the legs the configuration actually selected are held to one sample per query; a leg
    # that is switched off must record NOTHING, which is asserted just as hard. Both directions
    # matter: the first catches a leg that silently stopped recording (it would read exactly like
    # a leg that costs nothing), the second catches a leg running when the split says it is not,
    # which would attribute its cost to the residual.
    active = [(LEG_DENSE, dense_metric, dense_total), (LEG_META, meta_metric, meta_total)]
    idle: list[tuple[str, list[float], int]] = []
    (active if wants_lexical else idle).append((LEG_SPARSE, sparse_metric, sparse_total))
    (active if wants_learned else idle).append(
        (LEG_LEARNED_SPARSE, learned_metric, learned_total)
    )

    for label, _samples, observed in active:
        if observed != n:
            raise AssertionError(
                f"leg {label!r} recorded {observed} samples for {n} queries. The per-query "
                "denominator does not hold, so no per-query mean may be quoted."
            )
    for label, _samples, observed in idle:
        if observed:
            raise AssertionError(
                f"leg {label!r} is not in the {sparse_backend!r} pipeline but recorded {observed} "
                "samples. The split would book its cost as unattributed residual."
            )

    truncated = any(t > len(s) for _, s, t in active)

    # Nesting cross-check: the store's internal timer measures a subinterval of the retriever's
    # bracket around the same call, so stage >= metric for every query. A violation means the two
    # instruments are not timing what their names say.
    # `strict=True` matters more than it looks. `stages[...]` is a head-ordered list and the metric
    # samples come from a TAIL-retaining ring, so on eviction the two do not merely differ in
    # length, they MISALIGN — sample n-1023 would be compared against bracket 1, producing both
    # false violations and false passes. Unequal lengths must raise, not be silently absorbed.
    # Only the selected legs are paired. An idle leg still HAS a stage bracket (`retriever.py`
    # records the timing outside the backend `if`, so the key never disappears) but has no metric
    # samples, and `strict=True` would read that legitimate pairing of n brackets against 0
    # samples as a length mismatch.
    nesting_pairs = [("dense_retrieval", dense_metric)]
    if wants_lexical:
        nesting_pairs.append(("sparse_retrieval", sparse_metric))
    if wants_learned:
        nesting_pairs.append(("learned_sparse_retrieval", learned_metric))
    if not truncated:
        for stage_key, metric_samples in nesting_pairs:
            for bracket, inner in zip(stages.get(stage_key, []), metric_samples, strict=True):
                nesting_violation = max(nesting_violation, inner - bracket)

    def stage_mean(name: str) -> float:
        return mean(stages[name]) if stages.get(name) else 0.0

    embed_ms = stage_mean("query_embedding")
    dense_ms = stage_mean("dense_retrieval")
    sparse_ms = stage_mean("sparse_retrieval")
    learned_ms = stage_mean("learned_sparse_retrieval")
    fusion_ms = stage_mean("fusion")
    rerank_ms = stage_mean("reranking")
    meta_ms = mean(meta_metric) if meta_metric else 0.0
    total_ms = mean(totals)
    if total_ms <= 0:
        raise AssertionError("measured total is zero; the run did not time anything")

    # The learned leg belongs in BOTH sums, but at different GRANULARITY: its whole bracket in
    # the residual, its store round trip only in `store_ms`. Dropping it from either is the
    # failure this replaced. Omitting it from `store_ms` understates the store's share, and
    # omitting it from the residual hides that understatement inside "unattributed" — where the
    # guard below cannot see it, because that guard fires only on a NEGATIVE residual (parts
    # exceeding the whole). A leg missing from the split makes the residual too LARGE, which is
    # the silent direction, and it is the direction that flatters the "the store is cheap"
    # hypothesis this file exists to test. What must NOT reach `store_ms` is the encode, below.
    # ⛔ The learned bracket is NOT a store-only bracket, and this is the one asymmetry in the
    # file that would quietly corrupt the headline number. `retriever.py` opens it, runs
    # `encoder.encode(query)` — a tokenisation plus a transformer forward pass — and only THEN
    # calls the store. The dense leg's equivalent cost is a separate `query_embedding` stage that
    # `store_ms` deliberately excludes, so counting the learned bracket whole would book model
    # inference as store latency, overstating what a store swap could buy: the exact decision
    # this benchmark prices. So the store term comes from the INNER store timer (as `meta_ms`
    # already does), and the encode is published under its own name rather than hidden in either
    # the store's share or the residual.
    learned_store_ms = mean(learned_metric) if wants_learned and learned_metric else 0.0
    learned_encode_ms = 0.0
    # `not truncated` for the same reason the nesting cross-check above carries it: on eviction
    # `learned_ms` is a mean over ALL n brackets while `learned_store_ms` is a mean over the ring's
    # retained SUFFIX, so the two are windows on different populations and their difference is not
    # an encode. Without this gate a truncated run can raise "the instruments disagree" when its
    # actual fault is truncation, which the refusal below diagnoses correctly.
    if wants_learned and not truncated:
        # The store timer opens strictly INSIDE the retriever's bracket, so the bracket cannot be
        # the smaller of the two. If it is, the two instruments are not timing what their names
        # say and the encode figure derived from their difference is meaningless: refuse, on the
        # same standard as the residual guard below, which raises on the structurally identical
        # parts-exceed-the-whole fault.
        #
        # A gap at or just above zero is NOT refused, even though a real SPLADE forward pass is
        # never 0.0 at 3 dp. `measure()` cannot know what encoder it was handed — a stub that
        # returns fixed weights encodes in microseconds and is a legitimate way to exercise the
        # arm — so a zero encode is a fact about the caller's encoder, not proof of a broken
        # instrument. It is reported rather than judged.
        gap = learned_ms - learned_store_ms
        if gap < -_RESIDUAL_TOLERANCE_MS:
            raise AssertionError(
                f"the learned leg's store timer ({learned_store_ms:.3f} ms) exceeds its stage "
                f"bracket ({learned_ms:.3f} ms) by {-gap:.3f} ms, but it opens inside that "
                "bracket. The instruments disagree; no encode split may be derived from them."
            )
        learned_encode_ms = max(0.0, gap)  # the real inversion raised above; this is float noise

    # `store_ms` counts only the legs that RAN, because it is what a store swap could address and
    # a switched-off leg offers nothing to remove. The residual subtracts EVERY bracket the
    # pipeline recorded, including the sub-microsecond one around a skipped leg and the SPLADE
    # encode, because those intervals really did elapse inside the measured total.
    store_ms = (
        dense_ms
        + meta_ms
        + (sparse_ms if wants_lexical else 0.0)
        + learned_store_ms
    )
    residual = total_ms - (
        embed_ms + dense_ms + sparse_ms + learned_ms + fusion_ms + rerank_ms + meta_ms
    )
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

    # Summed PER QUERY before taking the median, so this stays the median of one store-cost
    # series rather than a sum of independent medians. Only the selected legs contribute; an idle
    # leg has no samples to zip and adds 0 to every query by construction.
    per_query_store = [a + b for a, b in zip(dense_metric, meta_metric, strict=True)]
    for leg_samples in ([sparse_metric] if wants_lexical else []) + (
        [learned_metric] if wants_learned else []
    ):
        per_query_store = [
            running + one for running, one in zip(per_query_store, leg_samples, strict=True)
        ]
    store_p50 = percentile(sorted(per_query_store), 0.50)
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
        learned_sparse_ms_mean=round(learned_ms, 3),
        learned_sparse_encode_ms_mean=round(learned_encode_ms, 3),
        fusion_ms_mean=round(fusion_ms, 3),
        rerank_ms_mean=round(rerank_ms, 3),
        meta_ms_mean=round(meta_ms, 3),
        residual_ms_mean=round(residual, 3),
        store_ms_mean=round(store_ms, 3),
        store_ms_p50=store_p50,
        store_share=round(store_ms / total_ms, 4),
        total_ms_if_store_were_free=round(total_ms - store_ms, 3),
        sparse_fire_rate=None if fire_rate is None else round(fire_rate, 4),
        learned_sparse_fire_rate=(
            None if learned_fire_rate is None else round(learned_fire_rate, 4)
        ),
        truncated=truncated,
        max_nesting_violation_ms=round(nesting_violation, 3),
        notes=notes,
    )


def to_markdown(splits: list[LegSplit], ctx: str) -> str:
    def pct(value: float | None) -> str:
        # "n/a" is not decoration. This column exists to expose the issue #81 dead-leg condition,
        # whose signature is 0%, so a leg that was never in the pipeline must not render as 0%.
        return "n/a" if value is None else f"{value:.0%}"

    rows = [
        "| chunks | cand_k | rerank | total | embed | dense | sparse | learned | splade enc | "
        "meta | fusion | rerank | **store** | resid | **store share** | sparse fire | "
        "learned fire |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in splits:
        share = "truncated" if s.truncated else f"{s.store_share:.1%}"
        rows.append(
            f"| {s.n_chunks} | {s.candidate_k} | {'yes' if s.reranked else 'no'} | "
            f"{s.total_ms_mean:.1f} | {s.embed_ms_mean:.1f} | {s.dense_ms_mean:.1f} | "
            f"{s.sparse_ms_mean:.1f} | {s.learned_sparse_ms_mean:.1f} | "
            f"{s.learned_sparse_encode_ms_mean:.1f} | {s.meta_ms_mean:.1f} | "
            f"{s.fusion_ms_mean:.1f} | "
            f"{s.rerank_ms_mean:.1f} | **{s.store_ms_mean:.1f}** | {s.residual_ms_mean:.1f} | "
            f"**{share}** | {pct(s.sparse_fire_rate)} | {pct(s.learned_sparse_fire_rate)} |"
        )
    rows.append("")
    rows.append(
        "All figures are ms/query, means over `repeats x n_queries`, measured warm (each "
        "configuration gets a discarded full-pipeline warm-up pass). `store` = dense + meta + "
        "whichever sparse legs the backend selected (`sparse` is ts_rank, `learned` is SPLADE, "
        "and `splade` REPLACES the ts_rank leg rather than adding to it), where meta is "
        "`newest_indexed_at()`, the per-search round trip that sits outside every `stage_ms` "
        "bracket. A leg that did not run still shows a sub-microsecond bracket in its own column "
        "(the timing is taken outside the backend test) but contributes nothing to `store`; its "
        "fire column reads `n/a`, NOT 0%, which is the dead-leg alarm value. `learned` is the "
        "only leg column that is not store-only: it encloses the SPLADE query encode, shown "
        "beside it as `splade enc` and EXCLUDED from `store`, because model inference is not "
        "something a store swap could remove. ⚠️ `splade enc` is a SUBSET of `learned`, not an "
        "additional interval: the row reconciles as embed + dense + sparse + learned + meta + "
        "fusion + rerank + resid = total, with `splade enc` ignored. The `learned` remainder "
        "also carries one `sparse_row_count` "
        "round trip per query that the ts_rank leg does not pay, so the two sparse columns are "
        "not exactly like-for-like. `store share` is the ceiling on any store swap: a "
        "replacement that cost nothing would remove exactly this fraction."
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
    if args.queries < 1 or args.repeats < 1:
        ap.error(
            f"--queries ({args.queries}) and --repeats ({args.repeats}) must both be >= 1; "
            "below that the timing loop never runs and every per-query guard passes vacuously"
        )
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
