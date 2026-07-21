"""Ablation runner: index the eval corpus per embedder and score each retrieval config."""
from __future__ import annotations

import json
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import math

from recall.calibration import Calibration, from_samples
from recall.embeddings import Embedder
from recall.entailment import EntailmentJudge
from recall.eval.calibrate import measure_top_cosines
from recall.eval.metrics import (
    abstention_accuracy,
    false_abstain_rate,
    false_confident_rate,
    fraction_true,
    gap_false_confident_rate,
    mrr,
    near_miss_false_confident_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    successor_accuracy,
    superseded_trust_rate,
    wilson_ci,
)
from recall.index import Indexer
from recall.rerank import CrossEncoderReranker, Reranker
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore
from recall.timing import TimedEmbedder, TimedReranker, TimingStats, timed_call
from recall.trust import trusted_search
from recall.types import ScoredChunk, TrustedHit

EVAL_DIR = Path(__file__).parent
FUSIONS = ["dense", "hybrid", "hybrid+rerank"]


@dataclass
class AblationResult:
    embedder: str
    fusion: str
    p_at_5: float
    r_at_5: float
    mrr: float
    ndcg_at_10: float
    #: ANALYTIC, not measured: with no gap guard the system never abstains, so every
    #: unanswerable query is answered confidently and the rate is 1.0 by definition. Kept as the
    #: comparison baseline for `fcr_with_guard`, and rendered with a "by construction" marker so
    #: it is not read as an observation. Do not chart or cite it as a measurement.
    fcr_no_guard: float
    fcr_with_guard: float
    # Cost/latency metadata (mean wall time per call, ms). Additive; defaulted so existing
    # constructors/tests are unaffected. rerank_ms_mean is 0.0 for configs without a reranker.
    embed_ms_mean: float = 0.0
    rerank_ms_mean: float = 0.0


def _key(hit: ScoredChunk) -> str:
    md = hit.chunk.metadata
    return f"{md['file']}:{md['ord']}"


@contextmanager
def _throwaway_store(dsn: str, emb: Embedder, corpus_dir: Path, prefix: str):
    """Schema-created, corpus-indexed throwaway store for one eval stage.

    Setup runs INSIDE the guard: a failure while creating the schema or indexing must still
    drop the uuid-named table and close the connection — and the single copy of the teardown
    lives here instead of being repeated per runner.
    """
    table = prefix + uuid.uuid4().hex[:8]
    store = PgVectorStore(dsn, dim=emb.dim, table=table)
    try:
        store.ensure_schema()
        Indexer(store, emb).index_path(corpus_dir)
        yield store
    finally:
        try:
            store.drop_table()
        except Exception:
            pass  # best-effort drop of the throwaway uuid table
        finally:
            store.close()


def _score_config(
    store: PgVectorStore, embedder: Embedder, queries: list[dict], fusion: str,
    reranker: Reranker | None,
) -> AblationResult:
    timed_emb = TimedEmbedder(embedder)
    timed_rr = (
        TimedReranker(reranker) if (fusion == "hybrid+rerank" and reranker is not None) else None
    )
    retr = HybridRetriever(
        store, timed_emb,
        reranker=timed_rr,
        use_sparse=(fusion != "dense"),
    )
    ps, rs, ms, ns, unans_gaps = [], [], [], [], []
    for q in queries:
        if q.get("trust"):
            continue  # trust-sensitive queries are scored by run_trust_eval, not the ablations
        res = retr.search(q["query"], k=10)
        retrieved = [_key(h) for h in res.hits]
        if q["answerable"]:
            ps.append(precision_at_k(retrieved, q["relevant_ids"], 5))
            rs.append(recall_at_k(retrieved, q["relevant_ids"], 5))
            ms.append(mrr(retrieved, q["relevant_ids"]))
            ns.append(ndcg_at_k(retrieved, q["relevant_ids"], 10))
        else:
            unans_gaps.append(res.gap_warning)
    return AblationResult(
        embedder=embedder.name, fusion=fusion,
        p_at_5=mean(ps) if ps else 0.0,
        r_at_5=mean(rs) if rs else 0.0,
        mrr=mean(ms) if ms else 0.0,
        ndcg_at_10=mean(ns) if ns else 0.0,
        fcr_no_guard=1.0, fcr_with_guard=false_confident_rate(unans_gaps),
        embed_ms_mean=timed_emb.stats.mean_ms,
        rerank_ms_mean=timed_rr.stats.mean_ms if timed_rr else 0.0,
    )


def run_ablations(
    dsn: str, embedders: list[Embedder], corpus_dir: Path | None = None,
    queries_path: Path | None = None, fusions: list[str] | None = None,
) -> list[AblationResult]:
    """Index the eval corpus once per embedder, then score every fusion config against the queries."""
    corpus_dir = corpus_dir or (EVAL_DIR / "corpus")
    queries = json.loads(
        Path(queries_path or (EVAL_DIR / "queries.json")).read_text(encoding="utf-8")
    )
    fusions = list(fusions or FUSIONS)
    reranker: Reranker | None = None
    if "hybrid+rerank" in fusions:
        try:
            reranker = CrossEncoderReranker()
        except ImportError:
            fusions = [f for f in fusions if f != "hybrid+rerank"]

    results: list[AblationResult] = []
    for emb in embedders:
        with _throwaway_store(dsn, emb, corpus_dir, "eval_") as store:
            for fusion in fusions:
                results.append(_score_config(store, emb, queries, fusion, reranker))
    return results


#: Abstention arms. A = the calibrated cosine threshold (status quo). B = threshold plus
#: the entailment judge. C = the judge alone (threshold disabled) — the ablation proving any
#: near-miss win is the judge's, not the threshold's. Single definition: `arm_setup` and the
#: chart colors are keyed by these constants so a typo cannot desynchronize them.
ARM_THRESHOLD = "threshold"
ARM_STACKED = "threshold+entail"
ARM_ENTAIL_ONLY = "entail-only"
ARMS = [ARM_THRESHOLD, ARM_STACKED, ARM_ENTAIL_ONLY]


@dataclass
class NearMissEvalResult:
    embedder: str
    arm: str
    nearmiss_fcr: float           # near-miss queries answered confidently (lower is better)
    gap_fcr: float                # classic far-gap queries answered confidently — must not regress
    false_abstain: float          # answerable queries wrongly abstained — must not regress
    mrr_answerable: float
    entail_latency_ms_mean: float  # judge stage, averaged over the queries the judge actually
    #                                RAN on (threshold-abstained queries never reach it) — a
    #                                different denominator than query_latency_ms_mean, so in the
    #                                stacked arm this can EXCEED the all-queries total mean.
    #                                0.0 for the threshold arm.
    query_latency_ms_mean: float   # full trusted_search (+judge) wall time, mean over ALL queries


def _loo_calibrations(
    name: str, held_out: list[float], other: list[float], *, hold_out_unanswerable: bool
) -> list[Calibration | None]:
    """One refitted calibration per sample in `held_out`, each fitted WITHOUT that sample.

    Returns a list positionally aligned with `held_out`, so entry *i* is the calibration that
    may legitimately score query *i*. `hold_out_unanswerable` says which side of `from_samples`
    the held-out class belongs on.

    An entry is ``None`` when the class has fewer than two samples: the fold would fit on an
    empty class, and a threshold fitted to nothing is not a more honest number than the
    in-sample one — it is a different fabrication. The caller falls back to the full-sample
    calibration and the result stays in-sample, which is why the published table needs the
    sample counts beside it.
    """
    if len(held_out) < 2:
        return [None] * len(held_out)
    out: list[Calibration | None] = []
    for i in range(len(held_out)):
        rest = held_out[:i] + held_out[i + 1:]
        ans, unans = (other, rest) if hold_out_unanswerable else (rest, other)
        out.append(from_samples(name, ans, unans))
    return out


class _TimedJudge:
    """Wraps a judge to measure its wall time — the honest cost column of the results table.

    Built on the shared ``timing.timed_call`` utility (same primitive as ``TimedEmbedder`` /
    ``TimedReranker``); ``samples_ms`` is kept so per-call latencies remain available to callers.
    """

    def __init__(self, inner: EntailmentJudge) -> None:
        self._inner = inner
        self._stats = TimingStats()
        self.samples_ms: list[float] = []

    def judge(self, query: str, texts: list[str]) -> list[bool]:
        out = timed_call(self._stats, lambda: self._inner.judge(query, texts))
        self.samples_ms.append(self._stats.last_ms)
        return out


def run_nearmiss_eval(
    dsn: str, embedders: list[Embedder], judge: EntailmentJudge,
    corpus_dir: Path | None = None, queries_path: Path | None = None,
    nearmiss_path: Path | None = None, k: int = 10,
) -> list[NearMissEvalResult]:
    """Score the three abstention arms per embedder on answerable / far-gap / near-miss queries.

    The calibration is built ONLY from the labeled answerable/unanswerable queries in
    `queries.json` — the near-miss set is a held-out challenge set and must never tune the
    threshold it challenges. "Confident" for every arm means `trusted_search` did not abstain;
    the entailment arms share the SAME judge instance across embedders with no per-embedder
    adjustment — that transfer is the property under test.

    The near-miss set being held out does NOT make the other two columns held out. `gap_fcr` and
    `false_abstain` are measured on the very queries the threshold is fitted to, so a threshold
    that merely memorised them would score perfectly. Both are therefore measured under
    LEAVE-ONE-OUT: the calibration judging a query is refitted with that query's sample removed
    (`_loo_calibrations`), so no query is ever scored by a threshold that saw it. The entail-only
    arm is exempt — its threshold is a fixed -1.0 that is fitted to nothing.
    """
    corpus_dir = corpus_dir or (EVAL_DIR / "corpus")
    queries = json.loads(
        Path(queries_path or (EVAL_DIR / "queries.json")).read_text(encoding="utf-8")
    )
    nearmiss = json.loads(
        Path(nearmiss_path or (EVAL_DIR / "near_miss.json")).read_text(encoding="utf-8")
    )
    plain = [q for q in queries if not q.get("trust")]
    answerable = [q for q in plain if q["answerable"]]
    gaps = [q for q in plain if not q["answerable"]]

    results: list[NearMissEvalResult] = []
    for emb in embedders:
        with _throwaway_store(dsn, emb, corpus_dir, "nm_") as store:
            # measure_top_cosines walks `plain` in order and splits by label, so ans_cos is
            # positionally aligned with `answerable` and unans_cos with `gaps` — that alignment
            # is what lets a held-out fold be matched back to the query it belongs to.
            ans_cos, unans_cos = measure_top_cosines(store, emb, plain)
            cal = from_samples(emb.name, ans_cos, unans_cos)
            gap_cals = _loo_calibrations(emb.name, unans_cos, ans_cos, hold_out_unanswerable=True)
            ans_cals = _loo_calibrations(emb.name, ans_cos, unans_cos, hold_out_unanswerable=False)
            # threshold -1 passes every cosine: isolates the judge in the entail-only arm
            permissive = Calibration(embedder=emb.name, threshold=-1.0, scale=cal.scale)
            # third element: whether this arm's threshold was FITTED to the eval samples, and so
            # must be swapped for the leave-one-out refit when scoring them.
            arm_setup = {
                ARM_THRESHOLD: (cal, None, True),
                ARM_STACKED: (cal, judge, True),
                ARM_ENTAIL_ONLY: (permissive, judge, False),
            }
            # Each arm re-runs retrieval end-to-end ON PURPOSE: query_latency_ms_mean must
            # measure real per-arm wall time, and the judge is deliberately un-memoized so
            # its latency column counts model passes, not cache hits. Sharing arm A's
            # results with arm B would save ~1/3 of the retrieval cost at the price of
            # fabricating the latency columns.
            for arm in ARMS:
                arm_cal, arm_judge, arm_is_fitted = arm_setup[arm]
                timed = _TimedJudge(arm_judge) if arm_judge is not None else None
                q_times: list[float] = []

                def _search(text: str, loo_cal: Calibration | None = None):
                    # loo_cal is the refit that never saw this query; fall back to the arm's own
                    # calibration when the arm is unfitted, or when the class was too small to
                    # hold a sample out (see _loo_calibrations).
                    use = loo_cal if (arm_is_fitted and loo_cal is not None) else arm_cal
                    t0 = time.perf_counter()
                    res = trusted_search(store, emb, text, k=k, calibration=use,
                                         entailment=timed)
                    q_times.append((time.perf_counter() - t0) * 1000.0)
                    return res

                # near-miss queries are already held out by construction — they never enter the
                # calibration — so they are scored with the full-sample calibration.
                nm_confident = [not _search(q["query"]).abstained for q in nearmiss]
                gap_confident = [
                    not _search(q["query"], gap_cals[i]).abstained for i, q in enumerate(gaps)
                ]
                abst_flags, mrrs = [], []
                for i, q in enumerate(answerable):
                    res = _search(q["query"], ans_cals[i])
                    abst_flags.append(res.abstained)
                    ok_keys = [_tkey(h) for h in res.hits if h.verdict == "ok"]
                    mrrs.append(mrr(ok_keys, q["relevant_ids"]))
                results.append(
                    NearMissEvalResult(
                        embedder=emb.name,
                        arm=arm,
                        nearmiss_fcr=near_miss_false_confident_rate(nm_confident),
                        gap_fcr=gap_false_confident_rate(gap_confident),
                        false_abstain=false_abstain_rate(abst_flags),
                        mrr_answerable=mean(mrrs) if mrrs else 0.0,
                        entail_latency_ms_mean=(
                            mean(timed.samples_ms) if timed and timed.samples_ms else 0.0
                        ),
                        query_latency_ms_mean=mean(q_times) if q_times else 0.0,
                    )
                )
    return results


def nearmiss_results_to_markdown(results: list[NearMissEvalResult]) -> str:
    lines = [
        "| embedder | arm | near-miss FCR | gap FCR | false-abstain | MRR ans | "
        "judge ms (judged calls) | total ms/query |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.embedder} | {r.arm} | {_fmt_rate(r.nearmiss_fcr)} | {_fmt_rate(r.gap_fcr)} | "
            f"{_fmt_rate(r.false_abstain)} | {_fmt_rate(r.mrr_answerable, 3)} | "
            f"{r.entail_latency_ms_mean:.0f} | {r.query_latency_ms_mean:.0f} |"
        )
    return "\n".join(lines)


@dataclass
class TrustEvalResult:
    embedder: str
    str_baseline: float          # superseded-trust rate of plain search (top-1 is a stale id)
    str_recency: float           # STR of "trust the newest indexed hit" (stale docs re-synced)
    str_trust: float             # superseded-trust rate with the trust layer (stale id verdict ok)
    successor_acc: float         # expect=successor queries whose top trusted hit is the successor
    abstain_acc: float           # expect=abstain queries that actually abstained
    mrr_answerable_baseline: float
    mrr_answerable_trust: float  # must equal baseline: trust must not damage ordinary retrieval
    #: Fraction of trust-sensitive queries that returned ANY `ok` hit. Published beside
    #: `str_trust` because that rate alone cannot tell a working trust layer from a broken one:
    #: `str_trust` counts queries where a stale id was served as `ok`, so a system that returns
    #: nothing at all scores a perfect 0.00. Read the pair — 0.00 STR at 1.00 coverage is the
    #: claim; 0.00 STR at 0.00 coverage is a system that answered nothing.
    trust_coverage: float = float("nan")
    # 95% Wilson score intervals for the headline rates. NOT bootstrapped: the per-class samples
    # here are tiny (n=2 for abstention, n=4 for successor accuracy) and usually degenerate, and
    # a percentile bootstrap of an all-True sample returns [1.00, 1.00] — certainty from two
    # observations. (nan, nan) when the arm had no queries of that class.
    successor_acc_ci: tuple[float, float] = (float("nan"), float("nan"))
    abstain_acc_ci: tuple[float, float] = (float("nan"), float("nan"))
    str_trust_ci: tuple[float, float] = (float("nan"), float("nan"))
    trust_coverage_ci: tuple[float, float] = (float("nan"), float("nan"))
    #: Sample count behind each rate — an interval is unreadable without its n.
    n_trust_queries: int = 0
    n_successor: int = 0
    n_abstain: int = 0


def _tkey(hit: TrustedHit) -> str:
    return f"{hit.provenance.file}:{hit.provenance.ord}"


def run_trust_eval(
    dsn: str, embedders: list[Embedder], corpus_dir: Path | None = None,
    queries_path: Path | None = None, touch_stale: bool = True,
) -> list[TrustEvalResult]:
    """Score the validity-sensitive queries in three modes per embedder.

    baseline = plain hybrid search (no trust layer): a query counts as stale-trusted when its
    top-1 hit is a superseded/expired memory — exactly what a consumer would read as the answer.
    recency  = "trust the newest indexed hit" — the timestamp heuristic supersession is often
    mistaken for. With `touch_stale` (default) the stale docs get a timestamp-only store touch
    AFTER the initial pass (`store.touch_files` — no re-embed, so the baseline and trust
    measurements are unaffected BY CONSTRUCTION, for any embedder), simulating the re-sync/edit
    any living corpus performs constantly: the stale memory then carries the newest timestamp,
    and a per-document timestamp cannot see the supersession RELATION that makes it stale.
    trust    = trusted_search with an in-run calibration (built from the labeled answerable/
    unanswerable queries): a query counts as stale-trusted only if a stale id still carries
    verdict `ok`. Also reports successor accuracy, abstention accuracy, and the answerable-MRR
    regression check (trust must not change ordinary retrieval quality).
    """
    corpus_dir = corpus_dir or (EVAL_DIR / "corpus")
    queries = json.loads(
        Path(queries_path or (EVAL_DIR / "queries.json")).read_text(encoding="utf-8")
    )
    plain = [q for q in queries if not q.get("trust")]
    trust_qs = [q for q in queries if q.get("trust")]

    results: list[TrustEvalResult] = []
    for emb in embedders:
        with _throwaway_store(dsn, emb, corpus_dir, "trust_") as store:
            if touch_stale:
                store.touch_files(sorted(
                    {sid.rsplit(":", 1)[0] for q in trust_qs for sid in q["stale_ids"]}
                ))
            # in-run calibration from the labeled plain queries (per-embedder threshold —
            # a fixed constant does not transfer across embedders, see FINDINGS §2)
            cal = from_samples(emb.name, *measure_top_cosines(store, emb, plain))

            retr = HybridRetriever(store, emb)
            base_flags, rec_flags, trust_flags, succ_flags, abst_flags = [], [], [], [], []
            cov_flags: list[bool] = []
            for q in trust_qs:
                stale = set(q["stale_ids"])
                succ = set(q.get("successor_ids", []))
                bres = retr.search(q["query"], k=10)
                base_flags.append(bool(bres.hits) and _key(bres.hits[0]) in stale)
                if bres.hits:
                    # steelman of the timestamp approach: among the CONFIDENTLY-RELEVANT hits
                    # (same calibrated threshold the guards use) prefer the newest — recency as
                    # a tie-break, not a global newest-wins strawman. It still cannot see the
                    # supersession relation.
                    epoch = datetime.min.replace(tzinfo=timezone.utc)
                    pool = [h for h in bres.hits if h.score >= cal.threshold] or bres.hits[:1]
                    newest = max(pool, key=lambda h: h.indexed_at or epoch)
                    rec_flags.append(_key(newest) in stale)
                else:
                    rec_flags.append(False)
                tres = trusted_search(store, emb, q["query"], k=10, calibration=cal)
                ok_keys = [_tkey(h) for h in tres.hits if h.verdict == "ok"]
                trust_flags.append(any(k in stale for k in ok_keys))
                # coverage guards str_trust against its degenerate reading: a query that
                # returned no `ok` hit at all also contributes a "clean" 0 to trust_flags.
                cov_flags.append(bool(ok_keys))
                if q["expect"] == "successor":
                    succ_flags.append(bool(ok_keys) and ok_keys[0] in succ)
                else:
                    abst_flags.append(tres.abstained)

            base_mrrs, trust_mrrs = [], []
            for q in plain:
                if not q["answerable"]:
                    continue
                bres = retr.search(q["query"], k=10)
                base_mrrs.append(mrr([_key(h) for h in bres.hits], q["relevant_ids"]))
                tres = trusted_search(store, emb, q["query"], k=10, calibration=cal)
                trust_mrrs.append(mrr([_tkey(h) for h in tres.hits], q["relevant_ids"]))

            results.append(
                TrustEvalResult(
                    embedder=emb.name,
                    str_baseline=superseded_trust_rate(base_flags),
                    str_recency=superseded_trust_rate(rec_flags),
                    str_trust=superseded_trust_rate(trust_flags),
                    successor_acc=successor_accuracy(succ_flags),
                    abstain_acc=abstention_accuracy(abst_flags),
                    mrr_answerable_baseline=mean(base_mrrs) if base_mrrs else 0.0,
                    mrr_answerable_trust=mean(trust_mrrs) if trust_mrrs else 0.0,
                    trust_coverage=fraction_true(cov_flags),
                    successor_acc_ci=wilson_ci(succ_flags),
                    abstain_acc_ci=wilson_ci(abst_flags),
                    str_trust_ci=wilson_ci(trust_flags),
                    trust_coverage_ci=wilson_ci(cov_flags),
                    n_trust_queries=len(trust_flags),
                    n_successor=len(succ_flags),
                    n_abstain=len(abst_flags),
                )
            )
    return results


def _fmt_rate(x: float, digits: int = 2) -> str:
    return "n/a" if math.isnan(x) else f"{x:.{digits}f}"


def _fmt_ci(ci: tuple[float, float], digits: int = 2) -> str:
    lo, hi = ci
    if math.isnan(lo) or math.isnan(hi):
        return "n/a"
    return f"[{lo:.{digits}f}, {hi:.{digits}f}]"


def trust_results_to_markdown(results: list[TrustEvalResult]) -> str:
    lines = [
        "| embedder | STR baseline | STR recency | STR trust | trust coverage | successor acc | "
        "abstain acc | MRR ans (base) | MRR ans (trust) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.embedder} | {_fmt_rate(r.str_baseline)} | {_fmt_rate(r.str_recency)} | "
            f"{_fmt_rate(r.str_trust)} | {_fmt_rate(r.trust_coverage)} | "
            f"{_fmt_rate(r.successor_acc)} | {_fmt_rate(r.abstain_acc)} | "
            f"{_fmt_rate(r.mrr_answerable_baseline, 3)} | {_fmt_rate(r.mrr_answerable_trust, 3)} |"
        )
    lines.append("")
    lines.append(
        "**Read STR trust together with trust coverage.** STR counts queries where a stale "
        "memory was served with verdict `ok`, so a system that returns nothing scores a perfect "
        "0.00. The claim is 0.00 STR *at high coverage*; 0.00 STR at low coverage is a system "
        "that abstained its way to a good number."
    )
    # Point estimates above read as precise, but the eval set is tiny. Report an interval for the
    # headline rates so the interval — not the point — is what's compared. Wilson, not bootstrap:
    # resampling an all-True sample of n=2 returns [1.00, 1.00], i.e. certainty from no evidence.
    lines.append("")
    lines.append("95% Wilson score intervals for the headline rates (n in parentheses):")
    lines.append("")
    lines.append("| embedder | STR trust | trust coverage | successor acc | abstain acc |")
    lines.append("|---|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.embedder} | {_fmt_ci(r.str_trust_ci)} (n={r.n_trust_queries}) | "
            f"{_fmt_ci(r.trust_coverage_ci)} (n={r.n_trust_queries}) | "
            f"{_fmt_ci(r.successor_acc_ci)} (n={r.n_successor}) | "
            f"{_fmt_ci(r.abstain_acc_ci)} (n={r.n_abstain}) |"
        )
    return "\n".join(lines)


def save_trust_chart(results: list[TrustEvalResult], out_dir: Path) -> Path:
    """Paired-bar chart: superseded-trust rate, plain search vs trust layer, per embedder."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    embs = [r.embedder for r in results]
    fig, ax = plt.subplots(figsize=(max(5, len(embs) * 1.8), 4))
    x = range(len(embs))
    ax.bar([i - 0.2 for i in x], [r.str_baseline for r in results], width=0.4,
           label="plain search", color="#c44e52")
    ax.bar([i + 0.2 for i in x], [r.str_trust for r in results], width=0.4,
           label="trust layer", color="#55a868")
    ax.set_xticks(list(x))
    ax.set_xticklabels(embs, rotation=20, ha="right")
    ax.set_ylabel("superseded-trust rate")
    ax.set_ylim(0, 1)
    ax.set_title("How often a stale memory is presented as the answer (lower is better)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "trust_effect.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def save_nearmiss_chart(results: list[NearMissEvalResult], out_dir: Path) -> Path:
    """Grouped bars: near-miss FCR per arm, per embedder (lower is better)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    embs = list(dict.fromkeys(r.embedder for r in results))
    colors = {ARM_THRESHOLD: "#c44e52", ARM_STACKED: "#55a868", ARM_ENTAIL_ONLY: "#4c72b0"}
    width = 0.8 / max(1, len(ARMS))
    fig, ax = plt.subplots(figsize=(max(5, len(embs) * 2.2), 4))
    for j, arm in enumerate(ARMS):
        xs, vals = [], []
        for i, e in enumerate(embs):
            row = next((r for r in results if r.embedder == e and r.arm == arm), None)
            x = i + (j - 1) * width
            if row is None or math.isnan(row.nearmiss_fcr):
                # no data must NOT render as a zero-height bar — on a lower-is-better chart
                # that reads as a PERFECT score (same rationale as fraction_true's NaN)
                ax.annotate("n/a", (x, 0.02), ha="center", fontsize=8, rotation=90)
                continue
            xs.append(x)
            vals.append(row.nearmiss_fcr)
        ax.bar(xs, vals, width=width, label=arm, color=colors.get(arm))
    ax.set_xticks(range(len(embs)))
    ax.set_xticklabels(embs, rotation=20, ha="right")
    ax.set_ylabel("near-miss false-confident rate")
    ax.set_ylim(0, 1)
    ax.set_title("Near-miss queries answered confidently (lower is better)")
    ax.legend()
    fig.tight_layout()
    path = out_dir / "nearmiss_effect.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def results_to_markdown(results: list[AblationResult]) -> str:
    lines = [
        "| embedder | fusion | P@5 | R@5 | MRR | nDCG@10 | FCR no-guard | FCR guard |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.embedder} | {r.fusion} | {r.p_at_5:.3f} | {r.r_at_5:.3f} | {r.mrr:.3f} | "
            f"{r.ndcg_at_10:.3f} | {r.fcr_no_guard:.2f}† | {_fmt_rate(r.fcr_with_guard)} |"
        )
    lines.append("")
    lines.append(
        "_† FCR no-guard is ANALYTIC, not measured: with no gap guard the system never abstains, "
        "so every unanswerable query is answered confidently and the rate is 1.00 by definition. "
        "It is the reference point for FCR guard, not an observation._"
    )
    lines.append("")
    lines.append(
        "_P@5 is mechanically capped at 0.20: each query has exactly one relevant doc, so the "
        "best possible precision@5 is 1/5. Read it as \"answer found in the top 5\" (binary), not "
        "as classical precision — R@5 / MRR / nDCG@10 are the informative ranking metrics._"
    )
    lines.append("")
    lines.append("Cost/latency (mean wall time per call):")
    lines.append("")
    lines.append("| embedder | fusion | embed ms/query | rerank ms/query |")
    lines.append("|---|---|---|---|")
    for r in results:
        lines.append(
            f"| {r.embedder} | {r.fusion} | {r.embed_ms_mean:.1f} | {r.rerank_ms_mean:.1f} |"
        )
    return "\n".join(lines)


def save_charts(results: list[AblationResult], out_dir: Path) -> list[Path]:
    """Write nDCG-by-config and guard-effect charts. Requires `pip install recall[eval]`."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    labels = [f"{r.embedder}\n{r.fusion}" for r in results]
    fig, ax = plt.subplots(figsize=(max(6, len(results) * 1.1), 4))
    ax.bar(labels, [r.ndcg_at_10 for r in results], color="#4c72b0")
    ax.set_ylabel("nDCG@10")
    ax.set_ylim(0, 1)
    ax.set_title("Retrieval quality by config")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    p1 = out_dir / "ndcg_by_config.png"
    fig.savefig(p1, dpi=120)
    plt.close(fig)
    paths.append(p1)

    # Chart ONE NAMED config per embedder, not min() across fusions. Taking the minimum plots
    # the best case each embedder achieved under any fusion — a number that appears in no row of
    # the results table and cannot be reproduced from it. `hybrid` is the library's default, so
    # it is the config a reader would actually get. Embedders missing it are skipped rather than
    # silently backfilled from another fusion.
    chart_fusion = "hybrid" if any(r.fusion == "hybrid" for r in results) else results[0].fusion
    by_emb = {r.embedder: r for r in results if r.fusion == chart_fusion}
    embs = [e for e in dict.fromkeys(r.embedder for r in results) if e in by_emb]
    guard = [by_emb[e].fcr_with_guard for e in embs]
    fig, ax = plt.subplots(figsize=(max(5, len(embs) * 1.6), 4))
    x = range(len(embs))
    ax.bar([i - 0.2 for i in x], [1.0] * len(embs), width=0.4,
           label="no guard (1.0 by construction)", color="#c44e52")
    ax.bar([i + 0.2 for i in x], guard, width=0.4,
           label=f"with gap guard ({chart_fusion})", color="#55a868")
    ax.set_xticks(list(x))
    ax.set_xticklabels(embs, rotation=20, ha="right")
    ax.set_ylabel("false-confident rate")
    ax.set_ylim(0, 1)
    ax.set_title("Guard effect on unanswerable queries (lower is better)")
    ax.legend()
    fig.tight_layout()
    p2 = out_dir / "guard_effect.png"
    fig.savefig(p2, dpi=120)
    plt.close(fig)
    paths.append(p2)
    return paths
