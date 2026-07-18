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
    gap_false_confident_rate,
    mrr,
    near_miss_false_confident_rate,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    successor_accuracy,
    superseded_trust_rate,
)
from recall.index import Indexer
from recall.rerank import CrossEncoderReranker, Reranker
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore
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
    fcr_no_guard: float  # without the gap guard you always answer -> 1.0 on unanswerable
    fcr_with_guard: float


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
            store._conn.execute(f"DROP TABLE IF EXISTS {table}")
        except Exception:
            pass  # best-effort drop of the throwaway uuid table
        finally:
            store.close()


def _score_config(
    store: PgVectorStore, embedder: Embedder, queries: list[dict], fusion: str,
    reranker: Reranker | None,
) -> AblationResult:
    retr = HybridRetriever(
        store, embedder,
        reranker=reranker if fusion == "hybrid+rerank" else None,
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


class _TimedJudge:
    """Wraps a judge to measure its wall time — the honest cost column of the results table."""

    def __init__(self, inner: EntailmentJudge) -> None:
        self._inner = inner
        self.samples_ms: list[float] = []

    def judge(self, query: str, texts: list[str]) -> list[bool]:
        t0 = time.perf_counter()
        out = self._inner.judge(query, texts)
        self.samples_ms.append((time.perf_counter() - t0) * 1000.0)
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
            cal = from_samples(emb.name, *measure_top_cosines(store, emb, plain))
            # threshold -1 passes every cosine: isolates the judge in the entail-only arm
            permissive = Calibration(embedder=emb.name, threshold=-1.0, scale=cal.scale)
            arm_setup = {
                ARM_THRESHOLD: (cal, None),
                ARM_STACKED: (cal, judge),
                ARM_ENTAIL_ONLY: (permissive, judge),
            }
            # Each arm re-runs retrieval end-to-end ON PURPOSE: query_latency_ms_mean must
            # measure real per-arm wall time, and the judge is deliberately un-memoized so
            # its latency column counts model passes, not cache hits. Sharing arm A's
            # results with arm B would save ~1/3 of the retrieval cost at the price of
            # fabricating the latency columns.
            for arm in ARMS:
                arm_cal, arm_judge = arm_setup[arm]
                timed = _TimedJudge(arm_judge) if arm_judge is not None else None
                q_times: list[float] = []

                def _search(text: str):
                    t0 = time.perf_counter()
                    res = trusted_search(store, emb, text, k=k, calibration=arm_cal,
                                         entailment=timed)
                    q_times.append((time.perf_counter() - t0) * 1000.0)
                    return res

                nm_confident = [not _search(q["query"]).abstained for q in nearmiss]
                gap_confident = [not _search(q["query"]).abstained for q in gaps]
                abst_flags, mrrs = [], []
                for q in answerable:
                    res = _search(q["query"])
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
                )
            )
    return results


def _fmt_rate(x: float, digits: int = 2) -> str:
    return "n/a" if math.isnan(x) else f"{x:.{digits}f}"


def trust_results_to_markdown(results: list[TrustEvalResult]) -> str:
    lines = [
        "| embedder | STR baseline | STR recency | STR trust | successor acc | abstain acc | "
        "MRR ans (base) | MRR ans (trust) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.embedder} | {_fmt_rate(r.str_baseline)} | {_fmt_rate(r.str_recency)} | "
            f"{_fmt_rate(r.str_trust)} | "
            f"{_fmt_rate(r.successor_acc)} | {_fmt_rate(r.abstain_acc)} | "
            f"{_fmt_rate(r.mrr_answerable_baseline, 3)} | {_fmt_rate(r.mrr_answerable_trust, 3)} |"
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
            f"{r.ndcg_at_10:.3f} | {r.fcr_no_guard:.2f} | {r.fcr_with_guard:.2f} |"
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

    embs = list(dict.fromkeys(r.embedder for r in results))
    guard = [min(r.fcr_with_guard for r in results if r.embedder == e) for e in embs]
    fig, ax = plt.subplots(figsize=(max(5, len(embs) * 1.6), 4))
    x = range(len(embs))
    ax.bar([i - 0.2 for i in x], [1.0] * len(embs), width=0.4, label="no guard", color="#c44e52")
    ax.bar([i + 0.2 for i in x], guard, width=0.4, label="with gap guard", color="#55a868")
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
