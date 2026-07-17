"""Ablation runner: index the eval corpus per embedder and score each retrieval config."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from recall.calibration import from_samples
from recall.embeddings import Embedder
from recall.eval.metrics import (
    false_confident_rate,
    mrr,
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
        table = "eval_" + uuid.uuid4().hex[:8]
        store = PgVectorStore(dsn, dim=emb.dim, table=table)
        store.ensure_schema()
        Indexer(store, emb).index_path(corpus_dir)
        try:
            for fusion in fusions:
                results.append(_score_config(store, emb, queries, fusion, reranker))
        finally:
            try:
                store._conn.execute(f"DROP TABLE IF EXISTS {table}")
            except Exception:
                pass  # best-effort drop of the throwaway uuid table
            finally:
                store.close()
    return results


@dataclass
class TrustEvalResult:
    embedder: str
    str_baseline: float          # superseded-trust rate of plain search (top-1 is a stale id)
    str_trust: float             # superseded-trust rate with the trust layer (stale id verdict ok)
    successor_acc: float         # expect=successor queries whose top trusted hit is the successor
    abstain_acc: float           # expect=abstain queries that actually abstained
    mrr_answerable_baseline: float
    mrr_answerable_trust: float  # must equal baseline: trust must not damage ordinary retrieval


def _tkey(hit: TrustedHit) -> str:
    return f"{hit.provenance.file}:{hit.provenance.ord}"


def run_trust_eval(
    dsn: str, embedders: list[Embedder], corpus_dir: Path | None = None,
    queries_path: Path | None = None,
) -> list[TrustEvalResult]:
    """Score the validity-sensitive queries in two modes per embedder.

    baseline = plain hybrid search (no trust layer): a query counts as stale-trusted when its
    top-1 hit is a superseded/expired memory — exactly what a consumer would read as the answer.
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
        table = "trust_" + uuid.uuid4().hex[:8]
        store = PgVectorStore(dsn, dim=emb.dim, table=table)
        store.ensure_schema()
        Indexer(store, emb).index_path(corpus_dir)
        try:
            # in-run calibration from the labeled plain queries (per-embedder threshold —
            # a fixed constant does not transfer across embedders, see FINDINGS §2)
            ans, unans = [], []
            for q in plain:
                hits = store.query_dense(emb.embed([q["query"]])[0], k=1)
                top = hits[0].score if hits else 0.0
                (ans if q["answerable"] else unans).append(top)
            cal = from_samples(emb.name, ans, unans)

            retr = HybridRetriever(store, emb)
            base_flags, trust_flags, succ_flags, abst_flags = [], [], [], []
            for q in trust_qs:
                stale = set(q["stale_ids"])
                succ = set(q.get("successor_ids", []))
                bres = retr.search(q["query"], k=10)
                base_flags.append(bool(bres.hits) and _key(bres.hits[0]) in stale)
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
                    str_trust=superseded_trust_rate(trust_flags),
                    successor_acc=successor_accuracy(succ_flags),
                    abstain_acc=successor_accuracy(abst_flags),
                    mrr_answerable_baseline=mean(base_mrrs) if base_mrrs else 0.0,
                    mrr_answerable_trust=mean(trust_mrrs) if trust_mrrs else 0.0,
                )
            )
        finally:
            try:
                store._conn.execute(f"DROP TABLE IF EXISTS {table}")
            except Exception:
                pass  # best-effort drop of the throwaway uuid table
            finally:
                store.close()
    return results


def trust_results_to_markdown(results: list[TrustEvalResult]) -> str:
    lines = [
        "| embedder | STR baseline | STR trust | successor acc | abstain acc | "
        "MRR ans (base) | MRR ans (trust) |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r.embedder} | {r.str_baseline:.2f} | {r.str_trust:.2f} | "
            f"{r.successor_acc:.2f} | {r.abstain_acc:.2f} | "
            f"{r.mrr_answerable_baseline:.3f} | {r.mrr_answerable_trust:.3f} |"
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
