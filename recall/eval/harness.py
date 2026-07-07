"""Ablation runner: index the eval corpus per embedder and score each retrieval config."""
from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from statistics import mean

from recall.embeddings import Embedder
from recall.eval.metrics import (
    false_confident_rate,
    mrr,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recall.index import Indexer
from recall.rerank import CrossEncoderReranker, Reranker
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore
from recall.types import ScoredChunk

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
        p_at_5=mean(ps), r_at_5=mean(rs), mrr=mean(ms), ndcg_at_10=mean(ns),
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
            store._conn.execute(f"DROP TABLE IF EXISTS {table}")
            store.close()
    return results


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
