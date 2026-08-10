"""Arm R of PREREGISTRATION-mtrag-abstention.md -- which regime is MTRAG?

ONE signal: dense top-1 cosine, scored with the SHIPPED `recall.calibration.separability`
(Mann-Whitney AUC, threshold-free, so it cannot be inflated by fitting and scoring on the same
samples). Answerable versus unanswerable, over MTRAG's 842 human tasks.

⛔ The other five signals from [[project-recall-abstention-bounded-domain-2026-07-24]] are NOT
re-run. Cross-encoder rerank, RRF fusion, QNLI entailment, margin and ratio were measured on
500 questions and all failed (AUC 0.742 / 0.739 / 0.648 / 0.579 / 0.545), with two named dead
ends. Re-running them would repeat the 2026-07-28 mistake the EXPERIMENT-CONVENTION exists to
prevent. This file runs the shipped signal ONLY, to classify the corpus, because the same memo
establishes the near-miss regime is a property of how the questions were BUILT -- so which
regime a new corpus occupies is an open question about that corpus, not a re-measurement.

PREREGISTERED, before running (sec.2 P1):
    dense-cosine separability AUC <= 0.80   -> near-miss regime, Arm L is warranted
    AUC >= 0.90                             -> P1 FALSIFIED, and that is the GOOD outcome:
                                               the shipped signal already works here.
0.90 is also `recall.calibration.MIN_SEPARABILITY`, the bar the library certifies against.

Prior work: governed by [[project-recall-abstention-bounded-domain-2026-07-24]]; full search
recorded in benchmarks/archive/preregistrations/PREREGISTRATION-mtrag-abstention.md.

PROVENANCE, and why it is not master. The pgvector index was written by RE-call
3d3c905044bbc8b714a6173b1879a5b4e70e16b6, and query-side and document-side embeddings must come
from the same stack -- master's recall/embeddings.py differs by 370 lines (the #200 embedding
profile registry, the change that invalidated every embedding cache), so querying this index with
it would risk cosines that are wrong while still looking like numbers. Retrieval therefore runs on
the index's own stack. The ANALYSIS is master's regardless: `separability`,
`separability_interval`, `from_samples` and `best_threshold`, plus MIN_SEPARABILITY,
SEPARABILITY_Z, DEFAULT_SCALE and MIN_CALIBRATION_SAMPLES, were verified byte-identical between
that revision and origin/master before this ran.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from recall.calibration import (
    MIN_SEPARABILITY,
    from_samples,
    separability,
    separability_interval,
)
from recall.embeddings import FastEmbedEmbedder
from recall.retriever import HybridRetriever
from recall.store import PgVectorStore

EMBEDDER_MODEL = "BAAI/bge-small-en-v1.5"
DOMAINS = ("clapnq", "cloud", "fiqa", "govt")

#: MTRAG's `Collection` is a full index name (`mt-rag-clapnq-elser-512-100-20240503`), unlike
#: MTRAG-UN's bare domain, so the adapter's `task_domain` cannot be reused here.
COLLECTION_MARKERS = (("clapnq", "clapnq"), ("govt", "govt"), ("fiqa", "fiqa"), ("ibmcloud", "cloud"))

#: Preregistered thresholds. Not tunable at the command line, on purpose.
P1_NEAR_MISS_AT_OR_BELOW = 0.80
P1_FALSIFIED_AT_OR_ABOVE = 0.90


def domain_of(collection: str) -> str:
    low = collection.lower()
    for marker, domain in COLLECTION_MARKERS:
        if marker in low:
            return domain
    raise RuntimeError(f"cannot map Collection {collection!r} to a domain")


def last_user_turn(task: dict[str, Any]) -> str:
    """The shipped `last` query mode, mirroring benchmarks/mtrag/run.py:query_text."""
    turns = [str(t["text"]).strip() for t in task["input"] if t["speaker"] == "user"]
    if not turns:
        raise RuntimeError(f"task {task.get('task_id')} has no user turn")
    return turns[-1]


def answerability(task: dict[str, Any]) -> str:
    a = task.get("Answerability") or []
    return str(a[0]) if isinstance(a, list) and a else "?"


def load_dsn(env_file: Path | None) -> str:
    if os.environ.get("RECALL_DSN"):
        return os.environ["RECALL_DSN"]
    if env_file is not None:
        from dotenv import dotenv_values

        values = dotenv_values(env_file)
        dsn = values.get("RECALL_DSN") or values.get("DATABASE_URL")
        if dsn:
            return str(dsn)
    raise RuntimeError("set RECALL_DSN or pass --dsn-env-file")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag-json", type=Path, required=True, help="MTRAG mtrag-human/evaluations/RAG.json")
    ap.add_argument("--dsn-env-file", type=Path, default=None)
    ap.add_argument("--table-prefix", default="recall_mtrag_bge_v1")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0, help="smoke only; 0 = all tasks")
    args = ap.parse_args()

    tasks: list[dict[str, Any]] = json.loads(args.rag_json.read_text(encoding="utf-8"))["tasks"]
    if args.limit:
        tasks = tasks[: args.limit]
    print(f"tasks: {len(tasks)}   classes: {dict(Counter(answerability(t) for t in tasks))}", flush=True)

    dsn = load_dsn(args.dsn_env_file)
    embedder = FastEmbedEmbedder(EMBEDDER_MODEL)
    retrievers = {
        d: HybridRetriever(
            PgVectorStore(dsn, embedder.dim, table=f"{args.table_prefix}_{d}"),
            embedder,
            reranker=None,
            candidate_k=20,
            use_dense=True,
            use_sparse=False,  # dense cosine ONLY -- hit.score is then the cosine
        )
        for d in DOMAINS
    }

    rows: list[dict[str, Any]] = []
    started = time.perf_counter()
    for i, task in enumerate(tasks, 1):
        domain = domain_of(str(task["Collection"]))
        result = retrievers[domain].search(last_user_turn(task), k=10)
        rows.append(
            {
                "task_id": task["task_id"],
                "domain": domain,
                "cls": answerability(task),
                "top1_cosine": float(result.hits[0].score) if result.hits else None,
                "gap_warning": bool(result.gap_warning),
            }
        )
        if i % 100 == 0:
            print(f"  {i}/{len(tasks)}  {time.perf_counter()-started:.0f}s", flush=True)

    scored = [r for r in rows if r["top1_cosine"] is not None]
    empty = len(rows) - len(scored)
    ans = [float(r["top1_cosine"]) for r in scored if r["cls"] == "ANSWERABLE"]
    unans = [float(r["top1_cosine"]) for r in scored if r["cls"] == "UNANSWERABLE"]
    if not ans or not unans:
        print("VACUOUS: one class is empty, nothing was judged")
        return 2

    auc = separability(ans, unans)
    assert auc is not None
    lo, hi = separability_interval(auc, len(ans), len(unans))
    cal = from_samples(EMBEDDER_MODEL, ans, unans)

    verdict = (
        "P1 FALSIFIED -- the shipped signal already separates here; Arm L is NOT warranted"
        if auc >= P1_FALSIFIED_AT_OR_ABOVE
        else "P1 CONFIRMED -- near-miss regime, Arm L is warranted"
        if auc <= P1_NEAR_MISS_AT_OR_BELOW
        else "INCONCLUSIVE -- between the preregistered bounds"
    )

    print("\n=== ARM R: dense top-1 cosine separability, MTRAG ===")
    print(f"  scored {len(scored)} tasks ({empty} with no hits)")
    print(f"  answerable n={len(ans)}   mean {sum(ans)/len(ans):.4f}")
    print(f"  unanswerable n={len(unans)}   mean {sum(unans)/len(unans):.4f}")
    print(f"  separability AUC = {auc:.4f}   95% CI [{lo:.4f}, {hi:.4f}]")
    print(f"  MIN_SEPARABILITY = {MIN_SEPARABILITY} (library bar, applied to the LOWER bound)")
    print(f"  calibration certified = {cal.certified}   reason: {cal.certification_reason}")
    print("\n  PREREGISTERED P1: <=0.80 near-miss, >=0.90 falsified")
    print(f"  VERDICT: {verdict}")

    per_domain: dict[str, Any] = {}
    for d in DOMAINS:
        a = [float(r["top1_cosine"]) for r in scored if r["domain"] == d and r["cls"] == "ANSWERABLE"]
        u = [float(r["top1_cosine"]) for r in scored if r["domain"] == d and r["cls"] == "UNANSWERABLE"]
        per_domain[d] = {
            "n_answerable": len(a), "n_unanswerable": len(u),
            "separability": separability(a, u),
        }
    print("\n  per domain (diagnostic; unanswerable counts are small):")
    for d, v in per_domain.items():
        s = v["separability"]
        print(f"    {d:8} n={v['n_answerable']:>4}/{v['n_unanswerable']:<3} "
              f"AUC={'n/a' if s is None else f'{s:.4f}'}")

    diagnostics = {}
    for other in ("PARTIAL", "CONVERSATIONAL"):
        vals = [float(r["top1_cosine"]) for r in scored if r["cls"] == other]
        diagnostics[other] = {
            "n": len(vals),
            "mean": (sum(vals) / len(vals)) if vals else None,
            "separability_vs_unanswerable": separability(vals, unans) if vals else None,
        }
    print("\n  classes excluded from the primary contrast (diagnostic):")
    for k, v in diagnostics.items():
        m, s = v["mean"], v["separability_vs_unanswerable"]
        print(f"    {k:15} n={v['n']:>3}  mean={'n/a' if m is None else f'{m:.4f}'}  "
              f"AUC vs unans={'n/a' if s is None else f'{s:.4f}'}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(
            {
                "arm": "R",
                "signal": "dense top-1 cosine",
                "embedder": EMBEDDER_MODEL,
                "n_scored": len(scored),
                "n_no_hits": empty,
                "separability": auc,
                "separability_ci": [lo, hi],
                "min_separability": MIN_SEPARABILITY,
                "certified": cal.certified,
                "certification_reason": cal.certification_reason,
                "threshold": cal.threshold,
                "n_answerable": len(ans),
                "n_unanswerable": len(unans),
                "mean_answerable": sum(ans) / len(ans),
                "mean_unanswerable": sum(unans) / len(unans),
                "preregistered": {
                    "near_miss_at_or_below": P1_NEAR_MISS_AT_OR_BELOW,
                    "falsified_at_or_above": P1_FALSIFIED_AT_OR_ABOVE,
                },
                "verdict": verdict,
                "per_domain": per_domain,
                "diagnostics": diagnostics,
                "rows": rows,
            },
            indent=1,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
