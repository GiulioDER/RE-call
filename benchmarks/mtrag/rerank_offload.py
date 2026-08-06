"""Offload cross-encoder SCORING to a GPU while retrieval and credentials stay on the DB host.

Prior work (searched 2026-08-06, `docs_search(source_type="memory")`, no gap warning):

  - [[project-recall-mtrag-retrieval-coverage-bottleneck-2026-08-06]] -- the reranker is worth
    +0.0864 nDCG@5 on MTRAG dev, CI [+0.0671, +0.1061], measured at candidate_k=100 (fused pool
    ~200). That is the figure these arms must stay comparable with.
  - [[closed-hypothesis-recall-rerank-pool-interaction-2026-08-05]] -- ⚠️ on the PEPs corpus the
    SAME MiniLM gets WORSE as the pool widens: churn at pool 20, broke 12 / fixed 6 at pool 250,
    entire 95% CI below the preregistered threshold. "A wider pool did not give the cross-encoder
    more to select from, it gave it more rope."

🔑 The consequence for the arms here, which is not obvious and is easy to publish through:
`pool_bound()` DIFFERS BY ARM (dense_only 100, hybrid_* 200, hybrid_both 300). Reranking each
arm's whole pool therefore hands the cross-encoder a different width per arm, and the closed
hypothesis above says width alone moves the result. So:

  - `hybrid_lexical` vs `hybrid_splade` -- BOTH pool 200. The primary contrast is CLEAN and is
    also the width the +0.0864 baseline was measured at.
  - `hybrid_both` (300) and the single-leg arms (100) -- reranked numbers are CONFOUNDED with
    pool width. Read them as retrieval-only, or re-run them at a controlled width, but do not
    compare their reranked scores against the primary pair.

No prior work found for the offload mechanism itself (searched: cross-encoder reranker offload GPU
scoring pairs MTRAG candidate pool).

    dump      -- run retrieval for each arm, emit candidate pools + deduped pairs to score
    apply     -- fold GPU scores back in, reorder, and compute metrics
    validate  -- run the REAL CrossEncoderReranker locally and require identical ordering

`CrossEncoderReranker.rerank` is `model.predict(pairs)` followed by a stable descending sort, so
only the predict step is expensive and only it moves. The ordering logic stays here, which is what
makes the offload exact rather than approximate.

⚠️ `validate` is not optional. An offloaded ordering that merely looks reasonable would produce a
publishable-looking nDCG that RE-call itself would never compute. Numbers from an unvalidated
substitute are not results.

Two details that decide correctness:

1. The dump retrieves at ``k = arm.pool_bound()``, not 100. RE-call reranks the WHOLE fused pool
   and truncates afterwards, so dumping only the top 100 would rerank a pre-truncated pool and
   silently measure a different system.
2. Reordering uses ``sorted(range(n), key=lambda i: scores[i], reverse=True)`` exactly as
   `rerank()` does. Python's sort is stable, so ties resolve by fused rank; any other tie rule
   would diverge on near-ties.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

from benchmarks.mtrag.run import (
    DOMAINS,
    EMBEDDER_MODEL,
    SPARSE_ARMS,
    Arm,
    load_dsn,
    load_qrels,
    load_tasks,
    query_text,
    score_predictions,
    table_name,
    task_domain,
    utc_now,
)

#: Metric cutoff. Reranking reorders the pool; the reported metrics still cut at this depth.
EVAL_K = 100


def rerank_order(candidates: list[str], scores: dict[str, float]) -> list[str]:
    """Reorder `candidates` by `scores`, identically to `CrossEncoderReranker.rerank`.

    Stable descending sort over positions, so equal scores keep their fused order. A candidate
    with no score is a bug rather than a zero: scoring it as 0.0 would silently move it to the
    middle of the ranking, so it raises.
    """
    missing = [c for c in candidates if c not in scores]
    if missing:
        raise KeyError(f"{len(missing)} candidates have no score, e.g. {missing[:3]}")
    order = sorted(range(len(candidates)), key=lambda i: scores[candidates[i]], reverse=True)
    return [candidates[i] for i in order]


def _retrievers(dsn: str, arm: Arm, prefix: str, embedder: Any, sparse_model: str | None) -> dict:
    from recall.retriever import HybridRetriever
    from recall.store import PgVectorStore

    sparse_encoder = None
    if arm.sparse_backend in ("splade", "both"):
        from recall.sparse import SpladeEncoder

        sparse_encoder = SpladeEncoder.from_pretrained(
            sparse_model or "prithivida/Splade_PP_en_v1"
        )
    return {
        domain: HybridRetriever(
            PgVectorStore(dsn, embedder.dim, table=table_name(prefix, domain)),
            embedder,
            reranker=None,  # scoring is offloaded; this stage is retrieval only
            candidate_k=arm.candidate_k,
            use_dense=arm.use_dense,
            use_sparse=arm.use_sparse,
            sparse_backend=arm.sparse_backend,
            sparse_encoder=sparse_encoder,
        )
        for domain in DOMAINS
    }


def cmd_dump(args: argparse.Namespace) -> int:
    from recall.embeddings import FastEmbedEmbedder

    dsn = load_dsn(args.dsn_env_file)
    root = args.mtrag_root.resolve()
    out = args.output_dir.resolve()
    (out / "pools").mkdir(parents=True, exist_ok=True)

    embedder = FastEmbedEmbedder(EMBEDDER_MODEL)
    arms = [a for a in SPARSE_ARMS if not args.arms or a.name in set(args.arms)]

    queries: dict[str, str] = {}
    docs: dict[str, str] = {}
    pairs: set[tuple[str, str]] = set()

    for arm in arms:
        started = time.perf_counter()
        retrievers = _retrievers(dsn, arm, args.table_prefix, embedder, args.sparse_model)
        tasks = load_tasks(root, args.split, arm.query_mode)
        rows = []
        for position, task in enumerate(tasks, 1):
            domain = task_domain(task)
            query = query_text(task, arm.query_mode)
            # pool_bound, NOT the metric cutoff: rerank sees the whole fused pool.
            result = retrievers[domain].search(query, k=arm.pool_bound())
            candidates = [h.chunk.id for h in result.hits]
            rows.append({
                "task_id": task["task_id"], "domain": domain,
                "query": query, "candidates": candidates,
            })
            queries[task["task_id"]] = query
            for hit in result.hits:
                docs[hit.chunk.id] = hit.chunk.text
                pairs.add((task["task_id"], hit.chunk.id))
            if position % 100 == 0:
                print(json.dumps({
                    "event": "progress", "arm": arm.name, "queries": position,
                    "elapsed_s": round(time.perf_counter() - started, 1),
                }), flush=True)
        path = out / "pools" / f"{arm.name}.jsonl"
        with path.open("w", encoding="utf-8") as fh:
            for row in rows:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        pool_sizes = [len(r["candidates"]) for r in rows]
        print(json.dumps({
            "event": "arm_done", "arm": arm.name, "queries": len(rows),
            "pool_bound": arm.pool_bound(),
            "pool_min": min(pool_sizes), "pool_median": sorted(pool_sizes)[len(pool_sizes) // 2],
            "pool_max": max(pool_sizes),
            "elapsed_s": round(time.perf_counter() - started, 1),
        }), flush=True)

    for name, payload in (
        ("queries.jsonl", [{"qid": q, "text": t} for q, t in queries.items()]),
        ("docs.jsonl", [{"doc_id": d, "text": t} for d, t in docs.items()]),
        ("pairs.jsonl", [{"qid": q, "doc_id": d} for q, d in sorted(pairs)]),
    ):
        with (out / name).open("w", encoding="utf-8") as fh:
            for row in payload:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "event": "dump_done", "arms": [a.name for a in arms], "queries": len(queries),
        "unique_docs": len(docs), "pairs_to_score": len(pairs), "at": utc_now(),
    }), flush=True)
    return 0


def _load_scores(path: Path) -> dict[str, dict[str, float]]:
    scores: dict[str, dict[str, float]] = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("_header"):
                print(json.dumps({"event": "scores_header", **row}), flush=True)
                continue
            scores.setdefault(str(row["qid"]), {})[str(row["doc_id"])] = float(row["score"])
    return scores


def cmd_apply(args: argparse.Namespace) -> int:
    root = args.mtrag_root.resolve()
    out = args.output_dir.resolve()
    scores = _load_scores(args.scores)
    qrels = load_qrels(root, args.split)

    summaries = []
    for path in sorted((out / "pools").glob("*.jsonl")):
        arm_name = path.stem
        arm = next((a for a in SPARSE_ARMS if a.name == arm_name), None)
        predictions = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                ordered = rerank_order(row["candidates"], scores.get(row["task_id"], {}))
                predictions.append({
                    "task_id": row["task_id"],
                    "contexts": [{"document_id": d} for d in ordered[:EVAL_K]],
                })
        metrics = score_predictions(predictions, qrels, arm.pool_bound() if arm else None)
        summaries.append({"arm": arm_name, "reranked": True, "metrics": metrics["overall"],
                          "domains": metrics["domains"]})
        print(json.dumps({"event": "scored", "arm": arm_name, **metrics["overall"]}), flush=True)

    (out / "reranked_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    return 0


def cmd_validate(args: argparse.Namespace) -> int:
    """Require the offloaded ordering to match the real reranker EXACTLY on a sample.

    This is the gate. If it fails, the offloaded numbers are not RE-call's numbers and must not
    be reported as such.
    """
    from recall.rerank import CrossEncoderReranker
    from recall.types import Chunk, ScoredChunk

    out = args.output_dir.resolve()
    scores = _load_scores(args.scores)
    docs = {}
    with (out / "docs.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                row = json.loads(line)
                docs[str(row["doc_id"])] = str(row["text"])

    pool_path = next(iter(sorted((out / "pools").glob("*.jsonl"))))
    rows = []
    with pool_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    sample = rows[: args.sample]

    reranker = CrossEncoderReranker()
    mismatches = []
    for row in sample:
        candidates = row["candidates"]
        hits = [
            ScoredChunk(chunk=Chunk(id=c, source="s", text=docs[c], metadata={}), score=0.0)
            for c in candidates
        ]
        local = [h.chunk.id for h in reranker.rerank(row["query"], hits)]
        offloaded = rerank_order(candidates, scores.get(row["task_id"], {}))
        if local != offloaded:
            first = next(i for i, (a, b) in enumerate(zip(local, offloaded, strict=True)) if a != b)
            mismatches.append({"task_id": row["task_id"], "first_divergence_rank": first})

    verdict = "MATCH" if not mismatches else "MISMATCH"
    print(json.dumps({
        "event": "validate", "arm": pool_path.stem, "sampled": len(sample),
        "mismatches": len(mismatches), "detail": mismatches[:5], "verdict": verdict,
    }, indent=2), flush=True)
    return 0 if not mismatches else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)
    for name in ("dump", "apply", "validate"):
        p = sub.add_parser(name)
        p.add_argument("--mtrag-root", type=Path, required=(name != "validate"),
                       default=Path("."))
        p.add_argument("--output-dir", type=Path, required=True)
        p.add_argument("--split", choices=("dev", "test"), default="dev")
        if name == "dump":
            p.add_argument("--dsn-env-file", type=Path)
            p.add_argument("--table-prefix", default="recall_mtrag_bge_v1")
            p.add_argument("--sparse-model", default=None)
            p.add_argument("--arms", nargs="*")
        else:
            p.add_argument("--scores", type=Path, required=True)
        if name == "validate":
            p.add_argument("--sample", type=int, default=20)
    args = parser.parse_args(argv)
    return {"dump": cmd_dump, "apply": cmd_apply, "validate": cmd_validate}[args.cmd](args)


if __name__ == "__main__":
    sys.exit(main())
