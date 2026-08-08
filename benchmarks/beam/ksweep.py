"""Find the retrieval budget `k` for an embedder, for $0 — no answerer, no judge.

`k` was swept once, on `text-embedding-3-small`, and carried into every later plan. An optimum
fitted on one embedder is not evidence about another: voyage-4-large's cosines sit on an entirely
different scale (median 0.560 against bge-small's 0.825). This measures it per embedder.

What `k` controls is whether the facts the answer needs are IN the answerer's context. BEAM ships
those facts as rubric nuggets, so coverage of them is measurable without generating an answer or
paying a judge:

  1. retrieve ONCE per question at the maximum pool, reranked — top-k is then a prefix of
     top-max(k), the same argument `RESULTS.md` §7a makes for the depth curve, so one retrieval
     scores every depth without changing the configuration;
  2. embed each rubric nugget;
  3. at each depth, score every nugget by its best cosine against the retrieved memories.

**Read the GAIN column, not the level.** The curve rises with k by construction; the finding is
where it saturates. Past the knee, more memories add tokens and distraction without adding the
information the answer needs.

**This is a proxy and must not be read as a score.** It measures whether a fact is PRESENT, not
whether the answerer uses it — it cannot see distraction, and a judged sweep on
`text-embedding-3-small` kept improving (5 → 0.4421, 20 → 0.5010, 45 → 0.6104) well past where
coverage flattened. Where the two disagree, the judged number wins.

::

    python -m benchmarks.beam.ksweep --table bench_beam_voyage \\
        --embedder voyage:voyage-4-large --rows <a beam result artifact> \\
        --out results/beam_voyage/ksweep.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_DEPTHS = [5, 10, 20, 30, 45, 70, 100, 150, 250]
#: ef_search is derived as candidate_k * multiplier and pgvector caps it at 1000, so 250 is the
#: widest pool the default multiplier admits. See `recall.store._HNSW_EF_SEARCH_MAX`.
MAX_POOL = 250


def nuggets_of(row: dict[str, Any]) -> list[str]:
    """BEAM's rubric, as a flat list of required facts."""
    rubric = row.get("rubric")
    if isinstance(rubric, list):
        out = []
        for item in rubric:
            if isinstance(item, str):
                out.append(item)
            elif isinstance(item, dict):
                for field in ("nugget", "text", "fact"):
                    if isinstance(item.get(field), str):
                        out.append(item[field])
                        break
        return out
    return [rubric] if isinstance(rubric, str) else []


def cosine(a: list[float], b: list[float]) -> float:
    num = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return num / (na * nb) if na and nb else 0.0


def sweep(
    rows: list[dict[str, Any]],
    *,
    dsn: str,
    table: str,
    embedder_name: str,
    depths: list[int],
    reranker_name: str = "local",
) -> dict[str, Any]:
    from benchmarks._trust import bench_search
    from benchmarks.systems import resolve_embedder, resolve_reranker
    from recall.store import PgVectorStore

    embedder = resolve_embedder(embedder_name)
    reranker = resolve_reranker(reranker_name)

    mean_sim: dict[int, list[float]] = {k: [] for k in depths}
    weakest: dict[int, list[float]] = {k: [] for k in depths}
    started = time.time()

    for i, row in enumerate(rows, 1):
        nugs = nuggets_of(row)
        if not nugs or not row.get("question"):
            continue
        tenant = f"beam-1m-{row['conversation_idx']}"
        with PgVectorStore(dsn, dim=embedder.dim, tenant=tenant, table=table) as store:
            # This sweep reads only `h.chunk.text`, never a verdict or `abstained`, so the
            # calibration `bench_search` supplies changes nothing it measures: the threshold
            # reaches the retriever as `gap_threshold`, which sets a warning flag and culls no
            # hit, and it is the same 0.50 the uncalibrated fallback would have used anyway. It is
            # routed through the shared seam regardless, so that "arms that happen not to read
            # verdicts" is not a category anyone has to keep re-deciding a module at a time.
            result = bench_search(
                store, embedder, row["question"], k=MAX_POOL,
                candidate_k=MAX_POOL, reranker=reranker,
            )
        memories = [h.chunk.text for h in result.hits]
        if not memories:
            continue

        vectors = embedder.embed(nugs + memories)
        nug_vecs, mem_vecs = vectors[: len(nugs)], vectors[len(nugs):]
        for k in depths:
            window = mem_vecs[:k]
            if not window:
                continue
            best = [max(cosine(nv, mv) for mv in window) for nv in nug_vecs]
            mean_sim[k].append(statistics.mean(best))
            weakest[k].append(min(best))
        if i % 10 == 0:
            print(f"  {i}/{len(rows)} ({time.time() - started:.0f}s)", flush=True)

    return {
        "_provenance": {
            "generation": "post-#81/#84",
            "status": "current",
            "superseded_by": None,
            "backs": ["the `k` choice for this embedder — a proxy, NOT a judged score"],
            "note": (
                "Nugget-coverage proxy: measures whether the required facts are PRESENT in the "
                "answerer's context, not whether the answerer uses them. Where it disagrees with "
                "a judged sweep, the judged number wins."
            ),
        },
        "benchmark": "BEAM — nugget coverage vs retrieval budget",
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "embedder": embedder_name,
        "reranker": reranker_name,
        "table": table,
        "candidate_k": MAX_POOL,
        "depths": depths,
        "n": len(mean_sim[depths[0]]),
        "elapsed_s": round(time.time() - started, 1),
        "mean_nugget_sim": {str(k): mean_sim[k] for k in depths},
        "weakest_nugget_sim": {str(k): weakest[k] for k in depths},
    }


def report(result: dict[str, Any]) -> str:
    lines = [f"{'k':>5}{'mean nugget sim':>18}{'gain':>10}{'weakest':>10}"]
    prev = None
    for k in result["depths"]:
        vals = result["mean_nugget_sim"][str(k)]
        if not vals:
            continue
        m = statistics.mean(vals)
        w = statistics.mean(result["weakest_nugget_sim"][str(k)])
        gain = "" if prev is None else f"{m - prev:+.4f}"
        lines.append(f"{k:>5}{m:>18.4f}{gain:>10}{w:>10.4f}")
        prev = m
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows", type=Path, required=True,
                   help="a BEAM result artifact; its rows supply the questions and rubrics")
    p.add_argument("--table", required=True)
    p.add_argument("--embedder", required=True)
    p.add_argument("--dsn", default=None)
    p.add_argument("--reranker", default="local")
    p.add_argument("--conversations", default=None, help="e.g. 0,1,2 — default: all in --rows")
    p.add_argument("--depths", default=",".join(str(d) for d in DEFAULT_DEPTHS))
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()

    import os

    dsn = args.dsn or os.environ.get("RECALL_DSN")
    if not dsn:
        raise SystemExit("--dsn or RECALL_DSN required")

    rows = json.loads(args.rows.read_text(encoding="utf-8"))["rows"]
    if args.conversations:
        keep = {int(x) for x in args.conversations.split(",")}
        rows = [r for r in rows if r.get("conversation_idx") in keep]

    result = sweep(
        rows, dsn=dsn, table=args.table, embedder_name=args.embedder,
        depths=[int(d) for d in args.depths.split(",")], reranker_name=args.reranker,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(report(result))
    print(f"\n-> {args.out}")


if __name__ == "__main__":
    main()
