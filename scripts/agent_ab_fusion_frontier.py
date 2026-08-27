"""Can any weighting of the two retrieval legs separate a hazard from a coincidence?

    python -u scripts/agent_ab_fusion_frontier.py \
        --dsn postgresql://recall:recall@127.0.0.1:<port>/probe2_control \
        --archive ~/.claude/archive/agent-ab-skill-001 \
        --screen ~/.claude/archive/direction-screen-2026-08-27/direction-screen.json \
        --precision benchmarks/artifacts/agent_ab/draft-precision.json

Preregistered in `docs/preregistrations/2026-08-27-fusion-weight-frontier.md`.

**Collect once, sweep offline.** One retrieval pass captures each leg's full top-200 ranked list
per query; every fusion variant is then computed from those lists with no further database work.
That is the discipline that closed the calibration question in seconds
(`[[sweep-the-threshold-before-refitting-a-calibration]]`): buy the data once, explore the
parameter space for free.

Production fuses with UNWEIGHTED Reciprocal Rank Fusion, which rewards agreement between legs. On
a draft query the legs are not equally informative (lexical 14/14, dense 7/14), so consensus is
the wrong signal: a document both legs mildly like outranks the one document a leg is certain
about. Every variant here is a proposal — `recall/retriever.py` has no weight parameter at all.

Two endpoints per variant, measured together because either alone is misleading:

1. recall@5 — of the 14 registered miss sessions, how many surface the governing memo in the
   variant's top 5 for some draft query.
2. the (recall, false_trigger) frontier over that variant's fused SCORE, so the question
   "does a separating score exist" is answered rather than assumed. The current fusion has none.

⛔ A retrieval error is never scored as a miss, and a positive control must pass first.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
import sys  # noqa: E402

sys.path.insert(0, str(REPO_ROOT))

TOP_K = 5
CANDIDATE_K = 200
RRF_K = 60  # `recall/retriever.py`'s hardcoded damping constant.
LEXICAL_WEIGHTS = (1, 2, 3, 5, 10, 20)
SCORE_LAMBDAS = (0.5, 0.7, 0.8, 0.9, 1.0)
EMBEDDER_ID = "fastembed"
DIM = 384
MAX_QUERY_CHARS = 4096
CONTROL_MEMO = "python-write-text-crlf-churn"
CONTROL_QUERY = "Path.write_text on Windows injects CRLF against a tree configured eol=lf"


def ranked(retriever, query: str) -> list[tuple[str, str, float]]:
    """(chunk id, source name, score) for this leg, best first.

    ⛔ The chunk id is load-bearing and keying on the NAME instead was this script's worst bug.
    A document contributes SEVERAL chunks to a leg's list, so `fused[name] += 1/(k+rank)` sums a
    document's chunks and rewards documents that are diffusely present over the one document a leg
    ranks first: `lexical_only` then reordered the lexical leg it is supposed to reproduce, and
    dropped a memo from rank 1 to rank 5. Production fuses on `chunk.id`; so does this.
    """

    result = retriever.search(query, k=CANDIDATE_K)
    return [
        (str(h.chunk.id), Path(str(h.chunk.source)).name, float(h.score)) for h in result.hits
    ]


def rrf(
    legs: dict[str, list[tuple[str, str, float]]], weights: dict[str, float]
) -> list[tuple[str, float]]:
    """Reciprocal rank fusion over CHUNKS, returning (source name, score) best first."""

    fused: dict[str, float] = {}
    names: dict[str, str] = {}
    for leg, items in legs.items():
        w = weights.get(leg, 0.0)
        if not w:
            continue
        for rank, (chunk_id, name, _) in enumerate(items):
            names[chunk_id] = name
            fused[chunk_id] = fused.get(chunk_id, 0.0) + w / (RRF_K + rank + 1)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [(names[chunk_id], score) for chunk_id, score in ordered]


def score_fuse(
    legs: dict[str, list[tuple[str, str, float]]], lam: float
) -> list[tuple[str, float]]:
    """Min-max normalise each leg, then blend. RRF discards magnitude; a threshold needs it."""

    norm: dict[str, dict[str, float]] = {}
    names: dict[str, str] = {}
    for leg, items in legs.items():
        if not items:
            norm[leg] = {}
            continue
        vals = [s for _, _, s in items]
        lo, hi = min(vals), max(vals)
        span = (hi - lo) or 1.0
        norm[leg] = {}
        for chunk_id, name, score in items:
            names[chunk_id] = name
            norm[leg][chunk_id] = (score - lo) / span
    fused: dict[str, float] = {}
    for chunk_id in set(norm.get("lexical", {})) | set(norm.get("dense", {})):
        fused[chunk_id] = lam * norm.get("lexical", {}).get(chunk_id, 0.0) + (
            1 - lam
        ) * norm.get("dense", {}).get(chunk_id, 0.0)
    ordered = sorted(fused.items(), key=lambda kv: kv[1], reverse=True)
    return [(names[chunk_id], score) for chunk_id, score in ordered]


def variants(legs: dict[str, list[tuple[str, float]]]) -> dict[str, list[tuple[str, float]]]:
    out = {"lexical_only": rrf(legs, {"lexical": 1.0})}
    for w in LEXICAL_WEIGHTS:
        name = "unweighted_rrf" if w == 1 else f"weighted_rrf_w{w}"
        out[name] = rrf(legs, {"lexical": float(w), "dense": 1.0})
    for lam in SCORE_LAMBDAS:
        out[f"score_fusion_l{lam}"] = score_fuse(legs, lam)
    return out


def sessions(archive: Path, keep: set[str]) -> list[dict]:
    """The registered miss sessions, ON-ARM only.

    ⛔ The `variant` filter is load-bearing and its absence is not subtle in hindsight: the archive
    holds a `recall_on` AND a `recall_off` record for every task_id, so matching on task_id alone
    recovered 28 records for 14 sessions and printed `best recall@5: 21/14`. An impossible
    denominator is the lucky case; the same bug with a smaller overlap would have produced a
    plausible number. The population gate below now refuses rather than relying on the arithmetic
    looking wrong.
    """

    rows = []
    for line in (archive / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get("task_id") not in keep or record.get("variant") != "recall_on":
            continue
        if (record.get("metadata") or {}).get("locus") != "memory_only":
            continue
        drafts = []
        for call in record.get("tool_calls") or []:
            name = str(call.get("name", ""))
            args = call.get("args") or {}
            if name in ("Write", "Edit", "NotebookEdit"):
                payload = str(args.get("content") or args.get("new_string") or "")
                if payload.strip():
                    drafts.append(payload)
            elif name == "Bash" and args.get("command"):
                drafts.append(str(args["command"]))
        rows.append({
            "task_id": record["task_id"],
            "memo": str((record.get("metadata") or {}).get("governing_memo") or ""),
            "drafts": [d for d in drafts if len(d) <= MAX_QUERY_CHARS],
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--screen", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    from recall.embeddings import resolve_embedder
    from recall.generation_store import GenerationStore
    from recall.retriever import HybridRetriever

    screen = json.loads(Path(args.screen).expanduser().read_text(encoding="utf-8"))
    miss_ids = {s["task_id"] for s in screen["sessions"]}
    if len(miss_ids) != 14:
        raise SystemExit(f"expected the registered 14 miss sessions, found {len(miss_ids)}")
    positives = sessions(Path(args.archive).expanduser(), miss_ids)
    if len(positives) != 14:
        raise SystemExit(
            f"recovered {len(positives)} records for the registered 14 miss sessions "
            f"({len({p['task_id'] for p in positives})} distinct task ids). The population is not "
            "the one the record fixes, so this is not run."
        )

    precision = json.loads(Path(args.precision).expanduser().read_text(encoding="utf-8"))
    neg_queries = [
        d["draft"]
        for r in precision["negatives"]
        for d in r["per_draft"]
        if not d.get("refused") and len(d["draft"]) <= MAX_QUERY_CHARS
    ]
    print(f"positives: {len(positives)} sessions, "
          f"{sum(len(p['drafts']) for p in positives)} draft queries")
    print(f"negatives: {len(neg_queries)} draft queries")

    embedder = resolve_embedder(EMBEDDER_ID)
    store = GenerationStore(args.dsn, dim=DIM, tenant="default")
    store.check_schema()
    legs_r = {
        "dense": HybridRetriever(store, embedder, candidate_k=CANDIDATE_K, use_sparse=False),
        "lexical": HybridRetriever(store, embedder, candidate_k=CANDIDATE_K, use_dense=False),
    }

    for leg, retriever in legs_r.items():
        names = [n for _, n, _ in ranked(retriever, CONTROL_QUERY)[:TOP_K]]
        if f"{CONTROL_MEMO}.md" not in names:
            raise SystemExit(
                f"POSITIVE CONTROL FAILED on {leg}: quoting a memo's own content did not return "
                f"it in the top {TOP_K} (got {names}). Nothing below is a measurement."
            )
    print("positive control: OK on both legs")

    # ⛔ The control above validates the LEGS and says nothing about the FUSION, which is the half
    # this script actually contributes — and the half that was silently broken. The invariant that
    # catches it: `lexical_only` is the degenerate fusion of one leg, so it MUST reproduce that
    # leg's ordering exactly. When fusion was keyed on document name instead of chunk id, it did
    # not, and every variant was wrong while both leg controls passed.
    control_legs = {leg: ranked(r, CONTROL_QUERY) for leg, r in legs_r.items()}
    lex_order = [name for _, name, _ in control_legs["lexical"]]
    fused_order = [name for name, _ in rrf(control_legs, {"lexical": 1.0})]
    if lex_order[:20] != fused_order[:20]:
        raise SystemExit(
            "FUSION CONTROL FAILED: lexical_only does not reproduce the lexical leg's order.\n"
            f"  leg   : {lex_order[:5]}\n  fused : {fused_order[:5]}\n"
            "A one-leg fusion that reorders its own leg means every variant below is wrong."
        )
    print("fusion control: OK, lexical_only reproduces the lexical leg exactly\n")

    def capture(query: str) -> dict[str, list[tuple[str, float]]]:
        return {leg: ranked(r, query) for leg, r in legs_r.items()}

    print("capturing per-leg ranked lists (one pass; every variant is computed from these)...")
    pos_caps: list[tuple[str, str, list[dict]]] = []
    for entry in positives:
        caps = [capture(d) for d in entry["drafts"]]
        pos_caps.append((entry["task_id"], entry["memo"], caps))
        print(f"  {entry['task_id']:<26} {len(caps)} drafts")
    neg_caps = [capture(q) for q in neg_queries]
    print(f"  negatives: {len(neg_caps)} queries\n")

    names = list(variants(pos_caps[0][2][0]).keys())
    summary = {}
    for variant in names:
        # Per session: best fused score at which the governing memo sits in the top 5.
        best: list[float | None] = []
        for _, memo, caps in pos_caps:
            wanted = f"{memo}.md"
            scores = []
            for legs in caps:
                ordered = variants(legs)[variant][:TOP_K]
                scores.extend(s for n, s in ordered if n == wanted)
            best.append(max(scores) if scores else None)
        # Per negative query: the best fused score it returns at all inside the top 5.
        neg_best = []
        for legs in neg_caps:
            ordered = variants(legs)[variant][:TOP_K]
            neg_best.append(max((s for _, s in ordered), default=None))

        recall_at_0 = sum(1 for b in best if b is not None)
        pool = sorted({b for b in best if b is not None} | {n for n in neg_best if n is not None})
        frontier = []
        for t in pool:
            rec = sum(1 for b in best if b is not None and b >= t)
            ft = sum(1 for n in neg_best if n is not None and n >= t)
            frontier.append({"t": t, "recall": rec, "ft": ft, "ft_rate": round(ft / len(neg_best), 4)})
        viable = [f for f in frontier if f["recall"] >= 9 and f["ft_rate"] <= 0.35]
        at9 = [f for f in frontier if f["recall"] >= 9]
        min_ft_at_9 = min((f["ft_rate"] for f in at9), default=None)
        summary[variant] = {
            "recall_top5": recall_at_0,
            "of": len(best),
            "viable_points": len(viable),
            "min_ft_at_recall_9": min_ft_at_9,
            "frontier": frontier,
        }
        print(f"{variant:<22} recall@5 {recall_at_0:>2}/{len(best)}   viable points "
              f"{len(viable):>2}   min ft at recall>=9: "
              f"{'n/a' if min_ft_at_9 is None else f'{min_ft_at_9:.3f}'}")

    best_recall = max(v["recall_top5"] for v in summary.values())
    any_viable = sum(v["viable_points"] for v in summary.values())
    print(f"\nbest recall@5 over all variants: {best_recall}/14")
    print(f"variants with a viable point (recall>=9, ft<=0.35): "
          f"{sum(1 for v in summary.values() if v['viable_points'])}/{len(summary)}")
    print(f"total viable points: {any_viable}")

    out = Path(args.out) if args.out else (
        REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / "fusion-frontier.json"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "candidate_k": CANDIDATE_K, "top_k": TOP_K, "rrf_k": RRF_K,
                "lexical_weights": list(LEXICAL_WEIGHTS), "score_lambdas": list(SCORE_LAMBDAS),
                "population": {"sessions": len(positives), "negative_queries": len(neg_caps)},
                "summary": summary,
            },
            indent=2,
        ) + "\n",
        encoding="utf-8", newline="\n",
    )
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
