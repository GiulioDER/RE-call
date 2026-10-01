"""W0 R1-Coding: does a reranker close the Coding gap between the open proxy and C9?

Pre-registration: kept in the maintainer's private research log (W0 R1-Coding, 2026-10-01).

The second AML runs use a reranker whatever the embedder, so the cost of an open Qwen3-class
embedder is the reranked proxy (RP) against reranked C9 (RA), not against plain C9 (A). Both arms'
top 100 are collected with ``aml_w0_embedding_compare.py collect --dataset coding --keep-items``;
this re-orders them offline with Voyage ``rerank-2.5`` and scores MRR of the first gold session.

    python scripts/aml_w0_r1_coding.py rerank --a A.json.gz --p P.json.gz --amb-root AMB --out reranked.jsonl
    python scripts/aml_w0_r1_coding.py score --a A.json.gz --p P.json.gz --amb-root AMB \
        --reranked reranked.jsonl --out r1-coding.json
"""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import gzip
import json
from pathlib import Path
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RERANK_MODEL = "rerank-2.5"
#: USD per million tokens for rerank-2.5 (docs.voyageai.com/docs/pricing, read 2026-09-29); the
#: first 200M rerank tokens per account are free, so a cap computed at list price is conservative.
RERANK_USD_PER_MTOK = 0.05
#: The pre-registered Coding bar (W0's): RP minus RA must have a lower bound above minus this.
NONINFERIORITY_MRR = 0.05
#: Stage P's served MRR, which the re-collect must reproduce within REPRODUCE_TOLERANCE.
STAGE_P_MRR = {"A": 0.863, "P": 0.491}
REPRODUCE_TOLERANCE = 0.03
ARMS = ("A", "P", "RA", "RP")
CONTRASTS = (("RP", "RA"), ("RA", "A"), ("RP", "P"), ("RP", "A"), ("P", "A"))
#: The blend (pre-registered 2026-10-01): RRF of the served rank and the rerank rank, k 60, the
#: rerank term weighted w. w 1 is decided; 0.5 and 2 are reported only.
RRF_K = 60
BLEND_WEIGHTS = (0.5, 1.0, 2.0)
BLEND_ARMS = ("A", "P", "RA", "RP", "BA", "BP", "BA(0.5)", "BP(0.5)", "BA(2)", "BP(2)")
BLEND_CONTRASTS = (
    ("BA", "A"), ("BP", "A"), ("BP", "BA"), ("BP", "P"), ("BA", "RA"),
    ("BA(0.5)", "A"), ("BP(0.5)", "A"), ("BA(2)", "A"), ("BP(2)", "A"),
)


def base_of(arm: str) -> str:
    """Which collect an arm reads: RA and BA (any weight) re-order A's items, RP and BP P's."""
    return arm.split("(")[0][-1]


def blend_weight(arm: str) -> float:
    """The rerank weight of a blend arm: "BA" is 1, "BA(0.5)" is 0.5."""
    return float(arm.split("(")[1].rstrip(")")) if "(" in arm else 1.0


def blend_order(n: int, rerank_order: Sequence[int], weight: float) -> list[int]:
    """Indices 0..n-1 (the served order) by 1/(k + served rank) + weight/(k + rerank rank), best
    first; ties keep the served order."""
    rerank_rank = {index: rank for rank, index in enumerate(rerank_order, start=1)}
    score = [1 / (RRF_K + i + 1) + weight / (RRF_K + rerank_rank[i]) for i in range(n)]
    return sorted(range(n), key=lambda i: (-score[i], i))


def arm_items(arm: str, rows: dict[str, dict[str, Any]], orders: dict[tuple[str, str], list[int]], task: str) -> list[dict[str, Any]]:
    """The items an arm shows for a task: its base arm's kept items, served (A, P), re-ordered by
    the rerank (RA, RP) or by the blend of served and rerank order (BA, BP)."""
    base = base_of(arm)
    items = rows[base][task]["served_facts"]["kept"]
    if arm == base:
        return list(items)
    order = orders[(base, task)]
    if arm.startswith("B"):
        order = blend_order(len(items), order, blend_weight(arm))
    return [items[k] for k in order]


def verdict(contrasts: dict[str, dict[str, float]]) -> dict[str, Any]:
    """The pre-registered rule: RP is non-inferior to RA when RP minus RA's lower bound is above -0.05."""
    low = contrasts["RP-vs-RA"]["ci95_low"]
    return {"RP_noninferior_to_RA": low > -NONINFERIORITY_MRR, "rule": f"RP-vs-RA ci95_low {low} > -{NONINFERIORITY_MRR}"}


def blend_verdict(contrasts: dict[str, dict[str, float]]) -> dict[str, Any]:
    """The blend's pre-registered rules, each a lower bound above -0.05: BA minus A (the blend does
    not harm C9) and BP minus A (proxy plus blend is non-inferior to C9 as served)."""
    ba, bp = contrasts["BA-vs-A"]["ci95_low"], contrasts["BP-vs-A"]["ci95_low"]
    return {
        "blend_does_not_harm_c9": ba > -NONINFERIORITY_MRR,
        "proxy_blend_noninferior_to_c9": bp > -NONINFERIORITY_MRR,
        "rule": f"BA-vs-A ci95_low {ba}, BP-vs-A ci95_low {bp}, each > -{NONINFERIORITY_MRR}",
    }


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(r["question_id"]): r for r in json.loads(gzip.decompress(path.read_bytes()))["rows"]}


def _questions(amb_root: Path) -> dict[str, Any]:
    from aml_w0_embedding_compare import load_coding

    return {q.question_id: q for q in load_coding(amb_root, "r1-coding").questions}


def rerank(args: argparse.Namespace) -> None:
    from recall.rerank import VoyageReranker

    client = VoyageReranker(model=RERANK_MODEL)._voyage_client()
    questions = _questions(args.amb_root)
    done: set[tuple[str, str]] = set()
    tokens = 0
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add((record["arm"], record["id"]))
            tokens += int(record["tokens"])
    for arm, path in (("A", args.a), ("P", args.p)):
        for task, row in _rows(path).items():
            if (arm, task) in done:
                continue
            if tokens * RERANK_USD_PER_MTOK / 1e6 > args.max_usd:
                print(json.dumps({"stopped": "cap", "tokens": tokens}), flush=True)
                return
            texts = [item["content"] for item in row["served_facts"]["kept"]]
            result = client.rerank(questions[task].query, texts, model=RERANK_MODEL, top_k=len(texts))
            used = int(getattr(result, "total_tokens", 0) or 0) or sum(len(t) for t in texts) // 4
            tokens += used
            with args.out.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps({"arm": arm, "id": task, "order": [r.index for r in result.results], "tokens": used}) + "\n")
    print(json.dumps({"done": True, "tokens": tokens, "usd": round(tokens * RERANK_USD_PER_MTOK / 1e6, 4)}), flush=True)


def score(
    rows: dict[str, dict[str, dict[str, Any]]],
    orders: dict[tuple[str, str], list[int]],
    questions: dict[str, Any],
    *,
    blend: bool = False,
) -> dict[str, Any]:
    from aml_w0_embedding_compare import paired_bootstrap, score_items

    arms = BLEND_ARMS if blend else ARMS
    tasks = sorted(t for t in rows["A"] if t in rows["P"] and t in questions)
    per_arm = {arm: [score_items(arm_items(arm, rows, orders, t), questions[t]) for t in tasks] for arm in arms}
    means = {
        arm: {m: round(sum(r[m] for r in per_arm[arm]) / len(tasks), 4) for m in ("rr", "session_hit@10", "session_hit@100")}
        for arm in arms
    }
    contrasts = {}
    for treated, control in CONTRASTS + (BLEND_CONTRASTS if blend else ()):
        interval = paired_bootstrap([r["rr"] for r in per_arm[control]], [r["rr"] for r in per_arm[treated]], scale=1.0)
        contrasts[f"{treated}-vs-{control}"] = {k: round(v, 4) if isinstance(v, float) else v for k, v in interval.items()}
    reproduction = {
        arm: {"mrr": means[arm]["rr"], "stage_p": STAGE_P_MRR[arm], "ok": abs(means[arm]["rr"] - STAGE_P_MRR[arm]) <= REPRODUCE_TOLERANCE}
        for arm in ("A", "P")
    }
    result = {"n": len(tasks), "means": means, "contrasts": contrasts, "reproduction": reproduction, "verdict": verdict(contrasts)}
    if blend:
        result["blend_verdict"] = blend_verdict(contrasts)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    rr = sub.add_parser("rerank")
    sc = sub.add_parser("score")
    for p in (rr, sc):
        p.add_argument("--a", type=Path, required=True)
        p.add_argument("--p", type=Path, required=True)
        p.add_argument("--amb-root", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
    rr.add_argument("--max-usd", type=float, default=2.0)
    sc.add_argument("--reranked", type=Path, required=True)
    sc.add_argument("--blend", action="store_true", help="also score the pre-registered blend arms")
    args = parser.parse_args(argv)
    if args.mode == "rerank":
        rerank(args)
        return 0
    orders = {}
    for line in args.reranked.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        orders[(record["arm"], record["id"])] = record["order"]
    result = score({"A": _rows(args.a), "P": _rows(args.p)}, orders, _questions(args.amb_root), blend=args.blend)
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
