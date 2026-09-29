"""W0 R1 stage 1: re-order the proxy's own 100 items offline (fusion re-weight, Voyage rerank).

Pre-registration: kept in the maintainer's private research log (W0 R1 stage 1, 2026-09-29).

R0 stored, for each LoCoMo question, the 100 items served C9 (A) and the Qwen proxy (P) returned.
P's loss is ranking, not recall (turn_hit@100 99.74 against 99.87), so this measures, without
re-ingesting anything, whether re-ordering P's items recovers it:

- F(w): Reciprocal Rank Fusion (k 60) of P's stored rank with a local BM25 rank over the same 100
  items, BM25 weighted w. An approximation of a served re-weighting: its BM25 sees only the top 100.
- R: Voyage ``rerank-2.5`` over the 100 items (rerankers are unrestricted for the second Fulls).

The 815 questions outside the 720 draw choose F's weight; the draw judges every arm once.

    python scripts/aml_w0_r1_reorder.py rerank --a C-A.json.gz --p C-P.json.gz --locomo locomo10.json --out reranked.jsonl
    python scripts/aml_w0_r1_reorder.py score --a C-A.json.gz --p C-P.json.gz --reranked reranked.jsonl \
        --locomo locomo10.json --draw locomo-draw.json --out r1-stage1.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import gzip
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RRF_K = 60
WEIGHTS = (0.5, 1.0, 2.0)
RERANK_MODEL = "rerank-2.5"
#: USD per million tokens for rerank-2.5, read from docs.voyageai.com/docs/pricing on 2026-09-29
#: (the first 200M rerank tokens per account are free, so the cap is conservative).
RERANK_USD_PER_MTOK = 0.05
RECOVERY_POINTS = 1.5
_TOKEN = re.compile(r"\w+")


def _tokens(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


def bm25_order(query: str, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Indices of ``texts`` by BM25 score for ``query`` over these texts only, best first; ties keep
    the given order."""
    docs = [_tokens(t) for t in texts]
    n = len(docs)
    avg = sum(len(d) for d in docs) / n if n else 0.0
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    terms = set(_tokens(query))
    scores = []
    for d in docs:
        tf = Counter(d)
        s = 0.0
        for t in terms:
            if tf[t]:
                idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
                s += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / (avg or 1)))
        scores.append(s)
    return sorted(range(n), key=lambda i: (-scores[i], i))


def fuse(n: int, lexical: Sequence[int], weight: float) -> list[int]:
    """Indices 0..n-1 (the stored order) re-ordered by RRF of the stored rank and ``lexical``'s
    rank, the lexical term weighted ``weight``; ties keep the stored order."""
    lexical_rank = {index: rank for rank, index in enumerate(lexical, start=1)}
    score = [1 / (RRF_K + i + 1) + weight / (RRF_K + lexical_rank[i]) for i in range(n)]
    return sorted(range(n), key=lambda i: (-score[i], i))


def turn_rr(items: Sequence[dict[str, Any]], question: Any) -> float:
    """1 / the rank of the first item holding a gold turn (0 when none does), by W0's matcher."""
    from aml_locomo_route_compare import turn_present
    from aml_w0_embedding_compare import item_text

    for rank, item in enumerate(items, start=1):
        if any(turn_present(turn, item_text(item)) for turn in question.gold_turns):
            return 1.0 / rank
    return 0.0


def metrics(items: Sequence[dict[str, Any]], question: Any) -> dict[str, float]:
    from aml_w0_embedding_compare import score_items

    scored = score_items(items, question)
    return {"turn_hit@5": scored["turn_hit@5"], "turn_hit@10": scored["turn_hit@10"], "turn_rr": turn_rr(items, question)}


def mean_points(rows: Sequence[dict[str, float]], key: str) -> float:
    return round(100 * sum(r[key] for r in rows) / len(rows), 2) if rows else 0.0


def choose_weight(tune: dict[float, float]) -> float:
    """The weight with the highest tuning turn_hit@10; ties go to the smaller weight."""
    return max(sorted(tune), key=lambda w: (tune[w], -w))


def recovers(arm: dict[str, float], base: dict[str, float]) -> bool:
    return arm["turn_hit@10"] >= base["turn_hit@10"] + RECOVERY_POINTS and arm["turn_hit@5"] >= base["turn_hit@5"]


def _rows(path: Path) -> dict[str, dict[str, Any]]:
    return {str(r["id"]): r for r in json.loads(gzip.decompress(path.read_bytes()))["rows"]}


def rerank(args: argparse.Namespace) -> None:
    from recall.rerank import VoyageReranker

    from aml_w0_embedding_compare import load_locomo

    client = VoyageReranker(model=RERANK_MODEL)._voyage_client()
    questions = {q.question_id: q for q in load_locomo(args.locomo, "r1", None).questions}
    done: set[tuple[str, str]] = set()
    tokens = 0
    if args.out.exists():
        for line in args.out.read_text(encoding="utf-8").splitlines():
            record = json.loads(line)
            done.add((record["arm"], record["id"]))
            tokens += int(record["tokens"])
    for arm, path in (("A", args.a), ("P", args.p)):
        for ident, row in _rows(path).items():
            if (arm, ident) in done or ident not in questions:
                continue
            if tokens * RERANK_USD_PER_MTOK / 1e6 > args.max_usd:
                print(json.dumps({"stopped": "cap", "tokens": tokens}), flush=True)
                return
            texts = [str(item["content"]) if isinstance(item["content"], str) else "" for item in row["items"]]
            result = client.rerank(questions[ident].query, texts, model=RERANK_MODEL, top_k=len(texts))
            used = int(getattr(result, "total_tokens", 0) or 0) or sum(len(t) for t in texts) // 4
            tokens += used
            with args.out.open("a", encoding="utf-8") as sink:
                sink.write(json.dumps({"arm": arm, "id": ident, "order": [r.index for r in result.results], "tokens": used}) + "\n")
    print(json.dumps({"done": True, "tokens": tokens, "usd": round(tokens * RERANK_USD_PER_MTOK / 1e6, 3)}), flush=True)


def score(args: argparse.Namespace) -> dict[str, Any]:
    from aml_w0_embedding_compare import load_locomo

    questions = {q.question_id: q for q in load_locomo(args.locomo, "r1", None).questions}
    rows = {"A": _rows(args.a), "P": _rows(args.p)}
    reranked: dict[tuple[str, str], list[int]] = {}
    for line in args.reranked.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        reranked[(record["arm"], record["id"])] = record["order"]
    draw = {str(i) for i in json.loads(args.draw.read_text(encoding="utf-8"))["ids"]}
    ids = [i for i in rows["P"] if i in rows["A"] and i in questions and questions[i].gold_turns]
    splits = {"tune": [i for i in ids if i not in draw], "draw": [i for i in ids if i in draw]}

    def arm_rows(arm: str, split: str) -> list[dict[str, float]]:
        out = []
        for ident in splits[split]:
            base = "A" if arm.endswith("A") else "P"
            items = rows[base][ident]["items"]
            if arm in ("A", "P"):
                order = list(range(len(items)))
            elif arm.startswith("R"):
                order = reranked[(base, ident)]
            else:
                weight = float(arm.split("(")[1].rstrip(")").split(",")[0])
                texts = [str(i["content"]) if isinstance(i["content"], str) else "" for i in items]
                order = fuse(len(items), bm25_order(questions[ident].query, texts), weight)
            out.append(metrics([items[k] for k in order], questions[ident]))
        return out

    tune = {w: mean_points(arm_rows(f"F({w},P)", "tune"), "turn_hit@10") for w in WEIGHTS}
    chosen = choose_weight(tune)
    result: dict[str, Any] = {"n": {k: len(v) for k, v in splits.items()}, "tuning_turn_hit@10": tune, "chosen_weight": chosen, "draw": {}}
    for arm in ("A", "P", f"F({chosen},P)", "R(P)", "R(A)"):
        arm_metrics = arm_rows(arm, "draw")
        result["draw"][arm] = {k: mean_points(arm_metrics, k) for k in ("turn_hit@5", "turn_hit@10", "turn_rr")}
    d = result["draw"]
    result["decision"] = {
        f"F({chosen},P)": recovers(d[f"F({chosen},P)"], d["P"]),
        "R(P)": recovers(d["R(P)"], d["P"]),
        "R(A) improves A": recovers(d["R(A)"], d["A"]),
    }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="mode", required=True)
    rr = sub.add_parser("rerank")
    sc = sub.add_parser("score")
    for p in (rr, sc):
        p.add_argument("--a", type=Path, required=True)
        p.add_argument("--p", type=Path, required=True)
        p.add_argument("--locomo", type=Path, required=True)
        p.add_argument("--out", type=Path, required=True)
    rr.add_argument("--max-usd", type=float, default=6.0)
    sc.add_argument("--reranked", type=Path, required=True)
    sc.add_argument("--draw", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.mode == "rerank":
        rerank(args)
    else:
        result = score(args)
        args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
