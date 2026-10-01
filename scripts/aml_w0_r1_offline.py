"""W0 R1-Coding offline ideas: shallow rerank, session aggregation, lexical re-weighting.

Pre-registration: kept in the maintainer's private research log (W0 R1-Coding offline ideas,
2026-10-01).

Reads R1-Coding's recorded collects (100 kept items per task and arm) and its rerank-2.5 orders,
and scores three families of re-orderings for C9 (A) and the Qwen3 proxy (P), choosing each
family's variant per task by leave-one-out so no task is scored with a setting tuned on itself:

- D(K): the top K served items re-ordered by the reranker, the rest in served order. The reranker
  scores each pair on its own, so the recorded 100-item order restricted to the top K is the depth-K
  rerank.
- S(m): sessions ranked by an aggregate of their windows' served ranks; session-level MRR.
- L(w): RRF (k 60) of the served rank and a local BM25 rank over the same 100 items, BM25 weighted w.

    python scripts/aml_w0_r1_offline.py --a A.json.gz --p P.json.gz --amb-root AMB \
        --reranked reranked.jsonl --out offline.json
"""

from __future__ import annotations

import argparse
from collections import Counter
from collections.abc import Sequence
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

RRF_K = 60
BAR = 0.05
DEPTHS = (0, 5, 10, 20, 50, 100)
AGGREGATES = ("max", "sumrr", "max2", "top20")
LEXICAL_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0)
_TOKEN = re.compile(r"\w+")


def depth_order(n: int, rerank_order: Sequence[int], k: int) -> list[int]:
    """The top ``k`` served indices in the rerank's order, then the rest in served order."""
    return [i for i in rerank_order if i < k] + list(range(k, n))


def bm25_order(query: str, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> list[int]:
    """Indices of ``texts`` by BM25 score for ``query`` over these texts only; ties keep the given order."""
    docs = [_TOKEN.findall(t.lower()) for t in texts]
    n = len(docs)
    avg = sum(len(d) for d in docs) / n if n else 0.0
    df: Counter[str] = Counter()
    for d in docs:
        df.update(set(d))
    terms = set(_TOKEN.findall(query.lower()))
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


def lexical_order(query: str, texts: Sequence[str], weight: float) -> list[int]:
    """Served indices by 1/(k + served rank) + weight/(k + BM25 rank); ties keep the served order."""
    bm25_rank = {index: rank for rank, index in enumerate(bm25_order(query, texts), start=1)}
    score = [1 / (RRF_K + i + 1) + weight / (RRF_K + bm25_rank[i]) for i in range(len(texts))]
    return sorted(range(len(texts)), key=lambda i: (-score[i], i))


def session_order(sessions: Sequence[str], method: str) -> list[str]:
    """Distinct sessions ranked by an aggregate of their windows' served ranks (1-indexed); ties keep
    first appearance."""
    ranks: dict[str, list[int]] = {}
    for rank, session in enumerate(sessions, start=1):
        ranks.setdefault(session, []).append(rank)
    first = {s: r[0] for s, r in ranks.items()}

    def key(s: str) -> tuple[float, int]:
        r = ranks[s]
        if method == "max":
            value = 1.0 / r[0]
        elif method == "sumrr":
            value = sum(1.0 / (RRF_K + x) for x in r)
        elif method == "max2":
            value = 1.0 / r[0] + (0.5 / r[1] if len(r) > 1 else 0.0)
        elif method == "top20":
            value = float(sum(1 for x in r if x <= 20))
        else:
            raise ValueError(method)
        return (-value, first[s])

    return sorted(ranks, key=key)


def reciprocal_rank(ordered: Sequence[str], gold: frozenset[str]) -> float:
    return next((1.0 / rank for rank, s in enumerate(ordered, start=1) if s in gold), 0.0)


def loo_select(variants: dict[Any, list[float]]) -> tuple[list[float], list[Any]]:
    """Per task, the score of the variant with the highest mean over the OTHER tasks (ties go to the
    earlier variant); returns the per-task scores and the chosen variant per task."""
    names = list(variants)
    n = len(variants[names[0]])
    totals = {v: sum(variants[v]) for v in names}
    scores, chosen = [], []
    for i in range(n):
        best = max(names, key=lambda v: (totals[v] - variants[v][i], -names.index(v)))
        scores.append(variants[best][i])
        chosen.append(best)
    return scores, chosen


def family_scores(
    rows: dict[str, dict[str, Any]], orders: dict[tuple[str, str], list[int]], questions: dict[str, Any], arm: str, tasks: Sequence[str]
) -> dict[str, dict[Any, list[float]]]:
    """Per family, per variant, the per-task score for one arm (``"A"`` or ``"P"``)."""
    out: dict[str, dict[Any, list[float]]] = {"D": {k: [] for k in DEPTHS}, "S": {m: [] for m in AGGREGATES}, "L": {w: [] for w in LEXICAL_WEIGHTS}}
    for task in tasks:
        items = rows[arm][task]["served_facts"]["kept"]
        gold = questions[task].gold_sessions
        sessions = [str(i["session_id"]) for i in items]
        texts = [str(i["content"]) for i in items]

        def window_rr(order: Sequence[int], sessions: list[str] = sessions, gold: frozenset[str] = gold) -> float:
            return reciprocal_rank([sessions[k] for k in order], gold)

        for k in DEPTHS:
            out["D"][k].append(window_rr(depth_order(len(items), orders[(arm, task)], k)))
        for m in AGGREGATES:
            out["S"][m].append(reciprocal_rank(session_order(sessions, m), gold))
        for w in LEXICAL_WEIGHTS:
            out["L"][w].append(window_rr(lexical_order(questions[task].query, texts, w)))
    return out


BASELINE: dict[str, Any] = {"D": 0, "S": "max", "L": 0.0}


def verdicts(contrasts: dict[str, dict[str, float]]) -> dict[str, dict[str, bool]]:
    """The pre-registered rules per family: helps the proxy (LOO(P) minus P above 0), harmless to C9
    (LOO(A) minus A above -0.05), closes the gap (LOO(P) minus A above -0.05); lower bounds."""
    return {
        f: {
            "helps_proxy": contrasts[f"{f}:LOO(P)-vs-P"]["ci95_low"] > 0,
            "harmless_to_c9": contrasts[f"{f}:LOO(A)-vs-A"]["ci95_low"] > -BAR,
            "closes_gap": contrasts[f"{f}:LOO(P)-vs-A"]["ci95_low"] > -BAR,
        }
        for f in ("D", "S", "L")
    }


def score(rows: dict[str, dict[str, Any]], orders: dict[tuple[str, str], list[int]], questions: dict[str, Any]) -> dict[str, Any]:
    from aml_w0_embedding_compare import paired_bootstrap

    def interval(control: list[float], treated: list[float]) -> dict[str, Any]:
        iv = paired_bootstrap(control, treated, scale=1.0)
        return {k: round(v, 4) if isinstance(v, float) else v for k, v in iv.items()}

    def mean(xs: list[float]) -> float:
        return round(sum(xs) / len(xs), 4)

    tasks = sorted(t for t in rows["A"] if t in rows["P"] and t in questions)
    per = {arm: family_scores(rows, orders, questions, arm, tasks) for arm in ("A", "P")}
    result: dict[str, Any] = {"n": len(tasks), "variants": {}, "loo": {}, "contrasts": {}}
    for f in ("D", "S", "L"):
        base = BASELINE[f]
        loo = {arm: loo_select(per[arm][f]) for arm in ("A", "P")}
        result["variants"][f] = {arm: {str(v): mean(s) for v, s in per[arm][f].items()} for arm in ("A", "P")}
        result["loo"][f] = {arm: {"mrr": mean(loo[arm][0]), "chosen": dict(Counter(str(c) for c in loo[arm][1]))} for arm in ("A", "P")}
        result["contrasts"][f"{f}:LOO(P)-vs-P"] = interval(per["P"][f][base], loo["P"][0])
        result["contrasts"][f"{f}:LOO(A)-vs-A"] = interval(per["A"][f][base], loo["A"][0])
        result["contrasts"][f"{f}:LOO(P)-vs-A"] = interval(per["A"][f][base], loo["P"][0])
    result["verdicts"] = verdicts(result["contrasts"])
    return result


def main(argv: Sequence[str] | None = None) -> int:
    from aml_w0_r1_coding import _questions, _rows

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--a", type=Path, required=True)
    parser.add_argument("--p", type=Path, required=True)
    parser.add_argument("--amb-root", type=Path, required=True)
    parser.add_argument("--reranked", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    orders: dict[tuple[str, str], list[int]] = {}
    for line in args.reranked.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        orders[(record["arm"], record["id"])] = record["order"]
    result = score({"A": _rows(args.a), "P": _rows(args.p)}, orders, _questions(args.amb_root))
    args.out.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
