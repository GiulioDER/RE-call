"""Compare two official EnterpriseRAG-Bench evaluator result files.

The evaluator's aggregate score hides which questions changed. This report keeps the official
per-question fields, computes the leaderboard score as correctness times completeness, and reports
paired category deltas plus a deterministic bootstrap interval. It accepts only evaluator result
JSON files, not locally invented labels or gold answers.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


def load_results(path: Path) -> dict[str, Mapping[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("questions") if isinstance(payload, Mapping) else None
    if not isinstance(rows, list):
        raise ValueError(f"{path}: official evaluator result lacks a questions array")
    by_id: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not row.get("question_id"):
            raise ValueError(f"{path}: every question row needs question_id")
        question_id = str(row["question_id"])
        if question_id in by_id:
            raise ValueError(f"{path}: duplicate question_id {question_id}")
        by_id[question_id] = row
    return by_id


def _score(row: Mapping[str, Any]) -> float:
    correct = bool(row.get("answer_correct"))
    completeness = float(row.get("completeness_pct", 0.0))
    return completeness if correct else 0.0


def _recall(row: Mapping[str, Any]) -> float:
    value = row.get("document_recall_pct")
    return float(value) if value is not None else 0.0


def _extra(row: Mapping[str, Any]) -> float:
    value = row.get("invalid_extra_docs")
    return float(value) if value is not None else 0.0


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def bootstrap_ci(values: list[float], *, seed: int, samples: int = 10_000) -> tuple[float, float]:
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(_mean([values[rng.randrange(n)] for _ in range(n)]) for _ in range(samples))
    return means[int(samples * 0.025)], means[int(samples * 0.975)]


def compare(
    baseline_path: Path,
    candidate_path: Path,
    *,
    seed: int = 366,
) -> dict[str, Any]:
    baseline = load_results(baseline_path)
    candidate = load_results(candidate_path)
    if set(baseline) != set(candidate):
        missing = sorted(set(baseline) - set(candidate))
        extra = sorted(set(candidate) - set(baseline))
        raise ValueError(f"question sets differ: missing={missing[:5]} extra={extra[:5]}")

    pairs: list[dict[str, Any]] = []
    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for question_id, before in baseline.items():
        after = candidate[question_id]
        category = str(before.get("question_type") or after.get("question_type") or "unknown")
        if str(after.get("question_type") or category) != category:
            raise ValueError(f"question type changed for {question_id}")
        pair = {
            "question_id": question_id,
            "question_type": category,
            "baseline_score": _score(before),
            "candidate_score": _score(after),
            "score_delta": _score(after) - _score(before),
            "baseline_correct": bool(before.get("answer_correct")),
            "candidate_correct": bool(after.get("answer_correct")),
            "baseline_recall_pct": _recall(before),
            "candidate_recall_pct": _recall(after),
            "recall_delta_pct": _recall(after) - _recall(before),
            "baseline_invalid_extra_docs": _extra(before),
            "candidate_invalid_extra_docs": _extra(after),
            "invalid_extra_docs_delta": _extra(after) - _extra(before),
        }
        pairs.append(pair)
        by_category[category].append(pair)

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        deltas = [float(row["score_delta"]) for row in rows]
        recall_deltas = [float(row["recall_delta_pct"]) for row in rows]
        extra_deltas = [float(row["invalid_extra_docs_delta"]) for row in rows]
        return {
            "count": len(rows),
            "baseline_score": round(_mean([float(row["baseline_score"]) for row in rows]), 4),
            "candidate_score": round(_mean([float(row["candidate_score"]) for row in rows]), 4),
            "score_delta": round(_mean(deltas), 4),
            "score_delta_bootstrap_95_pct": [round(value, 4) for value in bootstrap_ci(deltas, seed=seed)],
            "baseline_correctness_pct": round(
                100 * _mean([float(row["baseline_correct"]) for row in rows]), 4
            ),
            "candidate_correctness_pct": round(
                100 * _mean([float(row["candidate_correct"]) for row in rows]), 4
            ),
            "recall_delta_pct": round(_mean(recall_deltas), 4),
            "invalid_extra_docs_delta": round(_mean(extra_deltas), 4),
            "candidate_wins": sum(delta > 0 for delta in deltas),
            "baseline_wins": sum(delta < 0 for delta in deltas),
            "ties": sum(delta == 0 for delta in deltas),
        }

    summary = summarize(pairs)
    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "question_count": len(pairs),
        "overall": summary,
        "by_category": {category: summarize(rows) for category, rows in sorted(by_category.items())},
        "changed_questions": [row for row in pairs if row["score_delta"] != 0],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=366)
    args = parser.parse_args()
    report = compare(args.baseline, args.candidate, seed=args.seed)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"wrote comparison report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
