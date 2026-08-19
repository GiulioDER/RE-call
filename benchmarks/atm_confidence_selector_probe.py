"""Evaluate confidence controlled list answer selection from saved retrieval scores."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

from benchmarks.atm_cardinality_selector_probe import split_bucket
from benchmarks.atm_list_selection_probe import parse_answer, score


THRESHOLD = 0.50


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def mean_scores(rows: list[dict[str, float]]) -> dict[str, float]:
    return {
        metric: statistics.fmean(row[metric] for row in rows)
        for metric in rows[0]
    } if rows else {}


def run(args: argparse.Namespace) -> dict[str, Any]:
    ground_truth = load_json(args.ground_truth)
    retrieval = load_json(args.retrieval_result)
    gold = {
        str(row["id"]): row
        for row in ground_truth
        if row.get("qtype") == "list_recall"
    }
    if len(gold) != 139:
        raise ValueError(f"expected 139 full list questions, found {len(gold)}")
    test_ids = {
        question_id for question_id in gold if split_bucket(question_id) == "test"
    }

    arms: dict[str, Any] = {}
    for arm_name, arm in retrieval["arms"].items():
        fixed_one: list[dict[str, float]] = []
        fixed_five: list[dict[str, float]] = []
        confidence_scores: list[dict[str, float]] = []
        details = []
        for row in arm["details"]:
            question_id = str(row["id"])
            if question_id not in test_ids:
                continue
            gold_answer = parse_answer(gold[question_id]["answer"])
            retrieved = row["retrieval_ids"]
            scores = row["retrieval_scores"]
            fixed_one.append(score(retrieved[:1], gold_answer))
            fixed_five.append(score(retrieved[:5], gold_answer))
            top_score = float(scores[0]) if scores else 0.0
            chosen_k = 1 if top_score >= THRESHOLD else 5
            selected = score(retrieved[:chosen_k], gold_answer)
            confidence_scores.append(selected)
            details.append(
                {
                    "id": question_id,
                    "top_score": top_score,
                    "chosen_k": chosen_k,
                    "scores": selected,
                }
            )
        arms[arm_name] = {
            "test_questions": len(details),
            "threshold": THRESHOLD,
            "fixed_top1": mean_scores(fixed_one),
            "fixed_top5": mean_scores(fixed_five),
            "confidence_controlled": mean_scores(confidence_scores),
            "cutoff_counts": {
                str(k): sum(row["chosen_k"] == k for row in details) for k in (1, 5)
            },
            "details": details,
        }

    return {
        "benchmark": "ATM-Bench",
        "split": "full_list_recall_test_partition",
        "measurement": "confidence_controlled_list_answer_selector",
        "threshold": THRESHOLD,
        "prediction": "top1_if_top_dense_cosine_at_least_threshold_else_top5",
        "judge": {"used": False, "model": None},
        "arms": arms,
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", type=Path, required=True)
    ap.add_argument("--retrieval-result", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for arm_name, arm in result["arms"].items():
        print(
            f"{arm_name}: confidence_J={arm['confidence_controlled']['jaccard']:.4f} "
            f"fixed1_J={arm['fixed_top1']['jaccard']:.4f} "
            f"fixed5_J={arm['fixed_top5']['jaccard']:.4f} "
            f"cutoffs={arm['cutoff_counts']}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
