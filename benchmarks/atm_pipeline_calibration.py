"""Calibrate the ATM corpus gap floor and compare frozen pipeline artifacts.

The official ATM question set contains answerable questions only. This probe therefore fits an
answer-preservation floor, not an answerable versus unanswerable abstention boundary. It chooses
no configuration and never reads holdout values while fitting the development threshold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from benchmarks.atm_list_selection_probe import parse_answer, score as list_score


def load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def is_dev(question_id: str) -> bool:
    value = int(hashlib.sha256(question_id.encode("utf-8")).hexdigest(), 16)
    return value % 10 < 7


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * fraction))))
    return float(ordered[index])


def mean_metric(rows: list[dict[str, Any]], section: str, metric: str) -> float:
    return statistics.fmean(float(row[section][metric]) for row in rows) if rows else 0.0


def list_summary(rows: list[dict[str, Any]], gold_by_id: dict[str, dict[str, Any]]) -> dict[str, float]:
    selected = [row for row in rows if gold_by_id.get(str(row["id"]), {}).get("qtype") == "list_recall"]
    if not selected:
        return {"jaccard@5": 0.0, "containment@5": 0.0, "answer_hit@5": 0.0, "questions": 0}
    values = [
        list_score(
            [str(value) for value in row["retrieval_ids"][:5]],
            parse_answer(gold_by_id[str(row["id"])]["answer"]),
        )
        for row in selected
    ]
    return {
        "jaccard@5": statistics.fmean(row["jaccard"] for row in values),
        "containment@5": statistics.fmean(row["gold_answer_containment"] for row in values),
        "answer_hit@5": statistics.fmean(row["answer_hit"] for row in values),
        "questions": len(selected),
    }


def summarize_arm(
    arm: dict[str, Any], gold_by_id: dict[str, dict[str, Any]], threshold: float
) -> dict[str, Any]:
    rows = list(arm["details"])
    dev = [row for row in rows if is_dev(str(row["id"]))]
    holdout = [row for row in rows if not is_dev(str(row["id"]))]

    def section(part: list[dict[str, Any]]) -> dict[str, Any]:
        top_scores = [max((float(value) for value in row["retrieval_scores"]), default=0.0) for row in part]
        return {
            "questions": len(part),
            "item_R@10": mean_metric(part, "retrieval_recall", "R@10"),
            "question_Recall@10": mean_metric(part, "question_hit", "Recall@10"),
            "complete_Recall@10GT": mean_metric(part, "complete_evidence", "Recall@10GT"),
            "top_dense_score_p05": percentile(top_scores, 0.05),
            "false_abstain_at_fitted_threshold": statistics.fmean(score < threshold for score in top_scores)
            if top_scores
            else 0.0,
            "list": list_summary(part, gold_by_id),
        }

    return {
        "threshold_fit": {
            "rule": "development fifth percentile of max returned dense cosine",
            "threshold": threshold,
            "development": section(dev),
        },
        "holdout": section(holdout),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    gold_rows = load(args.ground_truth)
    gold_by_id = {str(row["id"]): row for row in gold_rows}
    artifacts: dict[str, Any] = {}
    for path in args.retrieval:
        payload = load(path)
        dense = payload["arms"].get("dense")
        if dense is None:
            raise ValueError(f"{path} has no dense arm, required for threshold fitting")
        dev_scores = [
            float(row["max_dense_score"])
            for row in dense["details"]
            if row.get("max_dense_score") is not None
            if is_dev(str(row["id"]))
        ]
        threshold = percentile(dev_scores, 0.05)
        name = f"{payload['embedder']}|{payload.get('reranker', 'none')}"
        artifacts[name] = {
            "source": str(path),
            "embedder": payload["embedder"],
            "reranker": payload.get("reranker", "none"),
            "candidate_k": payload["candidate_k"],
            "arms": {
                arm_name: summarize_arm(arm, gold_by_id, threshold)
                for arm_name, arm in payload["arms"].items()
            },
        }
    return {
        "benchmark": "ATM-Bench",
        "measurement": "development_threshold_calibration_and_frozen_holdout",
        "split_rule": "sha256(question_id) integer modulo 10 < 7 is development",
        "threshold_limit": "answer-preservation floor only; no unanswerable ATM labels exist",
        "ground_truth": str(args.ground_truth),
        "artifacts": artifacts,
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", type=Path, required=True)
    ap.add_argument("--retrieval", type=Path, nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    return ap


def main() -> int:
    args = parser().parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for name, payload in result["artifacts"].items():
        hybrid = payload["arms"].get("hybrid") or payload["arms"]["dense"]
        dev = hybrid["threshold_fit"]["development"]
        test = hybrid["holdout"]
        print(
            f"{name}: threshold={hybrid['threshold_fit']['threshold']:.4f} "
            f"dev_complete_R@10GT={dev['complete_Recall@10GT']:.4f} "
            f"holdout_complete_R@10GT={test['complete_Recall@10GT']:.4f} "
            f"holdout_false_abstain={test['false_abstain_at_fitted_threshold']:.4f} "
            f"holdout_J@5={test['list']['jaccard@5']:.4f}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

