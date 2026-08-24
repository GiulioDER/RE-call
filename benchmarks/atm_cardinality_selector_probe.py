"""Evaluate a question only cardinality selector on ATM list questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any

from benchmarks.atm_list_selection_probe import parse_answer, score
from recall.embeddings import embed_passages, resolve_embedder


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def split_bucket(question_id: str) -> str:
    value = int(hashlib.sha256(question_id.encode("utf-8")).hexdigest(), 16) % 10
    return "dev" if value < 7 else "test"


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

    rows: list[dict[str, Any]] = []
    for question_id, row in sorted(gold.items()):
        answer = parse_answer(row["answer"])
        rows.append(
            {
                "id": question_id,
                "question": str(row["question"]),
                "gold_answer": answer,
                "bucket": split_bucket(question_id),
                "gold_bucket": "multi" if len(answer) > 1 else "singleton",
            }
        )
    dev = [row for row in rows if row["bucket"] == "dev"]
    test = [row for row in rows if row["bucket"] == "test"]
    if not dev or not test:
        raise ValueError("cardinality split must contain development and test rows")

    embedder = resolve_embedder(args.embedder)
    vectors = embed_passages(embedder, [row["question"] for row in rows])
    vector_by_id = {row["id"]: vector for row, vector in zip(rows, vectors)}

    arms: dict[str, Any] = {}
    for arm_name, arm in retrieval["arms"].items():
        retrieval_by_id = {str(row["id"]): row for row in arm["details"]}
        dev_vectors = [(row, vector_by_id[row["id"]]) for row in dev]
        adaptive_details: list[dict[str, Any]] = []
        fixed_one: list[dict[str, float]] = []
        fixed_five: list[dict[str, float]] = []
        for row in test:
            retrieved = retrieval_by_id[row["id"]]["retrieval_ids"]
            fixed_one.append(score(retrieved[:1], row["gold_answer"]))
            fixed_five.append(score(retrieved[:5], row["gold_answer"]))
            query_vector = vector_by_id[row["id"]]
            nearest, _nearest_vector = max(
                dev_vectors,
                key=lambda candidate: sum(
                    left * right for left, right in zip(query_vector, candidate[1])
                ),
            )
            predicted_bucket = nearest["gold_bucket"]
            chosen_k = 5 if predicted_bucket == "multi" else 1
            selected_scores = score(retrieved[:chosen_k], row["gold_answer"])
            adaptive_details.append(
                {
                    "id": row["id"],
                    "gold_bucket": row["gold_bucket"],
                    "predicted_bucket": predicted_bucket,
                    "nearest_dev_id": nearest["id"],
                    "chosen_k": chosen_k,
                    "scores": selected_scores,
                }
            )
        arms[arm_name] = {
            "dev_questions": len(dev),
            "test_questions": len(test),
            "dev_bucket_counts": {
                bucket: sum(row["gold_bucket"] == bucket for row in dev)
                for bucket in ("singleton", "multi")
            },
            "test_bucket_counts": {
                bucket: sum(row["gold_bucket"] == bucket for row in test)
                for bucket in ("singleton", "multi")
            },
            "fixed_top1": mean_scores(fixed_one),
            "fixed_top5": mean_scores(fixed_five),
            "adaptive": mean_scores([row["scores"] for row in adaptive_details]),
            "adaptive_cutoff_counts": {
                str(k): sum(row["chosen_k"] == k for row in adaptive_details)
                for k in (1, 5)
            },
            "adaptive_details": adaptive_details,
        }

    return {
        "benchmark": "ATM-Bench",
        "split": "full_list_recall",
        "measurement": "question_only_cardinality_selector",
        "development_rule": "sha256(question_id) mod 10 < 7",
        "selector": "nearest_development_question_cosine_label",
        "candidate_cutoffs": [1, 5],
        "embedder": args.embedder,
        "judge": {"used": False, "model": None},
        "arms": arms,
    }


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ground-truth", type=Path, required=True)
    ap.add_argument("--retrieval-result", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--embedder", default="fastembed:sentence-transformers/all-MiniLM-L6-v2")
    return ap


def main() -> int:
    args = parser().parse_args()
    result = run(args)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    for arm_name, arm in result["arms"].items():
        print(
            f"{arm_name}: adaptive_J={arm['adaptive']['jaccard']:.4f} "
            f"fixed1_J={arm['fixed_top1']['jaccard']:.4f} "
            f"fixed5_J={arm['fixed_top5']['jaccard']:.4f} "
            f"cutoffs={arm['adaptive_cutoff_counts']}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
