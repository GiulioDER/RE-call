"""Measure ATM list answer retention from saved retrieval rankings.

The probe is deterministic. It treats the first k retrieved evidence IDs as the answer, which
matches the official ATM list answer format without invoking an answer model or judge.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path
from typing import Any

K_VALUES = (1, 5, 10, 25, 50, 100)
LIST_SPLIT_PATTERN = re.compile(r"\s*(?:,|;|\band\b|/)\s*", re.IGNORECASE)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def parse_answer(value: Any) -> list[str]:
    if not isinstance(value, str):
        raise ValueError("list_recall answer must be a string")
    parts = LIST_SPLIT_PATTERN.split(value)
    answer: list[str] = []
    seen: set[str] = set()
    for part in parts:
        token = re.sub(r"\s+", " ", part.strip().lower())
        token = re.sub(r"\b(\d+)(st|nd|rd|th)\b", r"\1", token)
        token = re.sub(r"\([^)]+\)", "", token)
        token = re.sub(r"\s+", " ", token)
        token = re.sub(r"(\d),(\d)", r"\1\2", token)
        token = token.strip("\t\n\r \\\"'`.,;:!?()[]{}")
        if token and token not in seen:
            seen.add(token)
            answer.append(token)
    return answer


def score(predicted: list[str], gold: list[str]) -> dict[str, float]:
    predicted_set = set(predicted)
    gold_set = set(gold)
    intersection = len(predicted_set & gold_set)
    union = len(predicted_set | gold_set)
    return {
        "jaccard": intersection / union if union else 1.0,
        "gold_answer_containment": intersection / len(gold_set) if gold_set else 0.0,
        "answer_hit": 1.0 if predicted_set & gold_set else 0.0,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    ground_truth = load_json(args.ground_truth)
    retrieval = load_json(args.retrieval_result)
    by_id = {str(row["id"]): row for row in ground_truth if row.get("qtype") == "list_recall"}
    if not by_id:
        raise ValueError("ground truth contains no list_recall rows")

    arms: dict[str, Any] = {}
    for arm_name, arm in retrieval["arms"].items():
        details = []
        for row in arm["details"]:
            question_id = str(row["id"])
            if question_id not in by_id:
                continue
            gold = parse_answer(by_id[question_id]["answer"])
            retrieved = [str(value) for value in row["retrieval_ids"]]
            scores: dict[str, dict[str, float]] = {}
            for k in K_VALUES:
                scores[str(k)] = score(retrieved[:k], gold)
            details.append(
                {
                    "id": question_id,
                    "gold_answer_ids": gold,
                    "retrieved_ids": retrieved,
                    "scores": scores,
                }
            )
        summary: dict[str, Any] = {}
        for k in K_VALUES:
            rows = [row["scores"][str(k)] for row in details]
            summary[str(k)] = {
                metric: statistics.fmean(row[metric] for row in rows) for metric in rows[0]
            }
        arms[arm_name] = {"questions": len(details), "summary": summary, "details": details}

    return {
        "benchmark": "ATM-Bench",
        "split": "hard",
        "measurement": "deterministic_list_answer_retention",
        "gold_parser": "comma_or_newline_split_trim_terminal_punctuation",
        "prediction": "retrieval_ids[:k]",
        "judge": {"used": False, "model": None},
        "retrieval_result": str(args.retrieval_result),
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
        s5 = arm["summary"]["5"]
        s10 = arm["summary"]["10"]
        print(
            f"{arm_name}: J@5={s5['jaccard']:.4f} containment@5="
            f"{s5['gold_answer_containment']:.4f} J@10={s10['jaccard']:.4f} "
            f"containment@10={s10['gold_answer_containment']:.4f}"
        )
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
