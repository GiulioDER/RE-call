"""Compare RE-call EnterpriseRAG retrieval answer files by official question category."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


def _rows(path: Path) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row["question_id"])
            if question_id in result:
                raise ValueError(f"{path}: duplicate question_id {question_id}")
            result[question_id] = row
    return result


def _questions(path: Path) -> dict[str, tuple[str, set[str]]]:
    result: dict[str, tuple[str, set[str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            result[str(row["question_id"])] = (
                str(row.get("question_type", "unknown")),
                {str(value) for value in row.get("expected_doc_ids", [])},
            )
    return result


def _manifest(path: Path | None) -> Mapping[str, Any] | None:
    if path is None:
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def compare(
    questions_path: Path,
    baseline_path: Path,
    candidate_path: Path,
    *,
    baseline_manifest: Path | None = None,
    candidate_manifest: Path | None = None,
) -> dict[str, Any]:
    questions = _questions(questions_path)
    baseline = _rows(baseline_path)
    candidate = _rows(candidate_path)
    ids = set(baseline) & set(candidate) & set(questions)
    if not ids:
        raise ValueError("no common question ids between questions and answer files")

    pairs: list[dict[str, Any]] = []
    for question_id in sorted(ids):
        category, expected = questions[question_id]
        before = {str(value) for value in baseline[question_id].get("document_ids", [])}
        after = {str(value) for value in candidate[question_id].get("document_ids", [])}
        pairs.append(
            {
                "question_id": question_id,
                "question_type": category,
                "baseline_recall": len(before & expected) / len(expected) if expected else None,
                "candidate_recall": len(after & expected) / len(expected) if expected else None,
                "recall_delta": (
                    len(after & expected) / len(expected) - len(before & expected) / len(expected)
                    if expected
                    else None
                ),
                "baseline_exact": bool(expected) and expected <= before,
                "candidate_exact": bool(expected) and expected <= after,
                "baseline_extra": len(before - expected) if expected else len(before),
                "candidate_extra": len(after - expected) if expected else len(after),
            }
        )

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_category[pair["question_type"]].append(pair)

    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        answerable = [row for row in rows if row["baseline_recall"] is not None]
        recall_deltas = [float(row["recall_delta"]) for row in answerable]
        return {
            "count": len(rows),
            "answerable_count": len(answerable),
            "baseline_document_recall": (
                sum(float(row["baseline_recall"]) for row in answerable) / len(answerable)
                if answerable else None
            ),
            "candidate_document_recall": (
                sum(float(row["candidate_recall"]) for row in answerable) / len(answerable)
                if answerable else None
            ),
            "mean_recall_delta": sum(recall_deltas) / len(recall_deltas) if recall_deltas else None,
            "baseline_exact_coverage": sum(bool(row["baseline_exact"]) for row in answerable) / len(answerable) if answerable else None,
            "candidate_exact_coverage": sum(bool(row["candidate_exact"]) for row in answerable) / len(answerable) if answerable else None,
            "mean_extra_delta": (
                sum(row["candidate_extra"] - row["baseline_extra"] for row in rows) / len(rows)
                if rows else 0.0
            ),
            "questions_with_recall_gain": sum(delta > 0 for delta in recall_deltas),
            "questions_with_recall_loss": sum(delta < 0 for delta in recall_deltas),
        }

    return {
        "phase": "official_evaluator_posthoc",
        "questions": str(questions_path),
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "runtime": {
            "baseline": _manifest(baseline_manifest),
            "candidate": _manifest(candidate_manifest),
        },
        "common_questions": len(pairs),
        "overall": summary(pairs),
        "by_category": {category: summary(rows) for category, rows in sorted(by_category.items())},
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--candidate-manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = compare(
        args.questions,
        args.baseline,
        args.candidate,
        baseline_manifest=args.baseline_manifest,
        candidate_manifest=args.candidate_manifest,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"wrote retrieval comparison report to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
