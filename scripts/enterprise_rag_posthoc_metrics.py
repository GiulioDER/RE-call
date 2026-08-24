"""Score generated EnterpriseRAG answer rows against official gold fields.

This script is deliberately separate from ``benchmarks.enterprise_rag``. The runner strips gold
fields before retrieval and answer generation. Run this only after an answer file is complete, in
the official evaluator phase.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping


def _load_questions(path: Path) -> dict[str, tuple[str, set[str]]]:
    questions: dict[str, tuple[str, set[str]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            questions[str(row["question_id"])] = (
                str(row.get("question_type", "unknown")),
                {str(value) for value in row.get("expected_doc_ids", [])},
            )
    return questions


def _load_answers(path: Path) -> dict[str, Mapping[str, Any]]:
    answers: dict[str, Mapping[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            question_id = str(row["question_id"])
            if question_id in answers:
                raise ValueError(f"{path}: duplicate question_id {question_id}")
            answers[question_id] = row
    return answers


def score(questions_path: Path, answers_path: Path) -> dict[str, Any]:
    questions = _load_questions(questions_path)
    answers = _load_answers(answers_path)
    pairs: list[dict[str, Any]] = []
    for question_id, (question_type, expected) in questions.items():
        row = answers.get(question_id)
        if row is None:
            continue
        predicted = {str(value) for value in row.get("document_ids", [])}
        answerable = bool(expected)
        hits = len(predicted & expected)
        pairs.append(
            {
                "question_id": question_id,
                "question_type": question_type,
                "document_recall": hits / len(expected) if answerable else None,
                "exact_doc_set_coverage": bool(expected) and expected <= predicted,
                "invalid_extra_documents": len(predicted - expected) if answerable else len(predicted),
                "returned_documents": len(predicted),
            }
        )

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for pair in pairs:
        by_category[pair["question_type"]].append(pair)

    def summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
        answerable = [row for row in rows if row["document_recall"] is not None]
        return {
            "questions": len(rows),
            "answerable_questions": len(answerable),
            "document_recall": (
                sum(float(row["document_recall"]) for row in answerable) / len(answerable)
                if answerable
                else None
            ),
            "exact_doc_set_coverage": (
                sum(bool(row["exact_doc_set_coverage"]) for row in answerable) / len(answerable)
                if answerable
                else None
            ),
            "mean_invalid_extra_documents": (
                sum(float(row["invalid_extra_documents"]) for row in answerable) / len(answerable)
                if answerable
                else None
            ),
        }

    return {
        "phase": "official_evaluator_posthoc",
        "questions": str(questions_path),
        "answers": str(answers_path),
        "question_count": len(pairs),
        "overall": summary(pairs),
        "by_category": {name: summary(rows) for name, rows in sorted(by_category.items())},
        "pairs": pairs,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", required=True, type=Path)
    parser.add_argument("--answers", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    report = score(args.questions, args.answers)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["overall"], indent=2))
    print(f"wrote posthoc metrics to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
