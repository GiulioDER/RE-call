from __future__ import annotations

import json
from pathlib import Path

from scripts.enterprise_rag_retrieval_compare import compare


def test_retrieval_compare_reports_exact_and_extra_deltas(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question_type": "project_related",
                "expected_doc_ids": ["d1", "d2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    baseline = tmp_path / "baseline.jsonl"
    baseline.write_text(json.dumps({"question_id": "q1", "document_ids": ["d1", "noise"]}) + "\n")
    candidate = tmp_path / "candidate.jsonl"
    candidate.write_text(json.dumps({"question_id": "q1", "document_ids": ["d1", "d2"]}) + "\n")

    report = compare(questions, baseline, candidate)

    assert report["overall"]["candidate_document_recall"] == 1.0
    assert report["overall"]["candidate_exact_coverage"] == 1.0
    assert report["overall"]["mean_extra_delta"] == -1.0
