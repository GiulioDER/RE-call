from __future__ import annotations

import json
from pathlib import Path

from scripts.enterprise_rag_compare import compare

import pytest

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness


def _write(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"questions": rows}), encoding="utf-8")


def test_compare_reports_paired_category_deltas(tmp_path: Path) -> None:
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    _write(
        baseline,
        [
            {
                "question_id": "q1",
                "question_type": "project_related",
                "answer_correct": False,
                "completeness_pct": 100,
                "document_recall_pct": 50,
                "invalid_extra_docs": 2,
            },
            {
                "question_id": "q2",
                "question_type": "basic",
                "answer_correct": True,
                "completeness_pct": 80,
                "document_recall_pct": 100,
                "invalid_extra_docs": 1,
            },
        ],
    )
    _write(
        candidate,
        [
            {
                "question_id": "q1",
                "question_type": "project_related",
                "answer_correct": True,
                "completeness_pct": 75,
                "document_recall_pct": 100,
                "invalid_extra_docs": 1,
            },
            {
                "question_id": "q2",
                "question_type": "basic",
                "answer_correct": True,
                "completeness_pct": 90,
                "document_recall_pct": 100,
                "invalid_extra_docs": 3,
            },
        ],
    )

    report = compare(baseline, candidate, seed=1)

    assert report["question_count"] == 2
    assert report["overall"]["score_delta"] == 42.5
    assert report["by_category"]["project_related"]["candidate_wins"] == 1
    assert report["by_category"]["basic"]["invalid_extra_docs_delta"] == 2.0
