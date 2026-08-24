from __future__ import annotations

import json
from pathlib import Path

from scripts.enterprise_rag_experiment import build_plan
from scripts.enterprise_rag_posthoc_metrics import score

import pytest

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness


def test_posthoc_metrics_reads_gold_only_after_answers_exist(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    questions.write_text(
        json.dumps(
            {
                "question_id": "q1",
                "question_type": "project_related",
                "question": "Q",
                "expected_doc_ids": ["d1", "d2"],
            }
        )
        + "\n",
        encoding="utf-8",
    )
    answers = tmp_path / "answers.jsonl"
    answers.write_text(json.dumps({"question_id": "q1", "answer": "A", "document_ids": ["d1", "noise"]}) + "\n")

    report = score(questions, answers)

    assert report["phase"] == "official_evaluator_posthoc"
    assert report["overall"]["document_recall"] == 0.5
    assert report["overall"]["mean_invalid_extra_documents"] == 1.0


def test_experiment_plan_is_serial_and_uses_isolated_names(tmp_path: Path) -> None:
    questions = tmp_path / "questions.jsonl"
    documents = tmp_path / "documents.zip"
    ids = tmp_path / "ids.txt"
    for path in (questions, documents, ids):
        path.write_text("fixture", encoding="utf-8")
    args = type(
        "Args",
        (),
        {
            "questions": questions,
            "documents": documents,
            "question_ids": ids,
            "out_dir": tmp_path / "out",
            "dsn": "postgresql://isolated",
            "table_prefix": "bench_test",
            "tenant_prefix": "tenant_test",
            "embedder": "voyage:voyage-4-large",
            "k": "5,8",
            "candidate_k": "100,200",
            "sparse_backend": "lexical",
            "reranker": "none,voyage:rerank-2.5",
            "retrieval_captures": 3,
        },
    )()

    plan = build_plan(args)

    assert plan["runtime_policy"]["gold_fields_used_at_runtime"] is False
    assert len(plan["arms"]) == 8
    assert len({(arm["table"], arm["tenant"]) for arm in plan["arms"]}) == 8
