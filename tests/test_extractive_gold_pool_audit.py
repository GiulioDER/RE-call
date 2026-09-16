from __future__ import annotations

import hashlib
import json
from pathlib import Path

from scripts.audit_extractive_gold_pool import audit_pool


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _write_fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, Path]]:
    root = tmp_path / "memory"
    root.mkdir()
    queries: list[dict[str, object]] = []
    for index in range(1, 251):
        family = "project" if index % 2 else "feedback"
        source = f"recall/{family}-fixture-{index:03d}.md"
        span = f"Recorded fact number {index} has enough distinct words for deterministic testing."
        content = f"# Fixture {index}\n\n{span}\n"
        path = root / f"{family}-fixture-{index:03d}.md"
        path.write_text(content, encoding="utf-8", newline="\n")
        source_sha = _sha256_bytes(path.read_bytes())
        answer_sha = _sha256_bytes(span.encode("utf-8"))
        construction = "extractive_field" if index % 2 else "extractive_fallback"
        question = (
            f"What fact was recorded for fixture {index}?"
            if index % 2
            else f"What key statement is recorded in fixture {index}?"
        )
        queries.extend(
            [
                {
                    "id": f"extractive-positive-{index:03d}",
                    "query": question,
                    "expected_answerability": "answerable",
                    "gold_sources": [source],
                    "source_sha256": source_sha,
                    "gold_ordinal": 0,
                    "answer_span": span,
                    "answer_span_sha256": answer_sha,
                    "construction": construction,
                },
                {
                    "id": f"extractive-negative-{index:03d}",
                    "query": (
                        f"According to memory item ZXQXACT-{index:04d}, "
                        f"what fact was recorded for fixture {index}?"
                    ),
                    "expected_answerability": "unanswerable",
                    "gold_sources": [],
                    "source_sha256": None,
                    "gold_ordinal": None,
                    "answer_span": "NOT_FOUND",
                    "answer_span_sha256": _sha256_bytes(b"NOT_FOUND"),
                    "construction": "matched_absent_identifier_control",
                },
            ]
        )

    pool = tmp_path / "pool.json"
    pool.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "negative_prefix": "ZXQXACT",
                "answerable_queries": 250,
                "unanswerable_queries": 250,
                "queries": queries,
            }
        ),
        encoding="utf-8",
    )
    comparison = tmp_path / "comparison.json"
    comparison.write_text(
        json.dumps({"schema_version": 1, "queries": []}),
        encoding="utf-8",
    )
    return pool, comparison, {"recall": root}


def test_audit_accepts_positive_supervision_but_rejects_leaky_controls(
    tmp_path: Path,
) -> None:
    pool, comparison, roots = _write_fixture(tmp_path)

    result = audit_pool(pool, comparison, roots, enforce_frozen_hashes=False)

    assert result["decision"] == "GO_POSITIVE_SELECTOR_ONLY"
    assert result["gates"]["positive_supervision"]["passed"] is True
    assert result["gates"]["control_supervision"]["passed"] is False
    assert result["metrics"]["controls"]["identifier_rule"]["balanced_accuracy"] == 1.0
    assert result["metrics"]["controls"]["intro_rule"]["balanced_accuracy"] == 1.0
    assert result["metrics"]["split_overlaps"]["train_validation"]["source_sha256"] == 0
    assert result["metrics"]["split_overlaps"]["train_internal_test"]["source"] == 0


def test_audit_fails_positive_gate_on_source_hash_mismatch(tmp_path: Path) -> None:
    pool, comparison, roots = _write_fixture(tmp_path)
    payload = json.loads(pool.read_text(encoding="utf-8"))
    positive = next(
        row for row in payload["queries"] if row["expected_answerability"] == "answerable"
    )
    positive["source_sha256"] = "0" * 64
    pool.write_text(json.dumps(payload), encoding="utf-8")

    result = audit_pool(pool, comparison, roots, enforce_frozen_hashes=False)

    assert result["decision"] == "STOP_CURRENT_POOL"
    assert result["metrics"]["integrity"]["source_hash_mismatches"] == 1
    assert result["gates"]["positive_supervision"]["checks"]["source_hashes_match"] is False
