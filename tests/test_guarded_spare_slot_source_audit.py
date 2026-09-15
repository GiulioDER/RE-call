from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts.score_guarded_spare_slot_source_audit import score_source_audit


def _write(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8", newline="\n")


def _fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    pool_path = tmp_path / "pool.json"
    capture_path = tmp_path / "capture.json"
    review_path = tmp_path / "review.json"
    pool = {
        "queries": [
            {
                "id": "q1",
                "expected_answerability": "answerable",
                "gold_sources": ["gold-one.md"],
            },
            {
                "id": "q2",
                "expected_answerability": "answerable",
                "gold_sources": ["gold-two.md"],
            },
            {"id": "q3", "expected_answerability": "unanswerable", "gold_sources": []},
        ]
    }
    _write(pool_path, pool)
    pool_digest = hashlib.sha256(pool_path.read_bytes()).hexdigest()
    rows = [
        {
            "query_id": "q1",
            "expected_answerability": "answerable",
            "triggered": True,
            "base_hashes": ["h1"],
            "candidate_hashes": ["h1", "h2"],
            "review_evidence_ids_by_hash": {"h1": "E01", "h2": "E02"},
        },
        {
            "query_id": "q2",
            "expected_answerability": "answerable",
            "triggered": True,
            "base_hashes": ["h3"],
            "candidate_hashes": ["h3", "h4"],
            "review_evidence_ids_by_hash": {"h3": "E01", "h4": "E02"},
        },
        {
            "query_id": "q3",
            "expected_answerability": "unanswerable",
            "triggered": True,
            "base_hashes": ["h5"],
            "candidate_hashes": ["h5", "h6"],
            "review_evidence_ids_by_hash": {"h5": "E01", "h6": "E02"},
        },
    ]
    capture = {"query_pool_sha256": pool_digest, "rows": rows}
    _write(capture_path, capture)
    capture_digest = hashlib.sha256(capture_path.read_bytes()).hexdigest()
    review = {
        "query_pool_sha256": pool_digest,
        "capture_sha256": capture_digest,
        "queries": [
            {
                "query_id": "q1",
                "evidence_items": [
                    {"evidence_id": "E01", "source": "wrong.md"},
                    {"evidence_id": "E02", "source": "gold-one.md"},
                ],
            },
            {
                "query_id": "q2",
                "evidence_items": [
                    {"evidence_id": "E01", "source": "gold-two.md"},
                    {"evidence_id": "E02", "source": "wrong.md"},
                ],
            },
            {
                "query_id": "q3",
                "evidence_items": [
                    {"evidence_id": "E01", "source": "wrong.md"},
                    {"evidence_id": "E02", "source": "also-wrong.md"},
                ],
            },
        ],
    }
    _write(review_path, review)
    return pool_path, capture_path, review_path


def test_source_audit_scores_only_frozen_source_labels(tmp_path: Path) -> None:
    """Red proof for ``score_source_audit``: an off-by-one slice reported zero additions."""
    pool, capture, review = _fixture(tmp_path)

    result = score_source_audit(pool, capture, review)

    assert result["decision"] == "INVALID_REVIEW_INSTRUMENT"
    assert result["promotion_eligible"] is False
    assert result["integrity"]["human_labels_consumed"] is False
    assert result["metrics"] == {
        "answerable_triggered": 2,
        "answerable_base_source_hits": 1,
        "answerable_candidate_source_hits": 2,
        "answerable_source_hit_gains": 1,
        "answerable_source_hit_losses": 0,
        "answerable_added_gold_items": 1,
        "answerable_added_non_gold_items": 1,
        "unanswerable_triggered": 1,
        "unanswerable_added_non_gold_items": 1,
        "total_added_items": 3,
        "added_gold_source_precision": pytest.approx(1 / 3),
    }


def test_source_audit_refuses_a_changed_base_prefix(tmp_path: Path) -> None:
    pool, capture, review = _fixture(tmp_path)
    value = json.loads(capture.read_text(encoding="utf-8"))
    value["rows"][0]["candidate_hashes"] = ["h2", "h1"]
    _write(capture, value)
    review_value = json.loads(review.read_text(encoding="utf-8"))
    review_value["capture_sha256"] = hashlib.sha256(capture.read_bytes()).hexdigest()
    _write(review, review_value)

    with pytest.raises(ValueError, match="base prefix"):
        score_source_audit(pool, capture, review)


def test_source_audit_refuses_a_capture_digest_mismatch(tmp_path: Path) -> None:
    pool, capture, review = _fixture(tmp_path)
    value = json.loads(review.read_text(encoding="utf-8"))
    value["capture_sha256"] = "0" * 64
    _write(review, value)

    with pytest.raises(ValueError, match="capture digest"):
        score_source_audit(pool, capture, review)
