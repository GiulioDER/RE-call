"""Offline tests for the LOCOMO answer-key exclusion.

Everything here is synthetic and local: a tiny locomo-shaped file and a tiny audit-shaped file,
both written into ``tmp_path``. Nothing is fetched. The vendored real audit data is exercised by
one test that skips when ``locomo10.json`` is absent (it is gitignored — 2.8 MB of third-party
CC BY-NC data).

The test that matters most is `test_one_based_ids_are_rejected`. An off-by-one in the id mapping
still excludes exactly the right NUMBER of questions, just the wrong ones, and the corrected score
it produces is fabricated in a direction nobody can see from the output. The whole module exists
to make that failure loud, so it is tested as a failure and not only as a success.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from benchmarks.locomo_audit import (
    CITATION_ONLY_ERROR,
    audited_report,
    bad_question_ids,
    parse_audit_id,
    split_by_audit,
    verified_mapping,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_LOCOMO = REPO_ROOT / "locomo10.json"
VENDORED_AUDIT = REPO_ROOT / "benchmarks" / "audit_data" / "locomo_errors.json"


def _locomo_file(path: Path) -> Path:
    """Two conversations whose qa questions and answers are all distinct.

    Distinctness is the point: it means an index that is off by one lands on text that does not
    match, which is what the verification pass is supposed to notice.
    """
    conversations = [
        {
            "sample_id": "conv-26",
            "qa": [
                {"question": "Q zero of A?", "answer": "A zero", "category": 1},
                {"question": "Q one of A?", "answer": "A one", "category": 2},
                {"question": "Q two of A?", "answer": "A two", "category": 1},
            ],
        },
        {
            "sample_id": "conv-30",
            "qa": [
                {"question": "Q zero of B?", "answer": "B zero", "category": 4},
                {"question": "Q one of B?", "answer": "B one", "category": 3},
            ],
        },
    ]
    path.write_text(json.dumps(conversations), encoding="utf-8")
    return path


def _entry(
    audit_id: str, question: str, golden: str | None, error_type: str = "HALLUCINATION"
) -> dict[str, Any]:
    entry: dict[str, Any] = {
        "question_id": audit_id,
        "question": question,
        "category": 1,
        "error_type": error_type,
        "correct_answer": "something else",
    }
    if golden is not None:
        entry["golden_answer"] = golden
    return entry


def _audit_file(path: Path, entries: list[dict[str, Any]]) -> Path:
    path.write_text(json.dumps(entries), encoding="utf-8")
    return path


@pytest.fixture()
def locomo(tmp_path: Path) -> Path:
    return _locomo_file(tmp_path / "locomo.json")


def test_parse_audit_id_is_anchored() -> None:
    assert parse_audit_id("locomo_3_qa68") == (3, 68)
    assert parse_audit_id(" locomo_0_qa0 ") == (0, 0)
    for bad in ("locomo_3_q68", "xlocomo_3_qa68", "locomo_3_qa68x", "locomo__qa1"):
        with pytest.raises(ValueError, match="does not have the form"):
            parse_audit_id(bad)


def test_zero_based_ids_map_and_verify(locomo: Path, tmp_path: Path) -> None:
    """The real convention: ``qa{n}`` is ``qa[n]``, and our id is ``{sample_id}:{n}``."""
    audit = _audit_file(
        tmp_path / "audit.json",
        [
            _entry("locomo_0_qa1", "Q one of A?", "A one"),
            _entry("locomo_1_qa0", "Q zero of B?", "B zero"),
        ],
    )

    mapped = verified_mapping(locomo, audit)

    assert set(mapped) == {"conv-26:1", "conv-30:0"}
    assert mapped["conv-26:1"]["question_id"] == "locomo_0_qa1"


def test_one_based_ids_are_rejected(locomo: Path, tmp_path: Path) -> None:
    """An audit file written under the OTHER convention must raise, not silently shift by one.

    Same two findings as the test above, renumbered 1-based (``qa2`` meaning ``qa[1]``). Read
    0-based they point at neighbouring questions whose text does not match, which is exactly the
    off-by-one this verification exists to catch.
    """
    audit = _audit_file(
        tmp_path / "audit.json",
        [
            _entry("locomo_0_qa2", "Q one of A?", "A one"),
            _entry("locomo_1_qa1", "Q zero of B?", "B zero"),
        ],
    )

    with pytest.raises(ValueError) as excinfo:
        verified_mapping(locomo, audit)
    message = str(excinfo.value)
    assert "2 of 2 entries failed verification" in message
    assert "locomo_0_qa2" in message
    assert "locomo_1_qa1" in message


def test_mismatched_question_text_raises(locomo: Path, tmp_path: Path) -> None:
    audit = _audit_file(
        tmp_path / "audit.json",
        [
            _entry("locomo_0_qa0", "Q zero of A?", "A zero"),
            _entry("locomo_0_qa1", "A question that is not in the dataset at all?", "A one"),
        ],
    )

    with pytest.raises(ValueError, match="question text does not match conv-26:1"):
        verified_mapping(locomo, audit)


def test_mismatched_golden_answer_raises(locomo: Path, tmp_path: Path) -> None:
    """The question can line up and the ANSWER still disagree — a revised dataset, say."""
    audit = _audit_file(
        tmp_path / "audit.json", [_entry("locomo_0_qa1", "Q one of A?", "a different gold")]
    )

    with pytest.raises(ValueError, match="golden answer does not match conv-26:1"):
        verified_mapping(locomo, audit)


def test_entry_without_a_golden_answer_is_verified_on_the_question_alone(
    locomo: Path, tmp_path: Path
) -> None:
    audit = _audit_file(tmp_path / "audit.json", [_entry("locomo_0_qa2", "Q two of A?", None)])

    assert set(verified_mapping(locomo, audit)) == {"conv-26:2"}


def test_out_of_range_indices_raise(locomo: Path, tmp_path: Path) -> None:
    audit = _audit_file(
        tmp_path / "audit.json",
        [
            _entry("locomo_9_qa0", "Q zero of A?", "A zero"),
            _entry("locomo_1_qa7", "Q zero of B?", "B zero"),
        ],
    )

    with pytest.raises(ValueError) as excinfo:
        verified_mapping(locomo, audit)
    message = str(excinfo.value)
    assert "conversation 9 is not in the dataset" in message
    assert "qa index 7 is out of range for conv-30" in message


def test_duplicate_mapping_raises(locomo: Path, tmp_path: Path) -> None:
    audit = _audit_file(
        tmp_path / "audit.json",
        [
            _entry("locomo_0_qa1", "Q one of A?", "A one"),
            _entry("locomo_0_qa1", "Q one of A?", "A one"),
        ],
    )

    with pytest.raises(ValueError, match="already claimed by another entry"):
        verified_mapping(locomo, audit)


def test_citation_only_entries_are_excluded_by_default(locomo: Path, tmp_path: Path) -> None:
    """The 57-vs-99 split: a wrong citation does not corrupt an answer-graded score."""
    audit = _audit_file(
        tmp_path / "audit.json",
        [
            _entry("locomo_0_qa0", "Q zero of A?", "A zero", CITATION_ONLY_ERROR),
            _entry("locomo_0_qa1", "Q one of A?", "A one", "TEMPORAL_ERROR"),
        ],
    )

    assert bad_question_ids(locomo, audit) == {"conv-26:1"}
    assert bad_question_ids(locomo, audit, include_citation_only=True) == {
        "conv-26:0",
        "conv-26:1",
    }


def _outcome(question_id: str, *, correct: bool = True) -> dict[str, Any]:
    return {
        "question_id": question_id,
        "category": "cat1",
        "is_adversarial": False,
        "context": "",
        "answer": "a",
        "abstained": False,
        "correct": correct,
    }


def test_split_by_audit_partitions_without_loss() -> None:
    outcomes = [_outcome(f"conv-26:{i}") for i in range(5)]

    clean, excluded = split_by_audit(outcomes, {"conv-26:1", "conv-26:3", "conv-99:0"})

    assert [o["question_id"] for o in clean] == ["conv-26:0", "conv-26:2", "conv-26:4"]
    assert [o["question_id"] for o in excluded] == ["conv-26:1", "conv-26:3"]
    # Nothing lost and nothing invented: `conv-99:0` is a bad id no outcome carries, and it
    # contributes no row to either side.
    assert len(clean) + len(excluded) == len(outcomes)
    assert {o["question_id"] for o in clean + excluded} == {o["question_id"] for o in outcomes}


def test_split_by_audit_with_no_bad_ids_keeps_everything() -> None:
    outcomes = [_outcome("conv-26:0"), _outcome("conv-26:1")]

    clean, excluded = split_by_audit(outcomes, set())

    assert clean == outcomes
    assert excluded == []


def test_audited_report_gives_both_numbers(locomo: Path, tmp_path: Path) -> None:
    audit = _audit_file(
        tmp_path / "audit.json", [_entry("locomo_0_qa1", "Q one of A?", "A one", "HALLUCINATION")]
    )
    doc = {
        "arm": "recall",
        "aggregate": {"answerable_accuracy": {"n": 3, "rate": 0.6667, "ci95": [0.2, 0.94]}},
        "outcomes": [
            _outcome("conv-26:0", correct=True),
            _outcome("conv-26:1", correct=False),  # the audited-bad one
            _outcome("conv-26:2", correct=True),
        ],
    }

    report = audited_report(doc, locomo, audit)

    assert report["n_all"] == 3
    assert report["n_clean"] == 2
    assert report["excluded_ids"] == ["conv-26:1"]
    assert report["aggregate_all"]["answerable_accuracy"]["n"] == 3
    assert report["aggregate_clean"]["answerable_accuracy"]["n"] == 2
    assert report["aggregate_clean"]["answerable_accuracy"]["rate"] == 1.0


@pytest.mark.skipif(not REAL_LOCOMO.exists(), reason="locomo10.json is gitignored; fetch it first")
def test_vendored_audit_verifies_against_the_real_dataset() -> None:
    """The real join: all 156 vendored entries verify, and 99 of them are score-corrupting.

    99 is the audit's own headline (99 of 1,540, ceiling 93.57%) and 57 is its citation-only
    remainder; if either drifts, the vendored file changed under us and the published correction
    is no longer the one this repo documented.
    """
    mapped = verified_mapping(REAL_LOCOMO, VENDORED_AUDIT)

    assert len(mapped) == 156
    assert len(bad_question_ids(REAL_LOCOMO, VENDORED_AUDIT)) == 99
    assert len(bad_question_ids(REAL_LOCOMO, VENDORED_AUDIT, include_citation_only=True)) == 156
    assert "conv-26:1" in mapped  # locomo_0_qa1, "When did Melanie paint a sunrise?"
