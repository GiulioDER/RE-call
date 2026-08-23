"""The two ways the BEAM arm could publish a comparison that was never actually paired.

Both were found by the CCA audit of 2026-07-28, and both are of the same species as this repo's
two retractions: the run finishes, exits 0, and writes a well-formed artifact that describes a
measurement it did not make.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.beam.run import _coverage, _load_published

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness


def _published(tmp_path: Path, cells: dict[str, dict]) -> Path:
    path = tmp_path / "beam_1m_results.json"
    path.write_text(
        json.dumps(
            {
                "evaluations": [
                    {
                        "question_id": "q1",
                        "question": "when?",
                        "question_type": "temporal_reasoning",
                        "rubric": ["1985"],
                        "cutoff_results": cells,
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_published_answers_are_taken_from_the_named_cutoff_not_the_first_key(tmp_path: Path):
    """Mem0 ships more than one retrieval budget in one file; the headline is top_200.

    Verified before the fix: `_load_published` did `next(iter(cutoffs))`, so with `top_50`
    serialised first it re-judged Mem0's 50-memory answers and paired them against our k=200
    arm — a four-fold retrieval-budget mismatch on the one axis the suite exists to hold equal,
    recorded per-row as `cutoff_label` and never asserted on.
    """
    path = _published(
        tmp_path,
        {
            "top_50": {"score": 0.55, "generated_answer": "FIFTY", "memories_evaluated": 50},
            "top_200": {"score": 0.64, "generated_answer": "TWO HUNDRED", "memories_evaluated": 200},
        },
    )
    loaded = _load_published(path, 200)
    assert loaded["q1"]["generated_answer"] == "TWO HUNDRED"
    assert loaded["q1"]["cutoff_label"] == "top_200"
    assert loaded["q1"]["memories_evaluated"] == 200


def test_a_different_cutoff_selects_a_different_cell(tmp_path: Path):
    path = _published(
        tmp_path,
        {
            "top_50": {"score": 0.55, "generated_answer": "FIFTY", "memories_evaluated": 50},
            "top_200": {"score": 0.64, "generated_answer": "TWO HUNDRED", "memories_evaluated": 200},
        },
    )
    assert _load_published(path, 50)["q1"]["generated_answer"] == "FIFTY"


def test_a_missing_cutoff_is_an_error_rather_than_a_quiet_substitution(tmp_path: Path):
    # Falling back to "whatever is there" is what made the original defect invisible.
    path = _published(tmp_path, {"top_50": {"score": 0.55, "generated_answer": "FIFTY"}})
    with pytest.raises(SystemExit, match="top_200"):
        _load_published(path, 200)


def test_coverage_marks_a_short_run_incomplete_and_names_what_is_missing():
    # The field the RE-call arm's summary was missing entirely. `_run_pool` drops a question it
    # cannot score and continues, so without this a short run is indistinguishable from a
    # complete one except by reading `n` and knowing what it should have been.
    complete = _coverage({"q1", "q2"}, {"q1", "q2"})
    assert complete["complete"] is True

    short = _coverage({"q1", "q2", "q3"}, {"q1", "q3"})
    assert short["complete"] is False
    assert "q2" in short["missing_ids"]


# --- the invariant at the point of comparison ---------------------------------------------


def _artifact(tmp_path: Path, name: str, cutoff: int | None) -> Path:
    summary: dict = {"arm": name}
    if cutoff is not None:
        summary["cutoff"] = cutoff
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps({"summary": summary, "rows": []}), encoding="utf-8")
    return path


def test_pairing_refuses_two_arms_measured_at_different_retrieval_budgets(tmp_path: Path):
    """Selecting the right cell is necessary, not sufficient.

    Running with `--cutoff 50` against a stored k=200 artifact still aligns cleanly on question
    id and produces a well-formed McNemar table — the original defect's effect, reached by a
    different route. The budgets have to be checked where the arms are actually compared.
    """
    from benchmarks.beam.pair import require_same_cutoff

    a = _artifact(tmp_path, "recall", 200)
    b = _artifact(tmp_path, "mem0", 50)
    with pytest.raises(SystemExit, match="cutoff"):
        require_same_cutoff(a, b)


def test_pairing_accepts_two_arms_at_the_same_budget(tmp_path: Path):
    from benchmarks.beam.pair import require_same_cutoff

    require_same_cutoff(_artifact(tmp_path, "recall", 200), _artifact(tmp_path, "mem0", 200))


def test_an_unrecorded_budget_is_reported_not_treated_as_agreement(tmp_path: Path, capsys):
    # Artifacts predating the field must not pass silently — that would make the check weakest
    # exactly where it knows least.
    from benchmarks.beam.pair import require_same_cutoff

    require_same_cutoff(_artifact(tmp_path, "recall", 200), _artifact(tmp_path, "old", None))
    assert "could not be checked" in capsys.readouterr().out
