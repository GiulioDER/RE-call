"""MM-4's Stage 1 report scores S against M4r as amendment 2 fixes it.

Invariants: MRR is 1 over the rank of the first item from a clue session, 0 when none; a question's
metrics are the mean over its option rotations; a question missing either arm stops the report;
the paired difference is M4r minus S.

Red proof, 2026-09-27, each against ``scripts/aml_mm4_stage1_report.py`` with this file unchanged
(``PYTHONDONTWRITEBYTECODE=1``):
- ``reciprocal_rank`` counting ranks from 0 (``enumerate(items)``):
  ``test_mrr_is_one_over_the_first_clue_rank`` failed on ``== 0.5``.
- ``per_question`` keeping the last rotation instead of the mean:
  ``test_a_question_is_the_mean_of_its_rotations`` failed on ``== 0.5``.
- ``paired`` taking S minus M4r: ``test_the_difference_is_m4r_minus_s`` failed on ``== 1.0``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest

SPEC = importlib.util.spec_from_file_location(
    "aml_mm4_stage1_report", Path(__file__).parents[1] / "scripts" / "aml_mm4_stage1_report.py"
)
assert SPEC and SPEC.loader
report = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = report
SPEC.loader.exec_module(report)

QUESTIONS = {("Brand", "q1"): {"question": "Which logo?", "clues": ["s2"]}}


def _row(arm: str, sessions: list[str]) -> dict:
    return {"scenario": "Brand", "question_id": "q1", "arm": arm,
            "items": [{"session_id": s, "content": "text"} for s in sessions]}


def test_mrr_is_one_over_the_first_clue_rank() -> None:
    items = [{"session_id": "s1"}, {"session_id": "s2"}, {"session_id": "s2"}]
    assert report.reciprocal_rank(items, {"s2"}) == 0.5
    assert report.reciprocal_rank(items, {"s9"}) == 0.0


def test_a_question_is_the_mean_of_its_rotations() -> None:
    rows = [_row("S", ["s2"]), _row("S", ["s1"]), _row("M4r", ["s2"]), _row("M4r", ["s2"])]
    values = report.per_question(rows, QUESTIONS)
    assert values[("Brand", "q1")]["S"]["mrr"] == 0.5
    assert values[("Brand", "q1")]["M4r"]["recall_at_10"] == 1.0


def test_a_question_missing_an_arm_stops_the_report() -> None:
    with pytest.raises(SystemExit, match="lacks an arm"):
        report.per_question([_row("S", ["s2"])], QUESTIONS)


def test_the_difference_is_m4r_minus_s() -> None:
    values = {("Brand", "q1"): {"S": {"mrr": 0.0}, "M4r": {"mrr": 1.0}}}
    result = report.paired(values, "mrr")
    assert result["diff"] == 1.0 and result["better"] == 1 and result["worse"] == 0
