"""BUG-005: pair boundaries must preserve every observation or refuse ambiguity.

The primary runs these unchanged tests on baseline 8ef55caf0 before the repair.
Duplicate cases must fail because ValueError was not raised, then pass with the
guards in summarize_pairs and admit_pairs. The valid-pair controls also require
a deliberate mutation proof: replace each guard's `record.variant in arms` with
`bool(arms)`. That overbroad guard wrongly rejects the other legitimate arm.
"""

from __future__ import annotations

import pytest

from benchmarks.agent_ab.gate import admit_pairs
from benchmarks.agent_ab.schema import RECALL_OFF, RECALL_ON, SessionRecord
from benchmarks.agent_ab.summarize import summarize_pairs


def _record(task_id: str, variant: str, *, success: bool = True) -> SessionRecord:
    return SessionRecord(
        task_id=task_id,
        variant=variant,
        success=success,
        metadata={
            "recall_tools_available": ["mcp__recall__recall_search"]
            if variant == RECALL_ON else [],
        },
    )


@pytest.mark.parametrize("boundary", ["summary", "admission"])
@pytest.mark.parametrize("variant", [RECALL_ON, RECALL_OFF])
@pytest.mark.parametrize("order", ["first", "last", "reversed"])
def test_duplicate_observations_are_rejected(
    boundary: str, variant: str, order: str,
) -> None:
    """BUG-005 baseline: both public boundaries silently overwrite one observation."""
    first = _record("task-duplicate", variant)
    second = _record("task-duplicate", variant, success=False)
    other_variant = RECALL_OFF if variant == RECALL_ON else RECALL_ON
    other = _record("task-duplicate", other_variant)
    records = {
        "first": [first, second, other],
        "last": [other, first, second],
        "reversed": [second, other, first],
    }[order]

    with pytest.raises(ValueError, match="duplicate.*task-duplicate"):
        if boundary == "summary":
            summarize_pairs(records)
        else:
            admit_pairs(iter(records))


@pytest.mark.parametrize("boundary", ["summary", "admission"])
def test_identical_duplicate_is_rejected(boundary: str) -> None:
    """Identical records are still two observations; silently collapsing them loses provenance."""
    on = _record("task-duplicate", RECALL_ON)
    off = _record("task-duplicate", RECALL_OFF)
    with pytest.raises(ValueError, match="duplicate.*task-duplicate"):
        if boundary == "summary":
            summarize_pairs([on, off, on])
        else:
            admit_pairs(iter([on, off, on]))


@pytest.mark.parametrize("reverse", [False, True], ids=["on-first", "off-first"])
def test_single_pair_and_incomplete_reporting_are_preserved(reverse: bool) -> None:
    """Control proof: mutating either duplicate guard to bool(arms) rejects this valid pair."""
    on = _record("task-paired", RECALL_ON)
    off = _record("task-paired", RECALL_OFF, success=False)
    orphan = _record("task-orphan", RECALL_ON)
    records = [off, on, orphan] if reverse else [on, off, orphan]

    summary = None
    try:
        summary = summarize_pairs(records)
    except ValueError:
        pass
    assert summary is not None, "one observation per arm must remain summarizable"
    assert summary["record_count"] == 3
    assert summary["complete_pairs"] == 1
    assert summary["incomplete_task_ids"] == ["task-orphan"]
    assert summary["metrics"]["success"]["delta_mean"] == 1.0

    report = None
    try:
        report = admit_pairs(iter(records))
    except ValueError:
        pass
    assert report is not None, "one observation per arm must remain admissible"
    assert report.admitted == (on, off)
    assert report.discarded_task_ids == ("task-orphan",)
    assert any(
        verdict.task_id == "task-orphan"
        and verdict.variant == RECALL_OFF
        and not verdict.admitted
        and "no recall_off record" in verdict.reasons[0]
        for verdict in report.verdicts
    )
    admitted_summary = summarize_pairs(list(report.admitted))
    assert admitted_summary["complete_pairs"] == 1
    assert admitted_summary["metrics"]["success"]["delta_mean"] == 1.0
