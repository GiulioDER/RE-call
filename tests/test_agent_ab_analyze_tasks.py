"""The analyser's direction, pinned, because a sign error here is invisible in the output.

The trap benchmark's per-task helper counts a task as improved when its delta is NEGATIVE, because
a lower hazard rate is better. Task success runs the other way. A copy-paste between the two would
produce a table that reads perfectly and reports the headline backwards, and nothing about the
numbers would look wrong.

So these tests are about direction and about the degenerate cases, not about arithmetic.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.agent_ab_analyze_tasks import sign_test, success_by_task  # noqa: E402


def test_a_task_the_on_arm_wins_is_reported_as_improved():
    """The one that matters. On passes 4 of 4, off passes 0 of 4."""

    result = success_by_task({"t": [(True, False)] * 4})
    assert result["tasks"][0]["delta"] == 1.0
    assert result["improved"] == 1
    assert result["worsened"] == 0


def test_a_task_the_off_arm_wins_is_reported_as_worse():
    result = success_by_task({"t": [(False, True)] * 4})
    assert result["tasks"][0]["delta"] == -1.0
    assert result["improved"] == 0
    assert result["worsened"] == 1


def test_a_tie_is_neither_improved_nor_worse():
    result = success_by_task({"t": [(True, True), (False, False)]})
    assert (result["improved"], result["worsened"], result["unchanged"]) == (0, 0, 1)


def test_eight_tasks_all_improving_reach_significance():
    """The whole reason for eight rather than four.

    Four tasks bottom out at p=0.125 however large the effect. Eight reach 0.008, which is why the
    per-task view is the headline here and was only descriptive last time.
    """

    assert sign_test(4, 0) == 0.125
    assert sign_test(8, 0) < 0.01


def test_sign_test_is_none_when_nothing_moved():
    """A p-value over zero informative tasks is absent, not small."""

    assert sign_test(0, 0) is None


def test_the_cluster_interval_resamples_tasks_not_pairs():
    """Eight tasks that all improve by 1.0 have no spread, so no interval is invented."""

    identical = success_by_task({f"t{i}": [(True, False)] * 3 for i in range(8)})
    assert identical["cluster_ci"] is None
    assert identical["mean_delta"] == 1.0

    mixed = success_by_task(
        {"a": [(True, False)] * 3, "b": [(False, False)] * 3, "c": [(True, True)] * 3}
    )
    low, high = mixed["cluster_ci"]
    assert low <= mixed["mean_delta"] <= high
