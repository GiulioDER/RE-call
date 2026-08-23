"""The analyser's pairing, tested at the case that actually went wrong.

`records.jsonl` holds every session the runner attempted, including the ones the gate threw out.
Reading it whole and calling the result "admitted pairs" produced a cost table on
`agent-ab-additive-001` that averaged 15 pairs whose sessions had died on `402 Insufficient
credits`. The trap rates were right, because those read the admitted-only scores, which is exactly
why nobody noticed: the wrong number sat beside several right ones.

The correction moved two conclusions in OPPOSITE directions, which is the argument for the test
rather than a careful re-read: wall time went from -19,971 ms at p=0.0029 (significant) to
-29,850 ms at p=0.0782 (not), and input tokens from p=0.0831 (not) to p=0.0494 (marginal).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_spec = importlib.util.spec_from_file_location(
    "agent_ab_analyze", REPO_ROOT / "scripts" / "agent_ab_analyze.py"
)
assert _spec and _spec.loader
analyze = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(analyze)

build_pairs = analyze.build_pairs


def _record(task: str, variant: str, **extra):
    return {"task_id": task, "variant": variant, "input_tokens": 100, **extra}


def _pair(task: str, **extra):
    return [_record(task, "recall_on", **extra), _record(task, "recall_off", **extra)]


def test_gate_discarded_tasks_are_excluded():
    """The regression. Mutation: ignoring `discarded` returns 2 pairs and inflates every cost."""

    records = _pair("kept#r1") + _pair("thrown#r1")
    pairs, dropped = build_pairs(records, {"thrown#r1"})
    assert set(pairs) == {"kept#r1"}
    assert dropped["gate_discarded"] == 2


def test_errored_sessions_are_excluded_not_averaged_in():
    """A session that failed has no measurement; averaging its zeros understates every cost."""

    records = _pair("good#r1") + [
        _record("bad#r1", "recall_on", error="api_error (HTTP 402)"),
        _record("bad#r1", "recall_off"),
    ]
    pairs, dropped = build_pairs(records, set())
    assert set(pairs) == {"good#r1"}
    assert dropped["errored"] == 1
    # The surviving half of a broken pair is unpaired, and a lone arm is not a comparison.
    assert dropped["unpaired"] == 1


def test_a_lone_arm_is_never_a_pair():
    records = [_record("solo#r1", "recall_on")]
    pairs, dropped = build_pairs(records, set())
    assert pairs == {}
    assert dropped["unpaired"] == 1


def test_no_admission_file_means_nothing_to_exclude():
    """A salvaged run has no admission.json and its records are already admitted-only.

    Mutation: treating `None` as "discard everything" would silently analyse zero pairs and report
    a clean, empty result.
    """

    records = _pair("a#r1") + _pair("b#r1")
    pairs, dropped = build_pairs(records, None)
    assert set(pairs) == {"a#r1", "b#r1"}
    assert dropped["gate_discarded"] == 0


def test_counts_reconcile_with_the_input():
    """Every record is either paired or accounted for in `dropped`; none vanish silently."""

    records = _pair("keep#r1") + _pair("drop#r1") + [
        _record("err#r1", "recall_on", error="boom"),
        _record("err#r1", "recall_off"),
    ]
    pairs, dropped = build_pairs(records, {"drop#r1"})
    used = len(pairs) * 2
    accounted = used + dropped["gate_discarded"] + dropped["errored"] + dropped["unpaired"]
    assert accounted == len(records)
