"""DAT-003: incomplete Codex usage stays unknown at the public runner boundary.

The unchanged tests are intended to fail against the pre-fix ``_usage_totals`` in
``benchmarks.agent_ab.codex_exec``: empty/partial usage invents zeros, missing turns
leave partial totals, and invalid counts either invent a number or discard the session.
The measured controls require production mutations: replace accumulation with assignment
for full totals, and reject falsy counts for the explicit-zero case.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from benchmarks.agent_ab.codex_exec import CodexExecConfig, run_codex_case
from benchmarks.agent_ab.schema import RECALL_ON


def _record(monkeypatch, turns):
    """Supply captured CLI output without executing Codex or making an API call."""

    events = [{"type": "turn.completed", **turn} for turn in turns]
    stream = "\n".join(json.dumps(event) for event in events).encode()

    class CapturedProcess:
        returncode = 0

        async def communicate(self):
            return stream, b""

    async def captured_exec(*args, **kwargs):
        return CapturedProcess()

    monkeypatch.setattr(asyncio, "create_subprocess_exec", captured_exec)
    return asyncio.run(
        run_codex_case(
            {"task_id": "usage-case", "prompt": "Read the supplied fixture."},
            RECALL_ON,
            CodexExecConfig(),
        )
    )


def _usage(input_tokens, output_tokens, cached_input_tokens, reasoning_output_tokens):
    return {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cached_input_tokens": cached_input_tokens,
        "reasoning_output_tokens": reasoning_output_tokens,
    }


def test_empty_usage_does_not_measure_zero_tokens(monkeypatch):
    record = _record(monkeypatch, [{"usage": {}}])

    assert record.success and record.error is None
    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.total_tokens is None
    assert record.metadata["cached_input_tokens"] is None
    assert record.metadata["reasoning_output_tokens"] is None


def test_partial_usage_preserves_independent_unknown_counters(monkeypatch):
    record = _record(monkeypatch, [{"usage": {"input_tokens": 7, "cached_input_tokens": 0}}])

    assert record.input_tokens == 7
    assert record.output_tokens is None
    assert record.total_tokens is None
    assert record.metadata["cached_input_tokens"] == 0
    assert record.metadata["reasoning_output_tokens"] is None


@pytest.mark.parametrize("missing_first", [False, True], ids=["missing-last", "missing-first"])
def test_a_turn_without_usage_keeps_aggregate_unknown(monkeypatch, missing_first):
    turns = [{"usage": _usage(7, 3, 2, 1)}, {}]
    if missing_first:
        turns.reverse()
    record = _record(monkeypatch, turns)

    assert record.input_tokens is None
    assert record.output_tokens is None
    assert record.total_tokens is None
    assert record.metadata["cached_input_tokens"] is None
    assert record.metadata["reasoning_output_tokens"] is None


def test_missing_counter_does_not_hide_other_complete_totals(monkeypatch):
    record = _record(
        monkeypatch,
        [
            {"usage": _usage(7, 3, 2, 1)},
            {"usage": {"output_tokens": 4, "cached_input_tokens": 1, "reasoning_output_tokens": 2}},
        ],
    )

    assert record.input_tokens is None
    assert record.output_tokens == 7
    assert record.total_tokens is None
    assert record.metadata["cached_input_tokens"] == 3
    assert record.metadata["reasoning_output_tokens"] == 3


@pytest.mark.parametrize(
    "invalid",
    [-1, True, "4", None, 1.5, float("nan"), float("inf"), -float("inf")],
    ids=["negative", "boolean", "string", "null", "fractional", "nan", "infinity", "negative-infinity"],
)
def test_invalid_usage_does_not_invent_a_total_or_discard_the_session(monkeypatch, invalid):
    record = None
    error = None
    try:
        record = _record(monkeypatch, [{"usage": _usage(invalid, 3, invalid, invalid)}])
    except (ValueError, OverflowError) as exc:
        error = exc

    assert error is None, "invalid usage must remain unknown without losing the completed session"
    assert record is not None and record.success and record.error is None
    assert record.input_tokens is None
    assert record.output_tokens == 3
    assert record.total_tokens is None
    assert record.metadata["cached_input_tokens"] is None
    assert record.metadata["reasoning_output_tokens"] is None


def test_fully_measured_turns_still_sum_each_counter(monkeypatch):
    record = _record(
        monkeypatch,
        [{"usage": _usage(7, 3, 2, 1)}, {"usage": _usage(5.0, 4.0, 1.0, 2.0)}],
    )

    assert record.success and record.error is None
    assert record.input_tokens == 12
    assert record.output_tokens == 7
    assert record.total_tokens == 19
    assert record.metadata["cached_input_tokens"] == 3
    assert record.metadata["reasoning_output_tokens"] == 3


def test_explicit_zero_usage_is_still_measured(monkeypatch):
    record = _record(monkeypatch, [{"usage": _usage(0, 0, 0, 0)}])

    assert record.input_tokens == 0
    assert record.output_tokens == 0
    assert record.total_tokens == 0
    assert record.metadata["cached_input_tokens"] == 0
    assert record.metadata["reasoning_output_tokens"] == 0
