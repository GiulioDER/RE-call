from __future__ import annotations

import asyncio

import pytest

from benchmarks.codex_ab import (
    RECALL_OFF,
    RECALL_ON,
    SessionRecord,
    read_jsonl,
    run_paired,
    write_jsonl,
)
from benchmarks.codex_ab.summarize import summarize_pairs


def test_session_record_round_trip_preserves_missing_measurements() -> None:
    record = SessionRecord(
        task_id="task-1",
        variant=RECALL_ON,
        success=True,
        user_input="Remember the decision",
        input_tokens=10,
        output_tokens=4,
        recall_call_count=1,
        recall_latency_ms=12.5,
        metadata={"category": "supersession"},
    )

    restored = SessionRecord.from_mapping(record.to_dict())

    assert restored == record
    assert restored.total_tokens == 14
    assert restored.wall_time_ms is None


def test_jsonl_round_trip(tmp_path) -> None:
    records = [
        SessionRecord(task_id="task-1", variant=RECALL_ON, success=True),
        SessionRecord(task_id="task-1", variant=RECALL_OFF, success=False, error="timeout"),
    ]
    path = tmp_path / "records.jsonl"

    write_jsonl(path, records)

    assert read_jsonl(path) == records


def test_recall_off_cannot_claim_recall_calls() -> None:
    with pytest.raises(ValueError, match="recall_off"):
        SessionRecord(task_id="task-1", variant=RECALL_OFF, success=True, recall_call_count=1)


def test_run_paired_starts_both_arms_together() -> None:
    started: set[str] = set()
    release = asyncio.Event()

    async def runner(row: dict[str, str], variant: str) -> SessionRecord:
        started.add(variant)
        if started == {RECALL_ON, RECALL_OFF}:
            release.set()
        await asyncio.wait_for(release.wait(), timeout=1)
        return SessionRecord(
            task_id=row["task_id"],
            variant=variant,
            success=variant == RECALL_ON,
            input_tokens=10 if variant == RECALL_ON else 20,
            output_tokens=2,
            model_turns=1,
            wall_time_ms=5 if variant == RECALL_ON else 8,
            recall_call_count=1 if variant == RECALL_ON else 0,
        )

    records = asyncio.run(run_paired([{"task_id": "task-1"}], runner))

    assert {(record.task_id, record.variant) for record in records} == {
        ("task-1", RECALL_ON),
        ("task-1", RECALL_OFF),
    }
    assert started == {RECALL_ON, RECALL_OFF}


def test_summary_excludes_incomplete_pairs_and_reports_deltas() -> None:
    records = [
        SessionRecord(
            task_id="task-1",
            variant=RECALL_ON,
            success=True,
            input_tokens=10,
            output_tokens=2,
            wall_time_ms=5,
            recall_call_count=1,
            recall_latency_ms=2,
        ),
        SessionRecord(
            task_id="task-1",
            variant=RECALL_OFF,
            success=False,
            input_tokens=20,
            output_tokens=2,
            wall_time_ms=8,
        ),
        SessionRecord(task_id="task-2", variant=RECALL_ON, success=True),
    ]

    summary = summarize_pairs(records)

    assert summary["complete_pairs"] == 1
    assert summary["incomplete_task_ids"] == ["task-2"]
    assert summary["metrics"]["success"]["delta_mean"] == 1.0
    assert summary["metrics"]["total_tokens"]["delta_mean"] == -10
    assert summary["metrics"]["wall_time_ms"]["delta_mean"] == -3
    assert summary["metrics"]["recall_call_count"]["delta_mean"] == 1
    assert summary["metrics"]["recall_latency_ms"]["n"] == 0
