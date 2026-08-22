"""Concurrent paired execution for a RE-call A/B benchmark."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Iterable, Mapping
from typing import Any

from .schema import RECALL_OFF, RECALL_ON, SessionRecord

Runner = Callable[
    [Mapping[str, Any], str], Awaitable[SessionRecord | Mapping[str, Any]]
]


def _error_record(row: Mapping[str, Any], variant: str, error: BaseException) -> SessionRecord:
    return SessionRecord(
        task_id=str(row["task_id"]),
        variant=variant,
        success=False,
        user_input=str(row.get("user_input", "")),
        error=f"{type(error).__name__}: {error}",
    )


async def _run_one(row: Mapping[str, Any], variant: str, runner: Runner) -> SessionRecord:
    try:
        result = await runner(row, variant)
        if isinstance(result, SessionRecord):
            record = result
        else:
            record_data = dict(result)
            record_data["task_id"] = row["task_id"]
            record_data["variant"] = variant
            record = SessionRecord.from_mapping(record_data)
        if record.task_id != str(row["task_id"]) or record.variant != variant:
            raise ValueError("runner returned a record for a different task or variant")
        return record
    except Exception as error:
        return _error_record(row, variant, error)


async def run_paired(
    rows: Iterable[Mapping[str, Any]],
    runner: Runner,
    *,
    pair_concurrency: int = 1,
) -> list[SessionRecord]:
    """Run both arms for each row, starting the two arms together.

    ``pair_concurrency`` limits the number of simultaneous task pairs. The two
    variants inside one pair always run concurrently. The default of one pair
    avoids turning host contention into an untracked benchmark variable.
    """

    if pair_concurrency < 1:
        raise ValueError("pair_concurrency must be at least one")
    materialized = [dict(row) for row in rows]
    task_ids = [str(row.get("task_id", "")) for row in materialized]
    if any(not task_id for task_id in task_ids):
        raise ValueError("every row must contain a nonempty task_id")
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("task_id values must be unique within one paired run")

    semaphore = asyncio.Semaphore(pair_concurrency)

    async def run_pair(row: Mapping[str, Any]) -> list[SessionRecord]:
        async with semaphore:
            return list(
                await asyncio.gather(
                    _run_one(row, RECALL_ON, runner),
                    _run_one(row, RECALL_OFF, runner),
                )
            )

    pairs = await asyncio.gather(*(run_pair(row) for row in materialized))
    return [record for pair in pairs for record in pair]
