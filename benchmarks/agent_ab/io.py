"""JSONL persistence for paired benchmark records."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .schema import SessionRecord


def write_jsonl(path: str | Path, records: Iterable[SessionRecord]) -> None:
    """Write canonical session records as one JSON object per line."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(record.to_dict(), sort_keys=True) for record in records]
    target.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def read_jsonl(path: str | Path) -> list[SessionRecord]:
    """Read and validate canonical session records from a JSONL file."""

    target = Path(path)
    records: list[SessionRecord] = []
    for line_number, line in enumerate(target.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON on line {line_number} of {target}") from error
        if not isinstance(value, dict):
            raise TypeError(f"JSONL line {line_number} of {target} must be an object")
        records.append(SessionRecord.from_mapping(value))
    return records
