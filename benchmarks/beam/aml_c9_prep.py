"""Flatten a BEAM split into the JSONL the live C9 BEAM probe reads (``aml_c9_probe.py``).

One line per conversation: its chat batches, each with the batch's date and its messages in
order, and its probing questions with type and rubric. Batches are BEAM's own top-level chat
batches, which is also where BEAM puts its ``time_anchor``; each batch becomes one AML Add, the
nearest public equivalent of the platform adding one session at a time.

Runs wherever ``pyarrow`` imports (the probe itself is stdlib only, so it can run beside the
served C9 without a virtualenv):

    python -m benchmarks.beam.aml_c9_prep --data 100K.parquet --size 100K --out beam100k.jsonl
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from benchmarks.beam.dataset import _flatten_turns, _parse_probing, _questions_of


def batch_millis(date: str) -> int | None:
    """BEAM anchors read like ``March-15-2024``; anything else carries no timestamp."""
    try:
        stamp = datetime.strptime(date.strip(), "%B-%d-%Y").replace(tzinfo=timezone.utc)
    except ValueError:
        return None
    return int(stamp.timestamp() * 1_000)


def conversation_record(row: dict, size: str, index: int) -> dict:
    batches = []
    for batch in row["chat"]:
        turns = _flatten_turns([batch])
        if not turns:
            continue
        date = next((turn["date"] for turn in turns if turn["date"]), "")
        base = batch_millis(date)
        messages = []
        for ordinal, turn in enumerate(turns):
            message = {"role": turn["role"], "content": turn["content"]}
            if base is not None:
                # One minute per turn keeps the order explicit inside a batch without leaving
                # the batch's calendar day for any realistic batch length.
                message["timestamp"] = base + ordinal * 60_000
            messages.append(message)
        batches.append({"date": date, "messages": messages})
    questions = [
        {
            "id": question.question_id,
            "type": question.question_type,
            "question": question.question,
            "rubric": question.rubric,
        }
        for question in _questions_of(_parse_probing(row["probing_questions"]), size, index)
    ]
    return {"conversation": index, "batches": batches, "questions": questions}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--size", default="100K")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    import pyarrow.parquet as pq

    table = pq.read_table(args.data)
    with args.out.open("w", encoding="utf-8") as handle:
        for index in range(table.num_rows):
            row = table.slice(index, 1).to_pylist()[0]
            handle.write(json.dumps(conversation_record(row, args.size, index), ensure_ascii=False))
            handle.write("\n")
    print(f"wrote {table.num_rows} conversations to {args.out}")


if __name__ == "__main__":
    main()
