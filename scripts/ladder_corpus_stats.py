"""Corpus statistics for the Answerability Ladder pre-registration — sizes only, never outcomes.

Prior work: [[project-recall-abstention-bounded-domain-2026-07-24]] — six abstention signals
measured, all failed; this measures the axis they were measured on, not a seventh signal.

Ring widths must be fixed before the builder runs, and a width wider than the corpus is not a
ring — it is d=max wearing a different number. So the widths are derived from cluster sizes, which
are a property of the data and not of any result. Nothing here reads a question, an answer, or a
retrieval score.
"""
from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def conversation_turn_counts(path: Path) -> list[int]:
    """Turns per LOCOMO conversation — the cluster size a d=max excision would remove."""
    data = json.loads(path.read_text(encoding="utf-8"))
    counts: list[int] = []
    for sample in data:
        conversation = sample.get("conversation", {})
        turns = 0
        for key, value in conversation.items():
            if re.fullmatch(r"session_\d+", key) and isinstance(value, list):
                turns += sum(1 for t in value if t.get("dia_id"))
        counts.append(turns)
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--locomo", type=Path, default=Path("locomo10.json"))
    args = parser.parse_args(argv)

    counts = conversation_turn_counts(args.locomo)
    print(f"conversations: {len(counts)}")
    print(
        f"turns per conversation: min={min(counts)} "
        f"median={statistics.median(counts)} max={max(counts)}"
    )
    print(f"total turns: {sum(counts)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
