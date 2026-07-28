"""Score the two LLM judges against the human labels on the contested items.

Both judges graded the same 10-conversation run; they disagreed on 199 answerable questions. On
every one of those, exactly one judge is wrong — so hand-labelling them is what turns "the judges
disagree" into "this judge is right". Everything upstream of this file (which memory system won,
which judge to publish) rests on nothing stronger than one model's opinion until these labels
exist.

The labels are free text rather than Y/N, because the annotator found the binary too coarse — the
recurring case being a prediction that covers most of a list but drops one item. That extra
resolution is the useful part, and mapping it back to a binary is a judgement call, so the rules
are stated here rather than buried in a notebook:

- **"extra detail is still correct."** ``Yes +``/``Yes + day``/``Yes + data`` mean the prediction
  was right and said MORE than the gold. The judge prompt explicitly allows that ("additional
  items or extra detail beyond the gold answer do NOT make it incorrect"), so these are correct.
- **"missing any gold item is incorrect."** Every ``partial missing …`` label, and any label
  containing "missing", is incorrect — including one written as ``yes missing advices from
  friend``, which the annotator confirmed should read as incorrect. This is the same rule the
  judge prompt states, applied consistently.
- **undecidable rows are EXCLUDED, not guessed.** ``no gold answer`` (the dataset's gold is
  unusable) and ``not sure golden is vague`` are dropped from the denominator. Counting a dataset
  defect as a judge error would blame the judge for LOCOMO's problem; guessing would manufacture
  ground truth that does not exist.

Run::

    python -m benchmarks.labelling.score_labels
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

HERE = Path(__file__).parent
LABELS = HERE / "judge_labelling_labelled.csv"
KEY = HERE / "judge_labelling_key.json"

#: Labels that mean "this row cannot arbitrate a judge" — a dataset defect or declared uncertainty.
EXCLUDE_MARKERS = ("no gold answer", "not sure")


def human_verdict(label: str) -> bool | None:
    """Map one free-text label to correct / incorrect / undecidable (None).

    Order matters: the exclusion markers are checked first, and "missing" is checked before the
    leading "yes", because ``yes missing advices from friend`` is an incorrect row whose text
    starts with "yes".
    """
    text = label.strip().casefold()
    if not text:
        return None
    if any(marker in text for marker in EXCLUDE_MARKERS):
        return None
    if "missing" in text:
        return False
    return text.startswith("yes")


def score() -> dict[str, Any]:
    key: dict[str, dict[str, Any]] = json.loads(KEY.read_text(encoding="utf-8"))
    with LABELS.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    judges = ("gpt4o_mini_said", "gpt4o_said")
    totals: dict[str, dict[str, int]] = {j: {"right": 0, "wrong": 0} for j in judges}
    by_arm: dict[str, dict[str, dict[str, int]]] = {}
    excluded = 0

    for row in rows:
        verdict = human_verdict(row["human_label_raw"])
        if verdict is None:
            excluded += 1
            continue
        entry = key[row["item"]]
        arm = str(entry["arm"])
        for judge in judges:
            bucket = "right" if bool(entry[judge]) == verdict else "wrong"
            totals[judge][bucket] += 1
            by_arm.setdefault(arm, {}).setdefault(judge, {"right": 0, "wrong": 0})[bucket] += 1

    scored = totals[judges[0]]["right"] + totals[judges[0]]["wrong"]
    return {
        "contested_items": len(rows),
        "scored": scored,
        "excluded_undecidable": excluded,
        "judge_accuracy": {
            j: {"correct": totals[j]["right"], "n": scored, "rate": round(totals[j]["right"] / scored, 4)}
            for j in judges
        },
        "by_arm": {
            arm: {
                j: {
                    "correct": counts[j]["right"],
                    "n": counts[j]["right"] + counts[j]["wrong"],
                    "rate": round(counts[j]["right"] / (counts[j]["right"] + counts[j]["wrong"]), 4),
                }
                for j in judges
            }
            for arm, counts in sorted(by_arm.items())
        },
    }


def main() -> int:
    report = score()
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
