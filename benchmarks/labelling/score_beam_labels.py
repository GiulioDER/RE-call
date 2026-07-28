"""Score the two memory systems against the human labels, paired on question.

`score_labels.py` arbitrates two JUDGES on one system's answers. This scores two SYSTEMS under one
judge — the annotator — so the statistic is different: McNemar on the questions where the two arms
were labelled differently, because the questions they both got right (or both got wrong) carry no
information about which is better.

Verdicts are read the way `score_labels.py` reads them, and for the reason recorded there: the
annotator found strict Y/N too coarse, so free text is accepted and mapped. Anything containing
"missing" is INCORRECT (the judge prompt's own rule: covering only some gold items is wrong), and
undecidable rows are EXCLUDED rather than guessed — counting a dataset defect as a system error
would blame the system for BEAM's problem.

::

    python -m benchmarks.labelling.score_beam_labels --labels beam_labelled.csv --key beam_key.json
"""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

CORRECT = {"y", "yes", "correct", "true", "1"}
INCORRECT = {"n", "no", "wrong", "incorrect", "false", "0"}


def read_verdict(raw: str) -> bool | None:
    """True / False / None (undecidable or blank), by the rules in score_labels.py."""
    v = (raw or "").strip().lower()
    if not v:
        return None
    # Undecidable is tested FIRST and deliberately: "no gold answer" begins with "no" and is not a
    # verdict on the system — it records that BEAM's gold is unusable. Reading it as "incorrect"
    # would charge a dataset defect to whichever arm happened to be labelled.
    if v.startswith("no gold") or "not sure" in v or "vague" in v or "undecidable" in v:
        return None
    # "covers most of the gold but drops an item" -> incorrect, per the judge prompt's own rule.
    if "missing" in v:
        return False
    first = v.split()[0]
    if first in CORRECT:
        return True
    if first in INCORRECT:
        return False
    return None


def score(labels: list[dict[str, str]], key: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_question: dict[str, dict[str, bool]] = defaultdict(dict)
    per_arm: dict[str, list[bool]] = defaultdict(list)
    excluded = 0

    for row in labels:
        item = row.get("item", "").strip()
        meta = key.get(item)
        if meta is None:
            continue
        verdict = read_verdict(row.get("your_verdict_Y_or_N", ""))
        if verdict is None:
            excluded += 1
            continue
        per_arm[meta["arm"]].append(verdict)
        by_question[meta["question_id"]][meta["arm"]] = verdict

    paired = {q: v for q, v in by_question.items() if len(v) == 2}
    recall_only = sum(1 for v in paired.values() if v.get("recall") and not v.get("mem0"))
    mem0_only = sum(1 for v in paired.values() if v.get("mem0") and not v.get("recall"))

    return {
        "labelled": sum(len(v) for v in per_arm.values()),
        "excluded_undecidable": excluded,
        "per_arm": {
            arm: {"n": len(v), "correct": sum(v), "accuracy": (sum(v) / len(v)) if v else None}
            for arm, v in per_arm.items()
        },
        "paired_questions": len(paired),
        "discordant": {"recall_only": recall_only, "mem0_only": mem0_only},
        "mcnemar_note": (
            "exact binomial on the discordant pairs; concordant pairs carry no information "
            "about which system is better"
        ),
    }


def mcnemar_exact(a: int, b: int) -> float | None:
    """Two-sided exact binomial p on discordant pairs. None when there are none."""
    n = a + b
    if n == 0:
        return None
    from math import comb

    tail = sum(comb(n, i) for i in range(0, min(a, b) + 1))
    return float(min(1.0, 2.0 * tail / (2**n)))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--key", type=Path, required=True)
    args = p.parse_args()

    with args.labels.open(encoding="utf-8-sig", newline="") as fh:
        labels = list(csv.DictReader(fh))
    key = json.loads(args.key.read_text(encoding="utf-8"))

    result = score(labels, key)
    d = result["discordant"]
    p_value = mcnemar_exact(d["recall_only"], d["mem0_only"])

    print(json.dumps(result, indent=1))
    print(f"\ndiscordant: recall-only {d['recall_only']}  mem0-only {d['mem0_only']}")
    print(f"McNemar exact p = {p_value if p_value is None else round(p_value, 5)}")
    if p_value is None:
        print("No discordant pairs — the labels cannot separate the systems.")


if __name__ == "__main__":
    main()
