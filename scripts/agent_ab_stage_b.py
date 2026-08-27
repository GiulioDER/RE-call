#!/usr/bin/env python3
"""Score the write-time hook A/B against its registered decision rule.

    python scripts/agent_ab_stage_b.py benchmarks/artifacts/agent_ab/stage-b-001

Both arms are the instruction arm; `recall_on` carries the write-time hook and `recall_off` is the
control. The endpoint is the checker's verdict, and the test is McNemar on discordant pairs, which
is the only part of a paired design that carries information: pairs where both arms agree tell you
about the task, not about the treatment.

The exact binomial form is used rather than the chi-square approximation, because the discordant
counts here are single digits and the approximation is not trustworthy there.

⚠️ This scores whatever pairs are COMPLETE. A run stopped partway is not the registered n, and the
report says so rather than presenting a smaller sample as the planned one.
"""

from __future__ import annotations

import collections
import json
import math
import sys
from pathlib import Path

REGISTERED_PAIRS = 48


def load(run: Path) -> list[dict]:
    for name in ("records.jsonl", "records.partial.jsonl"):
        path = run / name
        if path.is_file():
            rows = [json.loads(line) for line in
                    path.read_text(encoding="utf-8").splitlines() if line.strip()]
            if rows:
                return rows
    raise SystemExit(f"no records in {run}")


def passed(record: dict) -> bool | None:
    """The checker's verdict, or None when there is not one.

    Never defaulted to False: a session that died has no verdict, and scoring it as a failure
    would count apparatus deaths as evidence about the work.
    """

    check = record.get("metadata", {}).get("check")
    if not isinstance(check, dict) or "passed" not in check:
        return None
    return bool(check["passed"])


def mcnemar_exact(rescues: int, regressions: int) -> float:
    """Two-sided exact binomial p on the discordant pairs."""

    n = rescues + regressions
    if n == 0:
        return 1.0
    smaller = min(rescues, regressions)
    tail = sum(math.comb(n, i) for i in range(smaller + 1)) * (0.5 ** n)
    return min(1.0, 2 * tail)


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run = Path(sys.argv[1])
    records = load(run)

    verdicts: dict[tuple[str, str], bool | None] = {}
    families: dict[str, str] = {}
    errors: dict[tuple[str, str], str | None] = {}
    for r in records:
        verdicts[(r["task_id"], r["variant"])] = passed(r)
        families[r["task_id"]] = r.get("metadata", {}).get("base_task_id", "?")
        errors[(r["task_id"], r["variant"])] = r.get("error")

    tasks = sorted({task for task, _ in verdicts})
    complete = []
    incomplete = []
    for task in tasks:
        treat = verdicts.get((task, "recall_on"))
        control = verdicts.get((task, "recall_off"))
        if treat is None or control is None:
            incomplete.append(task)
            continue
        complete.append((task, treat, control))

    print(f"sessions recorded : {len(records)}")
    print(f"complete pairs    : {len(complete)} of the registered {REGISTERED_PAIRS}")
    if len(complete) < REGISTERED_PAIRS:
        print(f"  ⚠️ {REGISTERED_PAIRS - len(complete)} pairs short of the registered n. "
              f"This is a partial run and must be reported as one.")
    if incomplete:
        print(f"  pairs with a missing or dead arm: {len(incomplete)}")

    control_failures = sum(1 for _, _, c in complete if not c)
    treat_failures = sum(1 for _, t, _ in complete if not t)
    print()
    print(f"control failures  : {control_failures} of {len(complete)}")
    print(f"treatment failures: {treat_failures} of {len(complete)}")

    rescues = [t for t, tr, c in complete if tr and not c]
    regressions = [t for t, tr, c in complete if c and not tr]
    print()
    print(f"RESCUES     (control failed, hook passed): {len(rescues)}")
    for task in rescues:
        print(f"    {task}")
    print(f"REGRESSIONS (control passed, hook failed): {len(regressions)}")
    for task in regressions:
        print(f"    {task}")

    p = mcnemar_exact(len(rescues), len(regressions))
    print()
    print(f"McNemar exact, two-sided: p = {p:.4f} on {len(rescues) + len(regressions)} "
          f"discordant pairs")
    net = len(rescues) - len(regressions)
    print(f"net (rescues minus regressions): {net:+d}")

    print()
    print("by family")
    per: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"pairs": 0, "control_fail": 0, "treat_fail": 0})
    for task, treat, control in complete:
        row = per[families.get(task, "?")]
        row["pairs"] += 1
        row["control_fail"] += 0 if control else 1
        row["treat_fail"] += 0 if treat else 1
    for family, row in sorted(per.items()):
        print(f"    {family:24s} pairs {row['pairs']:>2}  control fail {row['control_fail']:>2}"
              f"  hook fail {row['treat_fail']:>2}")

    trace = run / "hook-trace.jsonl"
    if trace.is_file():
        rows = [json.loads(line) for line in
                trace.read_text(encoding="utf-8").splitlines() if line.strip()]
        by_session = collections.Counter(r["session_id"] for r in rows)
        counts = sorted(by_session.values())
        vocab = collections.Counter(str(r.get("vocabulary_would_fire")) for r in rows)
        print()
        print(f"hook: {len(rows)} injections across {len(by_session)} sessions")
        if counts:
            print(f"    per session: min {counts[0]}, median {counts[len(counts) // 2]}, "
                  f"max {counts[-1]}")
        print(f"    vocabulary_would_fire: {dict(vocab)}")
        gated = sum(1 for r in rows if r.get("vocabulary_would_fire") is True)
        if rows:
            print(f"    a df<=2 gate would have kept {gated} of {len(rows)} "
                  f"({gated / len(rows):.0%})")

    print()
    print("registered decision rule, on rescues x regressions")
    if len(rescues) >= 6:
        band = "BUILD" if len(regressions) <= 1 else (
            "BUILD WITH A GATE" if len(regressions) <= 4 else "KILL")
    elif len(rescues) >= 3:
        band = "UNDERPOWERED" if len(regressions) <= 4 else "KILL"
    else:
        band = "KILL, and this is the terminal outcome"
    print(f"    rescues {len(rescues)}, regressions {len(regressions)} -> {band}")

    # The cell is not the whole rule. The same record committed in advance: "nothing ships on a
    # non-significant positive". The >= 6 boundary was drawn where 6 rescues and ZERO regressions
    # reach p about 0.03; one regression moves the same 6 to p = 0.125, and the cell alone does
    # not notice. Applying only the cell would ship on a result the record already refused.
    if band.startswith("BUILD") and p > 0.05:
        print(f"    ⛔ OVERRIDDEN by the record's standing commitment: p = {p:.4f} is not "
              f"significant, and nothing ships on a non-significant positive. The cell is "
              f"reached on COUNTS; the commitment is on evidence.")
    if len(complete) < REGISTERED_PAIRS:
        print("    ⚠️ read against a partial n: the rule was written for "
              f"{REGISTERED_PAIRS} pairs and this run has {len(complete)}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
