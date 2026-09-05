#!/usr/bin/env python3
"""Score an A/A base-rate run against the four registered predictions.

    python scripts/agent_ab_base_rate.py benchmarks/artifacts/agent_ab/base-rate-001

The run has no hook anywhere and both arms are the instruction arm, so **every session is a control
session** and every disagreement between the arms is session-to-session variance. That makes this
three measurements at once: the control failure count the power table needs, the variance floor any
McNemar effect has to clear, and a null check on the harness itself.

⚠️ The null check is the one that can invalidate the rest. A systematic winner between two
identical arms is apparatus bias, and nothing measured against that apparatus could then be
attributed to a treatment.
"""

from __future__ import annotations

import collections
import json
import sys
from pathlib import Path


def load(run: Path) -> list[dict]:
    records = run / "records.jsonl"
    if not records.is_file():
        records = run / "records.partial.jsonl"
    if not records.is_file():
        raise SystemExit(f"no records in {run}")
    return [json.loads(line) for line in
            records.read_text(encoding="utf-8").splitlines() if line.strip()]


def passed(record: dict) -> bool | None:
    """The checker's verdict, or None when the session never produced one.

    Read with `[...]` where the shape is guaranteed and `.get` only where absence is meaningful:
    a missing verdict must not silently become a failure, because that scores a dead session as
    evidence about the work. See `[[missing-input-becomes-a-clean-null]]`.
    """

    check = record.get("metadata", {}).get("check")
    if not isinstance(check, dict) or "passed" not in check:
        return None
    return bool(check["passed"])


def main() -> int:
    if len(sys.argv) != 2:
        raise SystemExit(__doc__)
    run = Path(sys.argv[1])
    records = load(run)

    verdicts = {(r["task_id"], r["variant"]): passed(r) for r in records}
    # `metadata["family"]` is "primary" for every ts-* task, so grouping on it collapses all
    # eight into one row and B4 reads as a perfectly flat spread no matter what the data says.
    # The eight task families this prediction is about are `base_task_id`.
    families = {r["task_id"]: r.get("metadata", {}).get("base_task_id", "?") for r in records}

    incomplete = [k for k, v in verdicts.items() if v is None]
    print(f"sessions: {len(records)}, without a checker verdict: {len(incomplete)}")
    if incomplete:
        for key in incomplete[:10]:
            print(f"  no verdict: {key}")

    scored = {k: v for k, v in verdicts.items() if v is not None}
    failures = sum(1 for v in scored.values() if not v)
    print()
    print(f"B1  control sessions failing: {failures} of {len(scored)}"
          f"   [band 11 to 20 of 48]")

    tasks = sorted({task for task, _ in scored})
    pairs = [(t, scored.get((t, "recall_on")), scored.get((t, "recall_off")))
             for t in tasks]
    complete = [(t, a, b) for t, a, b in pairs if a is not None and b is not None]
    discordant = [(t, a, b) for t, a, b in complete if a != b]
    print(f"B2  discordant pairs: {len(discordant)} of {len(complete)}   [band 4 to 10 of 24]")

    on_wins = sum(1 for _, a, b in discordant if a and not b)
    off_wins = len(discordant) - on_wins
    if discordant:
        share = max(on_wins, off_wins) / len(discordant)
        print(f"B3  discordant split: recall_on {on_wins}, recall_off {off_wins}, "
              f"max share {share:.0%}   [falsified above 70%]")
        if share > 0.70:
            print("    WARNING: a systematic winner between two IDENTICAL arms is apparatus "
                  "bias, and stage B could not attribute anything to the hook.")
    else:
        print("B3  no discordant pairs, so no direction to test")

    by_family: dict[str, list[bool]] = collections.defaultdict(list)
    for (task, _), verdict in scored.items():
        by_family[families.get(task, "?")].append(verdict)
    print()
    print("B4  failures by family   [worst should be >= 3x the best]")
    rates = {}
    for family, results in sorted(by_family.items()):
        failed = sum(1 for v in results if not v)
        rates[family] = failed / len(results)
        print(f"    {family:28s} {failed:>2} of {len(results):>2}  {rates[family]:>5.0%}")
    if rates:
        worst, best = max(rates.values()), min(rates.values())
        ratio = (worst / best) if best else float("inf")
        print(f"    spread: worst {worst:.0%}, best {best:.0%}, ratio "
              f"{'inf' if ratio == float('inf') else f'{ratio:.1f}x'}")

    print()
    print("consequence for stage B")
    if failures >= 11:
        print("  >= 11: the registered power table roughly holds; stage B at 48 pairs as written")
    elif failures >= 6:
        print("  6 to 10: UNDERPOWERED before it starts; stage B needs a larger n or is not run")
    else:
        print("  <= 5: the apparatus removed the failures the hook was meant to rescue; "
              "stage B measures nothing and is not run")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
