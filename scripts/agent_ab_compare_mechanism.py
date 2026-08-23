"""Compare mechanism rates between two task-success runs, as preregistered.

    python scripts/agent_ab_compare_mechanism.py --run-id agent-ab-skill-001 \
        --baseline ~/.claude/archive/agent-ab-tasksuccess-001

Written and committed BEFORE `agent-ab-skill-001` produced a number, alongside
`docs/preregistrations/2026-08-22-hazard-query-instruction.md`, so the analysis is the registered
one rather than the one that fit the result. It answers the record's predictions 1 to 4 and
nothing else; success and cost stay with `agent_ab_analyze_tasks.py`.

The comparison is CROSS-RUN and the record says so: the two runs share tasks, fixtures, corpus
generation, model and CLI version, and differ in when they were executed and in the on arm's
instruction text. Nothing here can tell those two differences apart; the preregistration's
confound section owns that, not this script.

Metric definitions are IMPORTED from the analyzer, not restated, so the two runs are measured by
the same code path: `searched` is recall_call_count > 0, `retrieved_governing_memo` is the task's
memo appearing in what the session retrieved, None when it never looked.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agent_ab.gate import admit_pairs  # noqa: E402
from benchmarks.agent_ab.schema import RECALL_ON, SessionRecord  # noqa: E402

# Reuse the analyzer's definitions so both runs are measured by the same code.
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from agent_ab_analyze_tasks import retrieved_governing_memo, searched  # noqa: E402

MEMORY_ONLY = "memory_only"


def fisher_one_sided_greater(a: int, b: int, c: int, d: int) -> float:
    """P(table at least as extreme) for H1: rate in row 1 (a/(a+b)) EXCEEDS row 2 (c/(c+d)).

    Exact hypergeometric tail, so no scipy. Margins fixed at (a+b, c+d, a+c, b+d); extremeness
    is a larger `a`.
    """

    row1, row2, col1 = a + b, c + d, a + c
    total = row1 + row2
    lo = max(0, col1 - row2)
    hi = min(col1, row1)

    def pmf(k: int) -> float:
        return (
            math.comb(row1, k) * math.comb(row2, col1 - k) / math.comb(total, col1)
        )

    return sum(pmf(k) for k in range(a, hi + 1)) if lo <= a <= hi else 1.0


def load_records(source: Path) -> list[SessionRecord]:
    path = source / "records.jsonl"
    if not path.is_file():
        raise SystemExit(f"no records.jsonl under {source}")
    return [
        SessionRecord.from_mapping(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def mechanism(records: list[SessionRecord]) -> dict:
    """The three preregistered rates for one run, with their denominators, plus per-task reach."""

    admitted = list(admit_pairs(records).admitted)
    on = [
        r
        for r in admitted
        if r.variant == RECALL_ON and r.metadata.get("locus") == MEMORY_ONLY
    ]
    looked = [r for r in on if searched(r)]
    reached = [r for r in looked if retrieved_governing_memo(r)]
    per_task: dict[str, dict[str, int]] = defaultdict(lambda: {"n": 0, "searched": 0, "reached": 0})
    for r in on:
        row = per_task[str(r.metadata.get("base_task_id") or r.task_id)]
        row["n"] += 1
        row["searched"] += int(searched(r))
        row["reached"] += int(bool(retrieved_governing_memo(r)))
    return {
        "on_sessions": len(on),
        "searched": len(looked),
        "reached": len(reached),
        "search_rate": len(looked) / len(on) if on else None,
        "memo_rate_given_searched": len(reached) / len(looked) if looked else None,
        "reach_rate": len(reached) / len(on) if on else None,
        "per_task": dict(sorted(per_task.items())),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--baseline",
        required=True,
        help="directory holding the baseline run's records.jsonl (e.g. the 001 archive)",
    )
    args = parser.parse_args()

    artifacts = REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / args.run_id
    new = mechanism(load_records(artifacts))
    base = mechanism(load_records(Path(args.baseline).expanduser()))

    tests = {
        "p1_reach": fisher_one_sided_greater(
            new["reached"], new["on_sessions"] - new["reached"],
            base["reached"], base["on_sessions"] - base["reached"],
        ),
        "p2_search": fisher_one_sided_greater(
            new["searched"], new["on_sessions"] - new["searched"],
            base["searched"], base["on_sessions"] - base["searched"],
        ),
        "p3_memo_given_searched": fisher_one_sided_greater(
            new["reached"], new["searched"] - new["reached"],
            base["reached"], base["searched"] - base["reached"],
        ),
    }

    print(f"{'':<28}{'baseline':>14}{'this run':>14}")
    for label, key, num, den in (
        ("P1 reach rate", "reach_rate", "reached", "on_sessions"),
        ("P2 search rate", "search_rate", "searched", "on_sessions"),
        ("P3 memo | searched", "memo_rate_given_searched", "reached", "searched"),
    ):
        b = f"{base[num]}/{base[den]}" if base[den] else "n/a"
        n = f"{new[num]}/{new[den]}" if new[den] else "n/a"
        print(f"  {label:<26}{b:>14}{n:>14}")
    for name, p in tests.items():
        print(f"  {name:<26} one-sided Fisher p = {p:.4f}")
    print("\n  per-task reach (this run), P4 is ts-lf-rewrite:")
    for task, row in new["per_task"].items():
        print(f"    {task:<34} searched {row['searched']}/{row['n']}  reached {row['reached']}/{row['n']}")

    payload = {"baseline": base, "run": new, "one_sided_fisher_p": tests}
    out = artifacts / "mechanism-comparison.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
