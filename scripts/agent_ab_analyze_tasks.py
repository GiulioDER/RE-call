"""Report the task-success run's endpoints, in the order they were preregistered.

    python -u scripts/agent_ab_analyze_tasks.py --run-id agent-ab-tasksuccess-001

Nothing here decides what counts. The task loci were fixed by `scripts/agent_ab_qualify_tasks.py`
and committed at `09aa03f0`; the predictions were committed before the run; admissibility comes
from the gate's verdict rather than from any exit status.

⚠️ **The direction is opposite to the trap benchmark's, and that is the one thing most likely to go
wrong silently here.** For a hazard, lower is better, so `stats.summarize_by_task` counts a task as
improved when its delta is NEGATIVE. For task success, higher is better. Reusing that helper would
have inverted the headline while producing a perfectly plausible table, so the per-task view is
implemented here with the direction written out, and `tests/test_agent_ab_analyze_tasks.py` asserts
it against a fixture where the on arm plainly wins.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path
from typing import Any, Sequence

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agent_ab.gate import admit_pairs  # noqa: E402
from benchmarks.agent_ab.schema import RECALL_OFF, RECALL_ON, SessionRecord  # noqa: E402
from benchmarks.agent_ab.stats import compare_binary, compare_continuous  # noqa: E402

MEMORY_ONLY = "memory_only"


def sign_test(improved: int, worsened: int) -> float | None:
    """Two-sided exact sign test over TASKS, ignoring ties.

    Returned so the per-task view has a real p-value rather than a descriptive shrug. That was not
    possible with four tasks, where the test bottoms out at 0.125 whatever the effect; with eight it
    reaches 0.008. `None` when nothing moved, because a p-value over zero informative tasks is not
    a small number, it is an absent one.
    """

    n = improved + worsened
    if n == 0:
        return None
    k = min(improved, worsened)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2.0 * tail)


def success_by_task(
    pairs_by_task: dict[str, Sequence[tuple[bool, bool]]],
    *,
    n: int = 10000,
    confidence: float = 0.95,
    seed: int = 12345,
) -> dict[str, Any]:
    """The per-TASK view: one success rate per arm per task, HIGHER IS BETTER.

    Repetitions of one task are not independent, so this collapses them and makes the task the unit
    of evidence. The bootstrap resamples whole tasks, which is the honest interval for "what would
    a new task of this kind do", and it is wide on purpose.
    """

    rows = []
    for task, pairs in sorted(pairs_by_task.items()):
        if not pairs:
            continue
        on_rate = sum(1 for on, _ in pairs if on) / len(pairs)
        off_rate = sum(1 for _, off in pairs if off) / len(pairs)
        rows.append(
            {
                "task": task,
                "n_reps": len(pairs),
                "on_rate": on_rate,
                "off_rate": off_rate,
                # on minus off, so POSITIVE means the memory layer succeeded more often.
                "delta": on_rate - off_rate,
            }
        )
    if not rows:
        return {"tasks": [], "n_tasks": 0, "note": "no tasks"}

    deltas = [row["delta"] for row in rows]
    improved = sum(1 for d in deltas if d > 0)
    worsened = sum(1 for d in deltas if d < 0)

    cluster_ci = None
    if len(rows) >= 2 and len(set(deltas)) > 1:
        rng = random.Random(seed)
        size = len(rows)
        means = sorted(
            sum(deltas[rng.randrange(size)] for _ in range(size)) / size for _ in range(n)
        )
        lo = (1.0 - confidence) / 2.0
        cluster_ci = [
            means[min(len(means) - 1, int(lo * len(means)))],
            means[min(len(means) - 1, int((1.0 - lo) * len(means)))],
        ]

    return {
        "tasks": rows,
        "n_tasks": len(rows),
        "mean_delta": sum(deltas) / len(deltas),
        "improved": improved,
        "worsened": worsened,
        "unchanged": len(deltas) - improved - worsened,
        "sign_test_p": sign_test(improved, worsened),
        "cluster_ci": cluster_ci,
        "direction": "positive delta means the on arm succeeded MORE often",
    }


def searched(record: SessionRecord) -> bool:
    return record.recall_call_count > 0


def retrieved_governing_memo(record: SessionRecord) -> bool | None:
    """Whether the task's own memo appears in what the session retrieved.

    `None` when the session never searched, so "did not look" stays distinguishable from "looked
    and the memo did not come back". Those are different failures and they point at different
    fixes.
    """

    memo = record.metadata.get("governing_memo")
    if not memo:
        return None
    if not record.recall_call_count:
        return None
    blob = "\n".join(record.retrieved_contexts) + "\n" + json.dumps(list(record.tool_calls))
    return memo in blob


def load(run_id: str) -> tuple[list[SessionRecord], Path]:
    artifacts = REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / run_id
    path = artifacts / "records.jsonl"
    if not path.is_file():
        path = artifacts / "records.partial.jsonl"
    if not path.is_file():
        raise SystemExit(f"no records under {artifacts}")
    records = [
        SessionRecord.from_mapping(json.loads(line))
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    return records, artifacts


def pairs_for(records: Sequence[SessionRecord], loci: set[str]) -> dict[str, list[tuple[bool, bool]]]:
    by_task: dict[str, dict[str, SessionRecord]] = {}
    for record in records:
        if record.metadata.get("locus") not in loci:
            continue
        by_task.setdefault(record.task_id, {})[record.variant] = record
    out: dict[str, list[tuple[bool, bool]]] = {}
    for task_id, arms in by_task.items():
        if RECALL_ON not in arms or RECALL_OFF not in arms:
            continue
        base = str(arms[RECALL_ON].metadata.get("base_task_id") or task_id)
        out.setdefault(base, []).append((arms[RECALL_ON].success, arms[RECALL_OFF].success))
    return out


def cost_pairs(records: Sequence[SessionRecord], field: str, loci: set[str]):
    by_task: dict[str, dict[str, SessionRecord]] = {}
    for record in records:
        if record.metadata.get("locus") not in loci:
            continue
        by_task.setdefault(record.task_id, {})[record.variant] = record
    pairs = []
    for arms in by_task.values():
        if RECALL_ON not in arms or RECALL_OFF not in arms:
            continue
        on_value = getattr(arms[RECALL_ON], field)
        off_value = getattr(arms[RECALL_OFF], field)
        if on_value is None or off_value is None:
            continue
        pairs.append((float(on_value), float(off_value)))
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()

    records, artifacts = load(args.run_id)
    report = admit_pairs(records)
    admitted = list(report.admitted)
    print(
        f"{len(records)} records, {report.summary()['admitted_pairs']} pairs admitted, "
        f"{report.discarded_pair_count} discarded\n"
    )

    checker_errors = sum(
        1 for r in admitted if (r.metadata.get("check") or {}).get("detail", {}).get("checker_error")
    )
    if checker_errors:
        print(f"[warn] {checker_errors} sessions were scored by a checker that RAISED\n")

    primary_pairs = pairs_for(admitted, {MEMORY_ONLY})
    control_pairs = pairs_for(admitted, {"both", "claude_md_only"})

    print("=" * 78)
    print("1. PRIMARY: task success on memory_only tasks")
    print("=" * 78)
    per_task = success_by_task(primary_pairs)
    print(f"  {'task':<26} {'off':>6} {'on':>6} {'delta':>7}  reps")
    for row in per_task["tasks"]:
        print(
            f"  {row['task']:<26} {row['off_rate']:>6.2f} {row['on_rate']:>6.2f} "
            f"{row['delta']:>+7.2f}  {row['n_reps']}"
        )
    ci = per_task.get("cluster_ci")
    print(
        f"\n  per-task (headline): {per_task['improved']} improved, {per_task['worsened']} worse, "
        f"{per_task['unchanged']} unchanged, over {per_task['n_tasks']} tasks"
    )
    print(f"  mean delta {per_task['mean_delta']:+.3f}", end="")
    if ci:
        print(f", cluster CI [{ci[0]:+.3f}, {ci[1]:+.3f}]", end="")
    p = per_task.get("sign_test_p")
    print(f", sign test p={p:.4f}" if p is not None else ", sign test p=n/a")

    flat = [pair for pairs in primary_pairs.values() for pair in pairs]
    per_pair = compare_binary("task_success", flat)
    print(
        f"\n  per-pair (consistency check, OVERSTATES confidence): n={per_pair.n_pairs}, "
        f"off {per_pair.off_mean:.3f} on {per_pair.on_mean:.3f}, "
        f"delta {per_pair.delta_mean:+.3f}, p={per_pair.p_value}"
    )

    print("\n" + "=" * 78)
    print("2. CONTROL: tasks whose fact is in CLAUDE.md")
    print("=" * 78)
    control_flat = [pair for pairs in control_pairs.values() for pair in pairs]
    if control_flat:
        control = compare_binary("control_success", control_flat)
        print(
            f"  n={control.n_pairs}, off {control.off_mean:.3f} on {control.on_mean:.3f}, "
            f"delta {control.delta_mean:+.3f}, p={control.p_value}"
        )
        for task, pairs in sorted(control_pairs.items()):
            on_rate = sum(1 for on, _ in pairs if on) / len(pairs)
            off_rate = sum(1 for _, off in pairs if off) / len(pairs)
            print(f"    {task:<26} off {off_rate:.2f} on {on_rate:.2f}")
    else:
        control = None
        print("  no admitted control pairs")

    print("\n" + "=" * 78)
    print("3. MECHANISM")
    print("=" * 78)
    on_records = [r for r in admitted if r.variant == RECALL_ON]
    on_primary = [r for r in on_records if r.metadata.get("locus") == MEMORY_ONLY]
    search_rate = (sum(1 for r in on_primary if searched(r)) / len(on_primary)) if on_primary else None
    hits = [retrieved_governing_memo(r) for r in on_primary]
    looked = [h for h in hits if h is not None]
    memo_rate = (sum(1 for h in looked if h) / len(looked)) if looked else None
    calls = [r.recall_call_count for r in on_primary]
    print(f"  on-arm sessions (memory_only): {len(on_primary)}")
    print(f"  search rate:                   {search_rate if search_rate is None else f'{search_rate:.3f}'}")
    print(
        f"  governing memo retrieved:      "
        f"{memo_rate if memo_rate is None else f'{memo_rate:.3f}'} of {len(looked)} that searched"
    )
    if calls:
        print(f"  recall calls: mean {statistics.mean(calls):.2f}, median {statistics.median(calls)}")
    off_calls = sum(r.recall_call_count for r in admitted if r.variant == RECALL_OFF)
    print(f"  off-arm recall calls (must be 0): {off_calls}")

    print("\n" + "=" * 78)
    print("4. COST (median beside mean, and they can disagree in sign)")
    print("=" * 78)
    costs = {}
    for field in ("input_tokens", "output_tokens", "wall_time_ms", "model_turns"):
        pairs = cost_pairs(admitted, field, {MEMORY_ONLY})
        if not pairs:
            continue
        result = compare_continuous(field, pairs)
        costs[field] = result.to_dict()
        print(
            f"  {field:<15} n={result.n_pairs:<3} mean {result.delta_mean:+12.1f}  "
            f"median {result.delta_median:+12.1f}  p={result.p_value}"
        )

    payload = {
        "run_id": args.run_id,
        "admission": report.summary(),
        "checker_errors": checker_errors,
        "primary_per_task": per_task,
        "primary_per_pair": per_pair.to_dict(),
        "control": control.to_dict() if control else None,
        "mechanism": {
            "on_sessions": len(on_primary),
            "search_rate": search_rate,
            "governing_memo_rate": memo_rate,
            "searched_sessions": len(looked),
            "off_arm_recall_calls": off_calls,
        },
        "cost": costs,
    }
    out = artifacts / "analysis.json"
    out.write_text(json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
