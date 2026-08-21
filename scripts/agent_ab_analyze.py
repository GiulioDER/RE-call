"""Turn one run's artifacts into the preregistered endpoints, and nothing else.

    python scripts/agent_ab_analyze.py --run-id agent-ab-traps-002

Reads `records.jsonl`, `trap-scores.json` and, when present, `ragas-scores.json`; writes
`analysis.json` and prints a table fit for pasting under a preregistration's Result heading.

Two rules this script exists to enforce.

**It reports the endpoints that were preregistered, in the order they were preregistered**, so the
analysis cannot quietly become a search for whichever metric came out well. The primary endpoint is
the trap hit rate on the `memory_only` loci; `both` and `claude_md_only` are controls, and
`claude_md_only` is expected to go the OTHER way.

**It refuses rather than rounds.** Every test comes from `benchmarks/agent_ab/stats.py`, which
returns `None` when a sample cannot support a number, and `None` is printed as `-`, never as 0.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agent_ab.schema import RECALL_OFF, RECALL_ON  # noqa: E402
from benchmarks.agent_ab.stats import (  # noqa: E402
    compare_binary,
    compare_continuous,
    summarize_by_task,
)

#: Loci come from the committed qualification artifact, not from `traps.py`. Which memory holds a
#: fact was MEASURED by `scripts/agent_ab_qualify.py` against the real corpus and the real
#: CLAUDE.md, and it is fixed before the run; a constant re-declared next to the detector could
#: drift away from what was actually qualified, and the drift would silently reassign a trap
#: between the primary endpoint and a control.
_QUALIFICATION = json.loads(
    (REPO_ROOT / "benchmarks" / "agent_ab" / "trap-qualification.json").read_text(encoding="utf-8")
)
LOCUS: dict[str, str] = {
    q["trap_id"]: q["locus"] for q in _QUALIFICATION["qualifications"]
}
#: Preregistered order. `claude_md_only` is last because it is the control RE-call should LOSE.
LOCI = ("memory_only", "both", "claude_md_only")


def _fmt(value: float | None, digits: int = 3) -> str:
    return "-" if value is None else f"{value:.{digits}f}"


def _p(value: float | None) -> str:
    return "-" if value is None else (f"{value:.4f}" if value >= 0.0001 else "<0.0001")


def build_pairs(records: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    by_task: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for record in records:
        by_task[record["task_id"]][record["variant"]] = record
    return {
        task: arms
        for task, arms in by_task.items()
        if RECALL_ON in arms and RECALL_OFF in arms
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--tasks",
        default=str(REPO_ROOT / "benchmarks" / "agent_ab" / "tasks" / "traps.jsonl"),
        help="the manifest the run used; it maps each task to its ONE designated trap",
    )
    args = parser.parse_args()

    artifacts = REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / args.run_id
    records = [
        json.loads(line)
        for line in (artifacts / "records.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trap_scores = json.loads((artifacts / "trap-scores.json").read_text(encoding="utf-8"))
    env_path = artifacts / "environment.json"
    environment = (
        json.loads(env_path.read_text(encoding="utf-8")) if env_path.is_file() else {}
    )

    # Each task has ONE designated trap, and only that one is its outcome. `trap-scores.json`
    # deliberately scores every detector against every transcript, which is useful for spotting a
    # session that walked into someone else's hazard, but reading "any trap triggered" as the
    # endpoint would count those incidental hits as failures on this task and inflate both arms.
    tasks = [
        json.loads(line)
        for line in Path(args.tasks).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    trap_of: dict[str, str] = {row["task_id"]: row["trap_id"] for row in tasks}

    triggered: dict[tuple[str, str], bool] = {}
    incidental = 0
    for score in trap_scores:
        base = score["task_id"].split("#")[0]
        own = trap_of.get(base)
        for entry in score["traps"]:
            if entry["trap_id"] == own:
                triggered[(score["task_id"], score["variant"])] = bool(entry["triggered"])
            elif entry["triggered"]:
                incidental += 1
    if incidental:
        print(f"note: {incidental} incidental trap hit(s) on tasks that did not target them")

    pairs = build_pairs(records)
    print(f"\nrun {args.run_id}: {len(pairs)} admitted pairs")
    print(
        f"corpus: calibrated={environment.get('recall_calibrated')} "
        f"trust={environment.get('recall_trust_state')} "
        f"transport={environment.get('recall_transport')} "
        f"cli={environment.get('claude_code_version')}"
    )

    analysis: dict[str, Any] = {"run_id": args.run_id, "pairs": len(pairs), "primary": {}, "controls": {}, "cost": {}}

    print("\n--- trap hit rate, by locus (PRIMARY = memory_only) ---")
    print(f"{'locus':<16}{'n':>4}{'on':>8}{'off':>8}{'delta':>9}{'95% CI':>20}{'p':>10}")
    for locus in LOCI:
        selected = [
            (triggered.get((task, RECALL_ON), False), triggered.get((task, RECALL_OFF), False))
            for task in pairs
            if LOCUS.get(trap_of.get(task.split("#")[0], ""), "") == locus
            and (task, RECALL_ON) in triggered
        ]
        if not selected:
            continue
        result = compare_binary(f"trap_hit_{locus}", selected)
        ci = (
            f"[{_fmt(result.delta_ci[0])}, {_fmt(result.delta_ci[1])}]"
            if result.delta_ci
            else "-"
        )
        print(
            f"{locus:<16}{result.n_pairs:>4}{_fmt(result.on_mean):>8}{_fmt(result.off_mean):>8}"
            f"{_fmt(result.delta_mean):>9}{ci:>20}{_p(result.p_value):>10}"
        )
        if result.note:
            print(f"                {result.note}")
        target = analysis["primary"] if locus == "memory_only" else analysis["controls"]
        target[locus] = result.to_dict()

        if locus == "memory_only":
            # The conservative reading, and the one headlined: one rate per trap, so the unit of
            # evidence is the hazard rather than the session. Repetitions of one task are
            # correlated, so the per-pair p-value above is a consistency check, not the claim.
            by_task: dict[str, list[tuple[bool, bool]]] = {}
            for task in pairs:
                base = task.split("#")[0]
                if LOCUS.get(trap_of.get(base, ""), "") != locus:
                    continue
                if (task, RECALL_ON) not in triggered:
                    continue
                by_task.setdefault(base, []).append(
                    (triggered[(task, RECALL_ON)], triggered.get((task, RECALL_OFF), False))
                )
            per_task = summarize_by_task(by_task)
            analysis["primary"]["by_task"] = per_task
            print()
            print("  PER-TASK (headline; one rate per trap, repetitions collapsed)")
            print(f"  {'trap':<26}{'reps':>5}{'on':>8}{'off':>8}{'delta':>9}")
            for entry in per_task["tasks"]:
                print(
                    f"  {entry['task']:<26}{entry['n_reps']:>5}{entry['on_rate']:>8.3f}"
                    f"{entry['off_rate']:>8.3f}{entry['delta']:>+9.3f}"
                )
            ci = per_task.get("cluster_ci")
            print(
                f"  improved {per_task['improved']}/{per_task['n_tasks']} traps, "
                f"mean delta {per_task['mean_delta']:+.3f}, "
                f"cluster CI {'-' if not ci else f'[{ci[0]:.3f}, {ci[1]:.3f}]'}"
            )
            print(f"  {per_task['note']}")

    print("\n--- cost ---")
    print(f"{'metric':<16}{'n':>4}{'on':>12}{'off':>12}{'delta':>12}{'p':>10}")
    for metric, getter in (
        ("input_tokens", lambda r: r.get("input_tokens")),
        ("output_tokens", lambda r: r.get("output_tokens")),
        ("wall_time_ms", lambda r: r.get("wall_time_ms")),
        ("model_turns", lambda r: r.get("model_turns")),
    ):
        selected = [
            (getter(arms[RECALL_ON]), getter(arms[RECALL_OFF])) for arms in pairs.values()
        ]
        result = compare_continuous(metric, selected)  # type: ignore[arg-type]
        print(
            f"{metric:<16}{result.n_pairs:>4}{_fmt(result.on_mean, 0):>12}"
            f"{_fmt(result.off_mean, 0):>12}{_fmt(result.delta_mean, 0):>12}"
            f"{_p(result.p_value):>10}"
        )
        analysis["cost"][metric] = result.to_dict()

    ragas_path = artifacts / "ragas-scores.json"
    if ragas_path.is_file():
        ragas = json.loads(ragas_path.read_text(encoding="utf-8"))
        lookup = {(s["task_id"], s["variant"]): s for s in ragas["scores"]}
        print(f"\n--- answer quality (ragas, judge {ragas['judge_model']}) ---")
        print(f"{'metric':<22}{'n':>4}{'on':>8}{'off':>8}{'delta':>9}{'p':>10}")
        for metric in ("answer_correctness", "factual_correctness"):
            selected = [
                (
                    lookup.get((task, RECALL_ON), {}).get(metric),
                    lookup.get((task, RECALL_OFF), {}).get(metric),
                )
                for task in pairs
            ]
            result = compare_continuous(metric, selected)  # type: ignore[arg-type]
            print(
                f"{metric:<22}{result.n_pairs:>4}{_fmt(result.on_mean):>8}"
                f"{_fmt(result.off_mean):>8}{_fmt(result.delta_mean):>9}{_p(result.p_value):>10}"
            )
            analysis.setdefault("quality", {})[metric] = result.to_dict()
    else:
        print("\n--- answer quality: not scored (no ragas-scores.json) ---")

    (artifacts / "analysis.json").write_text(json.dumps(analysis, indent=2), encoding="utf-8")
    print(f"\nwritten: {artifacts / 'analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
