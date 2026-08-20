"""Run one preregistered paired comparison, and write everything it produces.

    python scripts/agent_ab_run.py --run-id agent-ab-traps-001 --comparison headline --reps 5

`--comparison headline` is `recall` against `claude_md`, the number worth publishing.
`--comparison ceiling` is `recall` against `bare`, which bounds how much of the gap is memory
at all. They are separate runs with separate identifiers, because `SessionRecord.variant` names
only the two arms and the off arm's profile is what distinguishes the two comparisons.

Nothing here decides what counts. The trap set and its loci were fixed by
`scripts/agent_ab_qualify.py` and committed; the predictions were committed before this ran; and
`gate.py` decides admissibility from the session's own tool list rather than from its exit status.

**Start with `--reps 1 --limit 2`.** A smoke pair costs minutes and catches the whole class of
mistakes that otherwise costs a run.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agent_ab.arms import (  # noqa: E402
    ArmSpec,
    build_configs,
    write_claude_md_prompt,
    write_recall_prompt,
)
from benchmarks.agent_ab.claude_exec import run_claude_case  # noqa: E402
from benchmarks.agent_ab.gate import admit_pairs  # noqa: E402
from benchmarks.agent_ab.io import write_jsonl  # noqa: E402
from benchmarks.agent_ab.recall_server import WarmRecallServer  # noqa: E402
from benchmarks.agent_ab.runner import run_paired  # noqa: E402
from benchmarks.agent_ab.schema import RECALL_OFF, RECALL_ON  # noqa: E402
from benchmarks.agent_ab.summarize import (  # noqa: E402
    summarize_pairs,
    summarize_recall_overhead,
)
from benchmarks.agent_ab.traps import score_record  # noqa: E402

DEFAULT_DSN = "postgresql://recall:recall@127.0.0.1:5433/recall"
TASKS = REPO_ROOT / "benchmarks" / "agent_ab" / "tasks" / "traps.jsonl"
STATIC_MEMORY_SOURCES = ("CLAUDE.md",)


def load_tasks(path: Path, limit: int | None, reps: int) -> list[dict[str, Any]]:
    """Expand the manifest by repetition, keeping every task_id unique.

    `run_paired` requires unique ids within one run, so a repetition is a distinct task carrying
    its origin. Collapsing them back for a per-task view is the summariser's job, not this one's.
    """

    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if limit:
        rows = rows[:limit]
    expanded: list[dict[str, Any]] = []
    for rep in range(1, reps + 1):
        for row in rows:
            expanded.append({**row, "task_id": f"{row['task_id']}#r{rep}", "base_task_id": row["task_id"], "rep": rep})
    return expanded


def environment_capture(model: str, server: WarmRecallServer) -> dict[str, Any]:
    def run(*command: str) -> str:
        try:
            return subprocess.run(  # noqa: S603 - argv list, no shell
                command, capture_output=True, text=True, timeout=60
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - a missing tool must not abort a run
            return ""

    return {
        # Stamped here rather than inside the workflow, and in UTC, so two runs on the same day
        # are distinguishable in the artifact directory.
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "repo_revision": run("git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"),
        "repo_dirty": bool(run("git", "-C", str(REPO_ROOT), "status", "--porcelain")),
        "claude_code_version": run("claude", "--version"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "model": model,
        "recall_url": server.url,
        "recall_tenant": server.tenant,
        "recall_handshake_ms": server.handshake_ms,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--comparison", choices=("headline", "ceiling"), default="headline")
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tasks", default=str(TASKS))
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--port", type=int, default=5480)
    parser.add_argument("--model", default="anthropic/claude-haiku-4.5")
    parser.add_argument("--memory-index", default=None)
    parser.add_argument("--pair-concurrency", type=int, default=1)
    args = parser.parse_args()

    artifacts = REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / args.run_id
    if artifacts.exists() and any(artifacts.iterdir()):
        print(
            f"{artifacts} already has contents. A run identifier names one immutable run; "
            f"choose a new one rather than overwriting the evidence."
        )
        return 1
    (artifacts / "streams").mkdir(parents=True, exist_ok=True)
    workdir = artifacts / "workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    tasks = load_tasks(Path(args.tasks), args.limit, args.reps)
    print(f"run {args.run_id}: {args.comparison}, {len(tasks)} paired sessions\n")

    static_sources: tuple[str, ...] = STATIC_MEMORY_SOURCES
    if args.memory_index and Path(args.memory_index).is_file():
        static_sources = static_sources + (args.memory_index,)

    with WarmRecallServer(
        dsn=args.dsn, cwd=REPO_ROOT, tenant=args.tenant, port=args.port
    ) as server:
        check = await server.check()
        print(
            f"RE-call ready: {check['handshake_ms']} ms handshake, {check['tool_count']} tools, "
            f"trust_state={check['trust_state']}"
        )
        recall_prompt = write_recall_prompt(artifacts / "recall-prompt.txt", server)
        if args.comparison == "headline":
            off_profile = "claude_md"
            off_spec = ArmSpec.claude_md(
                write_claude_md_prompt(
                    artifacts / "static-memory-prompt.txt", static_sources, repo_root=REPO_ROOT
                )
            )
        else:
            off_profile = "bare"
            off_spec = ArmSpec.bare()

        configs = build_configs(
            {RECALL_ON: ArmSpec.recall(server, recall_prompt), RECALL_OFF: off_spec},
            model=args.model,
            cwd=workdir,
        )
        configs = {
            variant: replace(config, stream_dir=artifacts / "streams")
            for variant, config in configs.items()
        }
        environment = environment_capture(args.model, server)

        async def run_case(row: dict[str, Any], variant: str):
            record = await run_claude_case(row, variant, configs[variant])
            # Carry the run's own labels onto the record, so a records.jsonl can be grouped by
            # trap, locus and repetition without needing the manifest beside it.
            merged = {
                **record.metadata,
                "off_arm_profile": off_profile,
                "comparison": args.comparison,
                "base_task_id": row.get("base_task_id"),
                "rep": row.get("rep"),
                "trap_id": row.get("trap_id"),
                "locus": row.get("locus"),
            }
            return record.__class__.from_mapping({**record.to_dict(), "metadata": merged})

        records = await run_paired(tasks, run_case, pair_concurrency=args.pair_concurrency)

    report = admit_pairs(records)
    summary = summarize_pairs(list(report.admitted))
    trap_scores = [score_record(record) for record in report.admitted]

    write_jsonl(artifacts / "records.jsonl", records)
    for name, payload in (
        ("summary.json", summary),
        ("recall-overhead.json", summarize_recall_overhead(list(report.admitted))),
        ("admission.json", report.summary()),
        ("trap-scores.json", trap_scores),
        ("environment.json", environment),
    ):
        (artifacts / name).write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    print(
        f"\n{len(records)} records, {report.summary()['admitted_pairs']} pairs admitted, "
        f"{report.discarded_pair_count} discarded"
    )
    for task_id in report.discarded_task_ids:
        print(f"  discarded {task_id}")
    print(f"\nartifacts: {artifacts}")
    if not report.admitted:
        print("No admitted pairs. This is a wiring result, not a measurement.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
