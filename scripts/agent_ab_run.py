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
import contextlib
import json
import os
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
    write_claude_md_recall_prompt,
    write_recall_prompt,
)
from benchmarks.agent_ab.claude_exec import (  # noqa: E402
    resolve_claude_executable,
    run_claude_case,
)
from benchmarks.agent_ab.gate import admit_pairs  # noqa: E402
from benchmarks.agent_ab.io import write_jsonl  # noqa: E402
from benchmarks.agent_ab.recall_server import (  # noqa: E402
    StdioRecallSpec,
    WarmRecallServer,
)
from benchmarks.agent_ab.runner import run_paired  # noqa: E402
from benchmarks.agent_ab.schema import RECALL_OFF, RECALL_ON  # noqa: E402
from benchmarks.agent_ab.summarize import (  # noqa: E402
    summarize_pairs,
    summarize_recall_overhead,
)
from benchmarks.agent_ab.traps import score_record  # noqa: E402

#: The benchmark owns its corpus. NOT the shared recall-dogfood on 5433, which serves other
#: sessions and is uncalibrated, and not the session container's default database, which the
#: test suite DROPs tables in. Built by scripts/agent_ab_build_corpus.py.
DEFAULT_DSN = "postgresql://recall:recall@127.0.0.1:5406/agent_ab"
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
    for row in rows:
        # A manifest may set `reps` per task, so the primary endpoint can be run deep while the
        # controls stay cheap. Controls exist to show that nothing moved where nothing should;
        # they do not need the primary's precision to do that, and spending equal effort on them
        # buys hours of wall clock and no evidence.
        count = int(row.get("reps", reps))
        for rep in range(1, count + 1):
            expanded.append(
                {
                    **row,
                    "task_id": f"{row['task_id']}#r{rep}",
                    "base_task_id": row["task_id"],
                    "rep": rep,
                }
            )
    return expanded


def environment_capture(model: str, source: Any, check: dict[str, Any]) -> dict[str, Any]:
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
        # Resolved, not "claude": the npm shim is a batch file that cannot be started without a
        # shell, so a bare name here returns an empty string and the artifact silently loses the
        # one field that says which CLI produced the run.
        "claude_code_version": run(resolve_claude_executable(), "--version"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "model": model,
        "recall_tenant": source.tenant,
        # The whole trust picture, in the artifact, so a reader never has to ask
        # whether this run was calibrated.
        "recall_transport": check.get("transport", "http"),
        "recall_handshake_ms": check.get("handshake_ms"),
        "recall_trust_state": check.get("trust_state"),
        "recall_calibrated": check.get("calibrated"),
        "recall_generation_id": check.get("generation_id"),
        "recall_calibration_id": check.get("calibration_id"),
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--comparison",
        choices=("additive", "headline", "ceiling"),
        default="additive",
        help="additive = claude_md vs claude_md+recall, which isolates the memory layer and is "
        "the configuration a real user runs; headline = claude_md vs recall ALONE, which "
        "measures replacing the file; ceiling = bare vs recall alone",
    )
    parser.add_argument("--reps", type=int, default=1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--tasks", default=str(TASKS))
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--port", type=int, default=5480)
    parser.add_argument("--model", default="anthropic/claude-haiku-4.5")
    parser.add_argument("--memory-index", default=None)
    parser.add_argument("--pair-concurrency", type=int, default=1)
    parser.add_argument(
        "--transport",
        choices=("stdio", "http"),
        default="stdio",
        help="stdio serves a CALIBRATED generation; http is warm but development-only",
    )
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

    if args.transport == "stdio":
        source: StdioRecallSpec | WarmRecallServer = StdioRecallSpec(
            dsn=args.dsn, cwd=REPO_ROOT, tenant=args.tenant
        )
        context: Any = contextlib.nullcontext()
    else:
        source = WarmRecallServer(
            dsn=args.dsn, cwd=REPO_ROOT, tenant=args.tenant, port=args.port
        )
        context = source

    with context:
        check = await source.check()
        print(
            f"RE-call ready: {check['handshake_ms']} ms handshake, {check['tool_count']} tools, "
            f"trust_state={check.get('trust_state')} calibrated={check.get('calibrated')}"
        )
        if check.get("calibrated") is not True:
            # Not fatal, but it changes what the result means, so it is said once at the top of
            # the run and recorded in the environment capture rather than discovered afterwards.
            print(
                "  [note] this corpus is UNCALIBRATED: abstention is untuned and the result must "
                "say so wherever it is published."
            )
        recall_prompt = write_recall_prompt(artifacts / "recall-prompt.txt", source)
        static_prompt: Path | None = None
        if args.comparison in {"headline", "additive"}:
            off_profile = "claude_md"
            static_prompt = write_claude_md_prompt(
                artifacts / "static-memory-prompt.txt", static_sources, repo_root=REPO_ROOT
            )
            off_spec = ArmSpec.claude_md(static_prompt)
        else:
            off_profile = "bare"
            off_spec = ArmSpec.bare()

        if args.comparison == "additive":
            # The arm a real user runs: the hand-written file AND the memory layer. Both arms hold
            # the same static bundle, so the only difference is RE-call.
            if not isinstance(source, StdioRecallSpec):
                raise SystemExit("the additive comparison requires --transport stdio")
            assert static_prompt is not None
            combined = write_claude_md_recall_prompt(
                artifacts / "static-plus-recall-prompt.txt",
                static_prompt=static_prompt,
                server_name=source.server_name,
                tool_prefix=source.tool_prefix(),
            )
            recall_spec = ArmSpec.claude_md_recall(
                source, artifacts / "recall-mcp.json", combined
            )
            # The claim rests on the two arms holding the SAME static memory, so it is asserted
            # rather than assumed: the on arm's prompt must begin with the off arm's, byte for
            # byte. A silent divergence here would look like a treatment effect.
            static_text = static_prompt.read_text(encoding="utf-8").rstrip()
            if not combined.read_text(encoding="utf-8").startswith(static_text):
                raise SystemExit(
                    "the additive arm's prompt does not start with the static bundle the off arm "
                    "receives; the arms would differ by more than RE-call"
                )
            print(
                f"additive arms: static {len(static_text)} chars in BOTH; "
                f"on arm adds RE-call + {len(combined.read_text(encoding='utf-8')) - len(static_text)} chars"
            )
        elif isinstance(source, StdioRecallSpec):
            recall_spec = ArmSpec.recall_stdio(
                source, artifacts / "recall-mcp.json", recall_prompt
            )
        else:
            recall_spec = ArmSpec.recall(source, recall_prompt)
        configs = build_configs(
            {RECALL_ON: recall_spec, RECALL_OFF: off_spec},
            model=args.model,
            cwd=workdir,
        )
        configs = {
            variant: replace(config, stream_dir=artifacts / "streams")
            for variant, config in configs.items()
        }
        environment = environment_capture(args.model, source, check)

        # Appended to as each session finishes, and fsynced. The previous run died at 71 of 100
        # sessions and lost its index entirely, because records.jsonl was written once at the end;
        # the transcripts survived only because the adapter writes those before parsing. A run
        # that dies should cost the sessions it did not do, not the ones it did.
        progress_path = artifacts / "records.partial.jsonl"
        progress_lock = asyncio.Lock()
        completed = 0

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
            stamped = record.__class__.from_mapping(
                {**record.to_dict(), "metadata": merged}
            )
            nonlocal completed
            async with progress_lock:
                completed += 1
                with progress_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(stamped.to_dict(), sort_keys=True) + "\n")
                    handle.flush()
                    os.fsync(handle.fileno())
                print(
                    f"  [{completed}/{len(tasks) * 2}] {stamped.task_id} {variant} "
                    f"in={stamped.input_tokens} recall_calls={stamped.recall_call_count}",
                    flush=True,
                )
            return stamped

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
