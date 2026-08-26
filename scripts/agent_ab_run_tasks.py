"""Run the preregistered task-success comparison: real work, executable endpoints.

    python -u scripts/agent_ab_run_tasks.py --run-id agent-ab-tasksuccess-001 \
        --dsn postgresql://recall:recall@127.0.0.1:5407/agent_ab --tenant default \
        > benchmarks/artifacts/agent_ab/tasksuccess-001.log 2>&1

**Start with `--limit 2 --reps 1`.** A smoke pair costs minutes and catches the class of mistake
that otherwise costs the run.

This is a separate script from `scripts/agent_ab_run.py` because three things differ, and each of
them would be wrong to bolt onto the trap runner:

1. **Every session gets its own sandbox.** The trap runner passes ONE `cwd` to both arms and every
   task, which is right when nothing is written and wrong the moment anything is. Here each session
   is restored from a committed fixture into its own directory, and the tree digest is recorded so
   the gate can refuse a pair whose arms did not start from the same ground.
2. **The arms can write.** `BASE_TOOLS` is read-only, so `Write` and `Edit` are added, identically
   in both arms and recorded in the artifact.
3. **Scoring runs the artifact.** After each session its checker executes what the agent produced,
   against an oracle the sandbox never contained.

⛔ **Redirect to a log file and never pipe through `grep`.** A previous run died at 71 of 100
sessions and its cause is permanently unknown, because the invocation was piped through `grep -v`,
which swallowed stderr and masked the exit code. Records are appended and fsynced per session for
the same reason: a run that dies should cost the sessions it did not do, not the ones it did.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import platform
import re
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
)
from benchmarks.agent_ab.claude_exec import (  # noqa: E402
    resolve_claude_executable,
    run_claude_case,
)
from benchmarks.agent_ab.gate import admit_pairs  # noqa: E402
from benchmarks.agent_ab.io import write_jsonl  # noqa: E402
from benchmarks.agent_ab.recall_server import StdioRecallSpec  # noqa: E402
from benchmarks.agent_ab.sdk_exec import (  # noqa: E402
    SDKExecConfig,
    build_sdk_configs,
    run_sdk_case,
    sdk_version,
)
from benchmarks.agent_ab.runner import run_paired  # noqa: E402
from benchmarks.agent_ab.sandbox import restore  # noqa: E402
from benchmarks.agent_ab.schema import RECALL_OFF, RECALL_ON, SessionRecord  # noqa: E402
from benchmarks.agent_ab.summarize import summarize_recall_overhead  # noqa: E402
from benchmarks.agent_ab.tasksuccess import TASKS, TASKS_BY_ID, check_workspace  # noqa: E402

DEFAULT_DSN = "postgresql://recall:recall@127.0.0.1:5407/agent_ab"
STATIC_MEMORY_SOURCES = ("CLAUDE.md",)
QUALIFICATION = REPO_ROOT / "benchmarks" / "agent_ab" / "task-qualification.json"
#: Added to `BASE_TOOLS` for both arms. A task-success benchmark whose agent cannot write a file
#: measures nothing, and the pair of tools is identical on both sides so it cannot confound.
WRITE_TOOLS: tuple[str, ...] = ("Write", "Edit")


def openrouter_env() -> dict[str, str]:
    """The three variables that make `claude -p` work on this machine.

    Built here rather than left to the shell because forgetting them does not fail loudly:
    `claude -p` on the subscription token returns `401 OAuth access token has been revoked` **and
    exits 0**, so a harness reading exit codes scores the failure as success. That cost one
    experiment two days. `ANTHROPIC_API_KEY` must be present and EMPTY, not absent, or the CLI
    prefers it over the auth token.
    """

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit(
            "OPENROUTER_API_KEY is not set. The subscription token returns 401 and exits 0, so "
            "running without this would produce a run of null sessions that look successful."
        )
    return {
        "ANTHROPIC_BASE_URL": "https://openrouter.ai/api",
        "ANTHROPIC_AUTH_TOKEN": key,
        "ANTHROPIC_API_KEY": "",
    }


def _cli_version_tuple(text: str | None) -> tuple[int, ...] | None:
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", text or "")
    return tuple(int(part) for part in match.groups()) if match else None


async def preflight(config_env: dict[str, str], model: str, driver: str = "cli") -> None:
    """Prove the driver answers before spending a hundred sessions finding out that it does not."""

    from benchmarks.agent_ab.claude_exec import ClaudeExecConfig

    row = {"task_id": "preflight", "user_input": "Reply with the single word READY and nothing else."}
    if driver == "sdk":
        sdk_probe = SDKExecConfig(
            model=model,
            cwd=REPO_ROOT,
            timeout_s=180.0,
            env=config_env,
            bare=True,
            allowed_tools=(),
        )
        record = await run_sdk_case(row, RECALL_OFF, sdk_probe)
        # The SDK may resolve a different CLI than the shell does, and below 2.1.221 a session
        # runs without a pending stdio MCP server while reporting success. The init-reported
        # version is what THIS driver actually launched, so it is the one to assert on.
        version = _cli_version_tuple(str(record.metadata.get("claude_code_version") or ""))
        if version is not None and version < (2, 1, 221):
            raise SystemExit(
                f"the SDK launched Claude Code {version}, below the 2.1.221 stdio-MCP wait; "
                f"an on arm under this version silently runs without its server"
            )
    else:
        probe = ClaudeExecConfig(
            model=model,
            cwd=REPO_ROOT,
            timeout_s=180.0,
            env=config_env,
            bare=True,
            strict_mcp_config=False,
            allowed_tools=(),
        )
        record = await run_claude_case(row, RECALL_OFF, probe)
    text = (record.response or "").strip()
    if not text:
        raise SystemExit(f"preflight produced no response; error={record.error!r}")
    if "401" in text or "revoked" in text.lower():
        raise SystemExit(f"preflight came back with an auth failure: {text[:200]}")
    # Any API error, not just 401. Measured 2026-08-22: an exhausted OpenRouter balance answers
    # the preflight with "API Error: 402 This request requires more credits", which this check's
    # 401-only predecessor printed as the preflight RESPONSE and carried on, spending the whole
    # smoke on sessions that each died with the same 402. The admission gate caught it downstream
    # ("No admitted pairs"), but the gate is the last line, not the first.
    if text.startswith("API Error") or record.error:
        raise SystemExit(f"preflight came back with an API failure: {text[:200]}")
    print(f"preflight: {text[:60]!r}, {record.input_tokens} input tokens\n")


def load_tasks(limit: int | None, reps: int | None) -> list[dict[str, Any]]:
    """Expand the task set by repetition, keeping every task_id unique.

    `run_paired` requires unique ids within one run, so a repetition is a distinct task carrying
    its origin. Collapsing them back for the per-task view is the analyser's job.
    """

    rows = [task.to_row() for task in TASKS]
    if limit:
        rows = rows[:limit]
    expanded: list[dict[str, Any]] = []
    for row in rows:
        count = int(reps if reps is not None else row["reps"])
        for rep in range(1, count + 1):
            expanded.append(
                {**row, "task_id": f"{row['task_id']}#r{rep}", "base_task_id": row["task_id"], "rep": rep}
            )
    return expanded


def completed_pairs(progress: Path) -> tuple[set[str], list[dict[str, Any]]]:
    """Task ids whose BOTH arms are already recorded, and the rows to carry forward.

    A run that dies should cost the sessions it did not do, not the ones it did. `records.partial`
    is appended and fsynced per session for exactly this, and it is the only reason the first
    attempt at this run is not a total loss: it died at 46 of 112 sessions when the process that
    launched it exited, taking the `nohup`'d child with it.

    ⚠️ **A pair with only ONE arm recorded is not carried forward, it is re-run.** Half a pair is
    not evidence: pairing is the whole design, and splicing an arm measured before a crash onto one
    measured after it would silently mix two environments inside a single comparison. The orphan
    row is dropped from the carried set so it cannot be counted twice either.
    """

    if not progress.is_file():
        return set(), []
    rows = [
        json.loads(line)
        for line in progress.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    variants: dict[str, set[str]] = {}
    for row in rows:
        variants.setdefault(str(row["task_id"]), set()).add(str(row["variant"]))
    done = {task_id for task_id, seen in variants.items() if len(seen) == 2}
    carried = [row for row in rows if str(row["task_id"]) in done and not row.get("error")]
    # A session that errored is not a completed session, so its pair is re-run rather than kept.
    errored = {str(row["task_id"]) for row in rows if row.get("error")}
    done -= errored
    carried = [row for row in carried if str(row["task_id"]) not in errored]
    return done, carried


def load_loci() -> dict[str, str]:
    """Where each task's fact lives, read from the COMMITTED qualification.

    Read rather than recomputed on purpose. Re-qualifying inside the run would let the loci move
    between the predictions and the result, which is the one thing the qualification step exists
    to prevent.
    """

    if not QUALIFICATION.is_file():
        raise SystemExit(
            f"{QUALIFICATION} is missing. Run scripts/agent_ab_qualify_tasks.py and COMMIT it "
            f"before measuring: choosing which tasks count after seeing the result is the "
            f"difference between a benchmark and a press release."
        )
    payload = json.loads(QUALIFICATION.read_text(encoding="utf-8"))
    return {entry["trap_id"]: entry["locus"] for entry in payload["qualifications"]}


def environment_capture(
    model: str,
    spec: StdioRecallSpec,
    check: dict[str, Any],
    *,
    instruction_file: str | None = None,
    driver: str = "cli",
) -> dict[str, Any]:
    def run(*command: str) -> str:
        try:
            return subprocess.run(  # noqa: S603 - argv list, no shell
                command, capture_output=True, text=True, timeout=60
            ).stdout.strip()
        except Exception:  # noqa: BLE001 - a missing tool must not abort a run
            return ""

    return {
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "repo_revision": run("git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"),
        "repo_dirty": bool(run("git", "-C", str(REPO_ROOT), "status", "--porcelain")),
        "claude_code_version": run(resolve_claude_executable(), "--version"),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "model": model,
        "recall_tenant": spec.tenant,
        "recall_transport": check.get("transport", "stdio"),
        "recall_handshake_ms": check.get("handshake_ms"),
        "recall_trust_state": check.get("trust_state"),
        "recall_calibrated": check.get("calibrated"),
        "recall_generation_id": check.get("generation_id"),
        "recall_calibration_id": check.get("calibration_id"),
        "write_tools": list(WRITE_TOOLS),
        # Which driver launched the sessions. `cli` is claude_exec's subprocess over stream-json,
        # the driver of every archived baseline; `sdk` is sdk_exec over claude-agent-sdk.
        "driver": driver,
        "claude_agent_sdk_version": sdk_version() if driver == "sdk" else None,
        # Which instruction the on arm ran under. The formatted text itself is already kept in the
        # artifact as static-plus-recall-prompt.txt; this names the committed source it came from,
        # so two runs differing only here can be told apart without diffing prompt files.
        "instruction_file": instruction_file or "arms.RECALL_SYSTEM_PROMPT",
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--model", default="anthropic/claude-haiku-4.5")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--reps", type=int, default=None, help="override every task's reps")
    parser.add_argument("--pair-concurrency", type=int, default=1)
    parser.add_argument("--timeout-s", type=float, default=1800.0)
    parser.add_argument(
        "--instruction-file",
        default=None,
        help=(
            "COMMITTED file holding the on arm's RE-call instruction template, with {server} and "
            "{tool} placeholders. Default: the one-sentence RECALL_SYSTEM_PROMPT. This is the one "
            "field an instruction-variant run changes; everything else stays identical, so a "
            "behavioural difference against a baseline run on the same tasks is attributable to "
            "the instruction text and to nothing else."
        ),
    )
    parser.add_argument(
        "--driver",
        choices=("cli", "sdk"),
        default="cli",
        help=(
            "how sessions are launched: 'cli' (claude_exec subprocess over stream-json, the "
            "driver of every archived baseline) or 'sdk' (sdk_exec over claude-agent-sdk, which "
            "needs the agent extra installed). Everything else in the run is shared verbatim, "
            "which is what makes a driver-equivalence comparison a comparison."
        ),
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="continue a run that died, skipping pairs whose BOTH arms are already recorded",
    )
    args = parser.parse_args()

    artifacts = REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / args.run_id
    progress_path = artifacts / "records.partial.jsonl"
    if artifacts.exists() and any(artifacts.iterdir()) and not args.resume:
        print(
            f"{artifacts} already has contents. A run identifier names one immutable run; "
            f"choose a new one rather than overwriting the evidence, or pass --resume to "
            f"continue this one from its recorded sessions."
        )
        return 1
    already, carried = completed_pairs(progress_path) if args.resume else (set(), [])
    (artifacts / "streams").mkdir(parents=True, exist_ok=True)
    work_root = artifacts / "work"
    work_root.mkdir(parents=True, exist_ok=True)

    loci = load_loci()
    tasks = load_tasks(args.limit, args.reps)
    planned = len(tasks)
    if already:
        tasks = [row for row in tasks if str(row["task_id"]) not in already]
        print(
            f"resuming {args.run_id}: {len(already)} pairs already recorded and carried forward, "
            f"{len(tasks)} of {planned} still to run"
        )
    print(f"run {args.run_id}: additive, {len(tasks)} paired sessions, {len(tasks) * 2} in total\n")
    if not tasks:
        print("nothing left to run; re-deriving the artifacts from the recorded sessions")

    agent_env = openrouter_env()
    await preflight(agent_env, args.model, args.driver)

    spec = StdioRecallSpec(dsn=args.dsn, cwd=REPO_ROOT, tenant=args.tenant)
    check = await spec.check()
    print(
        f"RE-call ready: {check['handshake_ms']} ms handshake, {check['tool_count']} tools, "
        f"trust_state={check.get('trust_state')} calibrated={check.get('calibrated')}"
    )
    if check.get("calibrated") is not True:
        print("  [note] this corpus is UNCALIBRATED and the result must say so wherever published")

    static_prompt = write_claude_md_prompt(
        artifacts / "static-memory-prompt.txt", STATIC_MEMORY_SOURCES, repo_root=REPO_ROOT
    )
    instruction_template: str | None = None
    if args.instruction_file:
        instruction_path = Path(args.instruction_file)
        if not instruction_path.is_file():
            raise SystemExit(f"--instruction-file does not exist: {instruction_path}")
        instruction_template = instruction_path.read_text(encoding="utf-8")
    combined = write_claude_md_recall_prompt(
        artifacts / "static-plus-recall-prompt.txt",
        static_prompt=static_prompt,
        server_name=spec.server_name,
        tool_prefix=spec.tool_prefix(),
        instruction=instruction_template,
    )
    static_text = static_prompt.read_text(encoding="utf-8").rstrip()
    combined_text = combined.read_text(encoding="utf-8")
    if static_text not in combined_text:
        raise SystemExit(
            "the additive arm's prompt does not contain the off arm's bundle verbatim; the arms "
            "would differ by more than RE-call"
        )
    extra = len(combined_text) - len(static_text)
    if extra > 2000:
        raise SystemExit(f"the on arm's prompt adds {extra} chars, too large to be the instruction")
    print(f"additive arms: static {len(static_text)} chars in BOTH; on arm adds {extra}, first\n")

    off_spec = ArmSpec.claude_md(static_prompt)
    on_spec = ArmSpec.claude_md_recall(spec, artifacts / "recall-mcp.json", combined)
    # One template pair, built once so everything except memory is identical by construction. The
    # per-session `cwd` is the only field replaced afterwards.
    if args.driver == "sdk":
        templates: dict[str, Any] = build_sdk_configs(
            {RECALL_ON: on_spec, RECALL_OFF: off_spec},
            recall_spec=spec,
            model=args.model,
            cwd=work_root,
            timeout_s=args.timeout_s,
            env=agent_env,
            extra_allowed_tools=WRITE_TOOLS,
        )
    else:
        templates = build_configs(
            {RECALL_ON: on_spec, RECALL_OFF: off_spec},
            model=args.model,
            cwd=work_root,
            timeout_s=args.timeout_s,
            env=agent_env,
            extra_allowed_tools=WRITE_TOOLS,
        )
    templates = {
        variant: replace(config, stream_dir=artifacts / "streams")
        for variant, config in templates.items()
    }
    run_one = run_sdk_case if args.driver == "sdk" else run_claude_case

    progress_lock = asyncio.Lock()
    completed = 0

    async def run_case(row: dict[str, Any], variant: str):
        base_id = str(row["base_task_id"])
        task = TASKS_BY_ID[base_id]
        workdir = work_root / str(row["task_id"]).replace("#", "-") / variant
        if workdir.exists():
            # A half-finished pair from a previous attempt leaves its sandbox behind, and `restore`
            # refuses a directory with contents rather than cleaning it, because a stale artifact
            # is indistinguishable from this session's work. On a resume the directory is known to
            # be dead, so it is moved aside rather than deleted: whatever the dead session wrote is
            # still readable if the resumed pair later looks wrong.
            attempt = 1
            while (aside := workdir.parent / f"{workdir.name}.abandoned{attempt}").exists():
                attempt += 1
            workdir.rename(aside)
        digest = restore(task.workspace, workdir)
        config = replace(templates[variant], cwd=workdir)

        record = await run_one(row, variant, config)

        # Scored after the session, from the sandbox alone. `check_workspace` turns a checker
        # fault into passed=False with its traceback rather than aborting the run, and marks it so
        # "the artifact did not work" stays distinguishable from "the checker broke".
        verdict = check_workspace(task, workdir)
        merged = {
            **record.metadata,
            "off_arm_profile": "claude_md",
            "comparison": "additive",
            "base_task_id": base_id,
            "rep": row.get("rep"),
            "family": task.family,
            "locus": loci.get(base_id),
            "governing_memo": task.governing_memo,
            "workspace_digest": digest,
            "workdir": str(workdir.relative_to(REPO_ROOT)),
            "check": verdict.to_dict(),
        }
        stamped = record.__class__.from_mapping(
            # `success` becomes the ENDPOINT, not the CLI's exit status. A session that finished
            # cleanly and produced a broken artifact is a failure of the work, which is the whole
            # question; the CLI's own verdict stays in metadata under `cli_success`.
            {
                **record.to_dict(),
                "success": bool(verdict.passed),
                "metadata": {**merged, "cli_success": record.success},
            }
        )

        nonlocal completed
        async with progress_lock:
            completed += 1
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(stamped.to_dict(), sort_keys=True) + "\n")
                handle.flush()
                os.fsync(handle.fileno())
            mark = "PASS" if verdict.passed else "fail"
            print(
                f"  [{completed}/{len(tasks) * 2}] {stamped.task_id} {variant} {mark} "
                f"in={stamped.input_tokens} recall_calls={stamped.recall_call_count} "
                f"| {verdict.evidence[:80]}",
                flush=True,
            )
        return stamped

    fresh = await run_paired(tasks, run_case, pair_concurrency=args.pair_concurrency)

    # Carried-forward sessions rejoin here, so `records.jsonl` is the whole run rather than
    # whatever survived the last restart. They are reconstructed from their recorded rows, not
    # re-derived: the artifact must say what those sessions actually did, and a re-derivation would
    # be this run's opinion of a session that happened under a different process.
    records = [SessionRecord.from_mapping(row) for row in carried] + list(fresh)
    if carried:
        print(f"\ncarried forward {len(carried)} recorded sessions from earlier attempts")

    report = admit_pairs(records)
    admitted = list(report.admitted)
    parity_failures = digest_parity_failures(admitted)
    if parity_failures:
        print(f"\n[warn] {len(parity_failures)} pairs started from different trees: {parity_failures}")

    write_jsonl(artifacts / "records.jsonl", records)
    for name, payload in (
        ("admission.json", report.summary()),
        ("recall-overhead.json", summarize_recall_overhead(admitted)),
        ("environment.json", environment_capture(args.model, spec, check, instruction_file=args.instruction_file, driver=args.driver)),
        ("digest-parity.json", {"failures": parity_failures}),
    ):
        (artifacts / name).write_text(
            json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8", newline="\n"
        )

    print(
        f"\n{len(records)} records, {report.summary()['admitted_pairs']} pairs admitted, "
        f"{report.discarded_pair_count} discarded"
    )
    for task_id in report.discarded_task_ids:
        print(f"  discarded {task_id}")
    print(f"\nartifacts: {artifacts}")
    if not admitted:
        print("No admitted pairs. This is a wiring result, not a measurement.")
        return 1
    return 0


def digest_parity_failures(records: list[Any]) -> list[str]:
    """Task ids whose two arms did not start from the same sandbox.

    The same rule as the admission gate and for the same reason: a difference in starting state
    would look exactly like a treatment effect, and there would be nothing in the artifact to say
    otherwise.
    """

    by_task: dict[str, dict[str, str]] = {}
    for record in records:
        digest = record.metadata.get("workspace_digest")
        if digest:
            by_task.setdefault(record.task_id, {})[record.variant] = str(digest)
    return sorted(
        task_id
        for task_id, digests in by_task.items()
        if len(digests) == 2 and len(set(digests.values())) != 1
    )


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
