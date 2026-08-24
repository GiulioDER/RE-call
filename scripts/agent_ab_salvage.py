"""Rebuild a run's records from its saved transcripts when the runner died before writing them.

    python scripts/agent_ab_salvage.py --run-id agent-ab-traps-002

`run_claude_case` writes each session's stream to disk **before** parsing it, and the runner only
writes `records.jsonl` at the very end. So a runner that dies mid-run loses its index but not its
evidence: the transcripts are all there. This turns them back into records, scores the traps, and
writes the same artifacts a completed run would, minus the pairs that never finished.

Used in anger on 2026-08-21, when the headline run's process died at 71 of 100 sessions.

**Two things are different from a completed run, and both are recorded in the artifact rather than
smoothed over.**

`wall_time_ms` normally comes from the runner timing the subprocess. That measurement died with the
process, so this uses the session's own `duration_ms` from its result event. The two are not the
same: the runner's figure includes process spawn and MCP server startup, the session's does not.
Every salvaged record carries `metadata.salvaged = true` and `metadata.wall_time_source =
"stream_duration_ms"` so a reader can tell which they are looking at, and so the two are never
silently pooled.

A pair is only usable if BOTH arms finished. Unpaired survivors are reported and excluded, because
the whole design is paired and a lone arm is not a comparison.
"""

from __future__ import annotations

import argparse
import gzip
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agent_ab.arms import BASE_TOOLS  # noqa: E402
from benchmarks.agent_ab.claude_exec import (  # noqa: E402
    ClaudeExecConfig,
    build_record,
    parse_claude_stream_json,
    result_event,
)
from benchmarks.agent_ab.gate import admit_pairs  # noqa: E402
from benchmarks.agent_ab.io import write_jsonl  # noqa: E402
from benchmarks.agent_ab.schema import RECALL_OFF, RECALL_ON, VARIANTS  # noqa: E402
from benchmarks.agent_ab.summarize import (  # noqa: E402
    summarize_pairs,
    summarize_recall_overhead,
)
from benchmarks.agent_ab.traps import score_record  # noqa: E402


def split_name(name: str) -> tuple[str, str] | None:
    """`trap-x#r2.recall_on.jsonl.gz` -> `("trap-x#r2", "recall_on")`."""

    base = name[: -len(".jsonl.gz")] if name.endswith(".jsonl.gz") else name
    for variant in VARIANTS:
        if base.endswith(f".{variant}"):
            return base[: -len(variant) - 1], variant
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--tasks", default=str(REPO_ROOT / "benchmarks" / "agent_ab" / "tasks" / "traps.jsonl")
    )
    parser.add_argument("--recall-tool-prefix", default="mcp__recall-memory__")
    args = parser.parse_args()

    artifacts = REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / args.run_id
    streams = artifacts / "streams"
    if not streams.is_dir():
        raise SystemExit(f"no streams at {streams}")

    tasks = {
        row["task_id"]: row
        for row in (
            json.loads(line)
            for line in Path(args.tasks).read_text(encoding="utf-8").splitlines()
            if line.strip()
        )
    }

    records = []
    unreadable: list[str] = []
    for path in sorted(streams.iterdir()):
        parsed = split_name(path.name)
        if parsed is None:
            continue
        task_id, variant = parsed
        base = task_id.split("#")[0]
        row = dict(tasks.get(base, {}))
        row["task_id"] = task_id
        row["base_task_id"] = base
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as handle:
            stream = handle.read()
        try:
            events = parse_claude_stream_json(stream)
            result = result_event(events)
            # The runner's own timing died with it; the session's self-reported duration is the
            # only wall time that survived, and it measures a slightly different thing.
            duration = (result or {}).get("duration_ms")
            # strict_mcp_config=False only because this config never launches anything: it
            # exists to carry the tool prefix into build_record. Leaving it strict trips a real
            # guard that refuses "strict, but no servers", which is the right guard for a launch
            # and meaningless for a parse.
            config = ClaudeExecConfig(
                model="salvaged",
                allowed_tools=BASE_TOOLS,
                recall_tool_prefix=args.recall_tool_prefix,
                strict_mcp_config=False,
            )
            record = build_record(
                row,
                variant,
                stream=stream,
                wall_time_ms=float(duration) if duration is not None else 0.0,
                config=config,
                command=("salvaged",),
            )
            payload = record.to_dict()
            payload["metadata"] = {
                **payload.get("metadata", {}),
                "salvaged": True,
                "stream_path": path.name,
                "wall_time_source": "stream_duration_ms",
            }
            if duration is None:
                # Never 0.0: an unmeasured session must not read as an instant one.
                payload["wall_time_ms"] = None
            records.append(record.__class__.from_mapping(payload))
        except Exception as error:  # noqa: BLE001 - one bad transcript must not lose the rest
            unreadable.append(f"{path.name}: {type(error).__name__}: {error}"[:200])

    by_task: dict[str, set[str]] = defaultdict(set)
    for record in records:
        by_task[record.task_id].add(record.variant)
    complete = {t for t, v in by_task.items() if {RECALL_ON, RECALL_OFF} <= v}
    orphans = sorted(set(by_task) - complete)
    paired = [r for r in records if r.task_id in complete]

    print(f"salvaged {len(records)} records from {len(list(streams.iterdir()))} streams")
    print(f"complete pairs: {len(complete)}")
    if orphans:
        print(f"unpaired (excluded, one arm never finished): {len(orphans)} -> {orphans[:6]}")
    if unreadable:
        print(f"unreadable transcripts: {len(unreadable)}")
        for line in unreadable[:5]:
            print(f"  {line}")

    report = admit_pairs(paired, recall_tool_prefix=args.recall_tool_prefix)
    admitted = report.admitted
    print(
        f"gate: {len(admitted) // 2} pairs admitted, "
        f"{len(report.discarded_task_ids)} discarded"
    )
    for task_id in report.discarded_task_ids[:6]:
        print(f"  discarded: {task_id}")

    write_jsonl(artifacts / "records.jsonl", admitted)
    (artifacts / "summary.json").write_text(
        json.dumps(summarize_pairs(admitted), indent=2), encoding="utf-8"
    )
    (artifacts / "recall-overhead.json").write_text(
        json.dumps(summarize_recall_overhead(admitted), indent=2), encoding="utf-8"
    )
    (artifacts / "trap-scores.json").write_text(
        json.dumps([score_record(r) for r in admitted], indent=2), encoding="utf-8"
    )
    salvage: dict[str, Any] = {
        "run_id": args.run_id,
        "salvaged": True,
        "reason": "the runner process died before writing records.jsonl",
        "streams_seen": len(list(streams.iterdir())),
        "records_rebuilt": len(records),
        "complete_pairs": len(complete),
        "unpaired_excluded": orphans,
        "unreadable": unreadable,
        "gate_discarded": list(report.discarded_task_ids),
        "wall_time_source": "stream_duration_ms (the runner's own timing did not survive)",
    }
    (artifacts / "salvage.json").write_text(json.dumps(salvage, indent=2), encoding="utf-8")

    # environment.json is normally written by the runner at the end of a run, so a died run has
    # none. Most of it is recoverable from the transcripts themselves, which is better than a
    # placeholder: the CLI version and the MCP server status are recorded in every session's init
    # event. What cannot be recovered is marked null rather than guessed.
    if not (artifacts / "environment.json").exists():
        version, servers = None, None
        for record in admitted:
            meta = record.metadata or {}
            version = version or meta.get("claude_code_version")
            servers = servers or meta.get("mcp_servers")
        environment = {
            "reconstructed_from_transcripts": True,
            "note": "the runner died before writing environment.json; these fields come from the "
                    "sessions' own init events, and anything absent there is null, not guessed",
            "claude_code_version": version,
            "mcp_servers": servers,
            "recall_transport": "stdio",
            "recall_calibrated": None,
            "recall_trust_state": None,
            "recall_generation_id": None,
            "recall_calibration_id": None,
        }
        (artifacts / "environment.json").write_text(
            json.dumps(environment, indent=2), encoding="utf-8"
        )
    print(f"\nwritten: {artifacts / 'records.jsonl'} and summary/trap/salvage artifacts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
