"""The pre-measurement gate: prove the treatment reaches the on arm, and misses the off arm.

Run this before any paired measurement, and again whenever the CLI, the corpus or the server
configuration changes. It costs two short sessions. Skipping it costs the run.

    python scripts/agent_ab_gate.py

Four conditions, in order, each of which has failed in practice:

1. The warm RE-call server completes an MCP `initialize` and lists its tools.
2. A controlled `recall_search` succeeds, and its latency is recorded.
3. A `recall_on` session has the RE-call tools in its own tool list, and calls one.
4. A `recall_off` session has no RE-call tool and calls none.

Condition 3 is the one that has silently failed before. On 2026-08-20 an on-arm session against a
cold stdio server ran with no RE-call tool at all and reported `success` with no error, because
Claude Code 2.1.220 does not wait for a pending MCP server. That run is indistinguishable from a
genuine null result unless something checks the tool list, which is what this does.

Credentials come from the environment and are never printed:
`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` (or `OPENROUTER_API_KEY`).
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agent_ab.arms import ArmSpec, build_configs  # noqa: E402
from benchmarks.agent_ab.claude_exec import run_claude_case  # noqa: E402
from benchmarks.agent_ab.gate import check_session  # noqa: E402
from benchmarks.agent_ab.recall_server import WarmRecallServer  # noqa: E402
from benchmarks.agent_ab.schema import RECALL_OFF, RECALL_ON  # noqa: E402

DEFAULT_DSN = "postgresql://recall:recall@127.0.0.1:5433/recall"
PROBE_TASK = {
    "task_id": "gate-probe",
    "user_input": (
        "Why must a session in this repository start its own database container instead of "
        "using one that is already running? Answer in one sentence. If a memory search tool is "
        "available to you, use it first."
    ),
}


def _ok(label: str, detail: str = "") -> None:
    print(f"  [pass] {label}" + (f" — {detail}" if detail else ""))


def _fail(label: str, detail: str = "") -> None:
    print(f"  [FAIL] {label}" + (f" — {detail}" if detail else ""))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=os.environ.get("RECALL_AB_DSN", DEFAULT_DSN))
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--port", type=int, default=5480)
    parser.add_argument("--model", default=os.environ.get("AGENT_AB_MODEL", "anthropic/claude-haiku-4.5"))
    parser.add_argument(
        "--workdir",
        default=None,
        help="working directory for the probe sessions (default: a scratch directory)",
    )
    args = parser.parse_args()

    failures: list[str] = []
    workdir = Path(args.workdir) if args.workdir else REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / "gate-workdir"
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"\n1/2  warm RE-call server on 127.0.0.1:{args.port}, tenant {args.tenant!r}")
    with WarmRecallServer(dsn=args.dsn, cwd=REPO_ROOT, tenant=args.tenant, port=args.port) as server:
        try:
            report = await server.check()
        except Exception as error:  # noqa: BLE001 - the report is the point
            _fail("MCP handshake / recall_search", f"{type(error).__name__}: {error}")
            print(server.log_tail())
            return 1
        _ok("MCP initialize", f"{report['handshake_ms']} ms, {report['tool_count']} tools")
        _ok("recall_search", f"{report['search_ms']} ms, abstained={report['abstained']}")
        if report["trust_state"] != "healthy":
            print(
                f"  [note] trust_state={report['trust_state']!r} "
                f"failure_code={report['failure_code']!r} calibrated={report['calibrated']!r}: "
                f"this corpus is uncalibrated, which must be stated in any published result."
            )

        print(f"\n2/2  paired probe sessions ({args.model})")
        specs = {
            RECALL_ON: ArmSpec.recall(server),
            RECALL_OFF: ArmSpec.bare(),
        }
        configs = build_configs(specs, model=args.model, cwd=workdir)
        for variant in (RECALL_ON, RECALL_OFF):
            record = await run_claude_case(PROBE_TASK, variant, configs[variant])
            verdict = check_session(record, recall_tool_prefix=configs[variant].recall_tool_prefix)
            tools = record.metadata.get("recall_tools_available") or []
            detail = (
                f"tools={len(tools)} calls={record.recall_call_count} "
                f"servers={record.metadata.get('mcp_servers')}"
            )
            if verdict.admitted:
                _ok(f"{variant} session", detail)
            else:
                _fail(f"{variant} session", detail)
                for reason in verdict.reasons:
                    print(f"         reason: {reason}")
                failures.append(variant)
            for note in verdict.notes:
                print(f"         note: {note}")
            if variant == RECALL_ON and verdict.admitted and record.recall_call_count == 0:
                print(
                    "         [note] the tool was present and unused. That is admissible data, "
                    "but a probe designed to force a search should have forced one."
                )

    print()
    if failures:
        print(f"GATE FAILED for {failures}. Do not run a measurement until this passes.")
        return 1
    print("GATE PASSED. The on arm has RE-call, the off arm does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
