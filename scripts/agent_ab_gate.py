"""The pre-measurement gate: prove the treatment reaches the on arm, and misses the off arm.

Run this before any paired measurement, and again whenever the CLI, the corpus or the server
configuration changes. It costs two short sessions. Skipping it costs the run.

    python scripts/agent_ab_gate.py --dsn postgresql://recall:recall@127.0.0.1:5406/agent_ab

Four conditions, in order, each of which has failed in practice:

1. The RE-call server completes an MCP `initialize` and lists its tools.
2. A controlled `recall_search` succeeds, and its latency and trust state are recorded.
3. A `recall_on` session has the RE-call tools in its own tool list, and calls one.
4. A `recall_off` session has no RE-call tool and calls none.

Condition 3 is the one that has silently failed before. On 2026-08-20 an on-arm session against a
cold stdio server ran with no RE-call tool at all and reported `success` with no error, because
Claude Code 2.1.220 does not wait for a pending MCP server. **2.1.221 and later do wait**, and
report `failed` instead of `pending` when a server really cannot start, which is what makes the
stdio transport usable at all.

## Why the transport matters

`--transport stdio` is the default and is the only one that can serve a **calibrated** corpus.
RE-call reads generations only under `RECALL_ENV=production`, and production refuses the static
bearer token an HTTP listener would need, so a calibrated generation and a warm socket cannot be
had together without OIDC. stdio authenticates nobody because there is no remote caller, so it can.
The cost is roughly 11 s of per-session startup, which only the 2.1.221+ wait makes affordable.

`--transport http` keeps the 458 ms warm connect but runs in development mode against the legacy
store, so it is **uncalibrated** and says so.

Credentials come from the environment and are never printed:
`ANTHROPIC_BASE_URL`, `ANTHROPIC_AUTH_TOKEN` (or `OPENROUTER_API_KEY`).
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agent_ab.arms import ArmSpec, build_configs  # noqa: E402
from benchmarks.agent_ab.claude_exec import run_claude_case  # noqa: E402
from benchmarks.agent_ab.gate import check_session  # noqa: E402
from benchmarks.agent_ab.recall_server import StdioRecallSpec, WarmRecallServer  # noqa: E402
from benchmarks.agent_ab.schema import RECALL_OFF, RECALL_ON  # noqa: E402

DEFAULT_DSN = "postgresql://recall:recall@127.0.0.1:5406/agent_ab"
PROBE_TASK = {
    "task_id": "gate-probe",
    "user_input": (
        "How do I limit the number of CPU threads the embedder uses on this machine? Answer in "
        "one sentence. If a memory search tool is available to you, use it first."
    ),
}


@contextlib.contextmanager
def _null_context():
    yield


def _ok(label: str, detail: str = "") -> None:
    print(f"  [pass] {label}" + (f" - {detail}" if detail else ""))


def _fail(label: str, detail: str = "") -> None:
    print(f"  [FAIL] {label}" + (f" - {detail}" if detail else ""))


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=os.environ.get("RECALL_AB_DSN", DEFAULT_DSN))
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--transport", choices=("stdio", "http"), default="stdio")
    parser.add_argument("--port", type=int, default=5480, help="http transport only")
    parser.add_argument(
        "--model", default=os.environ.get("AGENT_AB_MODEL", "anthropic/claude-haiku-4.5")
    )
    parser.add_argument("--workdir", default=None)
    args = parser.parse_args()

    failures: list[str] = []
    workdir = (
        Path(args.workdir)
        if args.workdir
        else REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / "gate-workdir"
    )
    workdir.mkdir(parents=True, exist_ok=True)

    print(f"\n1/2  RE-call over {args.transport}, tenant {args.tenant!r}")
    if args.transport == "stdio":
        source: StdioRecallSpec | WarmRecallServer = StdioRecallSpec(
            dsn=args.dsn, cwd=REPO_ROOT, tenant=args.tenant
        )
        context: object = _null_context()
    else:
        source = WarmRecallServer(
            dsn=args.dsn, cwd=REPO_ROOT, tenant=args.tenant, port=args.port
        )
        context = source

    with context:  # type: ignore[attr-defined]
        try:
            report = await source.check()
        except Exception as error:  # noqa: BLE001 - the report is the point of this script
            _fail("MCP handshake / recall_search", f"{type(error).__name__}: {error}")
            if isinstance(source, WarmRecallServer):
                print(source.log_tail())
            return 1
        _ok("MCP initialize", f"{report['handshake_ms']} ms, {report['tool_count']} tools")
        _ok("recall_search", f"{report['search_ms']} ms, abstained={report['abstained']}")

        if report.get("trust_state") == "trusted" and report.get("calibrated") is True:
            _ok(
                "trust",
                f"calibrated, generation={str(report.get('generation_id'))[:24]}, "
                f"calibration={str(report.get('calibration_id'))[:24]}",
            )
        else:
            print(
                f"  [note] trust_state={report.get('trust_state')!r} "
                f"calibrated={report.get('calibrated')!r}: this corpus is UNCALIBRATED. That is "
                f"a fact about the result and must be stated wherever it is published."
            )

        print(f"\n2/2  paired probe sessions ({args.model})")
        if isinstance(source, StdioRecallSpec):
            recall_spec = ArmSpec.recall_stdio(source, workdir / "recall-mcp.json")
        else:
            recall_spec = ArmSpec.recall(source)
        specs = {RECALL_ON: recall_spec, RECALL_OFF: ArmSpec.bare()}
        configs = build_configs(specs, model=args.model, cwd=workdir)

        for variant in (RECALL_ON, RECALL_OFF):
            record = await run_claude_case(PROBE_TASK, variant, configs[variant])
            verdict = check_session(
                record, recall_tool_prefix=configs[variant].recall_tool_prefix
            )
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

    print()
    if failures:
        print(f"GATE FAILED for {failures}. Do not run a measurement until this passes.")
        return 1
    print("GATE PASSED. The on arm has RE-call, the off arm does not.")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
