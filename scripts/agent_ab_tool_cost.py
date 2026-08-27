"""Measure what one served tool costs, per turn, in input tokens.

Prior work: `benchmarks/agent_ab/sdk_exec.py` drives the sessions and computes the token fields
(reusing `claude_exec._usage_fields`, whose rule is fresh + cache-read + cache-creation from the
per-model aggregate), and `scripts/agent_ab_run_tasks.py::openrouter_env` supplies the model
credentials; neither is reimplemented here. Nothing in this repository measured tool-definition
overhead before: the two agent A/B runs measured task success and recorded token totals as a cost
surface, which is what surfaced the question this script answers.

Preregistered in `docs/preregistrations/2026-08-27-tool-definition-context-cost.md`.

This deliberately uses NO task machinery: no sandbox, no checker, no admission gate. The session
is one trivial prompt that calls nothing, so the only thing varying between arms is how many tool
definitions were injected into its context.

    python scripts/agent_ab_tool_cost.py --reps 5 \
        --dsn postgresql://recall:recall@127.0.0.1:5407/agent_ab --tenant default
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agent_ab.claude_exec import resolve_claude_executable  # noqa: E402
from benchmarks.agent_ab.recall_server import StdioRecallSpec  # noqa: E402
from benchmarks.agent_ab.schema import RECALL_OFF  # noqa: E402
from benchmarks.agent_ab.sdk_exec import SDKExecConfig, run_sdk_case, sdk_mcp_servers  # noqa: E402
from scripts.agent_ab_run_tasks import openrouter_env  # noqa: E402

#: One turn, calls nothing. The point is the context the session STARTS with.
PROMPT = "Reply with the single word READY and nothing else."


async def measure(name: str, servers, tools: int, reps: int, env, cli_path: str) -> dict:
    config = SDKExecConfig(
        model="anthropic/claude-haiku-4.5",
        cwd=REPO_ROOT,
        timeout_s=300.0,
        env=env,
        bare=True,
        cli_path=cli_path,
        mcp_servers=servers,
        allowed_tools=(),
    )
    rows = []
    for rep in range(1, reps + 1):
        record = await run_sdk_case(
            {"task_id": f"toolcost-{name}-r{rep}", "user_input": PROMPT}, RECALL_OFF, config
        )
        served = [t for t in (record.metadata.get("session_tools") or []) if t.startswith("mcp__")]
        rows.append(
            {
                "rep": rep,
                "input_tokens": record.input_tokens,
                "turns": record.model_turns,
                "mcp_tools_in_init": len(served),
                "error": record.error,
            }
        )
        print(
            f"  {name:12s} rep{rep}  tools={len(served):2d}  turns={record.model_turns}"
            f"  input={record.input_tokens}",
            flush=True,
        )
    good = [r for r in rows if r["error"] is None and r["input_tokens"] is not None]
    return {
        "arm": name,
        "tools_intended": tools,
        "tools_observed": sorted({r["mcp_tools_in_init"] for r in good}),
        "n": len(good),
        "median_input_tokens": statistics.median([r["input_tokens"] for r in good]) if good else None,
        "min_input_tokens": min((r["input_tokens"] for r in good), default=None),
        "max_input_tokens": max((r["input_tokens"] for r in good), default=None),
        "median_turns": statistics.median([r["turns"] for r in good if r["turns"] is not None])
        if good
        else None,
        "rows": rows,
    }


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reps", type=int, default=5)
    parser.add_argument("--dsn", default="postgresql://recall:recall@127.0.0.1:5407/agent_ab")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--out", default="benchmarks/artifacts/agent_ab/tool-cost.json")
    args = parser.parse_args()

    env = openrouter_env()
    cli_path = resolve_claude_executable()

    from recall_agent import RecallAgentMemory

    spec = StdioRecallSpec(dsn=args.dsn, cwd=REPO_ROOT, tenant=args.tenant)
    read_only = RecallAgentMemory(
        dsn=args.dsn, tenant=args.tenant, embedder=spec.embedder, use_generation_store=True
    )
    read_write = RecallAgentMemory(
        dsn=args.dsn, tenant=args.tenant, embedder=spec.embedder, use_generation_store=True
    )
    try:
        arms = [
            ("none", None, 0),
            ("read", {spec.server_name: read_only.sdk_mcp_server()}, 2),
            ("read+write", {spec.server_name: read_write.sdk_mcp_server(write_tools=True)}, 4),
            ("full", sdk_mcp_servers(spec), 18),
        ]
        results = []
        for name, servers, tools in arms:
            print(f"\narm {name} ({tools} tools intended):", flush=True)
            results.append(await measure(name, servers, tools, args.reps, env, cli_path))
    finally:
        read_only.close()
        read_write.close()

    baseline = next((r for r in results if r["arm"] == "none"), None)
    summary = {"prompt": PROMPT, "arms": results}
    if baseline and baseline["median_input_tokens"]:
        base = baseline["median_input_tokens"]
        for r in results:
            if r["median_input_tokens"] is None or not r["tools_observed"]:
                continue
            served = r["tools_observed"][0]
            r["delta_vs_none"] = r["median_input_tokens"] - base
            r["tokens_per_tool"] = (r["delta_vs_none"] / served) if served else None
        summary["arms"] = results

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    for r in results:
        print(
            f"  {r['arm']:12s} tools={r['tools_observed']}  n={r['n']}  "
            f"median_input={r['median_input_tokens']}  turns={r['median_turns']}  "
            f"delta={r.get('delta_vs_none')}  per_tool={r.get('tokens_per_tool')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
