"""Classify every task-success task by where its governing fact is actually reachable.

    python -u scripts/agent_ab_qualify_tasks.py \
        --dsn postgresql://recall:recall@127.0.0.1:5407/agent_ab --tenant default

Run this and commit the output **before** the measurement and before the predictions. Choosing
which tasks count after seeing which ones RE-call won is the difference between a benchmark and a
press release, and it is also the only way the control tasks stay in: they are the ones the memory
layer is not expected to win, and a result without them is not a comparison.

The classification is `traps.qualify`, unchanged, because "where does this fact live" means the
same thing for a task as for a hazard and two implementations of it would eventually disagree.

⚠️ **This qualifies over STDIO, not over the warm HTTP server, and the difference is not a detail.**
`scripts/agent_ab_qualify.py` uses `WarmRecallServer`, which strips `RECALL_ENV` and therefore runs
the server in development, where it reads the legacy `chunks` table rather than the promoted
generation. Measured 2026-08-21 against this benchmark's corpus: `recall_chunks_v1` holds 1006 rows
and `chunks` holds **0**. Qualifying over HTTP would have retrieved nothing for every probe and
classified all ten primary tasks as `neither`, which reads as "no task is winnable" rather than as
"the qualifier looked in the wrong table". Stdio is also the transport the on arm uses, so this
asks the question against the path that will actually answer it.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agent_ab.arms import write_claude_md_prompt  # noqa: E402
from benchmarks.agent_ab.recall_server import StdioRecallSpec  # noqa: E402
from benchmarks.agent_ab.tasksuccess import TASKS  # noqa: E402
from benchmarks.agent_ab.traps import qualify  # noqa: E402

DEFAULT_DSN = "postgresql://recall:recall@127.0.0.1:5407/agent_ab"
#: What a real session in this repository loads as static memory, and what both arms receive
#: byte for byte. The user-level global CLAUDE.md is deliberately excluded: it carries a host
#: inventory that must not enter a published artifact.
STATIC_MEMORY_SOURCES = ("CLAUDE.md",)
TOP_K = 5


async def retrieve(spec: StdioRecallSpec, queries: list[str], top_k: int) -> dict[str, list[str]]:
    """Ask every probe query down ONE stdio session.

    One session, not one per query: the server spends about 11 s loading its embedder, and paying
    that twelve times over would make re-running this expensive enough that nobody would.
    """

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=spec.python, args=["-m", "recall_mcp.server"], env=spec.env(), cwd=str(spec.cwd)
    )
    answers: dict[str, list[str]] = {}
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            for query in queries:
                result = await session.call_tool("recall_search", {"query": query})
                text = next(
                    (getattr(b, "text", "") for b in result.content if getattr(b, "text", None)),
                    "",
                )
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    # A refusal is prose, and it must not be silently read as "nothing matched":
                    # an empty result and a refused query classify identically and mean opposite
                    # things.
                    raise SystemExit(f"recall_search was refused for {query!r}: {text[:300]}")
                hits = payload.get("hits") or payload.get("results") or []
                answers[query] = [
                    Path(str(hit.get("source") or hit.get("source_id") or hit.get("path") or "")).name
                    for hit in hits[:top_k]
                ]
    return answers


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--memory-index", default=None)
    parser.add_argument(
        "--out", default=str(REPO_ROOT / "benchmarks" / "agent_ab" / "task-qualification.json")
    )
    args = parser.parse_args()

    artifacts = REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab"
    artifacts.mkdir(parents=True, exist_ok=True)
    sources: tuple[str, ...] = STATIC_MEMORY_SOURCES
    if args.memory_index and Path(args.memory_index).is_file():
        sources = sources + (args.memory_index,)
    prompt_file = write_claude_md_prompt(
        artifacts / "task-static-memory-prompt.txt", sources, repo_root=REPO_ROOT
    )
    claude_md_text = prompt_file.read_text(encoding="utf-8")
    print(f"static memory bundle: {len(claude_md_text):,} chars from {[Path(s).name for s in sources]}\n")

    spec = StdioRecallSpec(dsn=args.dsn, cwd=REPO_ROOT, tenant=args.tenant)
    check = await spec.check()
    print(
        f"RE-call: {check['handshake_ms']} ms handshake, {check['tool_count']} tools, "
        f"trust_state={check.get('trust_state')} calibrated={check.get('calibrated')} "
        f"generation={str(check.get('generation_id'))[:24]}"
    )
    if check.get("calibrated") is not True:
        print("  [note] this corpus is UNCALIBRATED; the result must say so wherever it is published")

    retrieved = await retrieve(spec, [task.probe_query for task in TASKS], args.top_k)
    qualifications = qualify(
        TASKS, search=lambda query: retrieved.get(query, []), claude_md_text=claude_md_text
    )

    width = max(len(q.trap_id) for q in qualifications)
    for qualification in qualifications:
        mark = "  " if qualification.eligible else "X "
        print(
            f"{mark}{qualification.trap_id:<{width}}  {qualification.locus:<14} "
            f"memory={qualification.in_memory!s:<5} claude_md={qualification.in_claude_md!s:<5}"
        )
        if not qualification.in_memory and qualification.declared_memo:
            print(
                f"    declared memo {qualification.declared_memo!r} did not come back; "
                f"retrieved {qualification.retrieved_sources}"
            )

    by_locus: dict[str, list[str]] = {}
    for qualification in qualifications:
        by_locus.setdefault(qualification.locus, []).append(qualification.trap_id)
    print(f"\n{sum(1 for q in qualifications if q.eligible)}/{len(qualifications)} eligible")
    for locus, ids in sorted(by_locus.items()):
        print(f"  {locus:<14} {ids}")

    payload: dict[str, Any] = {
        # Names only: an absolute path here would carry a home directory into a file committed as
        # evidence and published with the result.
        "static_memory_sources": [Path(s).name for s in sources],
        "static_memory_chars": len(claude_md_text),
        "tenant": args.tenant,
        "top_k": args.top_k,
        "transport": "stdio",
        "trust_state": check.get("trust_state"),
        "calibrated": check.get("calibrated"),
        "generation_id": check.get("generation_id"),
        "calibration_id": check.get("calibration_id"),
        "qualifications": [q.to_dict() for q in qualifications],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    # newline="\n" because this repository's .gitattributes sets eol=lf, and a plain write_text on
    # Windows translates every newline to CRLF and leaves the committed artifact permanently
    # modified. That is ts-lf-rewrite's own fact, arriving in the qualifier that classifies it: git
    # printed the CRLF warning on the first commit of this file.
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
