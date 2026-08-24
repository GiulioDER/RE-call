"""Classify every trap by where its governing fact is actually reachable, before any session runs.

    python scripts/agent_ab_qualify.py [--out benchmarks/artifacts/agent_ab/<run>/traps.json]

For each trap this asks the live corpus the question the task provokes, and checks whether the
governing memo comes back, then checks whether the static prompt bundle contains the fact. The
result is a per-trap locus:

    memory_only     the retrieval arm should win
    claude_md_only  the static arm should win
    both            neither arm has an advantage
    neither         no arm can learn it; excluded from the trap score

Run this and commit the output **before** the measurement. Choosing which traps count after seeing
which ones RE-call won is the difference between a benchmark and a press release. It is also the
only way the `claude_md_only` traps stay in: they are the ones the memory layer is expected to
lose, and a result without them is not a comparison, it is a highlight reel.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agent_ab.arms import write_claude_md_prompt  # noqa: E402
from benchmarks.agent_ab.recall_server import WarmRecallServer  # noqa: E402
from benchmarks.agent_ab.traps import TRAPS, qualify  # noqa: E402

DEFAULT_DSN = "postgresql://recall:recall@127.0.0.1:5433/recall"
#: What a real session in this repository loads as static memory. The user-level global CLAUDE.md
#: is deliberately excluded: it carries host inventory that must not enter a published artifact.
STATIC_MEMORY_SOURCES = ("CLAUDE.md",)


async def collect_sources(server: WarmRecallServer, query: str, top_k: int = 5) -> list[str]:
    from mcp import ClientSession
    from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

    client = create_mcp_http_client(headers={"Authorization": f"Bearer {server.token}"})
    async with client:
        async with streamable_http_client(server.url, http_client=client) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                result = await session.call_tool("recall_search", {"query": query})
                text = next((b.text for b in result.content if getattr(b, "text", None)), "{}")
    payload = json.loads(text)
    hits = payload.get("hits") or payload.get("results") or []
    sources = []
    for hit in hits[:top_k]:
        source = hit.get("source") or hit.get("source_id") or hit.get("path") or ""
        sources.append(Path(str(source)).name)
    return sources


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", default=DEFAULT_DSN)
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--port", type=int, default=5482)
    parser.add_argument("--out", default=None)
    parser.add_argument(
        "--memory-index",
        default=None,
        help="path to MEMORY.md, included in the static prompt bundle when present",
    )
    args = parser.parse_args()

    artifacts = REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab"
    artifacts.mkdir(parents=True, exist_ok=True)
    sources: tuple[str, ...] = STATIC_MEMORY_SOURCES
    if args.memory_index and Path(args.memory_index).is_file():
        sources = sources + (args.memory_index,)
    prompt_file = write_claude_md_prompt(
        artifacts / "static-memory-prompt.txt", sources, repo_root=REPO_ROOT
    )
    claude_md_text = prompt_file.read_text(encoding="utf-8")
    print(f"static memory bundle: {len(claude_md_text):,} chars from {list(sources)}\n")

    with WarmRecallServer(
        dsn=args.dsn, cwd=REPO_ROOT, tenant=args.tenant, port=args.port
    ) as server:
        retrieved = {}
        for trap in TRAPS:
            retrieved[trap.trap_id] = await collect_sources(server, trap.probe_query)

    qualifications = qualify(
        TRAPS, search=lambda query: _lookup(retrieved, query), claude_md_text=claude_md_text
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

    eligible = [q for q in qualifications if q.eligible]
    print(f"\n{len(eligible)}/{len(qualifications)} traps eligible")
    by_locus: dict[str, list[str]] = {}
    for qualification in qualifications:
        by_locus.setdefault(qualification.locus, []).append(qualification.trap_id)
    for locus, ids in sorted(by_locus.items()):
        print(f"  {locus:<14} {ids}")

    out = Path(args.out) if args.out else REPO_ROOT / "benchmarks" / "agent_ab" / "trap-qualification.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                # Names only: an absolute path here would carry a home directory into a
                # file that is committed as evidence and published with the result.
                "static_memory_sources": [Path(s).name for s in sources],
                "static_memory_chars": len(claude_md_text),
                "tenant": args.tenant,
                "qualifications": [q.to_dict() for q in qualifications],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")
    return 0 if eligible else 1


def _lookup(retrieved: dict[str, list[str]], query: str) -> list[str]:
    for trap in TRAPS:
        if trap.probe_query == query:
            return retrieved.get(trap.trap_id, [])
    return []


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
