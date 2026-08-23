"""Would decompose-expansion have rescued the queries that missed in agent-ab-skill-001?

    python -u scripts/agent_ab_probe_expansion.py \
        --archive ~/.claude/archive/agent-ab-skill-001 \
        --dsn postgresql://recall:recall@127.0.0.1:5407/agent_ab --tenant default

An offline feasibility probe, preregistered in
`docs/preregistrations/2026-08-23-decompose-expansion-probe.md`, that decides whether server-side
query expansion is worth building at all. Run `agent-ab-skill-001` ended with 15 on-arm sessions
that searched and still missed their governing memo, every one asking in goal vocabulary. The
probe takes those sessions' RECORDED queries verbatim, expands each through one fixed decomposition
prompt (below, committed, generic: it names no task, no memo, no corpus), and asks the same corpus
over the same stdio transport whether any expanded query retrieves the memo at the same top-5.

A session is RESCUED when at least one expansion of at least one of its recorded queries retrieves
its governing memo. This mirrors the feature's union semantics (original results plus expansions),
under which a session that already hit cannot be un-hit, so only the misses are informative.

⛔ The expansion prompt is the whole experiment. It must stay generic: one hint of any task's
vocabulary in it and the probe measures the hint.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import urllib.request
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmarks.agent_ab.recall_server import StdioRecallSpec  # noqa: E402
from benchmarks.agent_ab.schema import RECALL_ON, SessionRecord  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from agent_ab_analyze_tasks import retrieved_governing_memo, searched  # noqa: E402

TOP_K = 5
MEMORY_ONLY = "memory_only"
EXPANSION_MODEL = "anthropic/claude-haiku-4.5"
#: Fixed and generic on purpose. It teaches the decomposition MOVE, never any vocabulary: the
#: probe exists to learn whether a model with no knowledge of the corpus can bridge the gap.
EXPANSION_PROMPT = (
    "You expand search queries for a store of engineering postmortems. The store describes "
    "failures in the vocabulary of concrete operations and their symptoms: which tool ran, what "
    "kind of file was read or written, what looked wrong afterwards. It does not describe goals.\n"
    "Given a task-flavoured query, name the concrete operations the task would actually perform "
    "and how each one can go wrong. Reply with exactly 3 short search queries, one per line, no "
    "numbering, no other text.\n\nQuery: {query}"
)


def expand(query: str, key: str) -> list[str]:
    body = json.dumps(
        {
            "model": EXPANSION_MODEL,
            "temperature": 0,
            "max_tokens": 200,
            "messages": [{"role": "user", "content": EXPANSION_PROMPT.format(query=query)}],
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        "https://openrouter.ai/api/v1/chat/completions",
        data=body,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=120) as response:  # noqa: S310 - fixed https URL
        payload = json.loads(response.read().decode("utf-8"))
    text = payload["choices"][0]["message"]["content"]
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        raise SystemExit(f"expansion returned no queries for {query!r}: {text[:200]}")
    return lines[:3]


def miss_sessions(archive: Path) -> list[tuple[str, str, list[str]]]:
    """(task_id, governing_memo, recorded queries) for every on-arm session that searched and missed."""

    rows = []
    for line in (archive / "records.jsonl").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = SessionRecord.from_mapping(json.loads(line))
        if record.variant != RECALL_ON or record.metadata.get("locus") != MEMORY_ONLY:
            continue
        if not searched(record) or retrieved_governing_memo(record) is not False:
            continue
        queries = [
            str(call.get("args", {}).get("query", ""))
            for call in record.tool_calls
            if "recall_search" in str(call.get("name", "")) and call.get("args")
        ]
        rows.append((record.task_id, str(record.metadata["governing_memo"]), [q for q in queries if q]))
    return rows


async def retrieve(spec: StdioRecallSpec, queries: list[str]) -> dict[str, list[str]]:
    """Every query down ONE stdio session, top-5 source names each, refusals fatal."""

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
                    (getattr(b, "text", "") for b in result.content if getattr(b, "text", None)), ""
                )
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    raise SystemExit(f"recall_search was refused for {query!r}: {text[:300]}")
                hits = payload.get("hits") or payload.get("results") or []
                answers[query] = [
                    Path(str(h.get("source") or h.get("source_id") or h.get("path") or "")).name
                    for h in hits[:TOP_K]
                ]
    return answers


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", required=True)
    parser.add_argument("--dsn", default="postgresql://recall:recall@127.0.0.1:5407/agent_ab")
    parser.add_argument("--tenant", default="default")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    key = os.environ.get("OPENROUTER_API_KEY", "").strip()
    if not key:
        raise SystemExit("OPENROUTER_API_KEY is not set")

    sessions = miss_sessions(Path(args.archive).expanduser())
    print(f"{len(sessions)} on-arm sessions searched and missed their governing memo\n")

    distinct = sorted({q for _, _, queries in sessions for q in queries})
    expansions: dict[str, list[str]] = {}
    for query in distinct:
        expansions[query] = expand(query, key)
        print(f"  {query!r}")
        for e in expansions[query]:
            print(f"    -> {e!r}")

    spec = StdioRecallSpec(dsn=args.dsn, cwd=REPO_ROOT, tenant=args.tenant)
    check = await spec.check()
    if check.get("calibrated") is not True:
        print("  [note] corpus UNCALIBRATED; the record must say so")
    all_expanded = sorted({e for exp in expansions.values() for e in exp})
    retrieved = await retrieve(spec, all_expanded)

    rescued, per_task = [], defaultdict(lambda: [0, 0])
    for task_id, memo, queries in sessions:
        base = task_id.split("#")[0]
        hit = any(
            memo in " ".join(retrieved.get(e, []))
            for q in queries
            for e in expansions.get(q, [])
        )
        per_task[base][1] += 1
        per_task[base][0] += int(hit)
        rescued.append({"task_id": task_id, "governing_memo": memo, "rescued": hit})

    total = sum(1 for r in rescued if r["rescued"])
    print(f"\nrescued sessions: {total}/{len(rescued)}")
    for base, (won, n) in sorted(per_task.items()):
        print(f"  {base:<28} {won}/{n}")

    payload = {
        "archive": str(args.archive),
        "expansion_model": EXPANSION_MODEL,
        "expansion_prompt": EXPANSION_PROMPT,
        "generation_id": check.get("generation_id"),
        "expansions": expansions,
        "retrieved": retrieved,
        "sessions": rescued,
        "rescued": total,
        "of": len(rescued),
        "per_task": {k: {"rescued": v[0], "n": v[1]} for k, v in sorted(per_task.items())},
    }
    out = Path(args.out) if args.out else REPO_ROOT / "benchmarks" / "artifacts" / "agent_ab" / "expansion-probe.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
