#!/usr/bin/env python3
"""Search memory with the payload the agent is about to write, and inject what comes back.

Reads a `PreToolUse` payload on stdin; prints `additionalContext` or nothing. **It never denies.**

Preregistered in `docs/preregistrations/2026-08-27-write-time-hook.md`. Why it exists, in one
line: retrieval CAN find the governing memo (11 of 11 sessions that needed it, on executed ground
truth), and an instruction CANNOT make the agent query that way (0.067 adoption — it searched once,
at the start, and composed a keyword query instead of pasting its draft). So the search happens
mechanically, without asking the agent to remember.

⛔ **It must never block.** Denying a write would change task outcomes for reasons unrelated to
memory quality and make the A/B endpoint uninterpretable. The hook's only power is to add context.

⚠️ **It is expected to be mostly noise, and that is measured rather than hidden.** The memo is
needed in 23% of sessions and draft-time search fires on 29 of the 36 that do not, so at a median
of 10 payloads a session this fires ~10 times with maybe one useful hit. The registration's
endpoint is the NET of rescues minus regressions for exactly this reason.

Every injection is appended to a JSONL trace including whether the `df<=2` vocabulary trigger
WOULD have fired — recorded, never applied. A gated variant is then an offline re-analysis of the
trace instead of a second 112-session A/B.
"""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

#: Matches `recall_mcp.service.MAX_QUERY_CHARS`; the server refuses a longer query rather than
#: truncating it, so truncating here is the difference between a hit and a refusal.
MAX_QUERY_CHARS = 4096
#: Below this a payload is too short to carry a distinctive identifier, and querying with it
#: returns whatever shares a common word. Measured: median payload is 60 characters.
MIN_QUERY_CHARS = 12
TOP_K = 5
WRITE_TOOLS = ("Write", "Edit", "NotebookEdit")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")


def payload_of(tool_name: str, tool_input: dict) -> str:
    """The text the agent is about to commit, or "" when this tool commits none."""

    if tool_name in WRITE_TOOLS:
        return str(tool_input.get("content") or tool_input.get("new_string") or "")
    if tool_name == "Bash":
        return str(tool_input.get("command") or "")
    return ""


def trace(record: dict) -> None:
    """Append one line to the trace, if a destination is configured. Never fatal.

    A hook that dies because its logging failed would change the run it is measuring, which is the
    one thing this must not do.
    """

    destination = os.environ.get("RECALL_HOOK_TRACE", "").strip()
    if not destination:
        return
    try:
        path = Path(destination)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(record) + "\n")
    except OSError:
        pass


def search(query: str, dsn: str) -> list[tuple[str, str, float]]:
    """(source name, chunk text, ts_rank) for the lexical leg, by SQL. No model is loaded.

    ⛔ **Do NOT route this through `HybridRetriever`.** It requires an embedder, and this hook runs
    as one PROCESS PER TOOL CALL, so that is a fastembed ONNX load — measured at ~11 seconds in
    this project — on EVERY write. At a median of 10 payloads a session that is ~110 seconds of
    pure model loading in the treatment arm, and the A/B would be measuring latency rather than
    memory.

    The lexical leg is `ts_rank` over the `tsv` column and needs no vector at all. The dense leg
    is deliberately absent: it reaches 7 of 14 on draft queries against lexical's 14 of 14, so it
    would cost the model load to make retrieval worse.
    """

    import psycopg

    terms = " | ".join(
        sorted({t.lower() for t in TOKEN_RE.findall(query[:MAX_QUERY_CHARS])})[:200]
    )
    if not terms:
        return []
    tenant = os.environ.get("RECALL_HOOK_TENANT", "default")
    # 2s, not 10: this runs once per write, and an unreachable corpus at connect_timeout=10 costs
    # ten seconds on EVERY write — measured, by pointing it at a stopped container. A hook that is
    # slow when its backend is down is worse than one that is slow when it works, because the
    # failure is invisible: it returns nothing and the session merely drags.
    with psycopg.connect(dsn, connect_timeout=2) as conn:
        conn.execute("SET LOCAL statement_timeout = '5s'")
        # ⛔ Bind to the ACTIVE generation. `recall_chunks_v1` holds every generation ever built,
        # including retired ones, and `GenerationStore` is what normally applies this filter — raw
        # SQL does not inherit it. Without this the hook silently serves stale rows from a retired
        # corpus, which is the same class of defect as the CLI reading the legacy `chunks` table
        # while the server reads the promoted generation.
        active = conn.execute(
            "SELECT generation_id FROM recall_generations "
            "WHERE tenant_id = %s AND state = 'active' ORDER BY created_at DESC LIMIT 1",
            (tenant,),
        ).fetchone()
        if not active:
            return []
        rows = conn.execute(
            "SELECT source_uri, text, ts_rank(tsv, to_tsquery('english', %s)) AS rank "
            "FROM recall_chunks_v1 "
            "WHERE tenant_id = %s AND generation_id = %s AND tsv @@ to_tsquery('english', %s) "
            "ORDER BY rank DESC LIMIT %s",
            (terms, tenant, active[0], terms, TOP_K),
        ).fetchall()
    return [(Path(str(uri)).name, str(text), float(rank)) for uri, text, rank in rows]


def vocabulary_would_fire(query: str, vocab: set[str]) -> bool:
    """Recorded, NEVER applied. Whether the df<=2 trigger would have gated this injection."""

    return any(token.lower() in vocab for token in TOKEN_RE.findall(query))


def load_vocabulary() -> set[str]:
    path = os.environ.get("RECALL_HOOK_VOCAB", "").strip()
    if not path:
        return set()
    try:
        return set(json.loads(Path(path).read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError):
        return set()


def render(hits: list[tuple[str, str, float]]) -> str:
    lines = [
        "Project memory was searched with the text you are about to write. "
        f"{len(hits)} note(s) came back; most searches return nothing that applies, so read the "
        "first and move on if it is about a different operation."
    ]
    for name, text, score in hits:
        stem = name[:-3] if name.endswith(".md") else name
        lines.append(f"\n--- {stem} (score {score:.3f}) ---\n{text.strip()[:1200]}")
    return "\n".join(lines)


def main() -> int:
    try:
        event = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        return 0
    # `[]`, `3` and `"x"` are all VALID json and none of them has `.get`. Catching only a decode
    # error left an AttributeError to escape into the session, which is the one outcome this hook
    # is designed never to produce. Well-formed but wrong is the likelier malformation of the two.
    if not isinstance(event, dict):
        return 0

    # Every trace line carries the session it came from. Stage A's endpoint is injections PER
    # SESSION, and the whole run appends to ONE file, so without this the counts are a total that
    # no per-session number can be recovered from. `session_id` is the client's own id; `cwd` is
    # the harness's per-session sandbox, and is the key the harness can join on without having to
    # know what id the client chose.
    identity = {
        "session_id": str(event.get("session_id") or ""),
        "cwd": str(event.get("cwd") or ""),
        "at": datetime.now(timezone.utc).isoformat(),
    }

    tool_name = str(event.get("tool_name") or "")
    tool_input = event.get("tool_input")
    if not isinstance(tool_input, dict):
        return 0

    query = payload_of(tool_name, tool_input).strip()
    if len(query) < MIN_QUERY_CHARS:
        return 0

    dsn = os.environ.get("RECALL_HOOK_DSN", "").strip()
    if not dsn:
        # Unconfigured is silent BY DESIGN: this hook is an experiment, and a session that has not
        # opted in must behave exactly as it would without it.
        return 0

    try:
        hits = search(query, dsn)
    except Exception as error:  # noqa: BLE001 - a retrieval failure must not break the session
        trace({**identity, "tool": tool_name, "chars": len(query),
               "error": f"{type(error).__name__}: {error}"})
        return 0

    vocab = load_vocabulary()
    trace({
        **identity,
        "tool": tool_name,
        "chars": len(query),
        "query": query[:400],
        "hits": [{"source": n, "score": s} for n, _, s in hits],
        # Recorded so a gated variant is an offline re-analysis, not a second A/B.
        "vocabulary_would_fire": vocabulary_would_fire(query, vocab) if vocab else None,
    })
    if not hits:
        return 0

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": render(hits),
        }
    }))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
