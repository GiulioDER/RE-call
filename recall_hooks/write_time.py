"""`PreToolUse`: search project memory with the text the agent is about to write, and inject it.

**What this is for.** An agent that has to decide to search will mostly not search: measured in this
project, an explicit instruction to search before writing produced an adoption rate of 0.067. This
hook removes the decision. It fires on every write and injects what the corpus returns, so a memo
reaches the agent without the agent having to know the memo exists.

## The evidence, stated exactly, because it is weaker than a default usually deserves

Paired A/B, both arms identical except this hook, executed checker endpoints, 2026-08-27:

| | control | with the hook |
|---|---:|---:|
| task failures | 17 of 34 | **12 of 34** |

6 rescues, 1 regression, net +5, **McNemar exact two-sided p = 0.125**. The registered n was 48
pairs and the run stopped at 34 when its budget ran out.

⚠️ **That is not significant, and the pre-registration committed that nothing ships on a
non-significant positive.** It ships anyway, as an explicit owner decision recorded in
`docs/preregistrations/2026-08-27-write-time-hook.md`. Anyone reading this later should treat the
benefit as *plausible and unproven*, not as measured. What IS measured:

- **Cost is negligible in tokens**: -1% aggregate input tokens, median ratio 1.01, because the hook
  fires a median of 3 times a session rather than the 10 the design assumed.
- **It rescues nothing on the hardest family**: `ts-lf-rewrite` failed 6 of 6 in BOTH arms, with its
  governing memo in the corpus and in front of the agent on every write. Some failures are
  downstream of the memo arriving.

## Cost is latency, not tokens, and that is what the guards here are about

This runs as one PROCESS PER TOOL CALL, so every cost is paid on every `Write`, `Edit` and `Bash`.
Measured on this machine, 2026-08-27:

| condition | wall clock |
|---|---|
| payload below `min_chars` (no `psycopg` import) | **0.14s** |
| corpus reachable on the same machine | ~1.0s |
| corpus reachable on ANOTHER host, over an ssh tunnel | **2.21s median** |
| corpus UNREACHABLE | **2.9s** |

⚠️ The remote row is the honest one and it is expensive. `search` makes exactly ONE round trip
because at three it measured **2.54s median, 3.93s max** against the same corpus; folding them
gained only ~0.3s, which is the useful finding: the cost is the `psycopg` import (~0.9s) and the
connection handshake, not the queries. A remote corpus therefore costs roughly 2s on every
qualifying tool call, and whether that is worth 6 rescues in 34 tasks is a deployment decision, not
a default. `write_time.enabled: false` is the answer where it is not.

The third row is why `_cooldown` exists. An offline laptop would otherwise pay ~3s on every tool
call, silently, because a failing hook returns nothing and merely makes the session drag. After a
connection failure the hook stands down for `cooldown_seconds` and returns before importing
anything.

⛔ **Three properties this must never lose**, each of which is a way to break a session that is not
asking for memory:

1. **It never denies a tool call.** It emits `additionalContext` only. A memory layer that can veto
   a write is a memory layer that can wedge a session.
2. **It never raises.** Every failure path returns 0. A hook that raises is charged to the client.
3. **It never loads an embedder.** The lexical `ts_rank` leg needs no vector, and a fastembed ONNX
   load measured ~11 seconds in this project, which per tool call is not a feature.

Prior work: searched `recall_search("write-time memo injection hook PreToolUse")` against the
`memory` tenant on 2026-08-27; returns `write-time-memo-injection-result-2026-08-27` (this result),
`a-string-test-beat-the-llm-gate-at-deciding-when-to-search` and `search-with-the-draft-not-the-goal`,
which is why the query is the DRAFT TEXT rather than a goal description.
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Any

from . import claude_config_home, load_config

#: Matches `recall_mcp.service.MAX_QUERY_CHARS`. The server refuses a longer query rather than
#: truncating it, so truncating here is the difference between a hit and a refusal.
MAX_QUERY_CHARS = 4096
#: Below this a payload is `ls` or `cd ..`: no useful query, and the early return avoids the
#: `psycopg` import entirely, which is the difference between 0.14s and 1.0s on that call.
MIN_QUERY_CHARS = 12
#: How many memos to inject. Five was what the A/B measured; the hook has no relevance threshold,
#: so this is also the noise budget.
TOP_K = 5
#: Characters of each memo. Enough for the hazard sentence, not the whole document.
SNIPPET_CHARS = 1200
#: Seconds to stand down after a connection failure. See the latency table above.
COOLDOWN_SECONDS = 300

WRITE_TOOLS = ("Write", "Edit", "MultiEdit", "NotebookEdit")
COMMAND_TOOLS = ("Bash", "BashOutput")
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]{2,}")

#: Beside the hook config rather than inside it: `SessionEnd` rewrites that file asynchronously,
#: and a cooldown stamp written from a PreToolUse hook has no business racing it.
COOLDOWN_NAME = "recall-hook-write-time-cooldown"


def settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """The `write_time` block of the hook config, with defaults.

    Absent means ENABLED, because the installer writes the block and an older config that predates
    it should still get the feature it was upgraded for. `enabled: false` is the way to turn it
    off, and it is honoured everywhere.
    """

    config = load_config() if config is None else config
    block = config.get("write_time")
    block = block if isinstance(block, dict) else {}
    return {
        "enabled": bool(block.get("enabled", True)),
        "k": int(block.get("k", TOP_K)),
        "min_chars": int(block.get("min_chars", MIN_QUERY_CHARS)),
        "connect_timeout": float(block.get("connect_timeout", 2.0)),
        "cooldown_seconds": float(block.get("cooldown_seconds", COOLDOWN_SECONDS)),
    }


def _cooldown_path() -> Path:
    return claude_config_home() / COOLDOWN_NAME


def _in_cooldown() -> bool:
    """True while a recent connection failure says the corpus is unreachable.

    Read before anything is imported, because the whole point is to not pay the timeout again.
    A missing or unreadable stamp means "not in cooldown": the failure mode of this file must be
    to try the database, never to suppress the feature permanently.
    """

    try:
        raw = _cooldown_path().read_text(encoding="utf-8").strip()
    except OSError:
        return False
    try:
        return time.time() < float(raw)
    except ValueError:
        return False


def _start_cooldown(seconds: float) -> None:
    try:
        _cooldown_path().write_text(str(time.time() + seconds), encoding="utf-8", newline="\n")
    except OSError:
        # A hook that cannot write its own cooldown still has to run. The cost is that the next
        # call pays the timeout again, which is the behaviour without this file at all.
        pass


def _clear_cooldown() -> None:
    try:
        _cooldown_path().unlink()
    except OSError:
        pass


def payload_of(tool_name: str, tool_input: dict[str, Any]) -> str:
    """The text the agent is about to write, which is the query.

    Not a description of the goal: measured in this project, goal-shaped queries surface the
    governing memo for 1 of 14 sessions where the draft text surfaces it for 11 of 11. The draft
    carries the hazard's own vocabulary; a goal statement does not.
    """

    if tool_name in WRITE_TOOLS:
        return str(
            tool_input.get("content")
            or tool_input.get("new_string")
            or tool_input.get("new_source")
            or ""
        )
    if tool_name in COMMAND_TOOLS:
        return str(tool_input.get("command") or "")
    return ""


def _search_connection(
    connection: Any,
    query: str,
    config: dict[str, Any],
    options: dict[str, Any],
) -> list[tuple[str, str, float]]:
    """Run the lexical query on an already-open connection.

    This is kept separate from :func:`search` so a benchmark relay can amortize connection setup
    without creating a second SQL implementation. The caller owns the connection lifecycle.
    """

    terms = " | ".join(sorted({t.lower() for t in TOKEN_RE.findall(query[:MAX_QUERY_CHARS])})[:200])
    if not terms:
        return []
    tenant = str(config.get("tenant", "default"))
    rows = connection.execute(
        "SELECT source_uri, text, ts_rank(tsv, to_tsquery('english', %s)) AS rank "
        "FROM recall_chunks_v1 "
        "WHERE tenant_id = %s "
        "  AND generation_id = ("
        "        SELECT generation_id FROM recall_generations "
        "        WHERE tenant_id = %s AND state = 'active' "
        "        ORDER BY created_at DESC LIMIT 1) "
        "  AND tsv @@ to_tsquery('english', %s) "
        "ORDER BY rank DESC LIMIT %s",
        (terms, tenant, tenant, terms, int(options["k"])),
    ).fetchall()
    return [(Path(str(uri)).name, str(text), float(rank)) for uri, text, rank in rows]


def search(query: str, config: dict[str, Any], options: dict[str, Any]) -> list[tuple[str, str, float]]:
    """`(source name, chunk text, ts_rank)` for the lexical leg, by SQL. No model is loaded.

    ⛔ Do NOT route this through `HybridRetriever`: it requires an embedder, and this runs once per
    tool call. The dense leg is deliberately absent as well, and not only for cost: measured on
    draft queries it reaches 7 of 14 against lexical's 14 of 14, so it would pay a model load to
    make retrieval worse.
    """

    import psycopg

    # ⚠️ ONE round trip, deliberately. Measured against a corpus on another host: three round
    # trips cost a median of 2.54s per tool call, against ~1.0s for a corpus on the same machine,
    # and the difference is almost entirely latency rather than work. The statement timeout rides
    # on the connection's options instead of a `SET LOCAL`, and the generation lookup is a scalar
    # subquery instead of a separate SELECT.
    #
    # ⛔ Binding to the ACTIVE generation is NOT relaxed by folding it in. `recall_chunks_v1`
    # holds every generation ever built, retired ones included, and `GenerationStore` is what
    # normally applies this filter; raw SQL does not inherit it. A subquery that matches no row
    # yields NULL, `generation_id = NULL` matches nothing, and the caller gets no hits, which is
    # what the explicit `if not active: return []` did.
    with psycopg.connect(
        str(config["dsn"]),
        connect_timeout=options["connect_timeout"],
        options="-c statement_timeout=5s",
    ) as conn:
        return _search_connection(conn, query, config, options)


def render(hits: list[tuple[str, str, float]]) -> str:
    """The injected text.

    It says most searches return nothing that applies, because they do: measured, this fires on 29
    of 36 sessions that did not need a memo. Telling the agent that up front is what makes an
    irrelevant hit cheap to dismiss rather than a distraction to be reconciled.
    """

    lines = [
        "Project memory was searched with the text you are about to write. "
        f"{len(hits)} note(s) came back; most searches return nothing that applies, so read the "
        "first and move on if it is about a different operation."
    ]
    for name, text, score in hits:
        stem = name[:-3] if name.endswith(".md") else name
        lines.append(f"\n--- {stem} (score {score:.3f}) ---\n{text.strip()[:SNIPPET_CHARS]}")
    return "\n".join(lines)


def pre_tool_use(payload: dict[str, Any]) -> int:
    """Entry point. Returns 0 always: this hook cannot fail a tool call, only decline to speak."""

    tool_name = str(payload.get("tool_name") or "")
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return 0

    config = load_config()
    if not config.get("dsn"):
        # Unconfigured is silent BY DESIGN: a checkout that has not run the installer must behave
        # exactly as it would without this hook.
        return 0
    options = settings(config)
    if not options["enabled"]:
        return 0

    query = payload_of(tool_name, tool_input).strip()
    if len(query) < options["min_chars"]:
        return 0
    if _in_cooldown():
        return 0

    try:
        hits = search(query, config, options)
    except Exception:  # noqa: BLE001 - a retrieval failure must never break the session
        # Any failure to reach the corpus starts the cooldown, not only a timeout: a wrong DSN, a
        # revoked role and a stopped container all cost the same wall clock on every tool call,
        # and all of them are things the user fixes elsewhere rather than mid-session.
        _start_cooldown(options["cooldown_seconds"])
        return 0
    _clear_cooldown()
    if not hits:
        return 0

    print(json.dumps({
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "additionalContext": render(hits),
        }
    }))
    return 0


__all__ = [
    "MIN_QUERY_CHARS",
    "payload_of",
    "pre_tool_use",
    "render",
    "search",
    "settings",
]
