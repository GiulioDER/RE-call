"""The Claude Code hook entry points, kept deliberately outside the `recall` package.

**This package exists because of a measured cost, not for tidiness.** `recall/__init__.py` eagerly
imports the calibration, evidence and lineage modules, so `python -m recall.anything` pays about
one second before a line of hook code runs. Measured on this machine, 2026-08-19, with
`python -c pass` as the floor:

| Command | Wall clock |
|---|---|
| `python -c pass` | 0.38s |
| `python -c "import json, sys, os, pathlib"` | 0.44s |
| `python -c "import recall"` | 1.35s |
| `python -c "import psycopg"` | 1.14s |

Re-measure before trusting those:

```bash
python -X importtime -c "import recall" 2>&1 | tail -20
```

A `SessionStart` hook runs before the user's first turn of **every** session, so that second is
charged to opening Claude, forever, and the user attributes it to the client rather than to us.
Nothing here may import `recall` at module scope, and the installer in `recall/claude_code.py`
points the hook at `python -m recall_hooks` for exactly that reason.

The same logic removed the database from the session-start path. Counting chunks needs `psycopg`,
which is another second, and the count is a status line rather than a fact anyone acts on. So
`SessionEnd`, which runs asynchronously and can afford anything, refreshes a cached count into the
hook's own config file, and `SessionStart` reads that number from disk. A digest that is at most
one session stale costs nothing; a session launch that waits on a database the user has not started
costs the whole integration.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

#: Written by `recall.claude_code.install_hooks`, beside the client's own config rather than
#: inside it: `settings.json` is shared with every project and is a file people paste into issues.
HOOK_CONFIG_NAME = "recall-hook.json"


def claude_config_home() -> Path:
    raw = os.environ.get("CLAUDE_CONFIG_DIR", "").strip()
    if raw:
        return Path(raw).expanduser()
    return Path.home() / ".claude"


def config_path() -> Path:
    return claude_config_home() / HOOK_CONFIG_NAME


def load_config() -> dict[str, Any]:
    path = config_path()
    if not path.exists():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return document if isinstance(document, dict) else {}


def refresh_stats(config: dict[str, Any] | None = None) -> int:
    """Count this tenant's chunks and cache the number in the hook config. Returns the count.

    Called by the installer once, and by `SessionEnd` after every session. Never by
    `SessionStart`: see the module docstring.

    Not `PgVectorStore.count()`, which is this same SQL behind a constructor that requires the
    embedding dimension, and supplying that means resolving an embedder, and resolving an embedder
    can download a model.
    """
    config = load_config() if config is None else config
    dsn = config.get("dsn")
    if not dsn:
        return 0
    try:
        import psycopg
        from psycopg import sql

        # The table name cannot be a bound parameter, so it is quoted as an identifier. The
        # tenant is bound normally.
        statement = sql.SQL("SELECT count(*) FROM {} WHERE tenant_id = %s").format(
            sql.Identifier(str(config.get("table", "chunks")))
        )
        with psycopg.connect(dsn, connect_timeout=3) as conn:
            row = conn.execute(statement, (str(config.get("tenant", "default")),)).fetchone()
        count = int(row[0]) if row else 0
    except Exception:
        return int(config.get("chunks", 0) or 0)

    _save_config({**config, "chunks": count})
    return count


def _save_config(config: dict[str, Any]) -> None:
    """Replace the hook config atomically.

    `SessionEnd` runs asynchronously, so this write can land while a session that is *starting*
    reads the same file. `write_text` truncates at open, and the window between the truncation and
    the last byte is a torn read: the starting session sees invalid JSON, `load_config` returns an
    empty dict, and the digest silently vanishes for that session.

    `recall.atomic_write` exists and is the audited copy of this sequence. It is deliberately not
    imported: it lives in the `recall` package, whose `__init__` costs about a second, and this
    module is on the critical path of every session launch. That trade is the whole reason this
    package exists, so the duplication is intentional rather than an oversight to be tidied away.
    """
    path = config_path()
    tmp = path.with_name(f".{path.name}.tmp")
    try:
        tmp.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)  # atomic on POSIX and on Windows for a same-directory target
    except OSError:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def session_start(payload: dict[str, Any]) -> int:
    """Inject a short memory digest before the first turn.

    A digest and not a retrieval, deliberately. `SessionStart` carries no user prompt, so there is
    no query to retrieve against, and a similarity search with nothing to be similar to returns
    whatever happens to be nearest. The event that has a query is `UserPromptSubmit`, and that is
    where per-turn retrieval belongs once this is proven out.

    Fails open, always: a hook that runs before every session must never be the reason Claude does
    not start, so anything unexpected is silence and exit 0.
    """
    config = load_config()
    count = int(config.get("chunks", 0) or 0)
    if not count:
        return 0
    cwd = str(payload.get("cwd", ""))
    project = Path(cwd).name if cwd else "this project"
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": (
                    f"RE-call memory is available for {project}: {count} indexed chunks. "
                    "Call `recall_search` before proposing an idea, forming a hypothesis, or "
                    "repeating past work, and treat an `abstained: true` result as 'no supported "
                    "answer' rather than as an empty one. This corpus is uncalibrated, so trust "
                    "thresholds are defaults rather than fitted to it."
                ),
            }
        },
        sys.stdout,
    )
    return 0


def session_end(payload: dict[str, Any]) -> int:
    """Index what this session produced, then refresh the cached count for the next start.

    Runs with `async: true`, so nothing waits on it and nothing reads its output. That is the
    honest configuration rather than a concession: `SessionEnd` cannot block termination, so a
    synchronous index would be a promise the client is not obliged to keep.
    """
    config = load_config()
    dsn = config.get("dsn")
    cwd = payload.get("cwd")
    if not dsn or not cwd:
        return 0
    memory_dir = Path(str(cwd)) / "memory"
    if memory_dir.is_dir():
        try:
            from recall.setup import index_memory_directory

            index_memory_directory(
                dsn=str(dsn),
                embedder_name=str(config.get("embedder", "fastembed")),
                memory_dir=memory_dir,
                print_fn=lambda *args, **kwargs: None,
            )
        except Exception:
            return 0
    refresh_stats(config)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Dispatch for `python -m recall_hooks <event>`, invoked by the hooks themselves."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args:
        return 0
    try:
        payload = json.load(sys.stdin)
    except (json.JSONDecodeError, ValueError):
        payload = {}
    if not isinstance(payload, dict):
        payload = {}
    if args[0] == "session-start":
        return session_start(payload)
    if args[0] == "session-end":
        return session_end(payload)
    return 0


__all__ = [
    "claude_config_home",
    "config_path",
    "load_config",
    "main",
    "refresh_stats",
    "session_end",
    "session_start",
]
