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
import tempfile
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
        try:
            connection = psycopg.connect(dsn, connect_timeout=3)
        except Exception:
            # CONNECT failed: the database is down or unreachable, which is transient by
            # nature. Fail open with the cached count — the documented behaviour.
            return int(config.get("chunks", 0) or 0)
        with connection as conn:
            row = conn.execute(statement, (str(config.get("tenant", "default")),)).fetchone()
        count = int(row[0]) if row else 0
    except Exception as exc:
        # The QUERY was refused on a live connection: a dropped table, a revoked role, a
        # renamed tenant column. That is configuration rot, not an outage, and serving the
        # cached count forever would advertise a corpus this hook can never reach again.
        # Still fail open for this session, but say so once, to stderr, where hook output
        # goes; a silent stale number was the audited defect here. The print itself is
        # guarded: on a daemonised host stderr may be closed, and a fail-open path must not
        # turn into a crash because it tried to warn.
        try:
            print(
                f"recall-hooks: chunk count query refused ({type(exc).__name__}); "
                f"the cached figure may be stale until the config is fixed",
                file=sys.stderr,
            )
        except Exception:
            pass
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

    `mkstemp`, not a fixed temp name. The fixed `.recall-hook.json.tmp` this used was shared by
    every concurrent writer — `SessionEnd` runs async and `PreCompact` writes the same file — so
    writer A could promote writer B's half-written bytes over the real config. Because a corrupt
    config loads as `{}` and `refresh_stats` then returns early with no dsn, that corruption was
    PERMANENT and silent: the one shape of loss this function exists to prevent.
    """
    path = config_path()
    try:
        fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    except OSError:
        # Missing directory or unwritable parent — nothing to clean up, and this hook must not
        # raise into a session launch. The config simply is not updated this run.
        return
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(config, indent=2) + "\n")
        os.replace(tmp, path)  # atomic on POSIX and on Windows for a same-directory target
    except Exception:
        # `Exception`, not just `OSError`: a non-serialisable config raises TypeError from
        # json.dumps, which under `except OSError` would leak the temp file AND crash the hook.
        # Fail open (leave the old config in place) and clean the temp; a session launch must
        # not die because a stats write went wrong.
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


def _unsynced_notice(config: dict[str, Any]) -> str:
    """One sentence when memos have not reached hosted memory, or "".

    🔑 **This is where a silent failure stops being silent.** The hooks return 0 always and may not
    speak, so a sync that failed has nowhere to report — except here, at the one event already
    permitted to inject text, at zero extra cost because the config is being read anyway.

    Bounded to one sentence and only when there is BOTH a recorded error and pending work, so a
    healthy install sees nothing at all.
    """
    if not hosted_mode(config):
        return ""
    try:
        from .hosted import read_manifest
    except ImportError:  # pragma: no cover
        return ""
    manifest = read_manifest(config)
    pending = manifest.get("pending") or {}
    error = manifest.get("last_error") or {}
    if not pending or not error:
        return ""
    remedy = {
        "auth": "Run `recall-hooks login`.",
        "quota": "The account's upload quota is exhausted.",
        "refusal": "One file was refused; see the dashboard.",
    }.get(str(error.get("kind")), "It will retry next session.")
    return (
        f"WARNING: {len(pending)} memo(s) from previous sessions have not reached hosted "
        f"memory. {remedy} "
    )


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
    stuck = _unsynced_notice(config)
    count = int(config.get("chunks", 0) or 0)
    if not count and not stuck:
        return 0
    cwd = str(payload.get("cwd", ""))
    project = Path(cwd).name if cwd else "this project"
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": (
                    stuck
                    + f"RE-call memory is available for {project}: {count} indexed chunks. "
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


def hosted_mode(config: dict[str, Any]) -> bool:
    """Is this a hosted config?

    Explicit `mode` wins; otherwise an endpoint with no dsn means hosted. Every config on disk
    today has a dsn and no endpoint, so it resolves local and nothing changes for anyone.

    An EMPTY dsn is not a dsn, so an endpoint beside one still resolves hosted. Writing a real
    one into a hosted config is the thing that breaks: both `_index_and_refresh` and
    `write_time.pre_tool_use` guard on `config.get("dsn")` being truthy, so a real dsn would send
    an older `recall_hooks` at a database while a newer one syncs — two writers, one corpus. With
    the key absent or empty, an older client degrades to silence, which is this package's
    established failure mode.
    """
    mode = str(config.get("mode", "")).strip().lower()
    if mode in {"hosted", "local"}:
        return mode == "hosted"
    return bool(config.get("endpoint")) and not config.get("dsn")


def memory_dirs(cwd: str, override: str = "") -> list[tuple[str, Path]]:
    """Every memory store this project writes to, as `(root_id, path)`, nearest first.

    ⚠️ **BOTH are returned, and that is the fix for a real divergence.** The writer has always
    indexed `<cwd>/memory` while `prompt_time` reads
    `~/.claude/projects/<slug>/memory`, and nothing reconciled them: content written to one was
    invisible to the other. Both are real on machines today, so picking one silently drops whatever
    the other holds — which is the loss these hooks exist to prevent.

    The `root_id` namespaces them, because two roots with the same internal layout would otherwise
    collide into one uploaded name and each sync would overwrite the other's file.
    """
    roots: list[tuple[str, Path]] = []
    if cwd:
        local = Path(str(cwd)) / "memory"
        if local.is_dir():
            roots.append(("worktree", local))
    try:
        from .prompt_time import find_store
    except ImportError:  # pragma: no cover - the module ships beside this one
        return roots
    store = find_store(cwd, override)
    if store is not None and store.is_dir() and all(store != path for _id, path in roots):
        roots.append(("project", store))
    return roots


def _index_and_refresh(payload: dict[str, Any]) -> int:
    """Index this project's `memory/` and refresh the cached count. Never raises.

    Shared by `SessionEnd` and `PreCompact`, which want the same thing at different moments: make
    whatever the session wrote down searchable, and leave an accurate count behind.
    """
    config = load_config()
    cwd = payload.get("cwd")
    if not cwd:
        return 0
    if hosted_mode(config):
        # Imported inside the branch: this event is async, so the transport's cost is never
        # charged to a session launch, and a local-mode install never pays it at all.
        try:
            from .hosted import sync_memory_roots
        except ImportError:
            return 0
        roots = memory_dirs(str(cwd), str(config.get("memory_root", "")))
        if roots:
            try:
                sync_memory_roots(roots, config)
            except BaseException:
                # ⛔ `sync_memory_roots` documents that it never raises, and this catches it
                # anyway. "Documented not to" and "cannot" are different claims, and the cost of
                # being wrong here is a session that will not start. A test pins this by making
                # the sync raise on purpose.
                pass
        return 0
    dsn = config.get("dsn")
    if not dsn:
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


def session_end(payload: dict[str, Any]) -> int:
    """Index what this session produced, then refresh the cached count for the next start.

    Runs with `async: true`, so nothing waits on it and nothing reads its output. That is the
    honest configuration rather than a concession: `SessionEnd` cannot block termination, so a
    synchronous index would be a promise the client is not obliged to keep.
    """
    return _index_and_refresh(payload)


def pre_compact(payload: dict[str, Any]) -> int:
    """Make the session's memos searchable at the moment its context is about to be discarded.

    Compaction is the event this product exists for: it is where a long session loses the detail
    behind its conclusions. What it does NOT lose is anything already written to `memory/`, so the
    useful move is to close the write-to-searchable gap right here, rather than at `SessionEnd`
    which may be hours away. A memo written at 10:00 in a session that compacts at 11:00 and ends
    at 17:00 is otherwise unsearchable for seven hours, including by the very turn that most needs
    it: the one immediately after the compaction that just discarded its context.

    **This hook cannot do the thing it first looks like it should.** `PreCompact` supports no
    `additionalContext`, so it cannot prompt the model to write its conclusions down before they
    go, and `PostCompact` cannot either. Injecting context is `SessionStart` and
    `UserPromptSubmit`, which is why the `SessionStart` matcher here covers `compact`.

    ⛔ Never blocks. Exit code 2 on this event blocks compaction, which would hand a memory tool
    the power to wedge a session whose context window is already full. Every path returns 0, and
    the handler is registered `async` so a slow embedder cannot delay the compaction either.
    """
    return _index_and_refresh(payload)


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
    if args[0] == "pre-compact":
        return pre_compact(payload)
    # ⛔ The per-event imports are GUARDED, and not for tidiness. `python -m recall_hooks` puts
    # the current directory ahead of everything on `sys.path`, so a session whose cwd is a
    # checkout of this repository runs THAT checkout's copy of this package, whichever branch it
    # happens to be on. A branch predating one of these modules then raises `ImportError` out of
    # the hook on every prompt or every tool call. An older install must degrade to the behaviour
    # it had before the feature existed, which is silence.
    #
    # The deterministic fix belongs at the call site rather than here: invoke with `-P`, which
    # stops the cwd being prepended, and point `PYTHONPATH` at the copy you mean to run.
    if args[0] == "pre-tool-use":
        # Imported HERE, not at module scope. This dispatch is shared with SessionStart, whose
        # whole design is that it imports nothing it does not need, and `write_time` is only
        # reached on its own event.
        try:
            from .write_time import pre_tool_use
        except ImportError:
            return 0
        return pre_tool_use(payload)
    if args[0] == "user-prompt-submit":
        # Same reason as above. `prompt_time` imports `math` and `re` and touches the filesystem;
        # SessionStart pays for neither.
        try:
            from .prompt_time import user_prompt_submit
        except ImportError:
            return 0
        return user_prompt_submit(payload)
    return 0


__all__ = [
    "claude_config_home",
    "config_path",
    "load_config",
    "main",
    "pre_compact",
    "refresh_stats",
    "session_end",
    "session_start",
]
