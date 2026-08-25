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


def _warn(message: str) -> None:
    """Say something once, to stderr, and never let saying it become the failure.

    Every path in this module fails open, and a fail-open path that crashes because it tried to
    warn is worse than the silence it was trying to break. On a daemonised host stderr may be
    closed outright, so even `print` is guarded.
    """
    try:
        print(f"recall-hooks: {message}", file=sys.stderr)
    except Exception:
        pass


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
        except Exception as exc:
            # CONNECT failed: the database is down or unreachable. Transient by nature, so still
            # fail open with the cached count, but SAY SO, and record it.
            #
            # This path used to be the silent one, and silence here is the worst outcome
            # available: a DSN that never resolves (a placeholder left in the config, a container
            # nobody starts, a renamed host) makes the count stick at whatever it was, which for a
            # fresh install is the 0 the installer wrote. `session_start` then emits nothing, and
            # the whole integration is indistinguishable from one that was never installed.
            # Measured 2026-08-25 on the author's own machine, where exactly that had happened:
            # `postgresql://example/recall`, `chunks: 0`, no diagnostic anywhere.
            _warn(
                f"cannot reach the database ({type(exc).__name__}); serving the cached count of "
                f"{int(config.get('chunks', 0) or 0)}. Check the dsn in {config_path()}"
            )
            _remember(config, status="unreachable")
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
        _warn(
            f"chunk count query refused ({type(exc).__name__}); "
            f"the cached figure may be stale until the config is fixed"
        )
        _remember(config, status="refused")
        return int(config.get("chunks", 0) or 0)

    _save_config({**config, "chunks": count, "status": "ok"})
    return count


def _remember(config: dict[str, Any], *, status: str) -> None:
    """Record why the count could not be taken, without disturbing the count itself.

    Separate from the success path's write because the two carry opposite information: one is a
    fresh measurement, the other is a reason the measurement is stale. Writing `chunks` here would
    quietly re-stamp a stale number as current.
    """
    if config.get("status") == status:
        return  # nothing new to say; do not rewrite the file on every failed session
    _save_config({**config, "status": status})


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


def session_start(payload: dict[str, Any]) -> int:
    """Inject a short memory digest before the first turn.

    A digest and not a retrieval, deliberately. `SessionStart` carries no user prompt, so there is
    no query to retrieve against, and a similarity search with nothing to be similar to returns
    whatever happens to be nearest. The event that has a query is `UserPromptSubmit`, and that is
    where per-turn retrieval belongs once this is proven out. The user-invoked version of the
    retrieval this cannot do is the plugin's `/recall:session-open` command, which has the user's
    own words to search with.

    Fails open, always: a hook that runs before every session must never be the reason Claude does
    not start, so anything unexpected is silence and exit 0.

    **Two output channels, and they carry different things.** `additionalContext` on stdout is
    charged to the model's context, so only a corpus worth searching goes there. A store that is
    configured and cannot be reached goes to stderr instead: the user needs to know, the model does
    not, and a broken memory tool that spends context complaining about itself has made the session
    worse in exactly the way it was installed to avoid.
    """
    config = load_config()
    count = int(config.get("chunks", 0) or 0)
    if not count:
        # Nothing to advertise. Two very different reasons land here, and only one of them is
        # ordinary: an empty corpus, or a corpus this hook has never been able to reach. Say which
        # on stderr rather than in `additionalContext`, because a broken memory tool must not
        # spend the model's context complaining about itself, and hook stderr is where a user
        # looking for the answer will actually find it.
        if config.get("dsn") and config.get("status") in {"unreachable", "refused"}:
            _warn(
                f"memory is configured but the last count could not be taken "
                f"({config.get('status')}), so nothing was injected this session. "
                f"Check the dsn in {config_path()}"
            )
        return 0
    cwd = str(payload.get("cwd", ""))
    project = Path(cwd).name if cwd else "this project"
    json.dump(
        {
            "hookSpecificOutput": {
                "hookEventName": "SessionStart",
                "additionalContext": (
                    f"RE-call memory is available for {project}: {count} indexed chunks of "
                    "decisions and hazards earlier sessions paid for. Search it with "
                    "`recall_search` BEFORE your first file edit or state-changing command, and "
                    "search for the OPERATIONS you are about to perform rather than for your "
                    "goal: a memo about a failure is written in the failure's vocabulary, so "
                    "'pip install breaks the build' retrieves what 'add a dependency' does not. "
                    "Two or three short queries with different words beat one long one. Treat "
                    "`abstained: true` as 'no supported answer' rather than as an empty one, and "
                    "a `superseded` verdict as a retraction to follow rather than a claim to act "
                    "on."
                ),
            }
        },
        sys.stdout,
    )
    return 0


def _claude_project_slug(path: Path) -> str:
    r"""Claude Code's own encoding of a project path into a directory name under `projects/`.

    Every character that is not an ASCII letter or digit becomes `-`, so `C:\Users\me\proj`
    becomes `C--Users-me-proj`.

    ⚠️ Derived from the directories the client had already written on this machine (1,755 of them,
    read 2026-08-25) rather than from any documented contract. That is why every caller treats a
    miss as "no such store" rather than as an error: if the client ever changes the encoding, this
    resolves to a directory that does not exist, and the hook behaves exactly as it did before
    this function was written. Re-check the encoding against reality with:

    ```bash
    ls ~/.claude/projects | head
    ```
    """
    return "".join(
        ch if ("a" <= ch <= "z" or "A" <= ch <= "Z" or "0" <= ch <= "9") else "-"
        for ch in str(path)
    )


def _worktree_parent(cwd: Path) -> Path | None:
    r"""The main checkout behind a git worktree, read from `.git` rather than by running git.

    In a worktree, `.git` is a FILE holding `gitdir: <main>/.git/worktrees/<name>`. This matters
    because Claude Code keys its memory store on the directory the session opened in: a session in
    a worktree gets a slug of its own with nothing behind it, while the project's real store sits
    under the MAIN checkout's slug. Without this, the case that matters most indexes nothing, since
    an agent working in a worktree is the normal arrangement here rather than an exotic one.

    No subprocess: this runs on every session end, and `git rev-parse` costs more than reading one
    small file. Anything unexpected in that file returns None, which degrades to "not a worktree".
    """
    marker = cwd / ".git"
    try:
        if not marker.is_file():
            return None
        raw = marker.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return None
    if not raw.startswith("gitdir:"):
        return None
    gitdir = raw[len("gitdir:") :].strip().replace("\\", "/")
    head, separator, _ = gitdir.partition("/.git/worktrees/")
    if not separator or not head:
        return None
    return Path(head)


def _memory_directories(cwd: Path) -> list[Path]:
    """Every memory store this session could plausibly have written to, in preference order.

    Two conventions, and the hook has to serve both because different things write them:

    - `<project>/memory/`, which is what `recall setup` scaffolds and what the `CLAUDE.md` block
      tells the agent to write into.
    - `~/.claude/projects/<slug>/memory/`, which is where Claude Code's own memory feature puts
      what the model writes down. Nothing in recall creates it, and on a machine using that
      feature it is where the memos actually accumulate.

    Only the first was ever consulted, so on any project using the client's store,
    including recall's own, which has no in-repo `memory/` at all, `SessionEnd` and `PreCompact` indexed
    nothing and said nothing. A directory that does not exist is dropped rather than reported,
    because "this project does not use that convention" is the common case rather than a fault.
    """
    roots = [cwd]
    parent = _worktree_parent(cwd)
    if parent is not None:
        roots.append(parent)
    projects = claude_config_home() / "projects"
    candidates = [cwd / "memory"]
    candidates += [projects / _claude_project_slug(root) / "memory" for root in roots]
    seen: set[str] = set()
    found: list[Path] = []
    for candidate in candidates:
        # Case-insensitive on Windows, where one store is reachable under several spellings and
        # indexing it twice is wasted work rather than a correctness problem.
        key = str(candidate).lower() if os.name == "nt" else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        try:
            if candidate.is_dir():
                found.append(candidate)
        except OSError:
            continue
    return found


def _declared_embedder(cwd: Path) -> str | None:
    """What the PROJECT says its embedder is, from the environment or its own `.env`.

    The hook config records an embedder once, at install, and never reconciles it. That is fine
    until the project changes model, at which point the hook keeps indexing with the old one while
    `recall index` from the CLI uses the new one, and the tenant ends up holding vectors from two
    models at once.

    ⛔ Nothing raises when that happens. `bge-large`, `voyage-3` and `voyage-4` all emit 1024
    dimensions, so pgvector computes a cosine between them happily and returns a confidently
    ranked list that means nothing. **A dimension match is not a model match**, and the legacy
    `chunks` table this hook writes to records no profile to check against.

    Parsed here rather than through `recall._env` so the rule stays one function: take the first
    assignment, strip one layer of matching quotes, ignore comments. Anything else returns None
    and the recorded value stands.
    """
    from_env = os.environ.get("RECALL_EMBEDDER", "").strip()
    if from_env:
        return from_env
    roots = [cwd]
    parent = _worktree_parent(cwd)
    if parent is not None:
        roots.append(parent)
    for root in roots:
        try:
            lines = (root / ".env").read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            continue
        for line in lines:
            key, separator, raw = line.partition("=")
            # Compare the KEY exactly, not a prefix of the line, and note that this one comparison
            # does two jobs. `RECALL_EMBEDDER_EXTRA=x` is a different variable, and the first
            # version of this read it as a declaration; a commented `# RECALL_EMBEDDER=x` yields
            # the key `# RECALL_EMBEDDER`, which is equally unequal. There was a separate
            # `startswith("#")` skip here and it was removed rather than kept: no mutation of it
            # could make a test fail, because the exact comparison already rejects every comment
            # form, and unfalsifiable defensive code reads as a guard while guarding nothing.
            if not separator or key.strip() != "RECALL_EMBEDDER":
                continue
            value = raw.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            if value:
                return value
    return None


def _embedder_for(config: dict[str, Any], cwd: Path) -> str:
    """The embedder to index with, preferring the project's own declaration over the recorded one.

    The project wins because `recall index` from the CLI reads the same declaration: agreeing with
    it is what keeps one tenant from being written by two models. The disagreement is reported
    every time rather than once, because it stays wrong until somebody re-runs the installer.
    """
    recorded = str(config.get("embedder", "fastembed"))
    declared = _declared_embedder(cwd)
    if declared and declared != recorded:
        _warn(
            f"this project declares RECALL_EMBEDDER={declared} but the hook config records "
            f"{recorded}; indexing with {declared} to agree with the CLI. Re-run `recall setup` "
            f"to update {config_path()}"
        )
        return declared
    return recorded


#: Values that turn `auto_index` off and on in the hook config. Both spellings of each, because
#: this file is edited by hand and `false` and `"false"` are the same intention.
_AUTO_INDEX_OFF = frozenset({"false", "0", "no", "off"})
_AUTO_INDEX_ON = frozenset({"true", "1", "yes", "on"})


def _auto_index_enabled(config: dict[str, Any]) -> bool:
    """Whether `SessionEnd` and `PreCompact` may WRITE, as opposed to only counting.

    On by default, because for most users an automatic indexer is the entire point: it is what
    makes memory compound rather than decay. It is switchable because for some corpora it is not.
    A store indexed deliberately on another host, under a lock and a memory cap, does not want an
    extra writer arriving from whichever workstation happened to close a session, even though
    `Indexer.index_path` serialises them correctly.

    ⚠️ An unrecognised value WARNS rather than picking a side, and that is the whole design of this
    function. Both defaults are wrong in a different direction: failing on runs an indexer somebody
    tried to disable, failing off silently stops indexing somebody expects, which is the exact
    class of silent no-op the rest of this module was rewritten to eliminate. So a typo is loud and
    the behaviour it falls back to is stated in the message.
    """
    raw = config.get("auto_index", True)
    if isinstance(raw, bool):
        return raw
    text = str(raw).strip().lower()
    if text in _AUTO_INDEX_OFF:
        return False
    if text in _AUTO_INDEX_ON:
        return True
    _warn(
        f"auto_index is set to {raw!r} in {config_path()}, which is neither true nor false; "
        f"indexing anyway. Use true or false."
    )
    return True


def _index_and_refresh(payload: dict[str, Any]) -> int:
    """Index this project's memory stores and refresh the cached count. Never raises.

    Shared by `SessionEnd` and `PreCompact`, which want the same thing at different moments: make
    whatever the session wrote down searchable, and leave an accurate count behind.
    """
    config = load_config()
    dsn = config.get("dsn")
    cwd = payload.get("cwd")
    if not dsn or not cwd:
        return 0
    if not _auto_index_enabled(config):
        # Deliberately switched off. Still refresh the count: reading is not writing, and a digest
        # that reports the corpus as it currently stands is the half of this that costs the
        # corpus nothing. Silent rather than warning, because a setting somebody chose on purpose
        # must not nag on every session.
        refresh_stats(config)
        return 0
    embedder = _embedder_for(config, Path(str(cwd)))
    for memory_dir in _memory_directories(Path(str(cwd))):
        try:
            from recall.setup import index_memory_directory

            index_memory_directory(
                dsn=str(dsn),
                embedder_name=embedder,
                memory_dir=memory_dir,
                # ⛔ These two were omitted, so the hook wrote to `DEFAULT_TENANT` and
                # `DEFAULT_TABLE` no matter what the user configured, while `refresh_stats`
                # counted the CONFIGURED tenant. A store set up with `tenant: memory` therefore
                # had its memos written where nothing looks for them and its count read 0 forever:
                # an index that reports success and lands somewhere else. The same defect is
                # recorded in `index_memory_directory`'s own docstring as already fixed for its
                # other callers. This caller never passed them.
                tenant=str(config.get("tenant", "default")),
                table=str(config.get("table", "chunks")),
                # Not a discarding lambda. `index_memory_directory` catches its own exceptions and
                # explains them through `print_fn`, so discarding it made every indexing failure
                # silent, including the one that says the schema is applied for a different
                # embedder's dimension, the failure most likely to happen here.
                print_fn=lambda *args, **kwargs: _warn(" ".join(str(a) for a in args)),
            )
        except Exception as exc:
            # The import itself, or an argument this version of `recall` does not accept. Warn and
            # try the next store rather than abandoning the refresh: a second store may index
            # fine, and the count is worth taking either way.
            _warn(f"could not index {memory_dir} ({type(exc).__name__})")
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
