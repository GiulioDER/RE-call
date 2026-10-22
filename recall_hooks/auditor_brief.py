"""`PreToolUse` on `Agent`: give an auditor subagent the prior bugs and lessons at the code it audits.

CCA's own auditors are briefed with the bugs CCA fixed and the lessons it accepted at the changed
code. This hook gives the same context to any auditor subagent a Claude Code session launches,
just before launch, by appending one block to the subagent's prompt through `updatedInput`
(`docs/AUDIT_MEMORY_DESIGN.md`, "Auditor brief hook"; owner decision, 2026-10-22).

What it reads, and what it does not:

- **Scope** is the repository of the session's `cwd`: its identity (`owner/name`, from the `origin`
  remote), the files the prompt names that exist in it, and the files changed against the merge
  base with `origin/HEAD` (else `origin/master`, else `origin/main`). A changed Python line is
  resolved to its enclosing symbol by `recall.anchors`, so those files are looked up by anchor;
  every other file at file level.
- **Source** is the CCA memo directory on this machine (`RECALL_AUDITOR_MEMOS`, else
  `$CCA_BUGS_DIR/memos`, else `~/.claude/cca-bugs/memos`), read by `recall.records_files` through
  RE-call's own record parser and the matching rules `recall_records` answers with. No network, no
  ssh, no embedding: an exact key lookup needs no vectors. Parsed memos are cached on disk
  (`<claude config home>/cache/recall-auditor-brief-memos.json`, keyed on path, mtime and size),
  because admitting ~700 records costs more than the budget and each launch is a new process.
- **Kinds** `bug` and `lesson` only. A `false_positive` record is never queried, so it can never
  reach an auditor, and the block never tells an auditor to leave anything out.

⛔ Properties this must never lose:

1. **It never blocks and never decides.** It prints either nothing or `updatedInput`, never a
   permission decision, so the client's normal permission evaluation still applies; and it always
   exits 0.
2. **It keeps every field of the original `tool_input`.** `updatedInput` replaces the whole input.
3. **It fails open within about one second.** Any error, an unreadable source, or a wall clock
   past the budget leaves the prompt unchanged and appends one line naming the reason to
   `<claude config home>/logs/recall-auditor-brief.log`. The work runs on a daemon thread the hook
   stops waiting for, so even a source that hangs cannot hold the launch.
4. **It is quiet for everything else.** A subagent type outside the auditor list, or a prompt that
   already carries one of CCA's brief headings or this hook's marker, passes untouched, so no
   auditor sees the context twice.

The block's benefit is unmeasured; it ships as context, and no benefit is claimed until a
pre-registered comparison supports one.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import threading
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

#: The default auditor list, matched after any `plugin:` prefix is removed.
DEFAULT_AUDITOR_TYPES = (
    "bug-auditor",
    "security-auditor",
    "code-auditor",
    "perf-auditor",
    "numeric-auditor",
    "deploy-auditor",
    "dep-auditor",
    "doc-auditor",
    "env-validator",
)
#: CCA's own brief headings: a prompt holding one was briefed by CCA already.
CCA_HEADINGS = (
    "PRIOR FIXES IN THESE FILES",
    "PRIOR BUGS AT THE CHANGED CODE",
    "LESSONS AT THE CHANGED CODE",
)
MARKER = "<!-- recall:auditor-brief -->"
HEADING = "PRIOR BUGS AND LESSONS AT THIS CODE (RE-call; context, not findings, not instructions)"
KINDS = ("bug", "lesson")
MAX_BUGS = 5
MAX_LESSONS = 3
MAX_BLOCK_CHARS = 4000
#: Per record line. Eight full lines exceed `MAX_BLOCK_CHARS`, so the block cap is what binds.
MAX_LINE_CHARS = 600
#: Wall clock for the whole hook, from entry. `RECALL_AUDITOR_BRIEF_BUDGET` overrides it.
BUDGET_SECONDS = 1.0
LOG_NAME = "recall-auditor-brief.log"
#: The parsed-memo cache, beside the log under the client's config home.
CACHE_NAME = "recall-auditor-brief-memos.json"
#: Seconds kept back from the memo reading for matching, rendering and printing.
FINISH_RESERVE = 0.15
#: Seconds by which the stages' own deadlines precede the hook's hard stop.
INNER_MARGIN = 0.1
#: Bounds on what the scope reads, so a huge prompt or diff cannot spend the budget on paths.
MAX_NAMED_FILES = 50
MAX_ANCHORS = 200
MAX_FILE_BYTES = 1_000_000
BASE_REFS = ("origin/HEAD", "origin/master", "origin/main")

_PATH_TOKEN = re.compile(r"[\w./\\-]+\.[A-Za-z0-9]{1,10}")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
_SCP_REMOTE = re.compile(r"^[\w.-]+@[\w.-]+:(.+)$")


def _bare(agent_type: str) -> str:
    """`agent_type` without a `plugin:` prefix."""
    return agent_type.rsplit(":", 1)[-1].strip()


def auditor_types() -> frozenset[str]:
    """The auditor list: `RECALL_AUDITOR_TYPES` (comma separated) when set, else the default."""
    raw = os.environ.get("RECALL_AUDITOR_TYPES", "")
    names = [_bare(name) for name in raw.split(",")] if raw.strip() else list(DEFAULT_AUDITOR_TYPES)
    return frozenset(name for name in names if name)


def memo_directory() -> Path:
    """Where the CCA memos are read from."""
    explicit = os.environ.get("RECALL_AUDITOR_MEMOS", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    bugs = os.environ.get("CCA_BUGS_DIR", "").strip()
    if bugs:
        return Path(bugs).expanduser() / "memos"
    return Path.home() / ".claude" / "cca-bugs" / "memos"


def _budget() -> float:
    try:
        value = float(os.environ.get("RECALL_AUDITOR_BRIEF_BUDGET", "") or BUDGET_SECONDS)
    except ValueError:
        return BUDGET_SECONDS
    return value if value > 0 else BUDGET_SECONDS


def log_path() -> Path:
    from recall_hooks import claude_config_home

    return claude_config_home() / "logs" / LOG_NAME


def _log(reason: str) -> None:
    """Append one line, UTC time and reason, to the hook's log. Never raises."""
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = " ".join(reason.split())[:500]
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(UTC).isoformat(timespec='seconds')} {line}\n")
    except Exception:  # BROAD-CATCH: fail-open, a log that cannot be written is not a failure
        pass


# --- scope: the repository, the named files, the changed code -----------------------------------


def _start_git(cwd: str, *args: str) -> subprocess.Popen[str]:
    """`git -C cwd args`, started and not waited for: the scope runs its git calls side by side,
    because each one costs 100 to 200 ms to spawn on Windows (measured 2026-10-22)."""
    return subprocess.Popen(
        ["git", "-C", cwd, "-c", "core.quotepath=off", *args],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _finish_git(process: subprocess.Popen[str], deadline: float) -> str:
    """The output of a started git call. Raises `CalledProcessError` on failure, timeout past `deadline`."""
    try:
        out, err = process.communicate(timeout=max(deadline - time.monotonic(), 0.05))
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args, out, err)
    return out


def repo_identity(url: str) -> str | None:
    """`owner/name` from a remote URL (https, ssh or scp-like), or None."""
    url = url.strip().rstrip("/")
    url = url.removesuffix(".git")
    scp = _SCP_REMOTE.match(url)
    if "://" in url:
        path = urlsplit(url).path
    elif scp:
        path = scp.group(1)
    else:
        path = url
    parts = [part for part in path.replace("\\", "/").split("/") if part]
    return "/".join(parts[-2:]) if len(parts) >= 2 else None


def named_files(prompt: str, root: Path) -> list[str]:
    """Repository-relative paths of the files `prompt` names that exist in `root`, in order."""
    found: list[str] = []
    resolved_root = root.resolve()
    for token in _PATH_TOKEN.findall(prompt):
        token = token.replace("\\", "/").removeprefix("./")
        path = Path(token) if Path(token).is_absolute() else root / token
        try:
            relative = path.resolve().relative_to(resolved_root).as_posix()
        except (OSError, ValueError):
            continue
        if relative not in found and path.is_file():
            found.append(relative)
            if len(found) >= MAX_NAMED_FILES:
                break
    return found


def changed_lines(diff: str) -> dict[str, set[int] | None]:
    """Each file a `git diff --unified=0` touches, with its changed new-side lines (None: deleted).

    Headers are read only between `diff --git` and the first hunk, so a removed line whose text
    starts with `-- ` is never mistaken for a file header.
    """
    changes: dict[str, set[int] | None] = {}
    in_header = False
    old: str | None = None
    current: str | None = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            in_header, old, current = True, None, None
        elif in_header and line.startswith("--- "):
            name = line[4:]
            old = name[2:] if name.startswith("a/") else None
        elif in_header and line.startswith("+++ "):
            name = line[4:]
            new = name[2:] if name.startswith("b/") else None
            current = new or old
            if current is not None:
                changes.setdefault(current, set() if new else None)
        elif line.startswith("@@"):
            in_header = False
            match = _HUNK.match(line)
            lines = changes.get(current) if current is not None else None
            if match is None or lines is None:
                continue
            start, count = int(match.group(1)), int(match.group(2) or 1)
            # A pure deletion (count 0) sits after line `start`; that line is where it happened.
            lines.update(range(start, start + count) if count else {max(start, 1)})
    return changes


def _read_source(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _python_anchors(file: str, source: str, lines: set[int]) -> list[dict[str, Any]]:
    """One anchor per symbol enclosing a changed line; a module-level line anchors the file.

    `enclosing_symbol`'s rule (the innermost span containing the line) applied over ONE parse of
    the file, rather than one parse per changed line, which a large diff would make cost seconds.
    """
    from recall.anchors import code_sha, resolve_symbols, symbol_sha

    try:
        spans = resolve_symbols(source)
    except SyntaxError:
        # A file mid-edit: no symbols to resolve, so the file is anchored as a whole.
        return [{"file": file, "symbol": None, "code_sha": code_sha(source)}]
    symbols: list[str | None] = []
    for line in sorted(lines):
        containing = [s for s in spans if s.start_line <= line <= s.end_line]
        symbol = (
            min(containing, key=lambda s: (s.end_line - s.start_line, -s.start_line)).symbol
            if containing
            else None
        )
        if symbol not in symbols:
            symbols.append(symbol)
    return [{"file": file, "symbol": symbol, "code_sha": symbol_sha(source, symbol)} for symbol in symbols]


def scope(cwd: str, prompt: str, *, deadline: float) -> tuple[str, list[dict[str, Any]]]:
    """``(repo, anchors)`` for the repository holding `cwd`. Raises when `cwd` is not in one."""
    from recall.anchors import code_sha

    def diff_against(ref: str) -> subprocess.Popen[str]:
        # `--merge-base`: the work tree against the merge base of `ref` and HEAD, in one call.
        return _start_git(cwd, "diff", "--no-color", "--no-ext-diff", "--unified=0", "--merge-base", ref, "--")

    started = [
        _start_git(cwd, "rev-parse", "--show-toplevel"),
        _start_git(cwd, "config", "--get", "remote.origin.url"),
        diff_against(BASE_REFS[0]),
    ]
    try:
        root_text = _finish_git(started[0], deadline).strip()
        if not root_text:
            raise RuntimeError(f"{cwd} is not inside a git work tree")
        root = Path(root_text)
        repo = repo_identity(_finish_git(started[1], deadline))
        if repo is None:
            raise RuntimeError("the origin remote names no owner/name repository")
        diff: str | None = None
        for index, ref in enumerate(BASE_REFS):
            process = started[2] if index == 0 else diff_against(ref)
            try:
                diff = _finish_git(process, deadline)
                break
            except subprocess.CalledProcessError:
                continue  # no such ref, or no merge base with it: try the next
    finally:
        for process in started:
            if process.poll() is None:
                process.kill()
    changes = changed_lines(diff) if diff is not None else {}

    anchors: list[dict[str, Any]] = []
    for file, lines in changes.items():
        source = _read_source(root / file) if lines is not None else None
        if source is None:
            anchors.append({"file": file, "symbol": None, "code_sha": None})
        elif file.endswith(".py") and lines:
            anchors.extend(_python_anchors(file, source, lines))
        else:
            anchors.append({"file": file, "symbol": None, "code_sha": code_sha(source)})
    for file in named_files(prompt, root):
        if file in changes:
            continue
        source = _read_source(root / file)
        anchors.append({"file": file, "symbol": None, "code_sha": code_sha(source) if source is not None else None})

    unique: dict[tuple[str, str | None], dict[str, Any]] = {}
    for anchor in anchors:
        unique.setdefault((anchor["file"], anchor["symbol"]), anchor)
    return repo, list(unique.values())[:MAX_ANCHORS]


# --- the block ---------------------------------------------------------------------------------


def _clean(value: object, limit: int = MAX_LINE_CHARS) -> str:
    """`value` on one line with no backtick, so it can never open or close the block's fence."""
    text = str(value).replace("`", "'")
    # `str.split()` with no argument splits on every Unicode line break as well as whitespace
    # (U+2028, U+0085, the ASCII separators), so the joined text is one line.
    text = "".join(ch for ch in " ".join(text.split()) if ch.isprintable())
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _best_anchor(entry: dict[str, Any]) -> dict[str, Any]:
    for item in entry.get("anchor_matches", []):
        if item.get("match") == entry.get("anchor_match"):
            anchor = item.get("anchor")
            return anchor if isinstance(anchor, dict) else {}
    return {}


def _line(entry: dict[str, Any], now: datetime, retraction_files: set[str]) -> str | None:
    """One record as one line, or None when a retraction in force hides it."""
    from recall.supersession import parse_window, window_contains

    successor = entry.get("superseded_by")
    if successor is not None and successor in retraction_files:
        return None
    record = entry["recall_record"]
    anchor = _best_anchor(entry)
    label = "BUG" if entry["kind"] == "bug" else "LESSON"
    text = record.get("summary") if entry["kind"] == "bug" else record.get("rule")
    where = f"{anchor.get('file')} {anchor.get('symbol') or '(module)'}"
    notes = []
    if successor is not None:
        notes.append(f"superseded by {successor}")
    if not window_contains(parse_window(entry.get("valid_from"), entry.get("valid_until")), now):
        notes.append("outside its validity window")
    suffix = f" ({'; '.join(notes)})" if notes else ""
    return _clean(f"{label} {entry['id']} [{entry['anchor_match']}] {where}: {text or ''}{suffix}")


def render(entries: list[dict[str, Any]], *, now: datetime, retraction_files: set[str]) -> str:
    """The block for `entries` (best match first), or "" when none is shown. At most `MAX_BLOCK_CHARS`."""
    bugs: list[str] = []
    lessons: list[str] = []
    for entry in entries:
        bucket, cap = (bugs, MAX_BUGS) if entry.get("kind") == "bug" else (lessons, MAX_LESSONS)
        if len(bucket) < cap and (line := _line(entry, now, retraction_files)) is not None:
            bucket.append(line)
    lines = bugs + lessons
    if not lines:
        return ""
    head = [
        f"## {HEADING}",
        MARKER,
        "Records RE-call holds of bugs CCA fixed and lessons it accepted at the code this task "
        "touches. Each line: kind, id, [match: exact, symbol_changed or file_only], file, symbol.",
        "```text",
    ]
    tail = "```"
    note = "({} more not shown: size cap)"
    # Room for the fence's last line and for the note, whatever its count (at most two digits).
    budget = MAX_BLOCK_CHARS - len("\n".join(head)) - len(tail) - len(note.format(99)) - 2
    shown: list[str] = []
    used = 0
    for line in lines:
        if used + len(line) + 1 > budget:
            break
        shown.append(line)
        used += len(line) + 1
    if len(shown) < len(lines):
        shown.append(note.format(len(lines) - len(shown)))
    return "\n".join([*head, *shown, tail])


def cache_path() -> Path:
    from recall_hooks import claude_config_home

    return claude_config_home() / "cache" / CACHE_NAME


def build_block(cwd: str, prompt: str, *, deadline: float, now: datetime | None = None) -> str:
    """The block for an auditor launched in `cwd` with `prompt`, or "" when nothing matches.

    The memos are read on a second thread while the scope waits on git, so the two costs overlap.
    The reading stops a little before `deadline`, keeping what it parsed in the on-disk cache.
    """
    from recall import records_files

    loaded: dict[str, Any] = {}

    def load() -> None:
        try:
            loaded["records"] = records_files.load_memo_records(
                memo_directory(), cache_file=cache_path(), deadline=deadline - FINISH_RESERVE
            )
        except BaseException as exc:  # BROAD-CATCH: re-raised on the waiting thread
            loaded["error"] = exc

    reader = threading.Thread(target=load, name="recall-auditor-brief-memos", daemon=True)
    reader.start()
    repo, anchors = scope(cwd, prompt, deadline=deadline)
    reader.join(max(0.0, deadline - time.monotonic()))
    if reader.is_alive():
        raise TimeoutError("the memo directory was still being read at the deadline")
    if "error" in loaded:
        raise loaded["error"]
    if not anchors:
        return ""
    instant = now or datetime.now(UTC)
    records = loaded["records"]
    entries = records_files.match_memo_records(records, repo=repo, anchors=anchors, kinds=list(KINDS), now=instant)
    retractions = {
        memo.file for memo in records if str(memo.record.get("kind", "")).endswith("_retraction")
    }
    return render(entries, now=instant, retraction_files=retractions)


# --- the event ---------------------------------------------------------------------------------


def pre_agent(payload: dict[str, Any]) -> int:
    """Entry point for `python -m recall_hooks pre-agent`. Returns 0 always."""
    started = time.monotonic()
    agent_type = "?"
    try:
        if payload.get("tool_name") != "Agent":
            return 0
        tool_input = payload.get("tool_input")
        if not isinstance(tool_input, dict):
            return 0
        agent_type = str(tool_input.get("subagent_type") or "")
        if _bare(agent_type) not in auditor_types():
            return 0
        prompt = tool_input.get("prompt")
        if not isinstance(prompt, str) or MARKER in prompt or any(h in prompt for h in CCA_HEADINGS):
            return 0
        cwd = str(payload.get("cwd") or "")
        if not cwd:
            _log(f"{agent_type}: the event carries no cwd; prompt left unchanged")
            return 0
        budget = _budget()
        outcome: dict[str, Any] = {}

        def work() -> None:
            try:
                # The inner bounds fire first, so a slow stage is logged by name; the join below
                # is the backstop for a stage nothing inside bounds.
                outcome["block"] = build_block(cwd, prompt, deadline=started + budget - INNER_MARGIN)
            except BaseException as exc:  # BROAD-CATCH: reported by the waiting thread
                outcome["error"] = exc

        worker = threading.Thread(target=work, name="recall-auditor-brief", daemon=True)
        worker.start()
        worker.join(max(0.0, started + budget - time.monotonic()))
        if worker.is_alive():
            _log(f"{agent_type}: budget of {budget:.2f}s exceeded; prompt left unchanged")
            return 0
        if "error" in outcome:
            error = outcome["error"]
            _log(f"{agent_type}: {type(error).__name__}: {error}; prompt left unchanged")
            return 0
        block = outcome.get("block") or ""
        if not block:
            return 0
        updated = {**tool_input, "prompt": f"{prompt.rstrip()}\n\n{block}\n"}
        # `ensure_ascii` left at its default: memo text is full of non-ASCII and stdout may be cp1252.
        sys.stdout.write(
            json.dumps({"hookSpecificOutput": {"hookEventName": "PreToolUse", "updatedInput": updated}})
        )
    except Exception as exc:  # BROAD-CATCH: fail-open, a launch must never wait on this hook
        _log(f"{agent_type}: {type(exc).__name__}: {exc}; prompt left unchanged")
        return 0
    return 0


__all__ = [
    "CCA_HEADINGS",
    "DEFAULT_AUDITOR_TYPES",
    "HEADING",
    "MARKER",
    "auditor_types",
    "build_block",
    "changed_lines",
    "memo_directory",
    "named_files",
    "pre_agent",
    "render",
    "repo_identity",
    "scope",
]
