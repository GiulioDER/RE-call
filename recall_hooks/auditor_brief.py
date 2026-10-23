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

import difflib
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
#: What a person runs to fill the memo cache with no deadline.
WARM_COMMAND = "python -m recall_hooks warm-auditor-brief"
#: Bounds on what the scope reads, so a huge prompt or diff cannot spend the budget on paths.
MAX_NAMED_FILES = 50
MAX_FILE_BYTES = 1_000_000
BASE_REFS = ("origin/HEAD", "origin/master", "origin/main")

_PATH_TOKEN = re.compile(r"[\w./\\-]+\.[A-Za-z0-9]{1,10}")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
#: `[user@]host:path`; the host needs two characters, so a Windows drive path is not read as one.
_SCP_REMOTE = re.compile(r"^(?:[\w.-]+@)?[\w.-]{2,}:(?!//)(.+)$")
_DRIVE = re.compile(r"^[A-Za-z]:")


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
    except Exception:  # BROAD-CATCH: fail-open  (an unwritable log is not a failure)
        pass


# --- scope: the repository, the named files, the changed code -----------------------------------


#: Config every git call carries. The hook runs on every auditor launch, often in a repository the
#: user did not write, so no command that repository configures may run: no fsmonitor, no pager, no
#: external diff. `--no-ext-diff` and `--no-textconv` on the diff close the attribute-selected
#: drivers. Clean filters cannot be closed by config at all, so no call here reads the work tree
#: through git (the diff is commit to commit, and `cat-file --batch` applies no filter). No call
#: here runs a hook. In a partial clone, though, a missing blob makes the diff and `cat-file`
#: fetch it from the promisor remote, which runs the repository's own remote URL (an `ext::`
#: command, an `ssh` command): `GIT_NO_LAZY_FETCH` (git 2.45 and later) stops the fetch, and
#: `GIT_ALLOW_PROTOCOL=none`, which outranks the repository's `protocol.*.allow`, stops any
#: transport an older git would still start. The missing blob then fails the diff, which the scope
#: treats like a missing ref, or `cat-file` reports it missing, which reads as no blob (found by
#: the CCA architect gate, 2026-10-22).
_GIT_SAFE_CONFIG = ("core.quotepath=off", "core.fsmonitor=false", "core.pager=cat", "diff.external=")
_GIT_SAFE_ENV = {
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_PAGER": "cat",
    "PAGER": "cat",
    "GIT_OPTIONAL_LOCKS": "0",
    "GIT_NO_LAZY_FETCH": "1",
    "GIT_ALLOW_PROTOCOL": "none",
}
_GIT_UNSAFE_ENV = ("GIT_EXTERNAL_DIFF", "GIT_DIFF_OPTS")
#: The diff options every scope diff uses: commit to commit, no driver, fixed prefixes (a user's
#: `diff.noprefix` or `diff.mnemonicPrefix` would otherwise leave every header unparseable).
_DIFF_OPTIONS = (
    "diff", "--no-color", "--no-ext-diff", "--no-textconv", "--unified=0", "--src-prefix=a/", "--dst-prefix=b/",
)


def _git_env() -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key not in _GIT_UNSAFE_ENV}
    env.update(_GIT_SAFE_ENV)
    return env


def _start_git(cwd: str, *args: str, stdin: bool = False) -> subprocess.Popen[bytes]:
    """`git -C cwd args` under `_GIT_SAFE_CONFIG`, started and not waited for: the scope runs its
    git calls side by side, because each one costs 100 to 200 ms to spawn on Windows (measured
    2026-10-22)."""
    config = [part for item in _GIT_SAFE_CONFIG for part in ("-c", item)]
    return subprocess.Popen(
        ["git", "--no-pager", "-C", cwd, *config, *args],
        stdin=subprocess.PIPE if stdin else subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_env(),
    )


def _finish_git(process: subprocess.Popen[bytes], deadline: float, data: bytes | None = None) -> bytes:
    """The output of a started git call. Raises `CalledProcessError` on failure, timeout past `deadline`."""
    try:
        out, err = process.communicate(input=data, timeout=max(deadline - time.monotonic(), 0.05))
    except subprocess.TimeoutExpired:
        process.kill()
        process.communicate()
        raise
    if process.returncode != 0:
        raise subprocess.CalledProcessError(process.returncode, process.args, out, err)
    return out


def _stop(processes: list[subprocess.Popen[bytes]]) -> None:
    """Kill and reap every process still running, so none outlives the scope."""
    for process in processes:
        if process.poll() is None:
            process.kill()
            try:
                process.communicate(timeout=0.2)
            except (subprocess.TimeoutExpired, OSError):
                pass


def repo_identity(url: str) -> str | None:
    """`owner/name` from a remote URL (https, ssh, or scp-like with or without a user), or None."""
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
    """Repository-relative paths of the files `prompt` names that exist in `root`, in order.

    Every token is checked lexically before any filesystem call, so a prompt-chosen path outside
    the repository is never touched: on Windows even resolving a UNC path (``//host/share/x.py``)
    authenticates to that host, which could send the user's credentials anywhere. An absolute
    token counts only when it spells out a path under `root` (case-insensitively on Windows). A
    path that passes is then resolved, so a symlink out of the repository is refused and the
    name is the one git and the records use, in the file system's own case.
    """
    found: list[str] = []
    seen: set[str] = set()
    fold = str.casefold if os.name == "nt" else str
    root_text = fold(_DRIVE.sub("", root.as_posix()).rstrip("/"))
    try:
        resolved_root = root.resolve()
    except OSError:
        return found
    for token in _PATH_TOKEN.findall(prompt):
        token = token.replace("\\", "/")
        if token in seen:
            continue
        seen.add(token)
        if token.startswith("/"):
            if not root_text or not fold(token).startswith(root_text + "/"):
                continue
            token = token[len(root_text) + 1 :]
        parts = [part for part in token.split("/") if part not in ("", ".")]
        if not parts or ".." in parts:
            continue
        try:
            resolved = root.joinpath(*parts).resolve()
            relative = resolved.relative_to(resolved_root).as_posix()
            if relative in found or not resolved.is_file():
                continue
        except (OSError, ValueError):
            continue
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

    `enclosing_symbol`'s rule (the innermost span containing the line) and `symbol_sha`'s hash
    (`code_sha` of the span's normalised lines) applied over ONE parse of the file. Calling those
    two per line and per symbol re-parses the file each time: measured 2026-10-22 on a branch
    adding two modules, 97 anchors cost about 750 ms, most of the hook's budget.
    `tests/test_auditor_brief_hook.py` checks the hash against `symbol_sha` itself.
    """
    from recall.anchors import code_sha, normalize, resolve_symbols

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
    lines_of = normalize(source).split("\n")
    by_name = {span.symbol: span for span in spans}

    def sha(symbol: str | None) -> str:
        if symbol is None:
            return code_sha(source)
        span = by_name[symbol]
        return code_sha("\n".join(lines_of[span.start_line - 1 : span.end_line]))

    return [{"file": file, "symbol": symbol, "code_sha": sha(symbol)} for symbol in symbols]


def _parse_batch(output: bytes, files: list[str]) -> dict[str, bytes | None]:
    """`git cat-file --batch` output for `files`, in request order: each blob's bytes, or None when
    the path did not exist at that revision."""
    blobs: dict[str, bytes | None] = {}
    position = 0
    for file in files:
        end = output.find(b"\n", position)
        if end < 0:
            break
        header = output[position:end].split()
        position = end + 1
        if len(header) == 3 and header[2].isdigit():
            size = int(header[2])
            blobs[file] = output[position : position + size] if header[1] == b"blob" else None
            position += size + 1
        else:
            blobs[file] = None  # "<spec> missing": the file is new since the base
    return blobs


def _small(path: Path) -> bool:
    """Whether the work-tree copy at `path` exists and is at most `MAX_FILE_BYTES`.

    A missing copy is not small: a changed file deleted from the work tree, or outside a sparse
    checkout, still exists at HEAD, and its blob would be streamed whole. Like the pre-fix work-tree
    read, such a file is anchored at file level with no hash.
    """
    try:
        return path.stat().st_size <= MAX_FILE_BYTES
    except OSError:
        return False


def _blob_text(blob: bytes | None) -> str | None:
    """A blob as text, or None when it is absent, over `MAX_FILE_BYTES` or not UTF-8 (as `_read_source`)."""
    if blob is None or len(blob) > MAX_FILE_BYTES:
        return None
    try:
        return blob.decode("utf-8")
    except UnicodeDecodeError:
        return None


def _changed_since(base: str | None, current: str) -> set[int]:
    """The 1-based lines of `current` that differ from `base` (every line when `base` is None),
    compared over `recall.anchors.normalize`, the text the anchors hash and number."""
    from recall.anchors import normalize

    new = normalize(current).split("\n")
    if base is None:
        return set(range(1, len(new) + 1))
    lines: set[int] = set()
    for tag, _i1, _i2, j1, j2 in difflib.SequenceMatcher(None, normalize(base).split("\n"), new).get_opcodes():
        if tag != "equal":
            # A pure deletion (j1 == j2) sits after line j1; that line is where it happened.
            lines.update(range(j1 + 1, j2 + 1) if j2 > j1 else {max(j1, 1)})
    return lines


def start_scope(cwd: str) -> dict[str, subprocess.Popen[bytes]]:
    """Start the scope's git calls for `cwd`, side by side, before anything else is imported."""
    started: dict[str, subprocess.Popen[bytes]] = {}
    try:
        started["root"] = _start_git(cwd, "rev-parse", "--show-toplevel")
        started["url"] = _start_git(cwd, "config", "--get", "remote.origin.url")
        started["base"] = _start_git(cwd, "merge-base", "HEAD", BASE_REFS[0])
        started["diff"] = _start_git(cwd, *_DIFF_OPTIONS, "--merge-base", BASE_REFS[0], "HEAD", "--")
    except BaseException:
        _stop(list(started.values()))
        raise
    return started


def finish_scope(
    started: dict[str, subprocess.Popen[bytes]], cwd: str, prompt: str, *, deadline: float
) -> tuple[str, list[dict[str, Any]]]:
    """``(repo, anchors)`` for the repository holding `cwd`. Raises when `cwd` is not in one.

    The committed changes come from a commit-to-commit diff against the merge base, which never
    reads the work tree. The files the prompt names are compared in Python: the work-tree copy read
    from disk against the merge-base copy from `git cat-file --batch`, so their uncommitted changes
    count too. An uncommitted change in a file the prompt does not name is not in scope.
    """
    processes = list(started.values())
    try:
        root_text = _finish_git(started["root"], deadline).decode("utf-8", "replace").strip()
        if not root_text:
            raise RuntimeError(f"{cwd} is not inside a git work tree")
        root = Path(root_text)
        repo = repo_identity(_finish_git(started["url"], deadline).decode("utf-8", "replace"))
        if repo is None:
            raise RuntimeError("the origin remote names no owner/name repository")
        base: str | None = None
        diff: str | None = None
        for index, ref in enumerate(BASE_REFS):
            if index == 0:
                pair = (started["base"], started["diff"])
            else:
                pair = (
                    _start_git(cwd, "merge-base", "HEAD", ref),
                    _start_git(cwd, *_DIFF_OPTIONS, "--merge-base", ref, "HEAD", "--"),
                )
                processes.extend(pair)
            try:
                base = _finish_git(pair[0], deadline).decode("ascii", "replace").strip() or None
                diff = _finish_git(pair[1], deadline).decode("utf-8", "replace")
                break
            except subprocess.CalledProcessError:
                base, diff = None, None  # no such ref, or no merge base with it: try the next
        named = named_files(prompt, root)
        changes = changed_lines(diff) if diff is not None else {}
        from recall.records import MAX_QUERY_ANCHORS

        # One batch for every copy git must supply: a named Python file's merge-base copy (its
        # uncommitted change is found against it) and a changed file's HEAD copy (the commit diff's
        # line numbers count that copy's lines, not the work tree's).
        # `cat-file` streams a blob whole, so a file whose work-tree copy is over `MAX_FILE_BYTES`,
        # or absent, is not requested at all (the pre-fix code skipped it on the same stat); it is
        # anchored at file level with no hash, as `_read_source` would have left it.
        specs = [f"{base}:{file}" for file in named if base and file.endswith(".py") and _small(root / file)]
        specs += [
            f"HEAD:{file}"
            for file, lines in changes.items()
            if lines is not None and file not in named and _small(root / file)
        ]
        specs = [spec for spec in specs if "\n" not in spec][: len(named) + MAX_QUERY_ANCHORS]
        blobs: dict[str, bytes | None] = {}
        if specs:
            batch = _start_git(cwd, "cat-file", "--batch", stdin=True)
            processes.append(batch)
            request = "".join(f"{spec}\n" for spec in specs).encode("utf-8")
            blobs = _parse_batch(_finish_git(batch, deadline, request), specs)
    finally:
        _stop(processes)

    from recall.anchors import code_sha

    anchors: list[dict[str, Any]] = []
    cut = False
    # Named files first: the prompt asked for them, so the cap below never drops one for a diff entry.
    for file in named:
        if time.monotonic() > deadline:
            cut = True
            break
        source = _read_source(root / file)
        if source is None:
            anchors.append({"file": file, "symbol": None, "code_sha": None})
            continue
        lines: set[int] = set()
        if base and file.endswith(".py"):
            blob = blobs.get(f"{base}:{file}")
            if blob is None or len(blob) <= MAX_FILE_BYTES:
                lines = _changed_since(blob.decode("utf-8", "replace") if blob is not None else None, source)
        if lines:
            anchors.extend(_python_anchors(file, source, lines))
        else:
            anchors.append({"file": file, "symbol": None, "code_sha": code_sha(source)})
    for file, changed in changes.items():
        if file in named or cut:
            continue
        if len(anchors) >= MAX_QUERY_ANCHORS or time.monotonic() > deadline:
            cut = True  # the rest would not be queried, or there is no time left to read it
            break
        source = _blob_text(blobs.get(f"HEAD:{file}")) if changed is not None else None
        if source is None:
            anchors.append({"file": file, "symbol": None, "code_sha": None})
        elif file.endswith(".py") and changed:
            anchors.extend(_python_anchors(file, source, changed))
        else:
            anchors.append({"file": file, "symbol": None, "code_sha": code_sha(source)})

    unique: dict[tuple[str, str | None], dict[str, Any]] = {}
    for anchor in anchors:
        unique.setdefault((anchor["file"], anchor["symbol"]), anchor)
    if cut or len(unique) > MAX_QUERY_ANCHORS:
        _log(f"scope cut to at most {MAX_QUERY_ANCHORS} anchors (the query limit) or by the deadline")
    return repo, list(unique.values())[:MAX_QUERY_ANCHORS]


def scope(cwd: str, prompt: str, *, deadline: float) -> tuple[str, list[dict[str, Any]]]:
    """``(repo, anchors)`` for the repository holding `cwd`: `start_scope`, then `finish_scope`."""
    return finish_scope(start_scope(cwd), cwd, prompt, deadline=deadline)


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

    The git calls start first, then the memos are read on a second thread, so git's spawn, the
    `recall` imports and the memo reading overlap rather than queue. The reading stops a little
    before `deadline`, keeping what it parsed in the on-disk cache.
    """
    started = start_scope(cwd)
    loaded: dict[str, Any] = {}

    def load() -> None:
        try:
            from recall import records_files

            loaded["records"] = records_files.load_memo_records(
                memo_directory(), cache_file=cache_path(), deadline=deadline - FINISH_RESERVE
            )
        except TimeoutError as exc:
            loaded["error"] = TimeoutError(f"{exc}; `{WARM_COMMAND}` fills it with no deadline")
        except BaseException as exc:  # BROAD-CATCH: error-translation  (re-raised on the waiting thread)
            loaded["error"] = exc

    reader = threading.Thread(target=load, name="recall-auditor-brief-memos", daemon=True)
    reader.start()
    repo, anchors = finish_scope(started, cwd, prompt, deadline=deadline)
    reader.join(max(0.0, deadline - time.monotonic()))
    if reader.is_alive():
        raise TimeoutError("the memo directory was still being read at the deadline")
    if "error" in loaded:
        raise loaded["error"]
    if not anchors:
        return ""
    from recall import records_files

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
            except BaseException as exc:  # BROAD-CATCH: error-translation  (reported by the waiting thread)
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
    except Exception as exc:  # BROAD-CATCH: fail-open  (a launch must never wait on this hook)
        _log(f"{agent_type}: {type(exc).__name__}: {exc}; prompt left unchanged")
        return 0
    return 0


def warm() -> int:
    """`python -m recall_hooks warm-auditor-brief`: parse every memo into the on-disk cache.

    A command a person (or an installer) runs, so unlike the hook it has no deadline and fails
    loudly. It exists because a cold cache warms slowly inside the hook's budget: the first open
    of a memo file cost about 12 ms on the workstation this was written on (on-access scanning;
    a second read cost about 1 ms, measured 2026-10-22), so a 700-memo directory whose files were
    not opened recently gained as few as a dozen memos per launch, and every launch until it is
    warm leaves its prompt unchanged.
    """
    from recall import records_files

    directory = memo_directory()
    try:
        loaded = records_files.load_memo_records(directory, cache_file=cache_path())
    except OSError as exc:
        print(f"recall: cannot read the memo directory {directory}: {exc}", file=sys.stderr)
        return 1
    print(f"recall: cached {len(loaded)} record(s) from {directory}; {len(loaded.skipped)} memo(s) skipped")
    return 0


__all__ = [
    "CCA_HEADINGS",
    "DEFAULT_AUDITOR_TYPES",
    "HEADING",
    "MARKER",
    "auditor_types",
    "build_block",
    "changed_lines",
    "finish_scope",
    "memo_directory",
    "named_files",
    "pre_agent",
    "render",
    "repo_identity",
    "scope",
    "start_scope",
]
