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
  ssh, no embedding: an exact key lookup needs no vectors. Parsed memos are cached on disk, one
  file per memo directory under `<claude config home>/cache/`, keyed on path, mtime and size,
  because each launch is a new process and a memo file is slow to read the first time: measured
  2026-10-22 on the Windows workstation, the first open cost about 12 ms (on-access scanning) and
  a second read about 1 ms, so a directory of hundreds of memos does not fit the budget cold.
- **Kinds** `bug` and `lesson` only. A `false_positive` record is never queried, so it can never
  reach an auditor, nor can its file name, and the block never tells an auditor to leave anything
  out.

⛔ Properties this must never lose:

1. **It never blocks and never decides.** It prints either nothing or `updatedInput`, never a
   permission decision, so the client's normal permission evaluation still applies; and it always
   exits 0.
2. **It keeps every field of the original `tool_input`.** `updatedInput` replaces the whole input.
3. **It fails open within about three seconds.** Any error, an unreadable source, or a wall clock
   past the budget leaves the prompt unchanged and appends one line naming the reason to
   `<claude config home>/logs/recall-auditor-brief.log`. The work runs on a daemon thread the hook
   stops waiting for, so even a source that hangs cannot hold the launch.
4. **It is quiet for everything else.** A subagent type outside the auditor list, or a prompt that
   already carries one of CCA's brief headings or this hook's marker, passes untouched, so no
   auditor sees the context twice. So is a launch inside a CCA run whose `memory-arm.json` names
   the `holdout` arm (`cca_holdout_run`): CCA withheld its brief from that run on purpose, and the
   Codex bridge sends an empty prompt, so the heading check alone cannot see it.

The block's benefit is unmeasured; it ships as context, and no benefit is claimed until a
pre-registered comparison supports one.
"""

from __future__ import annotations

import difflib
import hashlib
import json
import math
import os
import re
import stat
import subprocess
import sys
import threading
import time
from collections.abc import Callable, Set
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Generic, TypeVar
from urllib.parse import urlsplit

#: The default auditor list, matched case-insensitively after any `plugin:` prefix is removed.
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
MAX_BUGS = 5
MAX_LESSONS = 3
#: How each record kind the block shows is presented: its line label, the record field its text
#: comes from, and how many of it the block holds. `KINDS`, the kinds queried, follows from it.
_KIND_VIEW: dict[str, tuple[str, str, int]] = {
    "bug": ("BUG", "summary", MAX_BUGS),
    "lesson": ("LESSON", "rule", MAX_LESSONS),
}
KINDS = tuple(_KIND_VIEW)
MAX_BLOCK_CHARS = 4000
#: Per record line. Eight full lines exceed `MAX_BLOCK_CHARS`, so the block cap is what binds.
MAX_LINE_CHARS = 600
#: Per log line, so one runaway error message cannot fill the log.
LOG_LINE_CHARS = 500
#: Wall clock for the whole hook, from entry. `RECALL_AUDITOR_BRIEF_BUDGET` overrides it, within
#: (0, `MAX_BUDGET_SECONDS`]: a thread join cannot wait longer than the platform's time limit, and
#: a budget past the client's hook timeout would have the client kill the hook with no log line.
#: Sized for a CCA L1 dispatch, which launches its auditors at once: measured 2026-10-27 on the
#: Windows workstation, nine hooks at once against a 21-file diff spent 0.75 to 1.14 s in git and
#: 0.3 s on anchors, so 1 s briefed 0 of 9 and 3 s 9 of 9, still inside a 5 s client timeout.
BUDGET_SECONDS = 3.0
MAX_BUDGET_SECONDS = 30.0
LOG_NAME = "recall-auditor-brief.log"
#: The parsed-memo cache, one file per memo directory, beside the log under the client's config home.
CACHE_PREFIX = "recall-auditor-brief-memos-"
#: Seconds kept back from the memo reading for matching, rendering and printing.
FINISH_RESERVE = 0.15
#: Seconds by which the stages' own deadlines precede the hook's hard stop.
INNER_MARGIN = 0.1
#: The shortest wait given to a git call, so a call started at the deadline still gets an answer.
MIN_GIT_WAIT_SECONDS = 0.05
#: How long a killed git process is given to exit before it is left to the operating system.
REAP_WAIT_SECONDS = 0.2
#: Bounds on what the scope reads, so a huge prompt or diff cannot spend the budget on paths.
MAX_NAMED_FILES = 50
MAX_NAMED_TOKENS = 400
# A junction's reparse tag. `stat.IO_REPARSE_TAG_MOUNT_POINT` exists only on Windows, and so does
# `st_reparse_tag`, so elsewhere the comparison is simply never true.
JUNCTION_REPARSE_TAG = 0xA0000003
MAX_FILE_BYTES = 1_000_000
BASE_REFS = ("origin/HEAD", "origin/master", "origin/main")
#: The hash a query anchor carries for a file whose content could not be read: it equals no real
#: hash and, unlike None, not a lesson's null hash either, so an unread file never matches `exact`.
UNKNOWN_SHA = "unavailable"

PATH_TOKEN_RE = re.compile(r"[\w./\\-]+\.[A-Za-z0-9]{1,10}")
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,(\d+))? @@")
BINARY_RE = re.compile(r'^Binary files ("?a/.+|/dev/null) and ("?b/.+|/dev/null) differ$')
#: `[user@]host:path`; the host needs two characters, so a Windows drive path is not read as one.
SCP_REMOTE_RE = re.compile(r"^(?:[\w.-]+@)?[\w.-]{2,}:(?!//)(.+)$")
DRIVE_RE = re.compile(r"^[A-Za-z]:")
#: git's C-quoting of a path (`core.quotepath=off` still quotes quotes, backslashes and controls).
_C_ESCAPES = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}

T = TypeVar("T")


def _bare(agent_type: str) -> str:
    """`agent_type` without a `plugin:` prefix, case-folded."""
    return agent_type.rsplit(":", 1)[-1].strip().casefold()


def auditor_types() -> frozenset[str]:
    """The auditor list: `RECALL_AUDITOR_TYPES` (comma separated) when it names one, else the default.

    A value that names no type (``,``) falls back to the default and says so, rather than switching
    the hook off in silence.
    """
    raw = os.environ.get("RECALL_AUDITOR_TYPES", "")
    if raw.strip():
        names = frozenset(name for name in (_bare(part) for part in raw.split(",")) if name)
        if names:
            return names
        _log(f"RECALL_AUDITOR_TYPES={raw!r} names no agent type; using the default list")
    return frozenset(DEFAULT_AUDITOR_TYPES)


def memo_directory() -> Path:
    """Where the CCA memos are read from.

    `CCA_BUGS_DIR` is read exactly as CCA reads it (`cca_checks.bugs.store`: the raw value), so the
    two always name the same directory.
    """
    explicit = os.environ.get("RECALL_AUDITOR_MEMOS", "").strip()
    if explicit:
        return Path(explicit).expanduser()
    bugs = os.environ.get("CCA_BUGS_DIR")
    if bugs:
        return Path(bugs) / "memos"
    return Path.home() / ".claude" / "cca-bugs" / "memos"


def _budget() -> float:
    """The budget in seconds: `RECALL_AUDITOR_BRIEF_BUDGET` within (0, `MAX_BUDGET_SECONDS`], else
    the default, logging any value it did not use as given."""
    raw = os.environ.get("RECALL_AUDITOR_BRIEF_BUDGET", "").strip()
    if not raw:
        return BUDGET_SECONDS
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value) or value <= 0:
        _log(f"RECALL_AUDITOR_BRIEF_BUDGET={raw!r} is not a positive number of seconds; using {BUDGET_SECONDS}")
        return BUDGET_SECONDS
    if value > MAX_BUDGET_SECONDS:
        _log(f"RECALL_AUDITOR_BRIEF_BUDGET={raw!r} is above {MAX_BUDGET_SECONDS} s; using {MAX_BUDGET_SECONDS}")
        return MAX_BUDGET_SECONDS
    return value


def log_path() -> Path:
    """The hook's log: `<claude config home>/logs/recall-auditor-brief.log`."""
    from recall_hooks import claude_config_home

    return claude_config_home() / "logs" / LOG_NAME


def cache_path(directory: Path) -> Path:
    """The parsed-memo cache for `directory`: one file per memo directory, so two configurations
    reading different directories never evict each other."""
    from recall_hooks import claude_config_home

    try:
        key = str(Path(directory).resolve())
    except OSError:
        key = str(directory)
    digest = hashlib.sha256(key.encode("utf-8", "surrogatepass")).hexdigest()[:12]
    return claude_config_home() / "cache" / f"{CACHE_PREFIX}{digest}.json"


def warm_command() -> str:
    """The warm command as the hook itself runs: this interpreter, `-P`, and this copy of
    `recall_hooks` on `PYTHONPATH`, so it imports the code (and so writes the cache) the hook reads."""
    package_parent = Path(__file__).resolve().parent.parent
    if sys.platform == "win32":
        def literal(value: str) -> str:
            return "'" + value.replace("'", "''") + "'"

        # The inline block comment labels the shell even after the diagnostic log flattens lines.
        return (
            "& { <# PowerShell #> $recallPriorPythonPath = $env:PYTHONPATH; "
            f"try {{ $env:PYTHONPATH = {literal(str(package_parent))}; "
            f"& {literal(sys.executable)} -P -m recall_hooks warm-auditor-brief "
            "} finally { $env:PYTHONPATH = $recallPriorPythonPath } }"
        )
    return f'PYTHONPATH="{package_parent}" "{sys.executable}" -P -m recall_hooks warm-auditor-brief'


def _log(reason: str) -> None:
    """Append one line, UTC time and reason, to the hook's log. Never raises."""
    try:
        path = log_path()
        path.parent.mkdir(parents=True, exist_ok=True)
        line = " ".join(reason.split())[:LOG_LINE_CHARS]
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"{datetime.now(UTC).isoformat(timespec='seconds')} {line}\n")
    except Exception:  # BROAD-CATCH: fail-open  (an unwritable log is not a failure)
        pass


class _Background(Generic[T]):
    """`fn` on a daemon thread, waited for with a bound: the shape both the hook's hard stop and
    the memo reader use. A thread still running at the bound is abandoned, never joined again."""

    def __init__(self, fn: Callable[[], T], name: str) -> None:
        self._value: T | None = None
        self._error: BaseException | None = None
        self._thread = threading.Thread(target=self._run, args=(fn,), name=name, daemon=True)
        self._thread.start()

    def _run(self, fn: Callable[[], T]) -> None:
        try:
            self._value = fn()
        except BaseException as exc:  # BROAD-CATCH: error-translation  (re-raised by `value`)
            self._error = exc

    def wait(self, timeout: float) -> bool:
        """Whether `fn` finished within `timeout` seconds."""
        self._thread.join(max(0.0, timeout))
        return not self._thread.is_alive()

    def value(self) -> T:
        """`fn`'s result, or its exception re-raised. Call only after `wait` returned True."""
        if self._error is not None:
            raise self._error
        return self._value  # type: ignore[return-value]  # set by `_run` before `wait` returned True


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
#: `diff.noprefix` or `diff.mnemonicPrefix` would otherwise leave every header unparseable), and no
#: rename detection, so a renamed file arrives as a deletion and an addition, each with a header.
_DIFF_OPTIONS = (
    "diff", "--no-color", "--no-ext-diff", "--no-textconv", "--no-renames", "--unified=0",
    "--src-prefix=a/", "--dst-prefix=b/",
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
        out, err = process.communicate(
            input=data, timeout=max(deadline - time.monotonic(), MIN_GIT_WAIT_SECONDS)
        )
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
                process.communicate(timeout=REAP_WAIT_SECONDS)
            except (subprocess.TimeoutExpired, OSError):
                pass


def repo_identity(url: str) -> str | None:
    """`owner/name` from a remote URL (https, ssh, or scp-like with or without a user), or None."""
    url = url.strip().rstrip("/")
    url = url.removesuffix(".git")
    scp = SCP_REMOTE_RE.match(url)
    if "://" in url:
        path = urlsplit(url).path
    elif scp:
        path = scp.group(1)
    else:
        path = url
    parts = [part for part in path.replace("\\", "/").split("/") if part]
    return "/".join(parts[-2:]) if len(parts) >= 2 else None


def _unlinked_lstat(root: Path, parts: list[str]) -> os.stat_result | None:
    """`lstat` of `root` joined with `parts`, or None when it is missing or any component below
    `root` is a symbolic link or a junction.

    The components are looked up one at a time from the top, so no call ever traverses a link: a
    link committed to an audited repository can point at a UNC path, and following it (to resolve
    or even to stat the path) makes Windows authenticate to that host, as SEC-002 found for a
    prompt-chosen path. `root` itself is the user's checkout and is not checked.

    Each component must also name exactly one child of the path above it. A changed file's name
    comes from a git tree, which a malicious commit can fill with anything: on Windows
    `root / "\\\\host\\share\\x.py"` IS that UNC path, a drive replaces `root`, and a backslash or
    a `..` climbs out of it. A colon is refused on Windows too (a drive or a data stream).
    """
    path = root
    result: os.stat_result | None = None
    for part in parts:
        child = path / part
        if part in ("", ".", "..") or child.parent != path or child.name != part or (os.name == "nt" and ":" in part):
            return None
        path = child
        try:
            result = os.lstat(path)
        except OSError:
            return None
        if stat.S_ISLNK(result.st_mode) or getattr(result, "st_reparse_tag", 0) == JUNCTION_REPARSE_TAG:
            return None
    return result


def named_files(prompt: str, root: Path) -> list[str]:
    """Repository-relative paths of the files `prompt` names that exist in `root`, in order.

    Every token is checked lexically before any filesystem call, so a prompt-chosen path outside
    the repository is never touched: on Windows even resolving a UNC path (``//host/share/x.py``)
    authenticates to that host, which could send the user's credentials anywhere. An absolute
    token counts only when it spells out a path under `root` (case-insensitively on Windows). A
    path that passes must then be a regular file reached through no link (`_unlinked_lstat`),
    since the repository's own content is not trusted either; only then is it resolved, so the
    name is the one git and the records use, in the file system's own case. At most
    `MAX_NAMED_TOKENS` distinct tokens reach the file system, so a prompt that pastes a long
    listing cannot spend the budget on it (measured 2026-10-22: 4000 tokens cost 6.6 s).
    """
    found: list[str] = []
    seen: set[str] = set()
    fold = str.casefold if os.name == "nt" else str
    root_text = fold(DRIVE_RE.sub("", root.as_posix()).rstrip("/"))
    try:
        resolved_root = root.resolve()
    except OSError:
        return found
    for token in PATH_TOKEN_RE.findall(prompt):
        token = token.replace("\\", "/")
        if token in seen:
            continue
        if len(seen) >= MAX_NAMED_TOKENS:
            break
        seen.add(token)
        if token.startswith("/"):
            if not root_text or not fold(token).startswith(root_text + "/"):
                continue
            token = token[len(root_text) + 1 :]
        parts = [part for part in token.split("/") if part not in ("", ".")]
        if not parts or ".." in parts:
            continue
        found_stat = _unlinked_lstat(root, parts)
        if found_stat is None or not stat.S_ISREG(found_stat.st_mode):
            continue
        try:
            resolved = root.joinpath(*parts).resolve()
            relative = resolved.relative_to(resolved_root).as_posix()
            if relative in found:
                continue
        except (OSError, ValueError):
            continue
        found.append(relative)
        if len(found) >= MAX_NAMED_FILES:
            break
    return found


def _unquote_c(text: str) -> str:
    """A git C-quoted path's inside, unescaped: `\\"`, `\\\\`, the control escapes, and octal bytes."""
    out = bytearray()
    index = 0
    while index < len(text):
        char = text[index]
        if char == "\\" and index + 1 < len(text):
            following = text[index + 1]
            if following in _C_ESCAPES:
                out.append(_C_ESCAPES[following])
                index += 2
                continue
            octal = text[index + 1 : index + 4]
            if len(octal) == 3 and all(digit in "01234567" for digit in octal):
                out.append(int(octal, 8))
                index += 4
                continue
        out.extend(char.encode("utf-8"))
        index += 1
    return out.decode("utf-8", "replace")


def _header_path(raw: str, prefix: str) -> str | None:
    """The path a `---`/`+++`/`Binary files` side names, without `prefix`, or None for `/dev/null`.

    git ends a header path holding a space with a tab, and C-quotes a path holding a quote, a
    backslash or a control character; both forms are read back to the plain path.
    """
    name = raw.removesuffix("\t")
    if len(name) >= 2 and name.startswith('"') and name.endswith('"'):
        name = _unquote_c(name[1:-1])
    return name[len(prefix) :] if name.startswith(prefix) else None


def _mode_only_path(rest: str) -> str | None:
    """The path of a `diff --git a/P b/P` line, when both sides name the same unquoted path."""
    if not rest.startswith("a/") or (len(rest) - 5) % 2:
        return None
    half = (len(rest) - 5) // 2
    path = rest[2 : 2 + half]
    return path if path and rest == f"a/{path} b/{path}" else None


def changed_lines(diff: str) -> dict[str, set[int] | None]:
    """Each file a `git diff --unified=0` touches, with its changed new-side lines (None: deleted).

    Headers are read only between `diff --git` and the first hunk, so a removed line whose text
    starts with `-- ` is never mistaken for a file header. A file whose change has no text hunks
    (a binary file, a mode change) is listed with no lines, so it is still in scope at file level.
    """
    changes: dict[str, set[int] | None] = {}
    in_header = False
    old: str | None = None
    current: str | None = None
    pending: str | None = None  # a mode-only path, kept only if its block names no other path

    def flush() -> None:
        if pending is not None and current is None:
            changes.setdefault(pending, set())

    for line in diff.splitlines():
        if line.startswith("diff --git "):
            flush()
            in_header, old, current = True, None, None
            pending = _mode_only_path(line[len("diff --git ") :])
        elif in_header and line.startswith("--- "):
            old = _header_path(line[4:], "a/")
        elif in_header and line.startswith("+++ "):
            new = _header_path(line[4:], "b/")
            current = new or old
            if current is not None:
                changes.setdefault(current, set() if new else None)
        elif in_header and (binary := BINARY_RE.match(line)):
            new = _header_path(binary.group(2), "b/")
            current = new or _header_path(binary.group(1), "a/")
            if current is not None:
                changes.setdefault(current, set() if new else None)
        elif line.startswith("@@"):
            in_header = False
            match = HUNK_RE.match(line)
            lines = changes.get(current) if current is not None else None
            if match is None or lines is None:
                continue
            start, count = int(match.group(1)), int(match.group(2) or 1)
            # A pure deletion (count 0) sits after line `start`; that line is where it happened.
            lines.update(range(start, start + count) if count else {max(start, 1)})
    flush()
    return changes


def _read_source(path: Path) -> str | None:
    try:
        if path.stat().st_size > MAX_FILE_BYTES:
            return None
        return path.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None


def _innermost_symbols(spans: list[Any], line_count: int) -> list[str | None]:
    """For each line 1..`line_count`, `enclosing_symbol`'s answer, from one pass over the spans.

    Spans are painted largest first, so a smaller span overwrites the one around it, and among
    spans of one size the later start is painted last: `enclosing_symbol`'s tie-break (the
    smallest span, then the latest start). One table instead of a scan of every span per line,
    which a large new module made cost hundreds of milliseconds.
    """
    owner: list[str | None] = [None] * (line_count + 1)
    for span in sorted(spans, key=lambda s: (-(s.end_line - s.start_line), s.start_line)):
        for line in range(max(span.start_line, 1), min(span.end_line, line_count) + 1):
            owner[line] = span.symbol
    return owner


def _python_anchors(file: str, source: str, lines: set[int]) -> list[dict[str, Any]]:
    """One anchor per symbol enclosing a changed line; a module-level line anchors the file.

    `enclosing_symbol`'s rule (the innermost span containing the line) and `symbol_sha`'s hash
    (`code_sha` of the span's normalised lines) applied over ONE parse of the file. Calling those
    two per line and per symbol re-parses the file each time: measured 2026-10-22 on a branch
    adding two modules, 97 anchors cost about 750 ms, most of the hook's budget.
    `tests/test_auditor_brief_hook.py` checks the hash against `symbol_sha` and the symbol against
    `enclosing_symbol` itself.
    """
    from recall.anchors import code_sha, normalize, resolve_symbols

    try:
        spans = resolve_symbols(source)
    except SyntaxError:
        # A file mid-edit: no symbols to resolve, so the file is anchored as a whole.
        return [{"file": file, "symbol": None, "code_sha": code_sha(source)}]
    lines_of = normalize(source).split("\n")
    owner = _innermost_symbols(spans, len(lines_of))
    symbols: dict[str | None, None] = {}
    for line in sorted(lines):
        symbols.setdefault(owner[line] if 0 < line < len(owner) else None, None)
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


def _small(root: Path, file: str) -> bool:
    """Whether the work-tree copy of `file` under `root` exists and is at most `MAX_FILE_BYTES`.

    A missing copy is not small: a changed file deleted from the work tree, or outside a sparse
    checkout, still exists at HEAD, and its blob would be streamed whole. Like the pre-fix work-tree
    read, such a file is anchored at file level, with the hash `UNKNOWN_SHA`. So is a copy reached
    through a link, which is never followed (`_unlinked_lstat`).
    """
    result = _unlinked_lstat(root, file.split("/"))
    return result is not None and result.st_size <= MAX_FILE_BYTES


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


@dataclass
class _GitScope:
    """What git told the scope: the repository, the merge base and its diff, and the blobs."""

    root: Path
    repo: str
    named: list[str]
    base: str | None = None
    changes: dict[str, set[int] | None] = field(default_factory=dict)
    blobs: dict[str, bytes | None] = field(default_factory=dict)


def _base_and_diff(
    started: dict[str, subprocess.Popen[bytes]], cwd: str, deadline: float, processes: list[subprocess.Popen[bytes]]
) -> tuple[str | None, str | None]:
    """The merge base and the commit-to-commit diff against the first of `BASE_REFS` that has one."""
    for index, ref in enumerate(BASE_REFS):
        if index == 0:
            pair = (started["base"], started["diff"])
        else:
            first = _start_git(cwd, "merge-base", "HEAD", ref)
            processes.append(first)
            second = _start_git(cwd, *_DIFF_OPTIONS, "--merge-base", ref, "HEAD", "--")
            processes.append(second)
            pair = (first, second)
        try:
            base = _finish_git(pair[0], deadline).decode("ascii", "replace").strip() or None
            return base, _finish_git(pair[1], deadline).decode("utf-8", "replace")
        except subprocess.CalledProcessError:
            continue  # no such ref, or no merge base with it: try the next
    _log(f"no base ref among {', '.join(BASE_REFS)} has a merge base with HEAD; scope is the named files only")
    return None, None


def _collect(
    started: dict[str, subprocess.Popen[bytes]], cwd: str, prompt: str, deadline: float
) -> _GitScope:
    """Finish the started git calls and fetch, in one `cat-file --batch`, every blob the anchors need."""
    from recall.records import MAX_QUERY_ANCHORS

    processes = list(started.values())
    try:
        root_text = _finish_git(started["root"], deadline).decode("utf-8", "replace").strip()
        if not root_text:
            raise RuntimeError(f"{cwd} is not inside a git work tree")
        root = Path(root_text)
        repo = repo_identity(_finish_git(started["url"], deadline).decode("utf-8", "replace"))
        if repo is None:
            raise RuntimeError("the origin remote names no owner/name repository")
        base, diff = _base_and_diff(started, cwd, deadline, processes)
        found = _GitScope(root=root, repo=repo, named=named_files(prompt, root), base=base)
        found.changes = changed_lines(diff) if diff is not None else {}
        # One batch for every copy git must supply: a named Python file's merge-base copy (its
        # uncommitted change is found against it) and a changed file's HEAD copy (the commit diff's
        # line numbers count that copy's lines, not the work tree's).
        # `cat-file` streams a blob whole, so a file whose work-tree copy is over `MAX_FILE_BYTES`,
        # or absent, is not requested at all (the pre-fix code skipped it on the same stat); it is
        # anchored at file level with `UNKNOWN_SHA`, which never matches a hash exactly.
        specs = [f"{base}:{file}" for file in found.named if base and file.endswith(".py") and _small(root, file)]
        specs += [
            f"HEAD:{file}"
            for file, lines in found.changes.items()
            if lines is not None and file not in found.named and _small(root, file)
        ]
        specs = [spec for spec in specs if "\n" not in spec][: len(found.named) + MAX_QUERY_ANCHORS]
        if specs:
            batch = _start_git(cwd, "cat-file", "--batch", stdin=True)
            processes.append(batch)
            request = "".join(f"{spec}\n" for spec in specs).encode("utf-8")
            found.blobs = _parse_batch(_finish_git(batch, deadline, request), specs)
        return found
    finally:
        _stop(processes)


def _anchors(found: _GitScope, deadline: float) -> tuple[list[dict[str, Any]], bool]:
    """The query anchors for `found`, named files first, and whether the limit or deadline cut them."""
    from recall.anchors import code_sha
    from recall.records import MAX_QUERY_ANCHORS

    anchors: list[dict[str, Any]] = []
    # Named files first: the prompt asked for them, so the cap below never drops one for a diff entry.
    for file in found.named:
        if time.monotonic() > deadline:
            return anchors, True
        source = _read_source(found.root / file)
        if source is None:
            anchors.append({"file": file, "symbol": None, "code_sha": UNKNOWN_SHA})
            continue
        lines: set[int] = set()
        if found.base and file.endswith(".py"):
            blob = found.blobs.get(f"{found.base}:{file}")
            if blob is None or len(blob) <= MAX_FILE_BYTES:
                lines = _changed_since(blob.decode("utf-8", "replace") if blob is not None else None, source)
        if lines:
            anchors.extend(_python_anchors(file, source, lines))
        else:
            anchors.append({"file": file, "symbol": None, "code_sha": code_sha(source)})
    for file, changed in found.changes.items():
        if file in found.named:
            continue
        if len(anchors) >= MAX_QUERY_ANCHORS or time.monotonic() > deadline:
            return anchors, True  # the rest would not be queried, or there is no time left to read it
        source = _blob_text(found.blobs.get(f"HEAD:{file}")) if changed is not None else None
        if source is None:
            anchors.append({"file": file, "symbol": None, "code_sha": UNKNOWN_SHA})
        elif file.endswith(".py") and changed:
            anchors.extend(_python_anchors(file, source, changed))
        else:
            anchors.append({"file": file, "symbol": None, "code_sha": code_sha(source)})
    return anchors, False


def finish_scope(
    started: dict[str, subprocess.Popen[bytes]], cwd: str, prompt: str, *, deadline: float
) -> tuple[str, list[dict[str, Any]]]:
    """``(repo, anchors)`` for the repository holding `cwd`. Raises when `cwd` is not in one.

    The committed changes come from a commit-to-commit diff against the merge base, which never
    reads the work tree. The files the prompt names are compared in Python: the work-tree copy read
    from disk against the merge-base copy from `git cat-file --batch`, so their uncommitted changes
    count too. An uncommitted change in a file the prompt does not name is not in scope.
    """
    from recall.records import MAX_QUERY_ANCHORS

    found = _collect(started, cwd, prompt, deadline)
    anchors, cut = _anchors(found, deadline)
    unique: dict[tuple[str, str | None], dict[str, Any]] = {}
    for anchor in anchors:
        unique.setdefault((anchor["file"], anchor["symbol"]), anchor)
    if cut or len(unique) > MAX_QUERY_ANCHORS:
        _log(f"scope cut to at most {MAX_QUERY_ANCHORS} anchors (the query limit) or by the deadline")
    return found.repo, list(unique.values())[:MAX_QUERY_ANCHORS]


def scope(cwd: str, prompt: str, *, deadline: float) -> tuple[str, list[dict[str, Any]]]:
    """``(repo, anchors)`` for the repository holding `cwd`: `start_scope`, then `finish_scope`."""
    return finish_scope(start_scope(cwd), cwd, prompt, deadline=deadline)


# --- the block ---------------------------------------------------------------------------------


def _clean(value: object) -> str:
    """`value` on one line with no backtick, so it can never open or close the block's fence."""
    text = str(value).replace("`", "'")
    # `str.split()` with no argument splits on every Unicode line break as well as whitespace
    # (U+2028, U+0085, the ASCII separators), so the joined text is one line.
    text = "".join(ch for ch in " ".join(text.split()) if ch.isprintable())
    return text if len(text) <= MAX_LINE_CHARS else text[: MAX_LINE_CHARS - 3] + "..."


#: A record anchored at a whole file (`symbol: null`) cannot be told apart from the query's file
#: level anchor by symbol, so `recall.records._match` calls it `symbol_changed` or `exact` on a
#: hash alone, which outranks a record anchored at a real symbol that only reaches `file_only`.
#: The brief ranks and prints it as `file_only`; `_match` itself is unchanged, because the
#: false-positive matcher depends on its `exact`. False restores the matcher's labels.
FILE_LEVEL_RANKS_AS_FILE_ONLY: bool = True


def _brief_view(entry: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """`(level, anchor)` the brief ranks and prints `entry` at: its best anchor, with a symbol-less
    anchor counted as `file_only`. On a tie a symbol anchor wins, so the line names a real symbol."""
    from recall.records import MATCH_ORDER

    best: tuple[tuple[int, bool], str, dict[str, Any]] | None = None
    for item in entry.get("anchor_matches", []):
        anchor = item.get("anchor")
        anchor = anchor if isinstance(anchor, dict) else {}
        level = str(item.get("match"))
        symbolless = FILE_LEVEL_RANKS_AS_FILE_ONLY and anchor.get("symbol") is None
        if symbolless:
            level = max(level, "file_only", key=MATCH_ORDER.index)
        key = (MATCH_ORDER.index(level), symbolless)
        if best is None or key < best[0]:
            best = (key, level, anchor)
    if best is None:
        return str(entry["anchor_match"]), {}
    return best[1], best[2]


def _brief_match(entry: dict[str, Any]) -> str:
    """The match level the brief ranks and prints `entry` at."""
    return _brief_view(entry)[0]


def _newest_first(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """`entries` by brief match level, then symbol anchors before symbol-less ones, then newest first
    by record date (none is oldest), then id.

    The second key keeps a record anchored at a real symbol that reached `file_only` ahead of a
    newer symbol-less record at the same level, so the cap cannot push it out. With
    `FILE_LEVEL_RANKS_AS_FILE_ONLY` False no anchor counts as symbol-less and the key is constant.
    """
    from recall.records import MATCH_ORDER

    def rank(entry: dict[str, Any]) -> tuple[int, bool]:
        level, anchor = _brief_view(entry)
        symbolless = FILE_LEVEL_RANKS_AS_FILE_ONLY and bool(anchor) and anchor.get("symbol") is None
        return MATCH_ORDER.index(level), symbolless

    ordered = sorted(entries, key=lambda e: str(e.get("id")))
    ordered.sort(key=lambda e: str((e.get("recall_record") or {}).get("date") or ""), reverse=True)
    ordered.sort(key=rank)
    return ordered


def _line(entry: dict[str, Any], now: datetime, retraction_files: set[str], unnamed_files: Set[str]) -> str | None:
    """One record as one line, or None when a retraction in force hides it.

    A successor in `unnamed_files` (a false-positive memo) is reported as "superseded" without its
    name: no false-positive record may reach an auditor, its file name included.
    """
    from recall.supersession import parse_window, window_contains

    successor = entry.get("superseded_by")
    if successor is not None and successor in retraction_files:
        return None
    record = entry["recall_record"]
    anchor = _brief_view(entry)[1]
    label, text_field, _cap = _KIND_VIEW[entry["kind"]]
    text = record.get(text_field)
    where = f"{anchor.get('file')} {anchor.get('symbol') or '(module)'}"
    notes = []
    if successor is not None:
        notes.append("superseded" if successor in unnamed_files else f"superseded by {successor}")
    if not window_contains(parse_window(entry.get("valid_from"), entry.get("valid_until")), now):
        notes.append("outside its validity window")
    suffix = f" ({'; '.join(notes)})" if notes else ""
    return _clean(f"{label} {entry['id']} [{_brief_match(entry)}] {where}: {text or ''}{suffix}")


def render(
    entries: list[dict[str, Any]],
    *,
    now: datetime,
    retraction_files: set[str],
    unnamed_files: frozenset[str] = frozenset(),
) -> str:
    """The block for `entries` (best match first), or "" when none is shown. At most `MAX_BLOCK_CHARS`."""
    buckets: dict[str, list[str]] = {kind: [] for kind in _KIND_VIEW}
    for entry in _newest_first(entries):
        kind = entry.get("kind")
        if kind not in _KIND_VIEW:
            continue
        bucket, cap = buckets[kind], _KIND_VIEW[kind][2]
        if len(bucket) < cap and (line := _line(entry, now, retraction_files, unnamed_files)) is not None:
            bucket.append(line)
    lines = [line for kind in _KIND_VIEW for line in buckets[kind]]
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
    # Room for the fence's last line and for the note at its longest (every line but none shown),
    # and for the two line breaks that join the head, the shown lines and the tail.
    joins = 2
    budget = MAX_BLOCK_CHARS - len("\n".join(head)) - len(tail) - len(note.format(len(lines))) - joins
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


def build_block(cwd: str, prompt: str, *, deadline: float, now: datetime | None = None) -> str:
    """The block for an auditor launched in `cwd` with `prompt`, or "" when nothing matches.

    The git calls start first, then the memos are read on a second thread, so git's spawn, the
    `recall` imports and the memo reading overlap rather than queue. The reading stops a little
    before `deadline`, keeping what it parsed in the on-disk cache.
    """
    started = start_scope(cwd)
    directory = memo_directory()

    def load() -> Any:
        from recall import records_files

        try:
            return records_files.load_memo_records(
                directory, cache_file=cache_path(directory), deadline=deadline - FINISH_RESERVE
            )
        except TimeoutError as exc:
            raise TimeoutError(f"{exc}; run `{warm_command()}` with the hook's environment to fill it") from exc

    reader: _Background[Any] = _Background(load, "recall-auditor-brief-memos")
    repo, anchors = finish_scope(started, cwd, prompt, deadline=deadline)
    if not reader.wait(deadline - time.monotonic()):
        raise TimeoutError("the memo directory was still being read at the deadline")
    records = reader.value()
    if not anchors:
        return ""
    from recall import records_files
    from recall.records import RECORD_KINDS

    instant = now or datetime.now(UTC)
    entries = records_files.match_memo_records(records, repo=repo, anchors=anchors, kinds=list(KINDS), now=instant)
    retraction_kinds = {kind for kind in RECORD_KINDS if kind.endswith("_retraction")}
    retractions = {memo.file for memo in records if memo.record.get("kind") in retraction_kinds}
    unnamed = frozenset(memo.file for memo in records if memo.record.get("kind") == "false_positive")
    return render(entries, now=instant, retraction_files=retractions, unnamed_files=unnamed)


# --- CCA's memory holdout ----------------------------------------------------------------------


#: Where CCA keeps the open run's state and each run's memory arm, relative to the project root.
#: Read from CCA's on-disk layout, never by importing CCA: `cca_gate_core.state_file`
#: (`.claude/audits/run-state.json`, key `run_id`), `deferred.runs_dir` and `holdout.ARM_FILE`.
CCA_STATE = (".claude", "audits", "run-state.json")
CCA_RUNS = (".claude", "audits", "runs")
CCA_ARM_FILE = "memory-arm.json"
CCA_HOLDOUT_ARM = "holdout"
#: CCA's own run id pattern (`pipeline_state._RUN_ID`), so a state file cannot name a path.
_CCA_RUN_ID_RE = re.compile(r"[A-Za-z0-9._-]+")
#: At most this many candidate roots are read, so a repository with hundreds of worktrees cannot
#: spend the budget on them.
MAX_CCA_ROOTS = 256


def _read_repo_json(root: Path, parts: tuple[str, ...]) -> Any:
    """The JSON at `root` joined with `parts`, or None when it is missing, unreadable, over
    `MAX_FILE_BYTES`, not a regular file, or reached through a link (`_unlinked_lstat`, SEC-002):
    these files sit inside the audited repository, which can commit a link to a UNC path."""
    result = _unlinked_lstat(root, list(parts))
    if result is None or not stat.S_ISREG(result.st_mode) or result.st_size > MAX_FILE_BYTES:
        return None
    try:
        return json.loads(root.joinpath(*parts).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _cca_roots(cwd: str, deadline: float) -> list[Path]:
    """The roots a CCA run for `cwd` may be open in: git's top level for `cwd` (else `cwd`
    itself), then every worktree of that repository. Codex starts an auditor in the main checkout
    while CCA runs in a linked worktree, so the top level alone misses most Codex runs."""
    roots = [Path(cwd)]
    processes: list[subprocess.Popen[bytes]] = []
    try:
        processes.append(_start_git(cwd, "rev-parse", "--show-toplevel"))
        processes.append(_start_git(cwd, "worktree", "list", "--porcelain"))
        try:
            top = _finish_git(processes[0], deadline).decode("utf-8", "replace").strip()
            if top:
                roots = [Path(top)]
        except subprocess.SubprocessError:
            pass  # not a git work tree, or git is slow: the root is `cwd` itself
        try:
            listing = _finish_git(processes[1], deadline).decode("utf-8", "replace")
        except subprocess.SubprocessError:
            listing = ""
        roots += [Path(line[len("worktree "):]) for line in listing.splitlines() if line.startswith("worktree ")]
    finally:
        _stop(processes)
    unique: dict[str, Path] = {}
    for root in roots:
        unique.setdefault(os.path.normcase(os.path.abspath(root)), root)
    return list(unique.values())[:MAX_CCA_ROOTS]


def cca_holdout_run(cwd: str, *, deadline: float) -> str | None:
    """The id of the CCA run open for `cwd` when its `memory-arm.json` names the `holdout` arm,
    else None. Never raises.

    CCA withholds its prior-bugs brief from about one DIFF run in three to measure what memory
    adds; briefing that run's auditors here would hand them the records CCA withheld. A run is open
    at a candidate root (`_cca_roots`) when its state names a valid run id and that run's arm file
    names the same run id. When every open run is in the holdout arm, the answer is their ids;
    when open runs disagree, the arm is ambiguous, which is logged and answers None. Everything
    else (no open run, unreadable JSON, a link, any other arm) answers None, so the brief is
    delivered as before.
    """
    try:
        arms: dict[str, str] = {}
        for root in _cca_roots(cwd, deadline):
            if time.monotonic() > deadline:
                break
            state = _read_repo_json(root, CCA_STATE)
            run_id = state.get("run_id") if isinstance(state, dict) else None
            if not isinstance(run_id, str) or not _CCA_RUN_ID_RE.fullmatch(run_id) or run_id in (".", ".."):
                continue
            arm = _read_repo_json(root, (*CCA_RUNS, run_id, CCA_ARM_FILE))
            if isinstance(arm, dict) and arm.get("run_id") == run_id:
                arms[run_id] = str(arm.get("arm"))
        if not arms:
            return None
        if len(set(arms.values())) > 1:
            listed = ", ".join(f"{run_id} {arm}" for run_id, arm in sorted(arms.items()))
            _log(f"CCA open runs disagree on the memory arm ({listed}): ambiguous, so no holdout is applied")
            return None
        if next(iter(arms.values())) == CCA_HOLDOUT_ARM:
            return ", ".join(sorted(arms))
    except Exception:  # BROAD-CATCH: fail-open  (an unreadable run is not a holdout)
        return None
    return None


# --- the event ---------------------------------------------------------------------------------


def _eligible(payload: dict[str, Any]) -> tuple[dict[str, Any], str, str, str] | None:
    """``(tool_input, agent_type, prompt, cwd)`` when this launch is one to brief, else None.

    Not briefed: another tool, an agent type outside the auditor list, a prompt that is not text or
    already holds a CCA brief heading or this hook's marker. `cwd` may be empty; the caller logs it.
    """
    if payload.get("tool_name") != "Agent":
        return None
    tool_input = payload.get("tool_input")
    if not isinstance(tool_input, dict):
        return None
    agent_type = str(tool_input.get("subagent_type") or "")
    if _bare(agent_type) not in auditor_types():
        return None
    prompt = tool_input.get("prompt")
    if not isinstance(prompt, str) or MARKER in prompt or any(h in prompt for h in CCA_HEADINGS):
        return None
    return tool_input, agent_type, prompt, str(payload.get("cwd") or "")


def pre_agent(payload: dict[str, Any]) -> int:
    """Entry point for `python -m recall_hooks pre-agent`. Returns 0 always."""
    started = time.monotonic()
    agent_type = "?"
    try:
        eligible = _eligible(payload)
        if eligible is None:
            return 0
        tool_input, agent_type, prompt, cwd = eligible
        if not cwd:
            _log(f"{agent_type}: the event carries no cwd; prompt left unchanged")
            return 0
        budget = _budget()
        deadline = started + budget - INNER_MARGIN
        # Beside the block rather than before it, so its git call costs the launch no extra time.
        holdout: _Background[str | None] = _Background(
            lambda: cca_holdout_run(cwd, deadline=deadline), "recall-auditor-brief-holdout"
        )
        # The inner bounds fire first, so a slow stage is logged by name; the wait below is the
        # backstop for a stage nothing inside bounds.
        worker: _Background[str] = _Background(
            lambda: build_block(cwd, prompt, deadline=deadline), "recall-auditor-brief"
        )
        if not worker.wait(started + budget - time.monotonic()):
            _log(f"{agent_type}: budget of {budget:.2f}s exceeded; prompt left unchanged")
            return 0
        try:
            block = worker.value()
        except BaseException as error:  # BROAD-CATCH: fail-open  (logged; the prompt is left as it was)
            _log(f"{agent_type}: {type(error).__name__}: {error}; prompt left unchanged")
            return 0
        if not block:
            return 0
        if not holdout.wait(started + budget - time.monotonic()):
            _log(f"{agent_type}: the CCA holdout check did not finish within the budget; brief delivered")
        elif (run_id := holdout.value()) is not None:
            _log(f"{agent_type}: CCA run {run_id} is in the memory holdout arm; prompt left unchanged")
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
    warm leaves its prompt unchanged. Run it the way the hook runs (`warm_command()`): another
    interpreter or copy of `recall` writes a cache tagged for that copy, which the hook ignores.
    """
    from recall import records_files

    directory = memo_directory()
    cache = cache_path(directory)
    try:
        loaded = records_files.load_memo_records(directory, cache_file=cache)
    except OSError as exc:
        print(f"recall: cannot read the memo directory {directory}: {exc}", file=sys.stderr)
        return 1
    print(
        f"recall: cached {len(loaded)} record(s) from {directory} in {cache}; "
        f"{len(loaded.skipped)} memo(s) skipped"
    )
    return 0


__all__ = [
    "CCA_HEADINGS",
    "DEFAULT_AUDITOR_TYPES",
    "HEADING",
    "MARKER",
    "UNKNOWN_SHA",
    "auditor_types",
    "build_block",
    "cache_path",
    "cca_holdout_run",
    "changed_lines",
    "finish_scope",
    "log_path",
    "memo_directory",
    "named_files",
    "pre_agent",
    "render",
    "repo_identity",
    "scope",
    "start_scope",
    "warm",
    "warm_command",
]
