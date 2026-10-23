"""General knowledge: owner-curated entries injected before a command or after a failure.

**What this module is.** The model and the reader for a small prebuilt index, plus the two hook
entry points that use it. An entry is a piece of curated procedure or warning (how to run a test
suite on a given host, which flag a tool needs) that a session cannot derive from the repository.
The content is the operator's own and is never shipped in this repository; every fixture in the
tests is invented. With no index the feature does nothing.

**The index.** `<knowledge home>/index.json`, written atomically by the operator's own tooling and
only ever read here:

    {"schema": 1, "built_at": "<ISO UTC>",
     "entries": [{"id": ..., "kind": ..., "scope": ...,
                  "triggers": [{"command": ...} | {"command_regex": ...} | {"host": ...}],
                  "symptoms": [...], "body": "..."}]}

**Why a prebuilt index and not the entry files.** The matching runs inside a `PreToolUse` process
that is spawned once per tool call, so every cost is paid on every shell call. Parsing one JSON
file is cheap and bounded; walking a directory of entry files and parsing front matter is neither.
Nothing here touches the network, a database or an embedder, and nothing imports the `recall`
package (see `recall_hooks/__init__.py` for what that import costs).

**Where things live.**

| What | Where | Override |
|---|---|---|
| knowledge home | `<project config home>/general-knowledge` | `RECALL_GENERAL_KNOWLEDGE_HOME` |
| event log | `<project config home>/logs/general-knowledge.jsonl` | `..._GENERAL_KNOWLEDGE_LOG` |
| switch | the `general_knowledge` block of `<project config home>/recall-hook.json` | none |

The block is `{"enabled": bool, "min_score": float}`; the two variables are
`RECALL_GENERAL_KNOWLEDGE_HOME` and `RECALL_GENERAL_KNOWLEDGE_LOG`.

The project config home is `RECALL_PROJECT_CONFIG_HOME`, else `CLAUDE_CONFIG_DIR`, else
`~/.claude`, so all three rows move with those two variables as well as with their own.

A MISSING `recall-hook.json`, or one without the block, means enabled with `min_score` 0.0. Every
other departure from the documented shape means DISABLED: a file that cannot be read or parsed, a
document that is not an object, a block that is not an object, or an `enabled` that is not a real
bool. This is the injection switch, so a hand edit that meant "off" may not turn it on by being
broken. All three resolve from the project config home, never from `recall_hooks.load_config`,
because the Codex adapter redirects `load_config` to Codex's own directory and Claude and Codex
must share one corpus, one switch and one log.

**The two events.**

* `PreToolUse` (`pre_tool_use_context`), shell tools only (`Bash`, `PowerShell`). Trigger keyed: an
  entry fires when a program, a `command_regex` match or a host alias named by one of its
  triggers appears in the command about to run. No ranking. Logs one `trigger` event whenever any
  in-scope entry matched, with `ts`, `project`, `session`, `client`, `trigger` (the first trigger
  that fired per matched entry, as `<kind>:<value>`), `injected` and `suppressed` ids.
* `PostToolUseFailure` (`post_tool_use_failure`), Claude only, since Codex has no post-tool event.
  Error keyed: the entries are ranked against the error text by BM25 over their symptoms and
  bodies (`prompt_time`, lexical, imported only on this path). Logs one `failure` event for every
  failure, matched or not, with `ts`, `project`, `session`, `client`, `signature` (command family
  plus the normalised error line), `injected` and `suppressed`: the signatures nothing matched are
  what shows which entries are worth writing. A user interrupt is not a failure and is skipped.

At most `MAX_PER_CALL` (2) entries are injected per call, and each entry at most once per session.

Two tool calls of one session can run in parallel, and both append to the event log at once. On
Windows an append is not guaranteed to be atomic, so a line can rarely be lost or interleaved; the
log is a diagnostic aid, and that loss is accepted rather than paid for with a lock on every call.

**A slow `command_regex` is refused at load, as defence in depth.** A pattern runs against every
shell command in every project, so one that backtracks catastrophically stalls every tool call.
`_trigger` therefore drops (never raises on) a `command_regex` longer than `MAX_REGEX_CHARS` (200)
characters, or one with a quantified group whose body itself holds a quantifier, the nested shape
of `(a+)+`, `(a*)*` or `(?:x+){2,}`. The scan reads the pattern text only; it cannot see an
alternation whose branches overlap, such as `(a|aa)+`, which stays. The entry keeps its other
triggers. The operator's own tooling should also time every regex before it writes the index.

**The contract.** Never deny a tool call, never raise (every path returns 0), never load an
embedder, never touch the network. A missing, unreadable, corrupt or foreign-schema index is an
EMPTY index, and one malformed entry costs that entry only: a hook that raises is charged to the
client, and a hook may never be the reason a tool call does not run. Latency budget: 50 ms median
added to a `pre-tool-use` process, nothing on a success, at most 800 ms on a failure.

**Measured 2026-10-23** on a Xeon X5690, Python 3.14.6, with a 30 entry fixture index, whole
process medians (interpreter start included, which is most of the figure):

| Event | Batch | Enabled | Disabled | Added |
|---|---|---:|---:|---:|
| `pre-tool-use`, 20 runs per arm, interleaved ABAB | 1 | 333.9 ms | 324.2 ms | +9.7 ms |
| `pre-tool-use`, 20 runs per arm, interleaved ABAB | 2 | 317.5 ms | 319.1 ms | -1.6 ms |
| `post-tool-use-failure`, 10 runs | 1 | 300.8 ms | 287.0 ms | +13.8 ms |
| `post-tool-use-failure`, 10 runs | 2 | 300.8 ms | 283.0 ms | +17.8 ms |

The second pre-tool-use batch came out 1.6 ms faster with the feature on. That is noise, and it is
the reason both batches are printed: one batch is a sample, not a cost, so budget the range and
not either end of it. Re-measure by running the hook entry point as a subprocess against a 30
entry index with the feature on and off, alternating the arms, and compare the medians.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, NamedTuple

from . import HOOK_CONFIG_NAME, project_config_home

INDEX_NAME = "index.json"
INDEX_SCHEMA = 1
#: The most entries one tool call may inject. Each entry is paid for in the agent's context.
MAX_PER_CALL = 2
#: The most command text the matcher analyses, so its cost is bounded by this cap and not by the
#: input. A command can carry a heredoc body of hundreds of kilobytes, and parsing it costs time on
#: every `Bash` call (0.39 s for 100 KB of separators) for a program that is named in the first
#: lines. Program extraction and `command_regex` triggers both see only the first
#: `MAX_COMMAND_CHARS` characters.
MAX_COMMAND_CHARS = 16384
#: A per-session state file untouched for this long is deleted by the next claim (opportunistic).
STATE_MAX_AGE_SECONDS = 7 * 24 * 60 * 60
#: The longest session id used as a file name; a longer one is truncated and hash-suffixed.
_SESSION_ID_MAX = 100
_SESSION_UNSAFE = re.compile(r"[^A-Za-z0-9_-]")
#: Names Windows opens as devices whatever the extension, so they never name a state file as is.
_WINDOWS_DEVICE_NAMES = frozenset(
    {"con", "prn", "aux", "nul"}
    | {f"com{n}" for n in range(10)}
    | {f"lpt{n}" for n in range(10)}
)

#: The three ways an entry can be triggered: a trigger is a one-key object with one of these keys.
TRIGGER_KINDS = frozenset({"command", "command_regex", "host"})
#: The longest `command_regex` accepted at load; a longer one is dropped (see `_risky_regex`).
MAX_REGEX_CHARS = 200
#: A counted repeat that is a quantifier: `{m}`, `{m,}`, `{,n}` or `{m,n}`. Anything else after a
#: `{` is a literal brace to the engine, and so it is to the scan.
_COUNTED_REPEAT = re.compile(r"\{(\d*)(,?)(\d*)\}")

#: Compiled `command_regex` patterns, keyed by pattern text. Filled at load, so the matcher in the
#: hot path never compiles; a pattern that failed to compile is never stored, because its trigger
#: was dropped.
_COMPILED: dict[str, re.Pattern[str]] = {}


@dataclass(frozen=True)
class Trigger:
    """One condition under which an entry is relevant. `kind` is a member of `TRIGGER_KINDS`."""

    kind: str
    value: str


@dataclass(frozen=True)
class Entry:
    """One curated entry as read from the index. Immutable, so it can be shared across matches."""

    id: str
    kind: str
    scope: str
    triggers: tuple[Trigger, ...]
    symptoms: tuple[str, ...]
    body: str


def knowledge_home() -> Path:
    """Where the index lives.

    `RECALL_GENERAL_KNOWLEDGE_HOME`, else `<project config home>/general-knowledge`.
    """
    raw = os.environ.get("RECALL_GENERAL_KNOWLEDGE_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    return project_config_home() / "general-knowledge"


def log_path() -> Path:
    """The event log.

    `RECALL_GENERAL_KNOWLEDGE_LOG`, else `<project config home>/logs/general-knowledge.jsonl`.
    """
    raw = os.environ.get("RECALL_GENERAL_KNOWLEDGE_LOG", "").strip()
    if raw:
        return Path(raw).expanduser()
    return project_config_home() / "logs" / "general-knowledge.jsonl"


def settings(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """The `general_knowledge` block of the hook config, with defaults. Pure: `None` means `{}`.

    Absent means ENABLED, so installing the corpus is what turns the feature on and
    `enabled: false` is the one switch that turns it off. A `min_score` of the wrong type, or one
    too large for a float, falls back to its default rather than raising, for the same fail-open
    reason as the reader. A block that is PRESENT but not an object DISABLES (see below).

    **`enabled` fails CLOSED.** A PRESENT value that is not a real `bool` (a quoted `"false"`,
    `0`, `null`) DISABLES the feature: this is the injection switch, and a hand-edited config that
    meant "off" must never turn injection on because of its own typo. `bool("false")` is True, so
    coercing would do exactly that, and falling back to the default would too. Only a real `bool`
    sets it; only a finite int or float (never a bool, never NaN or infinity) sets `min_score`.
    """
    if not isinstance(config, dict):
        config = {}
    if "general_knowledge" not in config:
        return {"enabled": True, "min_score": 0.0}
    block = config["general_knowledge"]
    if not isinstance(block, dict):
        # A PRESENT block of the wrong shape (a string, a list, null) fails closed, for the same
        # reason `enabled` does: a hand edit that meant "off" must not leave injection on.
        return {"enabled": False, "min_score": 0.0}
    enabled = block.get("enabled", True)
    if not isinstance(enabled, bool):
        enabled = False
    min_score = block.get("min_score", 0.0)
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)):
        min_score = 0.0
    else:
        try:
            # `float(10**400)` and `math.isfinite(10**400)` both raise OverflowError: a huge int is
            # not a usable threshold, and this function may never raise.
            min_score = float(min_score)
        except OverflowError:
            min_score = 0.0
        if not math.isfinite(min_score):
            min_score = 0.0
    return {"enabled": enabled, "min_score": min_score}


def load_settings() -> dict[str, Any]:
    """`settings` over `<project config home>/recall-hook.json`. Never raises.

    Only a MISSING file is the defaults (enabled): that is a machine on which nobody wrote one.
    Any other failure DISABLES: a file that cannot be read (permissions, a directory in its place,
    bytes that are not UTF-8), one that does not parse, or a document that is not an object. The
    reason is the one `settings` gives for `enabled`: a hand edit that meant "off" and broke the
    file must not switch injection on. The cost is that a corrupt config silences general
    knowledge until it is fixed, which is the safe direction for an injection switch.

    Deliberately NOT `recall_hooks.load_config()`: see the module docstring. This reads the same
    file name from the project config home, which Claude and Codex share.
    """
    disabled = {"enabled": False, "min_score": 0.0}
    try:
        raw = (project_config_home() / HOOK_CONFIG_NAME).read_text(encoding="utf-8")
    except FileNotFoundError:
        return settings({})
    except (OSError, ValueError):
        return disabled
    try:
        document = json.loads(raw)
    except (ValueError, RecursionError):
        return disabled
    if not isinstance(document, dict):
        return disabled
    return settings(document)


def compiled(pattern: str) -> re.Pattern[str] | None:
    """The compiled form of a `command_regex` value, from the load-time cache; `None` if unknown."""
    return _COMPILED.get(pattern)


def _counted(pattern: str, index: int, *, need_comma: bool) -> bool:
    """True when a counted repeat starts at `index` (with a comma, when `need_comma`)."""
    found = _COUNTED_REPEAT.match(pattern, index)
    if found is None or not (found.group(1) or found.group(3)):
        return False
    return bool(found.group(2)) or not need_comma


def _risky_regex(pattern: str) -> bool:
    """True when `pattern` is too long, or nests a quantifier inside a quantified group.

    A conservative scan of the pattern TEXT, not a parse. It tracks group depth and skips escaped
    characters and character classes, so neither an escaped parenthesis nor a `(` inside `[...]`
    opens a group. It reports a group whose body holds a `+`, `*`, `?`, or a counted repeat with a
    comma (`{m,}`, `{m,n}`), when the group itself is followed by `+`, `*` or any counted repeat.
    That is the shape of `(a+)+` and `(?:x+){2,}`, which backtrack exponentially on a near miss. A
    quantifier nested deeper counts for every enclosing group. It does NOT see overlapping
    alternation (`(a|aa)+`), which needs more than a text scan. Never raises: a pattern the scan
    cannot read is treated as risky, which costs that one trigger only.
    """
    try:
        if len(pattern) > MAX_REGEX_CHARS:
            return True
        # One flag per open group, the first for the whole pattern: does its body hold a quantifier?
        stack: list[bool] = [False]
        i, n = 0, len(pattern)
        while i < n:
            char = pattern[i]
            if char == "\\":
                i += 2
                continue
            if char == "[":
                i += 1
                if i < n and pattern[i] == "^":
                    i += 1
                if i < n and pattern[i] == "]":
                    i += 1  # a `]` first in a class is a literal
                while i < n and pattern[i] != "]":
                    i += 2 if pattern[i] == "\\" else 1
                i += 1
                continue
            if char == "(":
                stack.append(False)
                i += 1
                if i < n and pattern[i] == "?":
                    # A group prefix (`(?:`, `(?P<name>`, `(?<=`, `(?i)`): no quantifier here.
                    i += 1
                    if pattern.startswith("P<", i):
                        end = pattern.find(">", i)
                        i = n if end < 0 else end + 1
                    elif pattern.startswith("#", i):
                        end = pattern.find(")", i)
                        i = n if end < 0 else end
                    else:
                        while i < n and pattern[i] in "aiLmsux-:=!<>P":
                            i += 1
                continue
            if char == ")":
                inner = stack.pop() if len(stack) > 1 else False
                i += 1
                follower = pattern[i] if i < n else ""
                counted = follower == "{" and _counted(pattern, i, need_comma=False)
                if inner and (follower in ("+", "*") or counted):
                    return True
                if inner:
                    stack[-1] = True
                continue
            if char in "+*?" or (char == "{" and _counted(pattern, i, need_comma=True)):
                stack[-1] = True
            i += 1
        return False
    except Exception:  # BROAD-CATCH: fail-open for the hook, closed for this one trigger
        return True


def _trigger(raw: Any) -> Trigger | None:
    """A `Trigger` from `{"<kind>": "<value>"}`, or `None` for anything else.

    Dropped, not raised: a map with more than one key, an unknown kind, a non-string or empty value,
    a regex that does not compile and a regex `_risky_regex` refuses are all the same case, one bad
    trigger that must not cost the entry its other triggers.
    """
    if not isinstance(raw, dict) or len(raw) != 1:
        return None
    ((kind, value),) = raw.items()
    if kind not in TRIGGER_KINDS or not isinstance(value, str) or not value:
        return None
    if kind == "command_regex" and value not in _COMPILED:
        if _risky_regex(value):
            # Too slow to run on every shell call. Never cached, so it is refused on every load.
            return None
        try:
            _COMPILED[value] = re.compile(value)
        except (re.error, OverflowError, RecursionError):
            # Not only `re.error`: a repeat count past the engine's limit raises OverflowError and
            # a deeply nested group raises RecursionError. Left to escape they would reach
            # `load_index`'s outer catch and empty the whole index for one bad trigger.
            return None
    return Trigger(kind, value)


def _entry(raw: Any) -> Entry | None:
    """An `Entry` from one index record, or `None` when the record is malformed."""
    if not isinstance(raw, dict):
        return None
    ident, body = raw.get("id"), raw.get("body")
    if not isinstance(ident, str) or not ident or not isinstance(body, str) or not body:
        return None
    raw_triggers, raw_symptoms = raw.get("triggers", []), raw.get("symptoms", [])
    if not isinstance(raw_triggers, list) or not isinstance(raw_symptoms, list):
        return None
    kind, scope = raw.get("kind", "procedure"), raw.get("scope", "global")
    if not isinstance(kind, str) or not isinstance(scope, str):
        return None
    triggers = tuple(t for t in map(_trigger, raw_triggers) if t is not None)
    symptoms = tuple(s for s in raw_symptoms if isinstance(s, str) and s.strip())
    return Entry(ident, kind, scope, triggers, symptoms, body)


def load_index(home: Path | None = None) -> list[Entry]:
    """Read `index.json` under `home` (default `knowledge_home()`). Never raises.

    Missing, unreadable, corrupt, not an object, or a schema other than `INDEX_SCHEMA` all give
    `[]`: a schema this reader does not know is not guessed at. A malformed entry is skipped and the
    rest are kept.
    """
    path = (knowledge_home() if home is None else home) / INDEX_NAME
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(document, dict) or document.get("schema") != INDEX_SCHEMA:
            return []
        records = document.get("entries")
        if not isinstance(records, list):
            return []
        return [e for e in map(_entry, records) if e is not None]
    except (OSError, ValueError):
        return []
    except Exception:  # BROAD-CATCH: fail-open
        # A pathologically nested document raises RecursionError, which is neither of the above.
        # A hook that raises is charged to the client, so an unreadable index is an empty one.
        return []


# ---------------------------------------------------------------------------------------------
# Project detection and trigger matching. Everything below returns a value for ANY string input:
# it runs inside the per-tool-call hook process, where an exception is charged to the client.
# ---------------------------------------------------------------------------------------------

#: Words that precede the real program and are skipped (case-insensitive). Their flags are NOT
#: understood: `sudo -u root pytest` yields the program `-u`. That is deliberate, a rule that
#: guessed which flags take an argument would be wrong in the other direction.
_WRAPPERS = frozenset({"sudo", "env", "time", "nohup"})
#: A leading `NAME=value` token (also `$env:NAME=value`): never a program, so it is skipped.
_ASSIGNMENT = re.compile(r"^[^=/\\]+=")
_PATH_SEPARATOR = re.compile(r"[\\/]")
_EXE_SUFFIX = re.compile(r"\.exe$", re.IGNORECASE)
#: Programs whose arguments name a remote host.
_HOST_PROGRAMS = frozenset({"ssh", "scp", "sftp", "rsync"})

#: One pass over a command. Alternation order matters: `&&` and `||` before the single `&` and `|`,
#: so they are one separator each. A word is a run of atoms: a double or single quoted string (an
#: unterminated quote takes the rest of the command, it never raises), a redirection such as `2>&1`
#: or `&>` (whose `&` is NOT a separator), or plain characters. A backslash is an ordinary
#: character, because on Windows it separates path components. Anything else (an exotic space) is
#: skipped by `finditer`, which acts as a token boundary.
_SCAN = re.compile(
    r"""(?P<sep>&&|\|\||;|\||\n|\r|&(?!>))
    |(?P<ws>[ \t\f\v]+)
    |(?P<word>(?:"[^"]*"?|'[^']*'?|[<>]+&?|&(?=>)|[^\s"'|;&<>]+)+)
    """,
    re.VERBOSE,
)
#: The pieces of one word: a quoted string (quotes dropped) or an unquoted run.
_ATOM = re.compile(r"\"([^\"]*)\"?|'([^']*)'?|([^\"']+)")


class Segment(NamedTuple):
    """One simple command of a command line: its program (normalised) and the tokens after it."""

    program: str
    args: tuple[str, ...]


def _unquote(word: str) -> str:
    return "".join(m.group(m.lastindex or 3) or "" for m in _ATOM.finditer(word))


def _program(token: str) -> str:
    """The basename of `token`, split on both `/` and `\\`, without `.exe`, lowercased."""
    return _EXE_SUFFIX.sub("", _PATH_SEPARATOR.split(token)[-1]).strip().lower()


def _segment(tokens: list[str]) -> Segment | None:
    for index, token in enumerate(tokens):
        if token.lower() in _WRAPPERS or _ASSIGNMENT.match(token):
            continue
        program = _program(token)
        return Segment(program, tuple(tokens[index + 1 :])) if program else None
    return None


def parse_command(command: str) -> list[Segment]:
    """The simple commands of `command`, split on `&&`, `||`, `;`, `|`, a lone `&` and newlines.

    Quote aware: a quoted token holding spaces is one token and its quotes are removed, and a
    separator inside quotes does not split. Not a shell parser (no escapes, no substitutions, no
    here documents), and it does not need to be: a trigger names a program, and a wrong guess only
    costs one injection that did not happen. A segment with no program (`& ` alone, a bare
    assignment) is dropped. Never raises.
    """
    segments: list[Segment] = []
    tokens: list[str] = []
    text = command[:MAX_COMMAND_CHARS] if isinstance(command, str) else ""
    for match in _SCAN.finditer(text):
        word = match.group("word")
        if word is not None:
            tokens.append(_unquote(word))
        elif match.group("sep") is not None:
            segment = _segment(tokens)
            if segment is not None:
                segments.append(segment)
            tokens = []
    segment = _segment(tokens)
    if segment is not None:
        segments.append(segment)
    return segments


def command_programs(command: str) -> list[str]:
    """The program of each simple command in `command`, in order (duplicates kept)."""
    return [segment.program for segment in parse_command(command)]


def repository_name(cwd: str) -> str | None:
    """The name of the repository `cwd` is inside, or `None`. Never raises.

    Walks up from `cwd` to the FIRST `.git` and stops there. A `.git` directory gives its parent's
    name. A `.git` FILE is read: `gitdir: <main>/.git/worktrees/<name>` (a linked worktree) gives
    the name of `<main>`, so every worktree of one repository is one project; any other well-formed
    `gitdir:` (a submodule, a separate git dir) gives the checkout's own name. A `.git` file that is
    unreadable or malformed gives `None` rather than a guess from a higher directory. Paths are
    taken as written, never resolved, so a network path costs no more than the stats it needs.
    """
    if not isinstance(cwd, str) or not cwd.strip():
        return None
    try:
        start = Path(os.path.abspath(cwd))  # absolute so '.' has parents; symlinks are not resolved
        if not start.is_dir():
            return None
        for directory in (start, *start.parents):
            dot_git = directory / ".git"
            if dot_git.is_dir():
                return directory.name or None
            if dot_git.is_file():
                return _name_from_gitdir_file(directory, dot_git)
    except (OSError, ValueError):
        return None
    except Exception:  # BROAD-CATCH: fail-open
        return None
    return None


def _name_from_gitdir_file(directory: Path, dot_git: Path) -> str | None:
    """The repository a `.git` file points at; the caller catches unreadable and undecodable."""
    with dot_git.open("rb") as handle:
        first = handle.read(4096).decode("utf-8").splitlines()[:1]
    if not first or not first[0].startswith("gitdir:"):
        return None
    target = first[0][len("gitdir:") :].strip()
    if not target:
        return None
    if not os.path.isabs(target):
        target = os.path.normpath(os.path.join(directory, target))
    parts = [part for part in _PATH_SEPARATOR.split(target) if part]
    for i in range(len(parts) - 1, 0, -1):
        if parts[i] == "worktrees" and parts[i - 1] == ".git":
            return parts[i - 2] if i >= 2 else None
    return directory.name or None


def in_scope(entry: Entry, project: str | None) -> bool:
    """`global` is always in scope; `project:<name>` only when `name` is the current project."""
    if entry.scope == "global":
        return True
    prefix = "project:"
    if project is None or not entry.scope.startswith(prefix):
        return False
    return entry.scope[len(prefix) :] == project


def _names_host(alias: str, args: tuple[str, ...]) -> bool:
    """True when an argument is `alias`, `alias:<path>`, or either of those after `user@`.

    `user@` is split off only from the part BEFORE the first colon, so an `@` in the path part
    (`examplehost:/var/x@1`) neither hides a host nor invents one (`other:/x@examplehost`).
    """
    wanted = alias.casefold()
    return any(arg.partition(":")[0].rpartition("@")[2].casefold() == wanted for arg in args)


def matched_trigger(
    entry: Entry, command: str, segments: list[Segment], project: str | None
) -> Trigger | None:
    """The FIRST trigger of `entry` that matches, or `None` (also `None` when out of scope).

    `segments` is `parse_command(command)`, passed in so a caller matching many entries parses once.
    The event log reports which trigger fired, and takes it from here instead of re-deriving the
    rules. Never raises.
    """
    if not in_scope(entry, project):
        return None
    for trigger in entry.triggers:
        if trigger.kind == "command":
            wanted = trigger.value.lower()
            if any(segment.program == wanted for segment in segments):
                return trigger
        elif trigger.kind == "host":
            if any(
                s.program in _HOST_PROGRAMS and _names_host(trigger.value, s.args)
                for s in segments
            ):
                return trigger
        elif trigger.kind == "command_regex":
            pattern = compiled(trigger.value)
            if pattern is None:
                continue
            try:
                found = pattern.search(command[:MAX_COMMAND_CHARS])
            except Exception:  # BROAD-CATCH: fail-open
                continue
            if found:
                return trigger
    return None


def match_triggers(entries: list[Entry], command: str, project: str | None) -> list[Entry]:
    """The entries whose scope and triggers fit `command`, in input order, each id once."""
    if not isinstance(command, str) or not command.strip():
        return []
    command = command[:MAX_COMMAND_CHARS]
    segments = parse_command(command)
    matched: list[Entry] = []
    seen: set[str] = set()
    for entry in entries:
        if entry.id in seen:
            continue
        if matched_trigger(entry, command, segments, project) is not None:
            seen.add(entry.id)
            matched.append(entry)
    return matched


def client() -> str:
    """Which client wrote an event: `RECALL_HOOK_CLIENT` (the Codex adapter sets it) or `claude`."""
    return os.environ.get("RECALL_HOOK_CLIENT", "claude")


def _session_key(session_id: Any) -> str:
    """The file stem for a session id, or "" when the payload names no usable session.

    The id comes from the client's payload, so it is untrusted input to a path: only
    `[A-Za-z0-9_-]` survives. A changed id gets a short hash of the ORIGINAL appended, so that
    `a/b` and `ab` stay two sessions instead of silently sharing one state file (and one
    session's first injection being suppressed by the other's). An id with nothing left after
    sanitising is no session at all.

    The hash is also appended when the id holds an upper-case letter, because on a
    case-insensitive filesystem (the Windows and macOS defaults) `Abc` and `abc` would otherwise
    open one file, and when it is a Windows device name (`CON`, `nul`, `COM1`), which Windows
    opens as a device rather than as `CON.json`. Client ids are lower-case UUIDs, which keep their
    plain name.
    """
    if not isinstance(session_id, str):
        return ""
    cleaned = _SESSION_UNSAFE.sub("", session_id)
    if not cleaned:
        return ""
    plain = (
        cleaned == session_id
        and len(cleaned) <= _SESSION_ID_MAX
        and cleaned == cleaned.lower()
        and cleaned.lower() not in _WINDOWS_DEVICE_NAMES
    )
    if plain:
        return cleaned
    digest = hashlib.sha256(session_id.encode("utf-8", "replace")).hexdigest()[:10]
    return f"{cleaned[: _SESSION_ID_MAX - 11]}-{digest}"


def _read_claimed(path: Path) -> list[str]:
    """The ids already injected in this session; unreadable, corrupt or odd-shaped means none."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, RecursionError):
        # RecursionError: deeply nested JSON. Counting it as "nothing claimed" means the next
        # claim rewrites the file, which repairs it; letting it reach the broad catch in
        # `claim_for_session` would leave the file broken and re-inject on every call.
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, str)]


def _write_claimed(path: Path, claimed: list[str]) -> None:
    """Replace the state file atomically (same-directory temp file, then `os.replace`).

    Two tool calls of one session can run in parallel, so a plain `write_text` could be read
    half-written by the other. Same sequence as `_save_config` in this package, for the same
    reason. Any failure leaves the old file and cleans the temp: the cost of a lost write is one
    repeated injection, never an error.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    tmp = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(claimed))
        os.replace(tmp, path)
    except BaseException:
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass
        raise


def _prune_state(state_dir: Path) -> None:
    """Delete state files in `state_dir` not modified for 7 days. One listing, never raises.

    Opportunistic: it rides on a claim instead of running on a schedule, and it touches nothing but
    regular `*.json` files, and the `*.tmp` files a killed atomic write can leave behind, directly
    inside the state directory.
    """
    try:
        cutoff = time.time() - STATE_MAX_AGE_SECONDS
        with os.scandir(state_dir) as listing:
            for item in listing:
                try:
                    if (
                        item.name.endswith((".json", ".tmp"))
                        and item.is_file(follow_symlinks=False)
                        and item.stat(follow_symlinks=False).st_mtime < cutoff
                    ):
                        os.unlink(item.path)
                except OSError:
                    continue
    except OSError:
        return


def claim_for_session(session_id: str, entry_ids: list[str], home: Path | None = None) -> list[str]:
    """The ids of `entry_ids` not yet injected in this session, at most `MAX_PER_CALL`; recorded.

    The caller's order is kept and duplicates are ignored. With no usable session id nothing can be
    remembered, so the first `MAX_PER_CALL` ids are returned and NOTHING is written: the cap still
    holds, only the once-per-session promise does not. State lives in `<home>/state/<id>.json`, one
    file per session, so concurrent sessions never see each other's claims.

    **Never raises, fails open on delivery.** An unreadable, corrupt or non-list state file counts
    as nothing claimed yet, and an unwritable state directory still returns the claimable ids; the
    worst case of either is a repeated injection, whereas a hook that raises is charged to the
    client.
    """
    try:
        unique = list(dict.fromkeys(i for i in entry_ids if isinstance(i, str)))
    except TypeError:
        return []
    key = _session_key(session_id)
    if not key:
        return unique[:MAX_PER_CALL]
    try:
        state_dir = (home if home is not None else knowledge_home()) / "state"
        path = state_dir / f"{key}.json"
        claimed = _read_claimed(path)
        taken = set(claimed)
        fresh = [i for i in unique if i not in taken][:MAX_PER_CALL]
        if fresh:
            try:
                _write_claimed(path, claimed + fresh)
            except Exception:  # BROAD-CATCH: fail-open
                pass  # unwritable state: deliver anyway, a repeat injection is the worst case
        _prune_state(state_dir)
        return fresh
    except Exception:  # BROAD-CATCH: fail-open
        return unique[:MAX_PER_CALL]


def log_event(record: dict[str, Any], path: Path | None = None) -> None:
    """Append `record` plus a UTC `ts` to the event log as one JSON line. Never raises.

    The log is a diagnostic aid; losing a line must never cost a tool call. An unwritable path or a
    record that cannot be serialised (`OSError`, `TypeError`, `ValueError`) is swallowed.
    """
    try:
        stamped = {"ts": datetime.now(timezone.utc).isoformat(timespec="seconds")}
        stamped.update((k, v) for k, v in record.items() if k != "ts")
        line = json.dumps(stamped)
        target = path if path is not None else log_path()
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "a", encoding="utf-8", newline="\n") as handle:
            handle.write(line + "\n")
    except (OSError, TypeError, ValueError, AttributeError):
        return


# ---------------------------------------------------------------------------------------------
# Failure side: a signature that groups "the same failure", and a lexical ranking of entries
# against the error text. Like the matcher above, everything here returns a value for ANY input.
# ---------------------------------------------------------------------------------------------

#: The most error text the failure path looks at, for both the signature and the ranking. A failed
#: command can print megabytes; the first few kilobytes hold the failure, and the 800 ms budget of
#: the failure path (spec 7.3) is only met if the rest is never touched.
MAX_ERROR_CHARS = 4096
#: The longest normalised detail line in a signature.
SIGNATURE_CHARS = 160
#: The longest command family in a signature. A family is a program name or `ssh:<alias>`, so a
#: longer one is a pasted blob rather than a program, and it must not make the log line unbounded.
FAMILY_CHARS = 64
#: How many times an entry's symptoms are repeated in its ranking document, so that a symptom
#: phrase outweighs the same words appearing once in the body.
SYMPTOM_WEIGHT = 3

#: Claude Code prefixes a failed Bash result with `Exit code N` on a line of its own.
_EXIT_HEADER = re.compile(r"^exit code (-?\d+)\s*$", re.IGNORECASE)
#: Terminal escape sequences, which PowerShell puts in its error text (observed live on
#: 2026-10-23: the error for an unknown command arrived wrapped in SGR colour codes). A full
#: sequence is ESC then either a CSI (`[`, parameters, a final byte), an OSC (`]` up to BEL or
#: ESC backslash) or one more character. The ESC itself was already gone in the live payload,
#: leaving `[31;1m`, so a bare SGR (`[` digits and semicolons `m`, at least one digit) goes too;
#: requiring a digit keeps text such as `array[m]` intact.
_ANSI_ESCAPE = re.compile(
    r"\x1b(?:\[[0-?]*[ -/]*[@-~]|\][^\x07\x1b]*(?:\x07|\x1b\\)?|[@-Z\\-_])"
    r"|\[\d{1,3}(?:;\d{1,3})*m"
)


def _without_ansi(text: str) -> str:
    """`text` with terminal escape sequences removed, so colour never changes a signature."""
    return _ANSI_ESCAPE.sub("", text)
#: Characters inside a path: not whitespace, quotes, shell metacharacters or `:`, so that
#: `a.py:42:` leaves its line number to the digit rule instead of being swallowed.
_PATH_BODY = r"""[^\s"'<>|*?:]"""
#: A path (other than a Windows one) must START a token: line start, or whitespace, a quote, an
#: opening bracket, `=` or `,` before it. That keeps `1/2` and `and/or` intact.
_PATH_START = r"""(?<![^\s"'(=\[,])"""
#: A Windows path (`c:\a\b` or `c:/a/b`; the text is lowercased first) and a UNC path
#: (`\\host\share`).
_WINDOWS_PATH = re.compile(r"(?<![a-z0-9])[a-z]:[\\/]" + _PATH_BODY + "*")
_UNC_PATH = re.compile(r"\\\\" + _PATH_BODY + "+")
#: A POSIX absolute path (a character must follow the slash, so a bare `/` stays) or `~/x`.
_POSIX_PATH = re.compile(_PATH_START + "(?:/" + _PATH_BODY + r"+|~[\\/]" + _PATH_BODY + "*)")
#: Hex runs of 7 or more with at least one digit AND at least one letter a to f, word bounded and
#: applied after the path rules and before the digit rule: a commit hash or object id has both.
#: An English word made of the letters a to f (`defaced`, `effaced`) has no digit, and a plain
#: number (`1234567`) has no letter, so it stays for `<n>`.
_HEX_RUN = re.compile(r"\b(?=[0-9a-f]*\d)(?=[0-9a-f]*[a-f])[0-9a-f]{7,}\b")
_DIGITS = re.compile(r"\d+")
_WHITESPACE = re.compile(r"\s+")

#: ssh options that take a value as the NEXT argument (`-p 22`). `-O` is added to the usual list
#: because `ssh -O exit host` would otherwise name the host `exit`.
_SSH_VALUE_OPTIONS = "pilOoFJLRDbceSmWwEBIQ"
#: `-p`, or a cluster of valueless flags ending in a value option (`-vp`), with the value next.
#: `-p22` and `-oBatchMode=yes` carry their value attached, so they match nothing and consume none.
_SSH_OPTION_WITH_VALUE = re.compile(r"^-[46AaCfGgKkMNnqsTtVvXxY]*[" + _SSH_VALUE_OPTIONS + r"]$")


def _ssh_host(args: tuple[str, ...]) -> str:
    """The destination of an `ssh` invocation: its first non-option argument, as an alias, or ""."""
    skip = False
    options_ended = False
    for arg in args:
        if skip:
            skip = False
            continue
        if not options_ended and arg == "--":
            options_ended = True
            continue
        if not options_ended and arg.startswith("-") and len(arg) > 1:
            skip = _SSH_OPTION_WITH_VALUE.match(arg) is not None
            continue
        host = arg
        if host.startswith("ssh://"):
            host = host[len("ssh://") :].partition("/")[0]
        host = host.rpartition("@")[2]
        if host.startswith("["):
            # A bracketed IPv6 destination: the address is everything up to the matching `]`, and
            # it holds colons, so the port split below would cut it to `[`.
            return host[1:].partition("]")[0].lower()
        return host.partition(":")[0].lower()
    return ""


#: Programs that only set up the shell for the command after them. An agent's command line
#: mostly starts with one (`cd repo && pytest`): measured over 663 failed shell calls in 8 weeks of
#: transcripts (2026-08-28 to 2026-10-23), 256 began with `cd`, so the first program named the
#: setup step instead of the thing that failed.
_PREAMBLE_PROGRAMS = frozenset(
    {"cd", "pushd", "export", "set", "source", ".", "set-location", "push-location"}
)
#: `timeout` options whose value is the next argument (`timeout -k 5 60 make`).
_TIMEOUT_VALUE_OPTIONS = frozenset({"-k", "-s", "--kill-after", "--signal"})
#: An ssh destination that is a configured alias (letters, digits, `-`). Anything else, an IPv4 or
#: IPv6 address or a dotted host name, identifies a machine, and the signature is written to a
#: log and into reports, so it is shown as `ssh:<host>`.
_BARE_ALIAS = re.compile(r"^[a-z0-9-]+$")
#: Interpreters whose `-m <module>` names the real program (`python -m ruff` is ruff).
_MODULE_RUNNERS = frozenset({"python", "python3", "py", "pythonw"})
#: The interpreter's valueless one-letter flags, which may be clustered (`-Bm`, `-Iu`).
_PYTHON_FLAGS = "bBdEiIOPqRsSuvx"
#: `-m`, alone or ending a cluster of valueless flags (`-Im`); the module is the next argument.
_PYTHON_MODULE_OPTION = re.compile(r"^-[" + _PYTHON_FLAGS + r"]*m$")
#: Valueless flags only (`-u`, `-OO`, `-Bq`).
_PYTHON_FLAG_CLUSTER = re.compile(r"^-[" + _PYTHON_FLAGS + r"]+$")
#: `-X` or `-W` with its value next (`-X utf8`), after any valueless flags.
_PYTHON_OPTION_WITH_VALUE = re.compile(r"^-[" + _PYTHON_FLAGS + r"]*[XW]$")
#: `-X` or `-W` with its value attached (`-Xutf8`, `-Wignore`).
_PYTHON_OPTION_ATTACHED = re.compile(r"^-[" + _PYTHON_FLAGS + r"]*[XW].")
#: The `py` launcher's own version selector (`py -3.12 -m ruff`, `py -V:3.12 -m ruff`).
_PY_LAUNCHER_VERSION = re.compile(r"^-(?:\d[\w.\-]*|V:\S+)$")
#: What a module name can be. Anything else is not a module, and since the family is logged it is
#: never taken for one: `python -m <value>` with an odd `<value>` keeps the interpreter's name.
_MODULE_NAME = re.compile(r"^[A-Za-z_][\w.]*$", re.ASCII)


def _python_module(program: str, args: tuple[str, ...]) -> str:
    """The module of `<interpreter> [options] -m <module>`, or "" when it does not run one.

    Only the interpreter's own options are read: the scan stops at the first positional argument
    (a script, whose arguments are its own, so `python tool.py -m <value>` runs no module), at
    `-c` (whose code is the next argument and every one after it is the code's), at `-` and `--`,
    and at any option it does not know, since stopping keeps the interpreter's name and that is
    the safe direction for a value that is logged.
    """
    index = 0
    while index < len(args):
        arg = args[index]
        if _PYTHON_MODULE_OPTION.match(arg):
            module = args[index + 1] if index + 1 < len(args) else ""
            return module.lower() if _MODULE_NAME.match(module) else ""
        if _PYTHON_OPTION_WITH_VALUE.match(arg) or arg == "--check-hash-based-pycs":
            index += 2
            continue
        if (
            _PYTHON_FLAG_CLUSTER.match(arg)
            or _PYTHON_OPTION_ATTACHED.match(arg)
            or (program == "py" and index == 0 and _PY_LAUNCHER_VERSION.match(arg))
        ):
            index += 1
            continue
        return ""
    return ""


def command_family(command: str) -> str:
    """The program a command runs, or `ssh:<alias>` for ssh; "" for none. Never raises.

    The first segment whose program is not a shell preamble (`cd`, `pushd`, `export`, `set`,
    `source`, `.`, and PowerShell's `Set-Location`, `Push-Location`); assignments are already
    dropped by `parse_command`. When every segment is preamble, the first one. Two wrappers are
    looked through: `timeout [options] <duration> <program>` names `<program>`, and
    `python -m <module>` (also `python3`, `py`) names `<module>`, so a failing `ruff` and a failing
    `pytest` are not both `python`. That `-m` must be the interpreter's own option and `<module>` a
    module name (`_python_module`): `python tool.py -m <value>` is `python`, because the family is
    written to a log and `<value>` is the script's argument, not a program.

    It reuses `parse_command` (and so its length cap) instead of finding the program a second way:
    the trigger matcher and the failure signature must agree on what a command is. For `ssh` the
    alias is part of the family because `ssh hosta` and `ssh hostb` fail for unrelated reasons.
    Known limit: in a chain (`python a.py && python -m ruff check`) the failing segment cannot be
    told from the command text, so the family is the first program even when a later one failed.
    """
    segments = parse_command(command)
    if not segments:
        return ""
    chosen = next((s for s in segments if s.program not in _PREAMBLE_PROGRAMS), segments[0])
    program, args = chosen.program, chosen.args
    if program == "timeout":
        positional: list[int] = []
        index = 0
        while index < len(args):
            if args[index] in _TIMEOUT_VALUE_OPTIONS:
                index += 2
                continue
            if not args[index].startswith("-"):
                positional.append(index)
            index += 1
        if len(positional) >= 2:
            program, args = _program(args[positional[1]]), args[positional[1] + 1 :]
    if program in _MODULE_RUNNERS:
        module = _python_module(program, args)
        if module:
            return module
    if program == "ssh":
        host = _ssh_host(args)
        if not host:
            return "ssh"
        return f"ssh:{host}" if _BARE_ALIAS.match(host) else "ssh:<host>"
    return program


#: Identifying values in an error line, replaced before anything else because a signature is
#: written to `general-knowledge.jsonl` and into reports. An SSH key fingerprint (`SHA256:...`, or
#: colon-separated hex pairs).
_FINGERPRINT = re.compile(
    r"\bsha(?:1|256|384|512):[a-z0-9+/=]{16,}|(?<![0-9a-f:])(?:[0-9a-f]{2}:){7,}[0-9a-f]{2}(?![0-9a-f:])"
)
#: Mailboxes and `name=value` credentials are found by scanning, not by one regex. A regex that
#: has to find where a name or a local part STARTS either backtracks (a credential word between two
#: unbounded name runs took 12.3 s on one 4 KB line) or, bounded and anchored, misses a name longer
#: than its bound and lets a non-credential pair swallow a credential nested in its value
#: (`url=...?token=x`). The scans anchor on the `=` or the `@`, walk left over name characters, and
#: are linear: every walk stops at the end of the previous one.
#:
#: The domain of a mailbox, from its `@`.
_EMAIL_DOMAIN = re.compile(r"@[a-z0-9-]+(?:\.[a-z0-9-]+)+")
_EMAIL_LOCAL_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789._%+-")
#: The characters of a `name` in `name=value`, a credential word in it, and the value after `=`.
_PAIR_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789_.-")
_CREDENTIAL_NAME = re.compile(r"token|secret|password|passwd|key|auth")
_PAIR_VALUE = re.compile(r"\"[^\"]*\"|'[^']*'|[^\s,;&]+")
#: IPv4, and IPv6 candidates (checked by `_ipv6_or_text`, so a time such as `12:30:45` stays).
_IPV4 = re.compile(r"(?<![\d.])(?:\d{1,3}\.){3}\d{1,3}(?![\d]|\.\d)")
_IPV6 = re.compile(r"(?<![\w:])[0-9a-f]{0,4}(?::[0-9a-f]{0,4}){2,7}(?![\w:])")
#: A long credential-like token: 20 or more base64 or URL-safe characters holding both a digit and
#: a letter. Applied after the hex rule, so a pure hex id stays `<hex>`.
_TOKEN = re.compile(
    r"(?<![\w<+/=-])(?=[a-z0-9+/_=-]*\d)(?=[a-z0-9+/_=-]*[a-z])[a-z0-9+/_-]{20,}={0,2}(?![\w+/=-])"
)


def _ipv6_or_text(match: re.Match[str]) -> str:
    """`<ip>` for an IPv6 address (it holds a digit and either `::` or eight groups), else as is."""
    value = match.group(0)
    if any(c.isdigit() for c in value) and ("::" in value or value.count(":") == 7):
        return "<ip>"
    return value


def _redact_emails(line: str) -> str:
    """`line` with every mailbox (a local part of any length, `@`, a dotted domain) as `<email>`."""
    out: list[str] = []
    kept = 0
    for domain in _EMAIL_DOMAIN.finditer(line):
        start = domain.start()
        if start < kept:
            continue
        while start > kept and line[start - 1] in _EMAIL_LOCAL_CHARS:
            start -= 1
        if start == domain.start():
            continue  # `@name.name` with no local part (a decorator, a handle): not a mailbox
        out.append(line[kept:start])
        out.append("<email>")
        kept = domain.end()
    out.append(line[kept:])
    return "".join(out)


def _redact_credential_pairs(line: str) -> str:
    """`line` with the value of every `name=value` whose name holds a credential word redacted.

    The name may be any length, and a pair that is NOT a credential keeps its value AND has it
    scanned, so `url=https://x/?token=abc` redacts the nested `token`.
    """
    out: list[str] = []
    kept = 0
    position = 0
    while True:
        equals = line.find("=", position)
        if equals < 0:
            break
        start = equals
        while start > kept and line[start - 1] in _PAIR_NAME_CHARS:
            start -= 1
        value = _PAIR_VALUE.match(line, equals + 1)
        if value is not None and _CREDENTIAL_NAME.search(line, start, equals):
            out.append(line[kept : equals + 1])
            out.append("<redacted>")
            kept = position = value.end()
            continue
        position = equals + 1
    out.append(line[kept:])
    return "".join(out)


def _redact(line: str) -> str:
    """`line` (lower case) with fingerprints, mailboxes, credential values and addresses replaced."""
    line = _FINGERPRINT.sub("<fingerprint>", line)
    line = _redact_emails(line)
    line = _redact_credential_pairs(line)
    line = _IPV6.sub(_ipv6_or_text, line)
    return _IPV4.sub("<ip>", line)


def _normalise(line: str) -> str:
    """Lowercase `line`; identifying values, paths, hex runs, long tokens and digit runs become
    placeholders; spaces collapse."""
    line = _redact(line.lower())
    line = _WINDOWS_PATH.sub("<path>", line)
    line = _UNC_PATH.sub("<path>", line)
    line = _POSIX_PATH.sub("<path>", line)
    line = _HEX_RUN.sub("<hex>", line)
    line = _TOKEN.sub("<token>", line)
    line = _DIGITS.sub("<n>", line)
    return _WHITESPACE.sub(" ", line).strip()


#: Words that mark the line of an error that says what went wrong, rather than progress, a
#: banner or a summary of counts.
_ERRORISH = re.compile(
    r"\b(error|errors|fatal|failed|failure|denied|not found|no such|cannot|can't|unable|refused|"
    r"timed out|timeout|exception|invalid|missing|not recognized|unrecognized|abort|aborted|"
    r"killed|panic)\b",
    re.IGNORECASE,
)
#: The last line of a Python traceback: `ValueError: ...`, `json.decoder.JSONDecodeError: ...`.
_EXCEPTION_LINE = re.compile(r"^[A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning|Failure)\b")
_TRACEBACK = re.compile(r"^traceback \(most recent call last\)", re.IGNORECASE)
#: A pytest-style `===== 1 failed, 2 passed in 3.1s =====` banner: error-ish words, no content.
_BANNER = re.compile(r"^=+ .* =+$")
#: Three letters in a row: a line with fewer, once paths, hex ids and numbers are placeholders, is
#: a bare number or path (`8`, `/tmp/x`) and says nothing about the failure.
_WORD = re.compile(r"[a-z]{3}")
_PLACEHOLDER = re.compile(r"<[a-z]+>")


def _informative(line: str) -> bool:
    """Whether `line` still holds a word once every placeholder (`<path>`, `<n>`...) is removed."""
    bare = _PLACEHOLDER.sub(" ", _normalise(line))
    return _WORD.search(bare) is not None


def _error_lines(text: str) -> list[str]:
    """The non-empty, stripped lines of `text` with a leading `Exit code N` line removed."""
    lines = [line for line in (raw.strip() for raw in text.splitlines()) if line]
    if lines and _EXIT_HEADER.match(lines[0]):
        lines = lines[1:]
    return lines


def _detail_line(error: str) -> str | None:
    """The line of a failure that says what went wrong, or None when the error has no lines.

    The FIRST line after `Exit code N`, which an earlier version took, is mostly boilerplate in
    agent commands: measured over 662 failed shell calls (2026-08-28 to 2026-10-23), 177 held a
    Python traceback, whose first line is `Traceback (most recent call last):` in every one of
    them, and a chain of checks prints its passing steps first (`All checks passed!`). The rule,
    applied to the last `MAX_ERROR_CHARS` characters of the error (where the failure is printed),
    over lines that still hold a word once numbers and paths are placeholders:

    1. In a Python traceback, its last `ExceptionType: message` line.
    2. Else the last line with an error word (`error`, `fatal`, `failed`, `denied`, `not found`,
       `no such`, `cannot`, `refused`, `timed out`, ...) that is not a `=== ... ===` banner.
    3. Else the last such line; else the first non-empty line of the error's start.
    """
    tail = _error_lines(_without_ansi(error[-MAX_ERROR_CHARS:]))
    lines = [line for line in tail if _informative(line)]
    if any(_TRACEBACK.match(line) for line in lines):
        for line in reversed(lines):
            if _EXCEPTION_LINE.match(line):
                return line
    for line in reversed(lines):
        if _ERRORISH.search(line) and not _BANNER.match(line):
            return line
    if lines:
        return lines[-1]
    head = _error_lines(_without_ansi(error[:MAX_ERROR_CHARS]))
    return head[0] if head else None


def failure_signature(command: str, error: str) -> str:
    """`<family>|<detail>`: the same failure of the same kind of command gives the same string.

    `family` is `command_family(command)` cut to `FAMILY_CHARS`. `detail` is the line
    `_detail_line` chooses, normalised by `_normalise` and cut to `SIGNATURE_CHARS`. When the
    error has no line besides the header it is `exit:N` (not normalised) or, with no exit header
    either, `exit:?`. The error is read only in its first and last `MAX_ERROR_CHARS` characters,
    BEFORE any splitting or matching, so the cost does not depend on how much the command printed.
    A non-string `command` or `error` counts as empty. Never raises.
    """
    family = command_family(command)[:FAMILY_CHARS]
    error = error if isinstance(error, str) else ""
    detail = _detail_line(error)
    if detail is not None:
        return f"{family}|{_normalise(detail)[:SIGNATURE_CHARS]}"
    first = _without_ansi(error[:MAX_ERROR_CHARS]).lstrip().partition("\n")[0].strip()
    header = _EXIT_HEADER.match(first)
    return f"{family}|exit:{header.group(1) if header is not None else '?'}"


def _without_exit_header(text: str) -> str:
    """`text` without its leading `Exit code N` line, the one `failure_signature` also skips."""
    head, _newline, rest = text.lstrip().partition("\n")
    return rest if _EXIT_HEADER.match(head.strip()) else text


def rank_for_failure(entries: list[Entry], error: str, min_score: float) -> list[Entry]:
    """The entries whose symptoms and body best match `error`, best first, at most `MAX_PER_CALL`.

    A lexical BM25 (`prompt_time.rank`, unchanged), never an embedder. Each entry's document is its
    symptoms repeated `SYMPTOM_WEIGHT` times followed by its body, so a symptom phrase counts for
    more than the same words in the body. Kept when `score >= min_score`. `prompt_time` is imported
    here, not at module level, so the `pre-tool-use` path never pays for it. Never raises: a broken
    entry list or an error that is not text gives `[]`.

    The query is the error WITHOUT a leading `Exit code N` line. Claude Code puts that header on
    every failed shell result, so leaving it in would add "exit" and "code" to every query, and any
    entry whose body says how a command exits would match every failure.
    """
    try:
        if not isinstance(error, str) or not entries:
            return []
        query = _without_exit_header(_without_ansi(error[:MAX_ERROR_CHARS]))
        if not query.strip():
            return []
        from . import prompt_time

        by_id: dict[str, Entry] = {}
        documents: list[tuple[str, str, list[str]]] = []
        for entry in entries:
            if entry.id in by_id:
                continue
            by_id[entry.id] = entry
            text = " ".join(list(entry.symptoms) * SYMPTOM_WEIGHT) + " " + entry.body
            documents.append((entry.id, "", prompt_time.tokenize(text)))
        ranked = prompt_time.rank(documents, query, k=MAX_PER_CALL)
        kept = [by_id[name] for name, _summary, score in ranked if score >= min_score]
        return kept[:MAX_PER_CALL]
    except Exception:  # BROAD-CATCH: fail-open
        return []


def render(entries: list[Entry]) -> str:
    """The text injected into the agent's context for `entries`; "" for none."""
    if not entries:
        return ""
    blocks = "\n\n".join(f"[{e.id}]\n{e.body}" for e in entries)
    return "General knowledge (owner-approved):\n\n" + blocks


#: The tools whose `command` is matched. `PowerShell` is a shell tool on Windows and carries the
#: same `tool_input["command"]` as `Bash`.
SHELL_TOOLS = frozenset({"Bash", "PowerShell"})


def pre_tool_use_context(payload: dict[str, Any]) -> str | None:
    """The general knowledge to inject before the shell command in `payload`, or None.

    Runs for every project, before and independently of the write-time gates: it needs no `dsn`, no
    `project_root` and no write-time `enabled`. It reads only the shared switch (`load_settings`),
    so a disabled feature costs one small file read and never touches the index or the log.

    One `"trigger"` event is logged whenever any in-scope entry matched, whether or not it was
    injected: `injected` holds the ids the session claimed, `suppressed` the matched ids it had
    already received (or that the per-call cap dropped). `trigger` names, per matched entry and in
    match order, the first trigger that fired, as `"<kind>:<value>"`.

    **Never raises.** The whole body is fail-open: a hook that raises is charged to the client, and
    a hook may never be the reason a tool call does not run.
    """
    try:
        if payload.get("tool_name") not in SHELL_TOOLS:
            return None
        tool_input = payload.get("tool_input")
        command = tool_input.get("command") if isinstance(tool_input, dict) else None
        if not isinstance(command, str) or not command.strip():
            return None
        if not load_settings()["enabled"]:
            return None
        cwd = payload.get("cwd")
        project = repository_name(cwd) if isinstance(cwd, str) else None
        entries = [e for e in load_index() if in_scope(e, project)]
        if not entries:
            return None
        command = command[:MAX_COMMAND_CHARS]
        segments = parse_command(command)
        matched: list[Entry] = []
        triggers: list[str] = []
        seen: set[str] = set()
        for entry in entries:
            if entry.id in seen:
                continue
            fired = matched_trigger(entry, command, segments, project)
            if fired is not None:
                seen.add(entry.id)
                matched.append(entry)
                triggers.append(f"{fired.kind}:{fired.value}")
        if not matched:
            return None
        session = payload.get("session_id")
        session = session if isinstance(session, str) else ""
        claimed_ids = claim_for_session(session, [e.id for e in matched])
        taken = set(claimed_ids)
        claimed = [e for e in matched if e.id in taken]
        log_event(
            {
                "event": "trigger",
                "project": project,
                "session": session,
                "client": client(),
                "trigger": triggers,
                "injected": [e.id for e in claimed],
                "suppressed": [e.id for e in matched if e.id not in taken],
            }
        )
        return render(claimed) or None
    except Exception:  # BROAD-CATCH: fail-open
        return None


def post_tool_use_failure(payload: dict[str, Any]) -> int:
    """`PostToolUseFailure`: inject the entries a failed shell command's error points at. Always 0.

    Skipped without output or log unless the tool is a shell tool, the failure is not a user
    interrupt (`is_interrupt` true means the person stopped the command, which says nothing about
    the command) and `error` is text. Then, with the shared switch on:

    * the signature is computed FIRST, so it is logged even if ranking fails;
    * in-scope entries are ranked against the error (`rank_for_failure`, lexical, the configured
      `min_score`) and claimed through the same once-per-session dedupe as `PreToolUse`;
    * ONE `"failure"` event is logged for EVERY failure, matched or not (spec 8): the signatures of
      the failures nothing matched are what shows which entries are worth writing. `injected` holds
      the claimed ids and `suppressed` the ranked ids this session had already claimed (dedupe
      only: the ranker returns at most `MAX_PER_CALL`, so the cap never drops one here);
    * the claimed entries are printed as `additionalContext`, after the log line, so a closed
      stdout can cost the injection but never the record.

    A missing or non-text `command` is the empty command: on this path the error is the evidence,
    and the signature's family is merely empty. The success path never reaches this function, so a
    command that works pays nothing; only the process spawned on a failure imports this module's
    ranker (`prompt_time`, imported inside `rank_for_failure`).

    **Never raises.** Same contract as `pre_tool_use_context`: the whole body is fail-open.
    """
    try:
        if payload.get("tool_name") not in SHELL_TOOLS or payload.get("is_interrupt") is True:
            return 0
        error = payload.get("error")
        if not isinstance(error, str):
            return 0
        config = load_settings()
        if not config["enabled"]:
            return 0
        tool_input = payload.get("tool_input")
        command = tool_input.get("command") if isinstance(tool_input, dict) else None
        command = command if isinstance(command, str) else ""
        cwd = payload.get("cwd")
        project = repository_name(cwd) if isinstance(cwd, str) else None
        session = payload.get("session_id")
        session = session if isinstance(session, str) else ""
        signature = failure_signature(command, error)
        ranked_ids: list[str] = []
        claimed: list[Entry] = []
        try:
            entries = [e for e in load_index() if in_scope(e, project)]
            ranked = rank_for_failure(entries, error, config["min_score"])
            ranked_ids = [e.id for e in ranked]
            taken = set(claim_for_session(session, ranked_ids))
            claimed = [e for e in ranked if e.id in taken]
        except Exception:  # BROAD-CATCH: fail-open, the signature below is still worth logging
            ranked_ids, claimed = [], []
        claimed_ids = {e.id for e in claimed}
        log_event(
            {
                "event": "failure",
                "project": project,
                "session": session,
                "client": client(),
                "signature": signature,
                "injected": [e.id for e in claimed],
                "suppressed": [i for i in ranked_ids if i not in claimed_ids],
            }
        )
        if claimed:
            print(json.dumps({
                "hookSpecificOutput": {
                    "hookEventName": "PostToolUseFailure",
                    "additionalContext": render(claimed),
                }
            }))
    except Exception:  # BROAD-CATCH: fail-open
        return 0
    return 0


__all__ = [
    "FAMILY_CHARS",
    "INDEX_NAME",
    "INDEX_SCHEMA",
    "MAX_COMMAND_CHARS",
    "MAX_ERROR_CHARS",
    "MAX_PER_CALL",
    "MAX_REGEX_CHARS",
    "SIGNATURE_CHARS",
    "SYMPTOM_WEIGHT",
    "Entry",
    "Segment",
    "STATE_MAX_AGE_SECONDS",
    "Trigger",
    "claim_for_session",
    "client",
    "command_family",
    "command_programs",
    "compiled",
    "failure_signature",
    "in_scope",
    "knowledge_home",
    "load_index",
    "load_settings",
    "log_event",
    "log_path",
    "match_triggers",
    "matched_trigger",
    "parse_command",
    "post_tool_use_failure",
    "pre_tool_use_context",
    "rank_for_failure",
    "render",
    "repository_name",
    "settings",
]
