"""General knowledge: owner-curated entries injected by the hooks before a command or after a failure.

**What this module is.** The model and the reader for a small prebuilt index. An entry is a piece of
curated procedure or warning (how to run a test suite on a given host, which flag a tool needs) that
a session cannot derive from the repository. Later layers match an entry to the command about to
run, or to the failure that just happened, and inject its body. This layer only turns
`<knowledge home>/index.json` into `Entry` objects.

**Why a prebuilt index and not the entry files.** The matching runs inside a `PreToolUse` process
that is spawned once per tool call, so every cost is paid on every `Bash` call. Parsing one JSON
file is cheap and bounded; walking a directory of entry files and parsing front matter is neither.
The index is written by a build step (a later task) and read here, and nothing here touches the
network, a database or an embedder, and nothing here imports the `recall` package (see
`recall_hooks/__init__.py` for what that import costs).

**Fail-open, always.** A missing, unreadable, corrupt or foreign-schema index is an EMPTY index, not
an error: a hook that raises is charged to the client, and a hook may never be the reason a tool
call or a session does not start. One malformed entry costs that entry only.

**One corpus for both clients.** The home, the log and the switch are resolved from the project
config home (`project_config_home`), never from `load_config`: the Codex adapter redirects
`load_config` to Codex's own directory, and Claude and Codex must share one corpus, one switch and
one log.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

from . import HOOK_CONFIG_NAME, project_config_home

INDEX_NAME = "index.json"
INDEX_SCHEMA = 1
#: The most entries one tool call may inject. Each entry is paid for in the agent's context.
MAX_PER_CALL = 2
#: The most command text the matcher analyses (ruling R12). A command can carry a heredoc body of
#: hundreds of kilobytes, and parsing it costs time on every `Bash` call (0.39 s for 100 KB of
#: separators) for a program that is named in the first lines. Program extraction and
#: `command_regex` triggers both see only the first `MAX_COMMAND_CHARS` characters.
MAX_COMMAND_CHARS = 16384

#: The three ways an entry can be triggered. A trigger is a one-key object whose key is one of these.
TRIGGER_KINDS = frozenset({"command", "command_regex", "host"})

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
    """Where the index lives: `RECALL_GENERAL_KNOWLEDGE_HOME`, else `<project config home>/general-knowledge`."""
    raw = os.environ.get("RECALL_GENERAL_KNOWLEDGE_HOME", "").strip()
    if raw:
        return Path(raw).expanduser()
    return project_config_home() / "general-knowledge"


def log_path() -> Path:
    """The event log: `RECALL_GENERAL_KNOWLEDGE_LOG`, else `<project config home>/logs/general-knowledge.jsonl`."""
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
    """`settings` over `<project config home>/recall-hook.json`; a missing or corrupt file is `{}`.

    Deliberately NOT `recall_hooks.load_config()`: see the module docstring. This reads the same
    file name from the project config home, which Claude and Codex share.
    """
    try:
        document = json.loads((project_config_home() / HOOK_CONFIG_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return settings({})
    return settings(document if isinstance(document, dict) else {})


def compiled(pattern: str) -> re.Pattern[str] | None:
    """The compiled form of a `command_regex` value, from the load-time cache; `None` if unknown."""
    return _COMPILED.get(pattern)


def _trigger(raw: Any) -> Trigger | None:
    """A `Trigger` from `{"<kind>": "<value>"}`, or `None` for anything else.

    Dropped, not raised: a map with more than one key, an unknown kind, a non-string or empty value
    and a regex that does not compile are all the same case, one bad trigger that must not cost the
    entry its other triggers.
    """
    if not isinstance(raw, dict) or len(raw) != 1:
        return None
    ((kind, value),) = raw.items()
    if kind not in TRIGGER_KINDS or not isinstance(value, str) or not value:
        return None
    if kind == "command_regex" and value not in _COMPILED:
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
    """The repository name a `.git` file points at; the caller catches unreadable and undecodable."""
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
    return project is not None and entry.scope.startswith(prefix) and entry.scope[len(prefix) :] == project


def _names_host(alias: str, args: tuple[str, ...]) -> bool:
    """True when an argument is `alias`, `alias:<path>`, or either of those after `user@`.

    `user@` is split off only from the part BEFORE the first colon, so an `@` in the path part
    (`examplehost:/var/x@1`) neither hides a host nor invents one (`other:/x@examplehost`).
    """
    wanted = alias.casefold()
    return any(arg.partition(":")[0].rpartition("@")[2].casefold() == wanted for arg in args)


def matched_trigger(entry: Entry, command: str, segments: list[Segment], project: str | None) -> Trigger | None:
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
            if any(s.program in _HOST_PROGRAMS and _names_host(trigger.value, s.args) for s in segments):
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


__all__ = [
    "INDEX_NAME",
    "INDEX_SCHEMA",
    "MAX_COMMAND_CHARS",
    "MAX_PER_CALL",
    "Entry",
    "Segment",
    "Trigger",
    "command_programs",
    "compiled",
    "in_scope",
    "knowledge_home",
    "load_index",
    "load_settings",
    "log_path",
    "match_triggers",
    "matched_trigger",
    "parse_command",
    "repository_name",
    "settings",
]
