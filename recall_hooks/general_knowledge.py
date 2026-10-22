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
from typing import Any

from . import HOOK_CONFIG_NAME, project_config_home

INDEX_NAME = "index.json"
INDEX_SCHEMA = 1
#: The most entries one tool call may inject. Each entry is paid for in the agent's context.
MAX_PER_CALL = 2

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
    `enabled: false` is the one switch that turns it off. A value of the wrong type falls back to
    its default rather than raising, for the same fail-open reason as the reader.

    The types are checked, not coerced: `bool("false")` is True, so coercing would turn a
    quoted `"false"` in a hand-edited config into the opposite of what its author meant. Only a
    real `bool` sets `enabled`, and only a finite int or float (never a bool, never NaN or
    infinity) sets `min_score`.
    """
    block = (config or {}).get("general_knowledge")
    block = block if isinstance(block, dict) else {}
    enabled = block.get("enabled", True)
    if not isinstance(enabled, bool):
        enabled = True
    min_score = block.get("min_score", 0.0)
    if isinstance(min_score, bool) or not isinstance(min_score, (int, float)) or not math.isfinite(min_score):
        min_score = 0.0
    return {"enabled": enabled, "min_score": float(min_score)}


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


__all__ = [
    "INDEX_NAME",
    "INDEX_SCHEMA",
    "MAX_PER_CALL",
    "Entry",
    "Trigger",
    "compiled",
    "knowledge_home",
    "load_index",
    "load_settings",
    "log_path",
    "settings",
]
