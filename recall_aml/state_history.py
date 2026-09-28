"""W3: the history of a keyed fact, ordered in code and rendered with its validity in the text.

Round two, 2026-09-28. D1 (value updates, current state) scored 27.90 on the first Textual Full
and C3 (trajectories) 45.78; nothing in C9 represents state (``supersedes`` was set on 0 of 263,662
compiled records). W1 (`recall_aml.conversation_records`) gives each fact a normalised
``subject|attribute`` key; this groups one conversation's facts by key and decides, in code:

* **update**: two or more different values of a ``state``, ``preference`` or ``plan``. The values
  are ordered by when they were said, and the text says which is the latest on record and when each
  earlier one stopped being current. Freshness is decided by timestamp, never by the model.
* **conflict**: a ``never`` statement and a statement that it happened, on one key. Both are kept
  with no winner, because BEAM's contradiction rubric asks the reader to report both and ask which
  is right, and D2 (contradictions, 69.02) is a strength to protect.
* **none**: anything else (one value, repeated values, events only).

Older values are labelled, never dropped: trajectory questions need the whole path. The rendered
history is built only from stored facts (verbatim values, dates, speakers); nothing is generated at
Search time.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Any

from recall_aml.conversation_records import ConversationFact

CHANGING = frozenset({"state", "preference", "plan"})
HAPPENED = frozenset({"event", "state", "plan", "preference"})


@dataclass(frozen=True)
class KeyHistory:
    key: str
    kind: str  # "update", "conflict" or "none"
    facts: tuple[ConversationFact, ...]  # in the order they were said

    @property
    def latest(self) -> ConversationFact | None:
        return self.facts[-1] if self.kind == "update" else None

    def rendered(self) -> str:
        subject, _, attribute = self.key.partition("|")
        if self.kind == "conflict":
            lines = [f"[two statements on record, both kept · {subject} / {attribute}]"]
            lines += [f"{_day(f)} {_said(f)};" for f in self.facts]
            return " ".join(lines)
        lines = [f"[history · {subject} / {attribute} · oldest first]"]
        for index, fact in enumerate(self.facts):
            if index + 1 < len(self.facts):
                lines.append(f"{_day(fact)} {_said(fact)} (no longer current: replaced {_day(self.facts[index + 1])});")
            else:
                lines.append(f"{_day(fact)} {_said(fact)} (latest on record)")
        return " ".join(lines)


def _day(fact: ConversationFact) -> str:
    return fact.mention_time.date().isoformat() if fact.mention_time else "undated"


def _said(fact: ConversationFact) -> str:
    speaker = f"{fact.speaker}: " if fact.speaker else ""
    return f'{speaker}"{fact.value}"'


def _value(fact: ConversationFact) -> str:
    return " ".join(fact.value.casefold().split())


def _counted(event_updates: bool) -> frozenset[str]:
    return CHANGING | {"event"} if event_updates else CHANGING


def classify(facts: Sequence[ConversationFact], *, event_updates: bool = False) -> str:
    """``event_updates`` (W1 and W3 v3): an ``event`` value counts toward a change when the key
    also holds a ``state``, ``plan`` or ``preference``. Measured on BEAM 100K: 3 of 13 linked
    update pairs were "call on April 21" (event) then "call April 22" (state), classified ``none``
    without it. Events alone (several trips on one key) stay ``none``."""
    relations = {f.relation for f in facts}
    if "never" in relations and relations & (HAPPENED - {"never"}):
        return "conflict"
    counted = [f for f in facts if f.relation in _counted(event_updates)]
    if len({_value(f) for f in counted}) >= 2 and relations & CHANGING:
        return "update"
    return "none"


def key_histories(
    facts: Sequence[ConversationFact], order: Callable[[ConversationFact], Any] | None = None,
    *, event_updates: bool = False,
) -> dict[str, KeyHistory]:
    """Every key's history, its facts sorted by ``order`` (default: when said, then as given).

    ``order`` lets a caller that knows more than the timestamp (an Add's sequence, a turn id) break
    ties between facts said on the same day, which BEAM's batch-level dates produce constantly.
    """
    by_key: dict[str, list[tuple[int, ConversationFact]]] = defaultdict(list)
    for position, fact in enumerate(facts):
        by_key[fact.key].append((position, fact))

    def default(item: tuple[int, ConversationFact]) -> tuple[datetime | None, int]:
        position, fact = item
        return (fact.mention_time, position)

    out: dict[str, KeyHistory] = {}
    for key, items in by_key.items():
        ordered = sorted(items, key=(lambda item: (order(item[1]), item[0])) if order else default)
        members = [fact for _, fact in ordered]
        kind = classify(members, event_updates=event_updates)
        if kind == "update":
            members = _distinct_changing(members, event_updates=event_updates)
        out[key] = KeyHistory(key=key, kind=kind, facts=tuple(members))
    return out


def _distinct_changing(ordered: Sequence[ConversationFact], *, event_updates: bool = False) -> list[ConversationFact]:
    """The changing values in order, each value kept at its latest statement (a repeat is a
    reaffirmation, not a new value)."""
    latest: dict[str, ConversationFact] = {}
    for fact in ordered:
        if fact.relation in _counted(event_updates):
            latest.pop(_value(fact), None)
            latest[_value(fact)] = fact
    return list(latest.values())
