"""Keys a memo's writer declares, and the three things RE-call does with them.

A memo states a fact in its `recall_graph` frontmatter (or through `recall_remember`'s `fact`), and the
fact's key is the ledger's conflict key: namespace, subject, predicate and context, compared exactly
(`recall.provenance_controller.fact_conflict_key`). Two memos about one slot conflict only when their
writer spelled the key the same way, and a conflict is `contested` until a `supersedes:` settles it.
Measured on MemoryAgentBench FactConsolidation (recall-lab WK-1 to WK-3, 2026-10-23): with declared
keys and edges RE-call's ledger answers at the dataset's ceiling, gpt-4o-mini extracting keys from the
prose afterwards could not keep them consistent, and a capable writer declaring keys as it wrote,
after looking up its own earlier keys by the subject it meant, matched the ceiling.

So this module gives a writer what that writer used:

- `subject_view`: the current facts of one subject, matched case- and spacing-blind, and the
  subjects whose names contain or are contained in it, so a writer can reuse an existing spelling;
- `successor_pairs`, rule R(a'): of the memos stating one single-valued key, each supersedes the
  memos of the latest earlier start, WHATEVER the value. "Another value" was the first wording, and
  a key whose history repeats a value then kept its first fact in force against the last
  (WK-1, 2026-10-23). The rule only proposes: `recall.reasoning_proposals` serves its pairs as
  `requires_review`, and nothing here writes an edge;
- `key_receipt`: the part of a write receipt that says which key a new memo's fact went under, what
  else is in force for it, and the edge rule R(a') proposes, if any.

Undated memos, and memos of one start, are never ordered against each other: a sort position would
name the newer one (`recall.reasoning_proposals._deterministic._date_successors` has the same rule).
"""

from __future__ import annotations

import unicodedata
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from itertools import groupby
from typing import Any

from recall.provenance_controller import canonical_json, predicate_cardinality
from recall.types import AtomicFact

__all__ = [
    "STATED_KEY_SUCCESSOR_RULE",
    "KeyedMemo",
    "fact_key",
    "key_receipt",
    "subject_name",
    "subject_view",
    "successor_pairs",
]

#: The rule id rule R(a') proposes under, in `recall.reasoning_proposals` and in a write receipt.
STATED_KEY_SUCCESSOR_RULE = "deterministic.stated_key_successor"
#: How many look-alike subjects `subject_view` lists.
SIMILAR_SUBJECTS = 5


def subject_name(text: str) -> str:
    """A subject as the lookup compares it: NFC, case folded, runs of whitespace made one space.

    Only the lookup folds. The key itself stays the exact string the writer declared, which is what
    the ledger compares, so a lookup can show a writer a spelling it must then repeat exactly.
    """
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def fact_key(fact: AtomicFact) -> tuple[str, str, str, str] | None:
    """The ledger's conflict key of a single-valued fact, or None: two values of a `set` or an `event`
    predicate hold at once, so neither replaces the other."""
    if predicate_cardinality(fact) != "single":
        return None
    return fact.namespace, fact.subject, fact.predicate, canonical_json(fact.context)


@dataclass(frozen=True)
class KeyedMemo:
    """One memo's statement of one fact: the memo's name, the fact, and when the fact starts."""

    name: str
    fact: AtomicFact
    valid_from: datetime | None


def _utc(value: datetime) -> datetime:
    return value if value.tzinfo is not None else value.replace(tzinfo=UTC)


def successor_pairs(
    memos: Iterable[KeyedMemo], *, incident_to: str | None = None,
) -> list[tuple[KeyedMemo, KeyedMemo]]:
    """Rule R(a'): (older, newer) for every memo and each memo of the next later start stating its key.

    Grouped by `fact_key`; per key, the dated statements sorted by start and grouped by equal start,
    each group paired with the next. A memo is never paired with itself, and a pair is listed once
    however many keys the two memos share. Statements with no start, and keys that are not
    single-valued, take part in nothing. With `incident_to`, generate only pairs involving that
    memo, without allocating the other pairs of adjacent groups.
    """
    by_key: dict[tuple[str, str, str, str], list[KeyedMemo]] = {}
    for memo in memos:
        key = fact_key(memo.fact)
        if key is not None and memo.valid_from is not None:
            by_key.setdefault(key, []).append(memo)
    pairs: list[tuple[KeyedMemo, KeyedMemo]] = []
    seen: set[tuple[str, str]] = set()
    for key in sorted(by_key):
        dated = sorted(by_key[key], key=lambda m: (_utc(m.valid_from), m.name))  # type: ignore[arg-type]
        groups = [list(g) for _start, g in groupby(dated, key=lambda m: _utc(m.valid_from))]  # type: ignore[arg-type]
        for earlier, later in zip(groups, groups[1:], strict=False):
            incident_later = [memo for memo in later if memo.name == incident_to] if incident_to is not None else later
            for older in earlier:
                candidates = later if incident_to is None or older.name == incident_to else incident_later
                for newer in candidates:
                    if older.name != newer.name and (older.name, newer.name) not in seen:
                        seen.add((older.name, newer.name))
                        pairs.append((older, newer))
    return pairs


def _same_subject(value: object, wanted: str) -> bool:
    return isinstance(value, str) and subject_name(value) == wanted


def subject_view(
    listing: Mapping[str, Any], subject: str, predicate: str | None = None, *, similar: int = SIMILAR_SUBJECTS
) -> dict[str, Any]:
    """`listing` (a `recall_current_facts` answer) narrowed to one subject, and its look-alikes.

    `facts` and `contested` keep only the entries whose subject matches `subject` under
    `subject_name`, and whose predicate equals `predicate` when one is given. `similar_subjects` lists,
    longest first, up to `similar` (subject, predicate) pairs of the whole listing whose subject
    contains, or is contained in, the one asked for without matching it: "Fatah" for "Fatah movement",
    so the writer can tell one entity spelled two ways from two entities. Every other field of the
    listing is kept as it was.
    """
    wanted = subject_name(subject)
    facts = [
        item for item in listing.get("facts", [])
        if isinstance(item.get("fact"), Mapping) and _same_subject(item["fact"].get("subject"), wanted)
        and (predicate is None or item["fact"].get("predicate") == predicate)
    ]
    contested = [
        group for group in listing.get("contested", [])
        if _same_subject(group.get("subject"), wanted) and (predicate is None or group.get("predicate") == predicate)
    ]
    pairs: set[tuple[str, str]] = set()
    for item in listing.get("facts", []):
        fact = item.get("fact")
        if isinstance(fact, Mapping) and isinstance(fact.get("subject"), str):
            pairs.add((fact["subject"], str(fact.get("predicate"))))
    for group in listing.get("contested", []):
        if isinstance(group.get("subject"), str):
            pairs.add((group["subject"], str(group.get("predicate"))))
    looks_like = []
    for name, pred in pairs:
        folded = subject_name(name)
        if wanted and folded and folded != wanted and (wanted in folded or folded in wanted):
            looks_like.append((name, pred))
    looks_like.sort(key=lambda pair: (-len(pair[0]), pair[0], pair[1]))
    return {
        **listing,
        "subject": subject,
        "predicate": predicate,
        "facts": facts,
        "contested": contested,
        "similar_subjects": [{"subject": name, "predicate": pred} for name, pred in looks_like[:similar]],
    }


def _statements(listing: Mapping[str, Any], key: tuple[str, str, str, str]) -> list[tuple[dict[str, Any], list[str], bool]]:
    """(fact payload, sources, standing) for every value of `key` the listing holds in force."""
    found: list[tuple[dict[str, Any], list[str], bool]] = []

    def same(payload: object) -> AtomicFact | None:
        if not isinstance(payload, Mapping):
            return None
        try:
            fact = AtomicFact.from_payload(dict(payload))
        except (TypeError, ValueError, KeyError):
            return None
        return fact if fact_key(fact) == key else None

    for item in listing.get("facts", []):
        if same(item.get("fact")) is not None:
            found.append((dict(item["fact"]), list(item.get("sources", [])), True))
    for group in listing.get("contested", []):
        for value in group.get("values", []):
            if same(value.get("fact")) is not None:
                found.append((dict(value["fact"]), list(value.get("sources", [])), False))
    return found


def key_receipt(
    listing: Mapping[str, Any], source: str, fact: AtomicFact, *, name_of: Callable[[str], str] = lambda name: name
) -> dict[str, Any]:
    """What a write receipt says about the key of the fact a new memo `source` states.

    `state` is `current` when this memo's value is the key's only value in force, `contested` when
    another value is in force beside it with no supersession between them, `not_in_force` when this
    memo's own fact is not current (outside its window, or superseded already), `not_single_valued`
    for a `set` or `event` predicate, and `unchecked` when the listing could not be checked against
    what the store serves. `in_force` lists every value of the key in force with the memos stating
    it. `proposals` holds the edges rule R(a') proposes between this memo and another in force for
    the key, each `requires_review`: a writer settles one by writing the correction with
    `supersedes`, a reviewer through `recall rewrite`. `name_of` turns a stored name into the name
    the caller knows the memo by.
    """
    receipt: dict[str, Any] = {
        "key": {"namespace": fact.namespace, "subject": fact.subject, "predicate": fact.predicate,
                "context": fact.context},
        "in_force": [],
        "proposals": [],
    }
    key = fact_key(fact)
    if key is None:
        receipt["state"] = "not_single_valued"
        return receipt
    if listing.get("served_state") in {"degraded", "unchecked"}:
        receipt["state"] = "unchecked"
        return receipt
    statements = _statements(listing, key)
    receipt["in_force"] = [
        {"value": payload.get("object"), "valid_from": payload.get("valid_from"),
         "memos": [name_of(name) for name in sources]}
        for payload, sources, _standing in statements
    ]
    mine = any(source in sources for _payload, sources, _standing in statements)
    if not mine:
        receipt["state"] = "not_in_force"
    elif any(not standing for _payload, _sources, standing in statements):
        receipt["state"] = "contested"
    else:
        values = {canonical_json(payload.get("object")) for payload, _sources, _standing in statements}
        receipt["state"] = "current" if len(values) == 1 else "contested"
    memos = []
    for payload, sources, _standing in statements:
        stated = AtomicFact.from_payload(payload)
        for name in sources:
            memos.append(KeyedMemo(name, stated, stated.valid_from))
    for older, newer in successor_pairs(memos, incident_to=source):
        receipt["proposals"].append({
            "rule_id": STATED_KEY_SUCCESSOR_RULE, "status": "requires_review", "relation": "supersedes",
            "superseded": name_of(older.name), "superseder": name_of(newer.name),
        })
    if receipt["state"] == "contested" and not receipt["proposals"]:
        receipt["note"] = ("another value is in force for this key and the memos cannot be ordered by "
                           "valid_from; write the correction with supersedes, or give the memos dates")
    return receipt
