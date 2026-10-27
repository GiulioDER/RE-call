"""Current facts, materialized: which assertions of the fact ledger are in force, and when.

`project_current` answers a read by decoding the tenant's whole event history and resolving
supersession over it, and `ProvenanceController.apply_fact` resolved it twice more for its conflict
check. This module keeps the answer beside the ledger as derived state instead: one row per
effective interval of each applied assertion, its own window less the windows of the successors
that hide it (`FactSupersession.in_force`), so a read at any instant is a range lookup and a
write's conflict check reads one conflict key's history rather than the tenant's.

Three tables, on PostgreSQL (migration 0030) and in a lite file alike:

- `recall_fact_current_events`: every ledger event the tables reflect, filed under the conflict
  key it bears on, with `live` set on an assertion no retraction names. Ids and hashes only: no
  table here holds a fact value, so the ledger stays the one place a value lives.
- `recall_fact_current`: the effective intervals, indexed on `(tenant_id, conflict_key,
  valid_from, valid_until)`. A correction for 2020 to 2022 splits an open-ended fact into two rows.
- `recall_fact_current_state`: per tenant, `replay` when the tables may not answer (an event names
  a fact of another conflict key, or a fact the ledger never recorded; no write path does either).

**Derived, and exact or silent.** A conflict key's rows are recomputed from that key's ledger
events with the ledger's own functions (`fact_supersession`, `retracted_keys`, `asserts_fact`),
which read nothing outside the key, so the tables can always be rebuilt from the immutable ledger
(`rebuild_tables`). A read uses them only when, in one snapshot, they file every event the ledger
holds for the tenant and every id they name resolves there (`current_view`, `key_history`);
otherwise it replays the ledger, which is the definition the tables must equal. That is chosen
over failing closed because a replay can never answer differently, while a lag is routine: an
older build writing the same database feeds nothing. The ledgers materialize each event in the
transaction that appends it, in a savepoint, so a failure here never fails an append; the tables
then lag and reads replay until `reconcile` (`MaterializationRecovery.reconcile` through the
outbox, or `CurrentFacts.reconcile`) or `rebuild` catches them up.
"""
# The SQL identifiers in this module are fixed module constants; values are bound parameters.
# ruff: noqa: S608
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Protocol

import psycopg

from recall.fact_ledger import (
    FACT_LEDGER_TABLE,
    LEDGER_COLUMNS,
    _event_from_row,
    decode_ledger_row,
)
from recall.observability import get_logger
from recall.provenance_controller import (
    FactEvent,
    ServedSources,
    _as_utc,
    absent_fact_ids,
    asserts_fact,
    fact_conflict_key,
    fact_supersession,
    retracted_keys,
    retraction_key,
)
from recall.types import AtomicFact

CURRENT_TABLE = "recall_fact_current"
CURRENT_EVENTS_TABLE = "recall_fact_current_events"
CURRENT_STATE_TABLE = "recall_fact_current_state"
#: Every event type the tables file. A refusal or an abstention changes no interval, but it is
#: part of its key's history (`ordered_after` reads it), so a key's history from the tables must
#: hold it, and the ledger's event count is only matched when every event is filed.
MATERIALIZED_EVENT_TYPES = frozenset(
    {"asserted", "superseded", "rejected", "abstained", "retracted", "withdrawn"}
)
#: The types whose filing cannot change a key's intervals or its `live` flags.
_INERT = frozenset({"rejected", "abstained"})
_log = get_logger("fact_current")

SQLITE_FACT_CURRENT_DDL = f"""
CREATE TABLE IF NOT EXISTS {CURRENT_EVENTS_TABLE} (
    tenant_id TEXT NOT NULL,
    event_id TEXT NOT NULL,
    conflict_key TEXT,
    fact_id TEXT,
    event_type TEXT NOT NULL,
    live INTEGER NOT NULL DEFAULT 0 CHECK (live IN (0, 1)),
    PRIMARY KEY (tenant_id, event_id)
);
CREATE INDEX IF NOT EXISTS recall_fact_current_events_key_idx
    ON {CURRENT_EVENTS_TABLE} (tenant_id, conflict_key);
CREATE INDEX IF NOT EXISTS recall_fact_current_events_fact_idx
    ON {CURRENT_EVENTS_TABLE} (tenant_id, fact_id);
CREATE TABLE IF NOT EXISTS {CURRENT_TABLE} (
    tenant_id TEXT NOT NULL,
    conflict_key TEXT NOT NULL,
    valid_from TEXT NOT NULL,
    valid_until TEXT NOT NULL,
    event_id TEXT NOT NULL,
    fact_id TEXT NOT NULL,
    PRIMARY KEY (tenant_id, event_id, valid_from),
    CHECK (valid_from < valid_until)
);
CREATE INDEX IF NOT EXISTS recall_fact_current_window_idx
    ON {CURRENT_TABLE} (tenant_id, conflict_key, valid_from, valid_until);
CREATE TABLE IF NOT EXISTS {CURRENT_STATE_TABLE} (
    tenant_id TEXT PRIMARY KEY,
    mode TEXT NOT NULL CHECK (mode IN ('table', 'replay')),
    reason TEXT,
    updated_at TEXT NOT NULL
);
-- A lite file's ledger is the store's own: indexing it here blocks only the one local writer.
-- PostgreSQL builds the same index CONCURRENTLY in its own migration, 0031.
CREATE INDEX IF NOT EXISTS recall_fact_ledger_order_idx
    ON {FACT_LEDGER_TABLE} (tenant_id, created_at, event_id);
"""

#: One effective interval: the assertion, its fact, and ``[low, high)``.
IntervalRow = tuple[str, str, datetime, datetime]


@dataclass(frozen=True)
class KeyProjection:
    """What one conflict key's history puts in the tables."""

    #: `FactSupersession.in_force` of every applied, unretracted assertion, one row per interval.
    intervals: tuple[IntervalRow, ...]
    #: The assertions no retraction names, DUPLICATE ones included: what `absent_fact_ids` and
    #: `live_sources` read a fact's cards from.
    live: frozenset[str]


def project_key(events: Sequence[FactEvent]) -> KeyProjection:
    """The rows of one conflict key, from its events in ledger order, by `project_current`'s rule.

    `project_current` lists an assertion at `now` when it asserts its fact, is not retracted and
    `now` lies in its window and outside every window that hides it; the last two are
    `FactSupersession.in_force`, so the intervals here hold exactly the instants it is listed at,
    before the per-fact dedupe and the served check a read applies (`current_view`).
    """
    index = fact_supersession(events)
    retracted = retracted_keys(events)
    rows: list[IntervalRow] = []
    live: set[str] = set()
    for event in events:
        if event.event_type != "asserted" or event.fact_id is None or retraction_key(event) in retracted:
            continue
        live.add(event.event_id)
        if asserts_fact(event) and event.fact is not None:
            rows.extend(
                (event.event_id, event.fact_id, low, high)
                for low, high in index.in_force(event)
                if low < high
            )
    return KeyProjection(tuple(rows), frozenset(live))


#: Given a fact id an event names: whether the ledger holds it, and its conflict key (None for a
#: fact with no value, which `recall_forget`'s redaction leaves; it bears on no key).
NamedKey = Callable[[str], tuple[bool, str | None]]


def event_keys(event: FactEvent, named_key: NamedKey) -> tuple[frozenset[str], str | None]:
    """The conflict keys `event` bears on, and why the tables cannot file it exactly, if they cannot.

    Its own fact's key, and the keys of the facts it names (a successor's predecessors, a
    retraction's fact). The controller only names facts its claim conflicts with, which share
    the claim's key, and a retraction names the fact it retracts, so one key is the rule; two,
    or a named fact the ledger never recorded, is reported, and the tenant replays.
    """
    keys: set[str] = set()
    if event.fact is not None:
        keys.add(fact_conflict_key(event.fact))
    for fact_id in event.supersedes_fact_ids:
        known, key = named_key(fact_id)
        if not known:
            return frozenset(keys), f"event {event.event_id} names fact {fact_id}, which the ledger does not hold"
        if key is not None:
            keys.add(key)
    if len(keys) > 1:
        return frozenset(keys), f"event {event.event_id} bears on {len(keys)} conflict keys"
    return frozenset(keys), None


def _payload_key(payload: object) -> str | None:
    """The conflict key of a stored fact payload; None for a redaction marker, which holds no value."""
    if isinstance(payload, str):
        payload = json.loads(payload)
    if not isinstance(payload, Mapping) or payload.get("redacted") is True:
        return None
    return fact_conflict_key(AtomicFact.from_payload(payload))


@dataclass(frozen=True)
class CurrentView:
    """A read the tables answered: `project_current`'s answer, and the events `live_sources` needs."""

    current: tuple[FactEvent, ...]
    #: Every live assertion of each fact in force at the instant, in ledger order.
    support: tuple[FactEvent, ...]


class Tables(Protocol):
    """The tables of one tenant on one connection; `PostgresTables` and `SQLiteTables`."""

    tenant_id: str

    def counts(self) -> tuple[int, int, str | None]:
        """The ledger's event count, the filed event count, and the tenant's mode."""
        ...

    def events(self, event_ids: Sequence[str] | None) -> list[FactEvent]:
        """Ledger events by id (all of the tenant's when None), in ledger order."""
        ...

    def key_events(self, key: str) -> list[FactEvent] | None:
        """The ledger events filed under `key`, in ledger order; None when one is not in the ledger."""
        ...

    def filed(self, event_ids: Sequence[str]) -> set[str]: ...

    def named_key(self, fact_id: str) -> tuple[bool, str | None]: ...

    def file(self, rows: Sequence[tuple[str, str | None, str | None, str]]) -> None:
        """Record `(event_id, conflict_key, fact_id, event_type)` rows, skipping filed ones."""
        ...

    def write_key(self, key: str, projection: KeyProjection) -> None: ...

    def set_replay(self, reason: str) -> None: ...

    def intervals_at(self, moment: datetime) -> list[tuple[str, str]] | None:
        """`(event_id, fact_id)` of the intervals holding `moment`, in ledger order; None when one
        names an event the ledger does not hold."""
        ...

    def support_ids(self, fact_ids: Sequence[str]) -> list[str]: ...

    def unfiled(self, *, limit: int | None = None) -> list[str]:
        """The ids of the tenant's ledger events the tables do not file, in ledger order."""
        ...

    def stored(self) -> tuple[set[FiledRow], set[StoredInterval], str | None]:
        """Every row the tables hold for the tenant, as `derive_tables` would write it, and the mode."""
        ...

    def clear(self) -> None: ...

    def lock_keys(self, keys: Sequence[str]) -> None: ...

    def lock_tenant(self, *, exclusive: bool) -> None: ...


def usable(tables: Tables) -> bool:
    """Whether the tables file every ledger event of the tenant and may answer for it.

    Filing only ever copies an id the ledger holds, so equal counts are equal sets; ids the
    ledger stopped holding (a redaction that rewrote them without a rebuild) are caught where
    they are read (`intervals_at`, `key_events`, `current_view`).
    """
    ledger, filed, mode = tables.counts()
    return ledger == filed and mode != "replay"


def current_view(tables: Tables, now: datetime, served: ServedSources | None) -> CurrentView | None:
    """`project_current(events, now, served)` from the tables, or None when they cannot answer.

    Of each fact's assertions in force at `now`, the first in ledger order is listed, as
    `project_current` lists it; a fact none of whose live assertions has a card the store serves
    is left out (`absent_fact_ids` over those assertions, which is what it reads of the history).
    """
    if _as_utc(now) == datetime.max.replace(tzinfo=UTC):
        # Replay preserves the distinction between an absent end and an explicit max end.
        return None
    if not usable(tables):
        return None
    rows = tables.intervals_at(now)
    if rows is None:
        return None
    first: dict[str, str] = {}
    for event_id, fact_id in rows:
        first.setdefault(fact_id, event_id)
    ids = list(dict.fromkeys(tables.support_ids(list(first))))
    support = tables.events(ids)
    if len(support) != len(ids) or not set(first.values()) <= {event.event_id for event in support}:
        return None
    listed = set(first.values())
    absent = absent_fact_ids(support, served, now)
    return CurrentView(
        tuple(event for event in support if event.event_id in listed and event.fact_id not in absent),
        tuple(support),
    )


def key_history(tables: Tables, fact: AtomicFact) -> list[FactEvent] | None:
    """The history of `fact`'s conflict key, or None when the tables cannot answer.

    What a write's conflict check reads (`assertion_is_duplicate`, `unresolved_conflicts`,
    `ordered_after`): each reads only events of the claim's key, so their answers over this
    history are their answers over the tenant's.
    """
    if not usable(tables):
        return None
    return tables.key_events(fact_conflict_key(fact))


def materialize_events(tables: Tables, event_ids: Sequence[str]) -> None:
    """File the ledger events `event_ids` and recompute every key they bear on. Idempotent.

    Each event is read from the ledger, never taken from a caller or an outbox snapshot, so the
    tables can only reflect what the ledger holds. Key locks are taken before the tenant's shared
    lock, the order every writer uses, so a rebuild (exclusive) can never deadlock against them.
    """
    filed = tables.filed(event_ids)
    wanted = [event_id for event_id in dict.fromkeys(event_ids) if event_id not in filed]
    if not wanted:
        return
    events = tables.events(wanted)
    if len(events) != len(wanted):
        missing = sorted(set(wanted) - {event.event_id for event in events})
        raise LookupError(f"events {missing} are not in the ledger")
    rows: list[tuple[str, str | None, str | None, str]] = []
    affected: set[str] = set()
    reasons: list[str] = []
    for event in events:
        keys, reason = event_keys(event, tables.named_key)
        if reason is not None:
            reasons.append(reason)
        key = min(keys) if keys else None
        rows.append((event.event_id, key, event.fact_id, event.event_type))
        if key is not None and event.event_type not in _INERT:
            affected.add(key)
    tables.lock_keys(sorted(affected))
    tables.lock_tenant(exclusive=False)
    tables.file(rows)
    for key in sorted(affected):
        history = tables.key_events(key)
        if history is None:
            raise LookupError(f"conflict key {key} files an event the ledger does not hold")
        tables.write_key(key, project_key(history))
    if reasons:
        tables.set_replay(reasons[0])


#: A filed event as the tables hold it: `(event_id, conflict_key, fact_id, event_type, live)`.
FiledRow = tuple[str, str | None, str | None, str, bool]
#: An interval as the tables hold it: `(event_id, fact_id, conflict_key, valid_from, valid_until)`;
#: a bound that cannot be read back is None.
StoredInterval = tuple[str, str, str, datetime | None, datetime | None]


@dataclass(frozen=True)
class DerivedTables:
    """What a tenant's whole ledger puts in the tables: what `rebuild_tables` writes."""

    rows: tuple[tuple[str, str | None, str | None, str], ...]
    projections: Mapping[str, KeyProjection]
    #: Why the tenant may not be answered from the tables, if it may not (`event_keys`).
    reasons: tuple[str, ...]

    def filed(self) -> dict[str, FiledRow]:
        live = {event_id for projection in self.projections.values() for event_id in projection.live}
        return {
            event_id: (event_id, key, fact_id, event_type, event_id in live)
            for event_id, key, fact_id, event_type in self.rows
        }

    def intervals(self) -> set[StoredInterval]:
        return {
            (event_id, fact_id, key, low, high)
            for key, projection in self.projections.items()
            for event_id, fact_id, low, high in projection.intervals
        }


def derive_tables(events: Sequence[FactEvent]) -> DerivedTables:
    """The rows of every conflict key of a tenant, from its events in ledger order. Writes nothing."""
    named: dict[str, str | None] = {}
    for event in events:
        if event.fact is not None and event.fact_id is not None:
            named.setdefault(event.fact_id, fact_conflict_key(event.fact))
        elif event.event_type in {"asserted", "rejected", "abstained"} and event.fact_id is not None:
            named.setdefault(event.fact_id, None)  # a redacted value: it bears on no key

    def lookup(fact_id: str) -> tuple[bool, str | None]:
        return fact_id in named, named.get(fact_id)

    rows: list[tuple[str, str | None, str | None, str]] = []
    groups: dict[str, list[FactEvent]] = {}
    reasons: list[str] = []
    for event in events:
        keys, reason = event_keys(event, lookup)
        if reason is not None:
            reasons.append(reason)
        key = min(keys) if keys else None
        rows.append((event.event_id, key, event.fact_id, event.event_type))
        if key is not None:
            groups.setdefault(key, []).append(event)
    return DerivedTables(
        tuple(rows), {key: project_key(history) for key, history in groups.items()}, tuple(reasons)
    )


def rebuild_tables(tables: Tables) -> None:
    """Replace the tenant's rows with ones derived from its whole ledger, under its exclusive lock."""
    tables.lock_tenant(exclusive=True)
    tables.clear()
    derived = derive_tables(tables.events(None))
    tables.file(derived.rows)
    for key, projection in derived.projections.items():
        tables.write_key(key, projection)
    if derived.reasons:
        tables.set_replay(derived.reasons[0])


@dataclass(frozen=True)
class Verification:
    """The tables against what the ledger derives now (`verify_tables`); ids, never values.

    A count can say the tables lag; only this can say they were edited. It proves nothing about
    the ledger itself, which its triggers and grants keep append-only.
    """

    tenant_id: str
    #: Events the ledger derives and the tables do not file, filed with other values, or filed
    #: but derived by nothing.
    events_missing: tuple[str, ...]
    events_changed: tuple[str, ...]
    events_extra: tuple[str, ...]
    #: Intervals derived and absent, or stored and not derived (event ids of each, one per interval).
    intervals_missing: tuple[str, ...]
    intervals_extra: tuple[str, ...]
    mode_stored: str
    mode_derived: str

    @property
    def matches(self) -> bool:
        return not (
            self.events_missing or self.events_changed or self.events_extra
            or self.intervals_missing or self.intervals_extra
        ) and self.mode_stored == self.mode_derived

    def to_payload(self, *, examples: int = 20) -> dict[str, object]:
        def part(ids: tuple[str, ...]) -> dict[str, object]:
            return {"count": len(ids), "examples": sorted(ids)[:examples]}

        return {
            "matches": self.matches,
            "events": {
                "missing": part(self.events_missing), "changed": part(self.events_changed),
                "extra": part(self.events_extra),
            },
            "intervals": {"missing": part(self.intervals_missing), "extra": part(self.intervals_extra)},
            "mode": {"stored": self.mode_stored, "derived": self.mode_derived},
        }


def verify_tables(tables: Tables) -> Verification:
    """Compare every row the tables hold with what the tenant's ledger derives now. Reads only.

    Run it in one snapshot: an append between the two reads would show as a difference.
    """
    derived = derive_tables(tables.events(None))
    expected = derived.filed()
    filed, stored_intervals, mode = tables.stored()
    have = {row[0]: row for row in filed}
    wanted = derived.intervals()
    return Verification(
        tenant_id=tables.tenant_id,
        events_missing=tuple(event_id for event_id in expected if event_id not in have),
        events_changed=tuple(
            event_id for event_id, row in expected.items() if event_id in have and have[event_id] != row
        ),
        events_extra=tuple(event_id for event_id in have if event_id not in expected),
        intervals_missing=tuple(interval[0] for interval in wanted - stored_intervals),
        intervals_extra=tuple(interval[0] for interval in stored_intervals - wanted),
        mode_stored=mode or "table",
        mode_derived="replay" if derived.reasons else "table",
    )


def reconcile_tables(tables: Tables, *, limit: int | None = None) -> int:
    """Materialize the tenant's ledger events the tables lack (at most `limit`); return how many."""
    missing = tables.unfiled(limit=limit)
    if missing:
        materialize_events(tables, missing)
    return len(missing)


def _sqlite_stamp(moment: datetime) -> str:
    """Fixed width and UTC, as the lite ledger stamps its events, so text order is time order."""
    return _as_utc(moment).astimezone(UTC).isoformat(timespec="microseconds")


def _parse_stamp(value: object) -> datetime | None:
    """A stored SQLite stamp back to the aware datetime it was written from; None for a malformed
    one, which no derived interval equals, so it is reported rather than raised."""
    try:
        return _as_utc(datetime.fromisoformat(str(value)))
    except ValueError:
        return None


def _placeholders(count: int) -> str:
    return ", ".join("?" for _ in range(count))


#: SQLite's default host-parameter limit is 32,766; ids are fetched in batches well under it.
_SQLITE_BATCH = 900


class SQLiteTables:
    """`Tables` on a SQLite connection that holds the ledger: a lite file, or `--sqlite-path`'s."""

    def __init__(self, conn: sqlite3.Connection, tenant_id: str) -> None:
        self._conn = conn
        self.tenant_id = tenant_id

    def counts(self) -> tuple[int, int, str | None]:
        row = self._conn.execute(
            f"SELECT (SELECT count(*) FROM {FACT_LEDGER_TABLE} WHERE tenant_id = ?), "
            f"(SELECT count(*) FROM {CURRENT_EVENTS_TABLE} WHERE tenant_id = ?), "
            f"(SELECT mode FROM {CURRENT_STATE_TABLE} WHERE tenant_id = ?)",
            (self.tenant_id, self.tenant_id, self.tenant_id),
        ).fetchone()
        return int(row[0]), int(row[1]), row[2]

    def events(self, event_ids: Sequence[str] | None) -> list[FactEvent]:
        sql = f"SELECT {LEDGER_COLUMNS} FROM {FACT_LEDGER_TABLE} WHERE tenant_id = ?"
        if event_ids is None:
            rows = self._conn.execute(sql + " ORDER BY created_at, event_id", (self.tenant_id,)).fetchall()
            return [decode_ledger_row(row) for row in rows]
        found: list[tuple[Any, ...]] = []
        ids = list(event_ids)
        for start in range(0, len(ids), _SQLITE_BATCH):
            batch = ids[start:start + _SQLITE_BATCH]
            found.extend(self._conn.execute(
                sql + f" AND event_id IN ({_placeholders(len(batch))})", (self.tenant_id, *batch)
            ).fetchall())
        # Batches cannot be ordered in SQL; `created_at, event_id` under SQLite's BINARY collation
        # is Python's string order, so this is the ledger's `ORDER BY created_at, event_id`.
        found.sort(key=lambda row: (row[12], row[0]))
        return [decode_ledger_row(row) for row in found]

    def key_events(self, key: str) -> list[FactEvent] | None:
        rows = self._conn.execute(
            f"SELECT {', '.join('l.' + name.strip() for name in LEDGER_COLUMNS.split(','))}, "
            f"f.event_id FROM {CURRENT_EVENTS_TABLE} f LEFT JOIN {FACT_LEDGER_TABLE} l "
            "ON l.event_id = f.event_id AND l.tenant_id = f.tenant_id "
            "WHERE f.tenant_id = ? AND f.conflict_key = ? ORDER BY l.created_at, l.event_id",
            (self.tenant_id, key),
        ).fetchall()
        if any(row[0] is None for row in rows):
            return None
        return [decode_ledger_row(row[:-1]) for row in rows]

    def filed(self, event_ids: Sequence[str]) -> set[str]:
        ids = list(event_ids)
        out: set[str] = set()
        for start in range(0, len(ids), _SQLITE_BATCH):
            batch = ids[start:start + _SQLITE_BATCH]
            out.update(str(row[0]) for row in self._conn.execute(
                f"SELECT event_id FROM {CURRENT_EVENTS_TABLE} WHERE tenant_id = ? "
                f"AND event_id IN ({_placeholders(len(batch))})",
                (self.tenant_id, *batch),
            ).fetchall())
        return out

    def named_key(self, fact_id: str) -> tuple[bool, str | None]:
        row = self._conn.execute(
            f"SELECT fact FROM {FACT_LEDGER_TABLE} WHERE tenant_id = ? AND fact_id = ? "
            "AND fact IS NOT NULL ORDER BY created_at, event_id LIMIT 1",
            (self.tenant_id, fact_id),
        ).fetchone()
        return (False, None) if row is None else (True, _payload_key(row[0]))

    def file(self, rows: Sequence[tuple[str, str | None, str | None, str]]) -> None:
        self._conn.executemany(
            f"INSERT INTO {CURRENT_EVENTS_TABLE} (tenant_id, event_id, conflict_key, fact_id, event_type) "
            "VALUES (?, ?, ?, ?, ?) ON CONFLICT (tenant_id, event_id) DO NOTHING",
            [(self.tenant_id, *row) for row in rows],
        )

    def write_key(self, key: str, projection: KeyProjection) -> None:
        self._conn.execute(
            f"DELETE FROM {CURRENT_TABLE} WHERE tenant_id = ? AND conflict_key = ?", (self.tenant_id, key)
        )
        self._conn.executemany(
            f"INSERT INTO {CURRENT_TABLE} (tenant_id, conflict_key, valid_from, valid_until, event_id, fact_id) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (self.tenant_id, key, _sqlite_stamp(low), _sqlite_stamp(high), event_id, fact_id)
                for event_id, fact_id, low, high in projection.intervals
            ],
        )
        self._conn.execute(
            f"UPDATE {CURRENT_EVENTS_TABLE} SET live = 0 WHERE tenant_id = ? AND conflict_key = ? AND live = 1",
            (self.tenant_id, key),
        )
        live = sorted(projection.live)
        for start in range(0, len(live), _SQLITE_BATCH):
            batch = live[start:start + _SQLITE_BATCH]
            self._conn.execute(
                f"UPDATE {CURRENT_EVENTS_TABLE} SET live = 1 WHERE tenant_id = ? "
                f"AND event_id IN ({_placeholders(len(batch))})",
                (self.tenant_id, *batch),
            )

    def set_replay(self, reason: str) -> None:
        self._conn.execute(
            f"INSERT INTO {CURRENT_STATE_TABLE} (tenant_id, mode, reason, updated_at) VALUES (?, 'replay', ?, ?) "
            "ON CONFLICT (tenant_id) DO UPDATE SET mode = 'replay', reason = excluded.reason, "
            "updated_at = excluded.updated_at",
            (self.tenant_id, reason[:2000], _sqlite_stamp(datetime.now(UTC))),
        )

    def intervals_at(self, moment: datetime) -> list[tuple[str, str]] | None:
        stamp = _sqlite_stamp(moment)
        rows = self._conn.execute(
            f"SELECT c.event_id, c.fact_id, l.event_id FROM {CURRENT_TABLE} c "
            f"LEFT JOIN {FACT_LEDGER_TABLE} l ON l.event_id = c.event_id AND l.tenant_id = c.tenant_id "
            "WHERE c.tenant_id = ? AND c.valid_from <= ? AND c.valid_until > ? "
            "ORDER BY l.created_at, l.event_id",
            (self.tenant_id, stamp, stamp),
        ).fetchall()
        if any(row[2] is None for row in rows):
            return None
        return [(str(row[0]), str(row[1])) for row in rows]

    def support_ids(self, fact_ids: Sequence[str]) -> list[str]:
        out: list[str] = []
        for start in range(0, len(fact_ids), _SQLITE_BATCH):
            batch = list(fact_ids[start:start + _SQLITE_BATCH])
            out.extend(str(row[0]) for row in self._conn.execute(
                f"SELECT event_id FROM {CURRENT_EVENTS_TABLE} WHERE tenant_id = ? AND live = 1 "
                f"AND fact_id IN ({_placeholders(len(batch))})",
                (self.tenant_id, *batch),
            ).fetchall())
        return out

    def unfiled(self, *, limit: int | None = None) -> list[str]:
        return [str(row[0]) for row in self._conn.execute(
            f"SELECT l.event_id FROM {FACT_LEDGER_TABLE} l WHERE l.tenant_id = ? AND NOT EXISTS ("
            f"SELECT 1 FROM {CURRENT_EVENTS_TABLE} f WHERE f.tenant_id = l.tenant_id "
            "AND f.event_id = l.event_id) ORDER BY l.created_at, l.event_id" + (" LIMIT ?" if limit is not None else ""),
            (self.tenant_id, limit) if limit is not None else (self.tenant_id,),
        ).fetchall()]

    def stored(self) -> tuple[set[FiledRow], set[StoredInterval], str | None]:
        filed = {
            (str(event_id), key, fact_id, str(event_type), bool(live))
            for event_id, key, fact_id, event_type, live in self._conn.execute(
                f"SELECT event_id, conflict_key, fact_id, event_type, live FROM {CURRENT_EVENTS_TABLE} "
                "WHERE tenant_id = ?", (self.tenant_id,),
            ).fetchall()
        }
        intervals = {
            (str(event_id), str(fact_id), str(key), _parse_stamp(low), _parse_stamp(high))
            for event_id, fact_id, key, low, high in self._conn.execute(
                f"SELECT event_id, fact_id, conflict_key, valid_from, valid_until FROM {CURRENT_TABLE} "
                "WHERE tenant_id = ?", (self.tenant_id,),
            ).fetchall()
        }
        return filed, intervals, self.counts()[2]

    def clear(self) -> None:
        for table in (CURRENT_TABLE, CURRENT_EVENTS_TABLE, CURRENT_STATE_TABLE):
            self._conn.execute(f"DELETE FROM {table} WHERE tenant_id = ?", (self.tenant_id,))

    def lock_keys(self, keys: Sequence[str]) -> None:
        """Nothing: a SQLite writer holds the whole file (`BEGIN IMMEDIATE`)."""

    def lock_tenant(self, *, exclusive: bool) -> None:
        """Nothing: a SQLite writer holds the whole file (`BEGIN IMMEDIATE`)."""


def fact_key_lock(tenant_id: str, key: str) -> str:
    """The advisory lock string every PostgreSQL writer of `key` holds (`PostgresFactLedger`)."""
    return f"fact\x1f{tenant_id}\x1f{key}"


class PostgresTables:
    """`Tables` on a PostgreSQL connection with the tenant GUC set; the caller owns the transaction."""

    def __init__(self, conn: psycopg.Connection, tenant_id: str) -> None:
        self._conn = conn
        self.tenant_id = tenant_id

    def counts(self) -> tuple[int, int, str | None]:
        row = self._conn.execute(
            f"SELECT (SELECT count(*) FROM {FACT_LEDGER_TABLE} WHERE tenant_id = %s), "
            f"(SELECT count(*) FROM {CURRENT_EVENTS_TABLE} WHERE tenant_id = %s), "
            f"(SELECT mode FROM {CURRENT_STATE_TABLE} WHERE tenant_id = %s)",
            (self.tenant_id, self.tenant_id, self.tenant_id),
        ).fetchone()
        assert row is not None
        return int(row[0]), int(row[1]), row[2]

    def events(self, event_ids: Sequence[str] | None) -> list[FactEvent]:
        sql = f"SELECT {LEDGER_COLUMNS} FROM {FACT_LEDGER_TABLE} WHERE tenant_id = %s"
        params: list[Any] = [self.tenant_id]
        if event_ids is not None:
            sql += " AND event_id = ANY(%s::text[])"
            params.append(list(event_ids))
        rows = self._conn.execute(sql + " ORDER BY created_at, event_id", params).fetchall()
        return [_event_from_row(row) for row in rows]

    def key_events(self, key: str) -> list[FactEvent] | None:
        columns = ", ".join("l." + name.strip() for name in LEDGER_COLUMNS.split(","))
        rows = self._conn.execute(
            f"SELECT {columns}, f.event_id FROM {CURRENT_EVENTS_TABLE} f "
            f"LEFT JOIN {FACT_LEDGER_TABLE} l ON l.event_id = f.event_id AND l.tenant_id = f.tenant_id "
            "WHERE f.tenant_id = %s AND f.conflict_key = %s ORDER BY l.created_at, l.event_id",
            (self.tenant_id, key),
        ).fetchall()
        if any(row[0] is None for row in rows):
            return None
        return [_event_from_row(row[:-1]) for row in rows]

    def filed(self, event_ids: Sequence[str]) -> set[str]:
        return {str(row[0]) for row in self._conn.execute(
            f"SELECT event_id FROM {CURRENT_EVENTS_TABLE} WHERE tenant_id = %s AND event_id = ANY(%s::text[])",
            (self.tenant_id, list(event_ids)),
        ).fetchall()}

    def named_key(self, fact_id: str) -> tuple[bool, str | None]:
        row = self._conn.execute(
            f"SELECT fact FROM {FACT_LEDGER_TABLE} WHERE tenant_id = %s AND fact_id = %s "
            "AND fact IS NOT NULL ORDER BY created_at, event_id LIMIT 1",
            (self.tenant_id, fact_id),
        ).fetchone()
        return (False, None) if row is None else (True, _payload_key(row[0]))

    def file(self, rows: Sequence[tuple[str, str | None, str | None, str]]) -> None:
        if not rows:
            return
        with self._conn.cursor() as cursor:
            cursor.executemany(
                f"INSERT INTO {CURRENT_EVENTS_TABLE} (tenant_id, event_id, conflict_key, fact_id, event_type) "
                "VALUES (%s, %s, %s, %s, %s) ON CONFLICT (tenant_id, event_id) DO NOTHING",
                [(self.tenant_id, *row) for row in rows],
            )

    def write_key(self, key: str, projection: KeyProjection) -> None:
        self._conn.execute(
            f"DELETE FROM {CURRENT_TABLE} WHERE tenant_id = %s AND conflict_key = %s", (self.tenant_id, key)
        )
        if projection.intervals:
            with self._conn.cursor() as cursor:
                cursor.executemany(
                    f"INSERT INTO {CURRENT_TABLE} "
                    "(tenant_id, conflict_key, valid_from, valid_until, event_id, fact_id) "
                    "VALUES (%s, %s, %s, %s, %s, %s)",
                    [
                        (self.tenant_id, key, low, high, event_id, fact_id)
                        for event_id, fact_id, low, high in projection.intervals
                    ],
                )
        self._conn.execute(
            f"UPDATE {CURRENT_EVENTS_TABLE} SET live = (event_id = ANY(%s::text[])) "
            "WHERE tenant_id = %s AND conflict_key = %s AND live IS DISTINCT FROM (event_id = ANY(%s::text[]))",
            (sorted(projection.live), self.tenant_id, key, sorted(projection.live)),
        )

    def set_replay(self, reason: str) -> None:
        self._conn.execute(
            f"INSERT INTO {CURRENT_STATE_TABLE} (tenant_id, mode, reason) VALUES (%s, 'replay', %s) "
            "ON CONFLICT (tenant_id) DO UPDATE SET mode = 'replay', reason = EXCLUDED.reason, "
            "updated_at = clock_timestamp()",
            (self.tenant_id, reason[:2000]),
        )

    def intervals_at(self, moment: datetime) -> list[tuple[str, str]] | None:
        instant = _as_utc(moment)
        rows = self._conn.execute(
            f"SELECT c.event_id, c.fact_id, l.event_id FROM {CURRENT_TABLE} c "
            f"LEFT JOIN {FACT_LEDGER_TABLE} l ON l.event_id = c.event_id AND l.tenant_id = c.tenant_id "
            "WHERE c.tenant_id = %s AND c.valid_from <= %s AND c.valid_until > %s "
            "ORDER BY l.created_at, l.event_id",
            (self.tenant_id, instant, instant),
        ).fetchall()
        if any(row[2] is None for row in rows):
            return None
        return [(str(row[0]), str(row[1])) for row in rows]

    def support_ids(self, fact_ids: Sequence[str]) -> list[str]:
        return [str(row[0]) for row in self._conn.execute(
            f"SELECT event_id FROM {CURRENT_EVENTS_TABLE} WHERE tenant_id = %s AND live "
            "AND fact_id = ANY(%s::text[])",
            (self.tenant_id, list(fact_ids)),
        ).fetchall()]

    def unfiled(self, *, limit: int | None = None) -> list[str]:
        return [str(row[0]) for row in self._conn.execute(
            f"SELECT l.event_id FROM {FACT_LEDGER_TABLE} l WHERE l.tenant_id = %s AND NOT EXISTS ("
            f"SELECT 1 FROM {CURRENT_EVENTS_TABLE} f WHERE f.tenant_id = l.tenant_id "
            "AND f.event_id = l.event_id) ORDER BY l.created_at, l.event_id" + (" LIMIT %s" if limit is not None else ""),
            (self.tenant_id, limit) if limit is not None else (self.tenant_id,),
        ).fetchall()]

    def stored(self) -> tuple[set[FiledRow], set[StoredInterval], str | None]:
        filed = {
            (str(event_id), key, fact_id, str(event_type), bool(live))
            for event_id, key, fact_id, event_type, live in self._conn.execute(
                f"SELECT event_id, conflict_key, fact_id, event_type, live FROM {CURRENT_EVENTS_TABLE} "
                "WHERE tenant_id = %s", (self.tenant_id,),
            ).fetchall()
        }
        # Read in UTC without a zone, so the year 1 and 9999 bounds load in any session time zone.
        intervals = {
            (str(event_id), str(fact_id), str(key), low.replace(tzinfo=UTC), high.replace(tzinfo=UTC))
            for event_id, fact_id, key, low, high in self._conn.execute(
                "SELECT event_id, fact_id, conflict_key, valid_from AT TIME ZONE 'UTC', "
                f"valid_until AT TIME ZONE 'UTC' FROM {CURRENT_TABLE} WHERE tenant_id = %s",
                (self.tenant_id,),
            ).fetchall()
        }
        return filed, intervals, self.counts()[2]

    def clear(self) -> None:
        for table in (CURRENT_TABLE, CURRENT_EVENTS_TABLE, CURRENT_STATE_TABLE):
            self._conn.execute(f"DELETE FROM {table} WHERE tenant_id = %s", (self.tenant_id,))

    def lock_keys(self, keys: Sequence[str]) -> None:
        for key in keys:
            self._conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (fact_key_lock(self.tenant_id, key),)
            )

    def lock_tenant(self, *, exclusive: bool) -> None:
        function = "pg_advisory_xact_lock" if exclusive else "pg_advisory_xact_lock_shared"
        self._conn.execute(
            f"SELECT {function}(hashtextextended(%s, 0))", (f"fact-current\x1f{self.tenant_id}",)
        )


#: The errors a read of the tables degrades on: a database migrated before 0030, a role not yet
#: granted them. The read replays and says so once per process; anything else raises.
_DEGRADE = (psycopg.errors.UndefinedTable, psycopg.errors.InsufficientPrivilege)
_warned: set[str] = set()


def _warn_once(what: str, exc: BaseException) -> None:
    if what not in _warned:
        _warned.add(what)
        _log.warning("current-facts tables unavailable for %s (%s); reads replay the ledger", what, exc)


def pg_current_view(
    conn: psycopg.Connection, tenant_id: str, now: datetime, served: ServedSources | None
) -> CurrentView | None:
    """`current_view` in one REPEATABLE READ snapshot of an autocommit connection, or None."""
    try:
        with conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            return current_view(PostgresTables(conn, tenant_id), now, served)
    except _DEGRADE as exc:
        _warn_once("reads", exc)
        return None


def pg_key_history(conn: psycopg.Connection, tenant_id: str, fact: AtomicFact) -> list[FactEvent] | None:
    """`key_history` inside the caller's transaction, in a savepoint, or None."""
    try:
        with conn.transaction():
            tables = PostgresTables(conn, tenant_id)
            # Writers already hold the key lock; purge/rebuild holds the tenant exclusively.
            # Retain its shared lock through the append, including this freshness check.
            tables.lock_tenant(exclusive=False)
            return key_history(tables, fact)
    except _DEGRADE as exc:
        _warn_once("conflict checks", exc)
        return None


def pg_materialize_appended(conn: psycopg.Connection, tenant_id: str, events: Iterable[FactEvent]) -> bool:
    """Materialize events the caller just appended, inside its transaction, in a savepoint.

    Returns whether they were. A failure is logged and rolled back to the savepoint, never raised:
    the append is the durable decision and the tables are derived from it, so they lag (and reads
    replay) rather than fail it.
    """
    ids = [event.event_id for event in events]
    if not ids:
        return True
    try:
        with conn.transaction():
            materialize_events(PostgresTables(conn, tenant_id), ids)
        return True
    except Exception as exc:  # BROAD-CATCH: fail-open  # derived state; reads replay until reconciled
        _log.warning("could not materialize %d fact event(s) for tenant %s: %s", len(ids), tenant_id, exc)
        return False


def sqlite_materialize_appended(conn: sqlite3.Connection, tenant_id: str, events: Iterable[FactEvent]) -> bool:
    """`pg_materialize_appended` for a SQLite connection inside its write transaction."""
    ids = [event.event_id for event in events]
    if not ids:
        return True
    conn.execute("SAVEPOINT recall_fact_current")
    try:
        materialize_events(SQLiteTables(conn, tenant_id), ids)
    except Exception as exc:  # BROAD-CATCH: fail-open  # derived state; reads replay until reconciled
        if not conn.in_transaction:
            raise  # SQLite ended the whole transaction (a full disk): the append is gone too
        conn.execute("ROLLBACK TO SAVEPOINT recall_fact_current")
        conn.execute("RELEASE SAVEPOINT recall_fact_current")
        _log.warning("could not materialize %d fact event(s) for tenant %s: %s", len(ids), tenant_id, exc)
        return False
    conn.execute("RELEASE SAVEPOINT recall_fact_current")
    return True


@dataclass(frozen=True)
class CurrentFactsStatus:
    """What `recall provenance rebuild --check` reports for a tenant."""

    tenant_id: str
    ledger_events: int
    filed_events: int
    intervals: int
    mode: str
    reason: str | None

    @property
    def current(self) -> bool:
        return self.ledger_events == self.filed_events and self.mode != "replay"

    def to_payload(self) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id, "ledger_events": self.ledger_events,
            "filed_events": self.filed_events, "intervals": self.intervals, "mode": self.mode,
            "reason": self.reason, "current": self.current,
            "reads": "table" if self.current else "replay",
        }


class PostgresCurrentFacts:
    """The current-facts `FactMaterializer` on PostgreSQL, with its rebuild, reconcile and purge.

    `materialize` is idempotent by event id and reads the event from the ledger, so delivering it
    through the outbox (`MaterializationRecovery`) twice, or after the ledger's own in-transaction
    materialization, changes nothing. `materializes` names every event type, so `reconcile`
    enqueues refusals and retractions as well as assertions.
    """

    materializes = MATERIALIZED_EVENT_TYPES

    def __init__(self, dsn: str, *, tenant_id: str, statement_timeout_ms: int = 60000) -> None:
        if not dsn or not tenant_id:
            raise ValueError("dsn and tenant_id must be non-empty")
        self.dsn = dsn
        self.tenant_id = tenant_id
        self.statement_timeout_ms = statement_timeout_ms

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.dsn, autocommit=True, connect_timeout=10)
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (self.tenant_id,))
        conn.execute(f"SET statement_timeout = {int(self.statement_timeout_ms)}")
        return conn

    def materialize(self, event: FactEvent) -> None:
        if event.tenant_id != self.tenant_id:
            raise ValueError("materializer tenant mismatch")
        with self._connect() as conn, conn.transaction():
            materialize_events(PostgresTables(conn, self.tenant_id), (event.event_id,))

    def unfiled_events(self) -> tuple[FactEvent, ...]:
        """The tenant's ledger events the tables lack, for `MaterializationRecovery.reconcile`."""
        with self._connect() as conn:
            tables = PostgresTables(conn, self.tenant_id)
            return tuple(tables.events(tables.unfiled()))

    def reconcile(self, *, limit: int | None = None) -> int:
        """Materialize missing events, at most `limit`, in one transaction; return how many."""
        with self._connect() as conn, conn.transaction():
            return reconcile_tables(PostgresTables(conn, self.tenant_id), limit=limit)

    def rebuild(self) -> CurrentFactsStatus:
        """Replace the tenant's rows with ones derived from its ledger, in one transaction."""
        with self._connect() as conn:
            with conn.transaction():
                rebuild_tables(PostgresTables(conn, self.tenant_id))
            return self._status(conn)

    def purge(self) -> None:
        """Delete the tenant's rows: reads replay until `rebuild`. The hook an erasure calls.

        `recall_forget`'s redaction (PR 950) rewrites event ids, fact ids and payloads in place,
        so the rows naming the old ones must go in the same transaction; `purge_in` and
        `rebuild_in` take the caller's connection for that.
        """
        with self._connect() as conn, conn.transaction():
            self.purge_in(conn)

    def purge_in(self, conn: psycopg.Connection) -> None:
        tables = PostgresTables(conn, self.tenant_id)
        tables.lock_tenant(exclusive=True)
        tables.clear()

    def rebuild_in(self, conn: psycopg.Connection) -> None:
        rebuild_tables(PostgresTables(conn, self.tenant_id))

    def status(self) -> CurrentFactsStatus:
        with self._connect() as conn:
            return self._status(conn)

    def verify(self) -> Verification:
        """`verify_tables` in one REPEATABLE READ snapshot."""
        with self._connect() as conn, conn.transaction():
            conn.execute("SET TRANSACTION ISOLATION LEVEL REPEATABLE READ, READ ONLY")
            return verify_tables(PostgresTables(conn, self.tenant_id))

    def _status(self, conn: psycopg.Connection) -> CurrentFactsStatus:
        ledger, filed, mode = PostgresTables(conn, self.tenant_id).counts()
        row = conn.execute(
            f"SELECT count(*), (SELECT reason FROM {CURRENT_STATE_TABLE} WHERE tenant_id = %s) "
            f"FROM {CURRENT_TABLE} WHERE tenant_id = %s",
            (self.tenant_id, self.tenant_id),
        ).fetchone()
        assert row is not None
        return CurrentFactsStatus(self.tenant_id, ledger, filed, int(row[0]), mode or "table", row[1])


class SQLiteCurrentFacts:
    """`PostgresCurrentFacts` for a SQLite file that holds the ledger, on a connection it is given.

    The lite store hands its own connection and lock (`recall.lite.facts`); every method runs in
    one `BEGIN IMMEDIATE` transaction the caller's `write` opens.
    """

    materializes = MATERIALIZED_EVENT_TYPES

    def __init__(
        self, conn: sqlite3.Connection, *, tenant_id: str, write: Callable[[], Any] | None = None
    ) -> None:
        self._conn = conn
        self.tenant_id = tenant_id
        self._write = write

    def _transaction(self) -> Any:
        if self._write is not None:
            return self._write()
        return _SQLiteImmediate(self._conn)

    def materialize(self, event: FactEvent) -> None:
        if event.tenant_id != self.tenant_id:
            raise ValueError("materializer tenant mismatch")
        with self._transaction():
            materialize_events(SQLiteTables(self._conn, self.tenant_id), (event.event_id,))

    def reconcile(self) -> int:
        with self._transaction():
            return reconcile_tables(SQLiteTables(self._conn, self.tenant_id))

    def rebuild(self) -> CurrentFactsStatus:
        with self._transaction():
            rebuild_tables(SQLiteTables(self._conn, self.tenant_id))
        return self.status()

    def purge(self) -> None:
        with self._transaction():
            SQLiteTables(self._conn, self.tenant_id).clear()

    def verify(self) -> Verification:
        """`verify_tables` in one read transaction (or the caller's, when one is open)."""
        if self._conn.in_transaction:
            return verify_tables(SQLiteTables(self._conn, self.tenant_id))
        self._conn.execute("BEGIN")
        try:
            return verify_tables(SQLiteTables(self._conn, self.tenant_id))
        finally:
            if self._conn.in_transaction:
                self._conn.execute("ROLLBACK")

    def status(self) -> CurrentFactsStatus:
        ledger, filed, mode = SQLiteTables(self._conn, self.tenant_id).counts()
        row = self._conn.execute(
            f"SELECT count(*), (SELECT reason FROM {CURRENT_STATE_TABLE} WHERE tenant_id = ?) "
            f"FROM {CURRENT_TABLE} WHERE tenant_id = ?",
            (self.tenant_id, self.tenant_id),
        ).fetchone()
        return CurrentFactsStatus(self.tenant_id, ledger, filed, int(row[0]), mode or "table", row[1])


class _SQLiteImmediate:
    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn

    def __enter__(self) -> None:
        self._conn.execute("BEGIN IMMEDIATE")

    def __exit__(self, exc_type: object, *_: object) -> None:
        if not self._conn.in_transaction:
            return
        self._conn.execute("ROLLBACK" if exc_type is not None else "COMMIT")


__all__ = [
    "CURRENT_EVENTS_TABLE", "CURRENT_STATE_TABLE", "CURRENT_TABLE", "CurrentFactsStatus",
    "CurrentView", "KeyProjection", "MATERIALIZED_EVENT_TYPES", "PostgresCurrentFacts",
    "PostgresTables", "SQLITE_FACT_CURRENT_DDL", "SQLiteCurrentFacts", "SQLiteTables",
    "DerivedTables", "Verification", "current_view", "derive_tables", "event_keys", "key_history",
    "materialize_events", "project_key", "rebuild_tables", "reconcile_tables", "usable",
    "verify_tables",
]
