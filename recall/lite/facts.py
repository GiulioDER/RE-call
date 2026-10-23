"""The fact ledger and evidence cards on a lite store: `PostgresFactLedger`'s rules on one SQLite file.

`SQLiteFactLedger` and `SQLiteEvidenceCardStore` (behind `recall provenance --sqlite-path`) open a
connection of their own, serve `current` and `events` from a snapshot taken at open (their writes
reload it), and follow the in-memory ledger's event ids. These adapters live on the `LiteStore`'s connection instead: they read the file on every call,
write in one `LiteStore._write()` transaction each, and mint the same event ids and the same
projection as PostgreSQL, so a client of `recall_apply_fact` and `recall_current_facts` cannot tell
the backends apart. Table names and columns are those of the SQLite adapters.
"""
# The SQL identifiers in this module are fixed module constants; values are bound parameters.
# ruff: noqa: S608
from __future__ import annotations

import json
import sqlite3
from collections.abc import Callable, Iterable
from datetime import UTC, datetime
from typing import Any

from recall.fact_current import (
    SQLITE_FACT_CURRENT_DDL,
    SQLiteTables,
    current_view,
    key_history,
    sqlite_materialize_appended,
)
from recall.fact_ledger import (
    FACT_LEDGER_TABLE,
    LEDGER_COLUMNS,
    RETRACTION_CODES,
    RETRACTION_WRITER,
    WITHDRAWAL_WRITER,
    SQLITE_FACT_LEDGER_DDL,
    assertion_event_id,
    assertion_is_duplicate,
    decision_event_id,
    decision_event_type,
    decode_ledger_row,
    plan_retractions,
    plan_withdrawals,
    project_candidates,
    project_current,
    retraction_ids,
    withdrawal_ids,
    supersession_ids,
)
from recall.lite.store import LiteStore
from recall.observability import get_logger
from recall.provenance_cards import (
    EVIDENCE_CARD_TABLE,
    SQLITE_EVIDENCE_CARD_DDL,
    _sqlite_put_cards,
    _sqlite_resolved_card,
)
from recall.provenance_controller import (
    CONTROLLER_POLICY_VERSION,
    CONTROLLER_SCHEMA_VERSION,
    DecisionCode,
    FactApplicationPermit,
    FactEvent,
    LedgerApplyResult,
    ServedSources,
    _as_utc,
    _consume_permit,
    absent_fact_ids,
    fact_identity,
    fact_supersession,
    ordered_after,
    replays_request,
)
from recall.types import AtomicFact, EvidenceCard

_SCHEMA = SQLITE_FACT_LEDGER_DDL + SQLITE_EVIDENCE_CARD_DDL
_log = get_logger("lite.facts")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _utc(moment: datetime) -> datetime:
    """`moment` in UTC: a naive one read as UTC, an aware one converted, as PostgreSQL's
    `timestamptz` stores every instant."""
    return _as_utc(moment).astimezone(UTC)


def _stamp(moment: datetime) -> str:
    """Fixed width and always UTC, so text order is time order (plain `isoformat()` drops zero
    microseconds, and a `+02:00` stamp sorts by its local clock, not its instant)."""
    return _utc(moment).isoformat(timespec="microseconds")


def ensure_fact_tables(store: LiteStore) -> None:
    """Create the fact tables in a file made before they existed.

    Outside `_write()` on purpose: `executescript` commits any open transaction first, and every
    statement is `IF NOT EXISTS`, so running it on each adapter construction costs a catalog read.
    A ledger table made before `retracted` or `withdrawn` existed is upgraded in place
    (`_admit_retraction`).
    The current-facts tables (`recall.fact_current`) come last, because that upgrade rebuilds the
    ledger table and drops the index they add to it.
    """
    store.ensure_schema(_SCHEMA)
    if not _admits_retraction(store):
        _admit_retraction(store)
    store.ensure_schema(SQLITE_FACT_CURRENT_DDL)


def _admits_retraction(store: LiteStore) -> bool:
    rows = store._metadata_values(
        "SELECT sql FROM sqlite_master WHERE type = 'table' AND name = ?", (FACT_LEDGER_TABLE,)
    )
    # `withdrawn` is the newest type the CHECK admits; a table that admits it admits `retracted`.
    return not rows or "'withdrawn'" in str(rows[0][0])


def _ledger_statements() -> list[str]:
    """`SQLITE_FACT_LEDGER_DDL` one statement at a time (a trigger body holds its own `;`)."""
    statements: list[str] = []
    pending = ""
    for line in SQLITE_FACT_LEDGER_DDL.splitlines(keepends=True):
        pending += line
        if sqlite3.complete_statement(pending):
            statements.append(pending.strip())
            pending = ""
    return statements


def _admit_retraction(store: LiteStore) -> None:
    """Rebuild a ledger table whose CHECK predates `retracted` or `withdrawn`, in one transaction,
    keeping its rows.

    SQLite cannot alter a CHECK, so the table is renamed, recreated from the current DDL, refilled
    and the old one dropped; the append-only triggers are recreated with it. A row the current
    CHECKs refuse (a file older than the fact-shape CHECK can hold one) rolls the whole rebuild
    back: the ledger stays as it was and serves as before, and only a retraction fails, which
    `recall_forget` reports.
    """
    old = FACT_LEDGER_TABLE + "_before_retraction"
    try:
        with store._write():
            if _admits_retraction(store):  # another handle upgraded it first
                return
            conn = store._conn
            conn.execute(f"ALTER TABLE {FACT_LEDGER_TABLE} RENAME TO {old}")
            for name in ("recall_fact_ledger_no_update", "recall_fact_ledger_no_delete"):
                conn.execute(f"DROP TRIGGER IF EXISTS {name}")
            conn.execute("DROP INDEX IF EXISTS recall_fact_ledger_current_idx")
            for statement in _ledger_statements():
                conn.execute(statement)
            columns = LEDGER_COLUMNS + ", fact_id"
            conn.execute(f"INSERT INTO {FACT_LEDGER_TABLE} ({columns}) SELECT {columns} FROM {old}")
            conn.execute(f"DROP TABLE {old}")
    except sqlite3.IntegrityError:
        _log.warning(
            "the fact ledger in %s holds a row the current schema refuses; it was not upgraded",
            store.path,
        )


class LiteFactLedger:
    """The `FactLedger` protocol on a lite store, with `PostgresFactLedger`'s semantics.

    Like it, reads and conflict checks come from the materialized current facts when those reflect
    every event in the file, and every append materializes what it wrote in its own transaction
    unless `materialize` is off. `last_read` says which way the latest read went.
    """

    def __init__(self, store: LiteStore, *, materialize: bool = True) -> None:
        self._store = store
        self.tenant_id = store.tenant
        self.materialize = materialize
        self.last_read: str | None = None
        ensure_fact_tables(store)

    def _check(self, tenant_id: str) -> None:
        if tenant_id != self.tenant_id:
            raise ValueError("ledger tenant mismatch")

    def _events(self, *, request_id: str | None = None) -> list[FactEvent]:
        sql = f"SELECT {LEDGER_COLUMNS} FROM {FACT_LEDGER_TABLE} WHERE tenant_id = ?"
        params: list[Any] = [self.tenant_id]
        if request_id is not None:
            sql += " AND request_id = ?"
            params.append(request_id)
        rows = self._store._metadata_values(sql + " ORDER BY created_at, event_id", params)
        return [decode_ledger_row(row) for row in rows]

    @property
    def events(self) -> tuple[FactEvent, ...]:
        """The tenant's immutable event stream, read from the file."""
        return tuple(self._events())

    def current(
        self, *, tenant_id: str, now: datetime, served: ServedSources | None = None
    ) -> tuple[FactEvent, ...]:
        return self.current_view(tenant_id=tenant_id, now=now, served=served)[0]

    def current_view(
        self, *, tenant_id: str, now: datetime, served: ServedSources | None = None
    ) -> tuple[tuple[FactEvent, ...], tuple[FactEvent, ...]]:
        """`PostgresFactLedger.current_view` on the file, in one read snapshot."""
        self._check(tenant_id)
        with self._store.snapshot():
            view = current_view(SQLiteTables(self._store._conn, self.tenant_id), now, served)
            events = self._events() if view is None else []
        if view is not None:
            self.last_read = "table"
            return view.current, view.support
        self.last_read = "replay"
        return project_current(events, now, served), tuple(events)

    def conflict_history(self, *, tenant_id: str, fact: AtomicFact) -> tuple[FactEvent, ...]:
        """`PostgresFactLedger.conflict_history` on the file, in one read snapshot."""
        self._check(tenant_id)
        with self._store.snapshot():
            return tuple(self._history(fact))

    def _history(self, fact: AtomicFact) -> list[FactEvent]:
        history = key_history(SQLiteTables(self._store._conn, self.tenant_id), fact)
        self.last_read = "replay" if history is None else "table"
        return self._events() if history is None else history

    def _materialized(self, events: tuple[FactEvent, ...] | list[FactEvent]) -> None:
        """Materialize what the open write transaction appended; never raises for the tables."""
        if self.materialize:
            sqlite_materialize_appended(self._store._conn, self.tenant_id, events)

    def request_event(self, *, tenant_id: str, request_id: str) -> FactEvent | None:
        self._check(tenant_id)
        events = self._events(request_id=request_id)
        return events[0] if events else None

    def _insert(
        self,
        *,
        event_id: str,
        event_type: str,
        generation_id: str,
        fact: AtomicFact | None,
        cards: tuple[EvidenceCard, ...],
        supersedes_fact_ids: tuple[str, ...],
        request_id: str,
        writer: str,
        decision_code: str,
        now: datetime,
        policy_version: str = CONTROLLER_POLICY_VERSION,
    ) -> FactEvent:
        fact_id = fact_identity(fact) if fact else None
        self._store._conn.execute(
            f"INSERT INTO {FACT_LEDGER_TABLE} ({LEDGER_COLUMNS}, fact_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                event_id,
                event_type,
                self.tenant_id,
                generation_id,
                _json(fact.to_payload()) if fact else None,
                _json([card.to_payload() for card in cards]),
                _json(list(supersedes_fact_ids)),
                request_id,
                writer,
                str(decision_code),
                policy_version,
                CONTROLLER_SCHEMA_VERSION,
                _stamp(now),
                fact_id,
            ),
        )
        return FactEvent(
            event_id=event_id, event_type=event_type, tenant_id=self.tenant_id,
            generation_id=generation_id, fact=fact, fact_id=fact_id, evidence_cards=cards,
            supersedes_fact_ids=supersedes_fact_ids, request_id=request_id, writer=writer,
            decision_code=str(decision_code), policy_version=policy_version,
            controller_version=CONTROLLER_SCHEMA_VERSION, created_at=_utc(now),
        )

    def apply_assertion(
        self,
        *,
        tenant_id: str,
        generation_id: str,
        fact: AtomicFact,
        cards: tuple[EvidenceCard, ...],
        request_id: str,
        writer: str,
        permit: FactApplicationPermit | None = None,
        supersedes_fact_ids: tuple[str, ...] = (),
        policy_version: str = CONTROLLER_POLICY_VERSION,
        controller_version: int = CONTROLLER_SCHEMA_VERSION,
        now: datetime | None = None,
        served: ServedSources | None = None,
        stated: bool = False,
    ) -> LedgerApplyResult:
        del controller_version
        self._check(tenant_id)
        moment = _as_utc(now or datetime.now(UTC))
        event_id = assertion_event_id(self.tenant_id, fact, request_id)
        # One BEGIN IMMEDIATE transaction holds the request check, the history read and the
        # insert: what the per-conflict-key advisory lock gives `PostgresFactLedger.apply_assertion`
        # (here it serializes every writer of the file, a superset of that).
        with self._store._write():
            _consume_permit(
                permit, tenant_id=tenant_id, generation_id=generation_id,
                request_id=request_id, fact=fact, cards=cards,
            )
            prior = self._events(request_id=request_id)
            if prior:
                if not replays_request(prior[0], fact, cards):
                    raise ValueError(DecisionCode.REQUEST_ID_REUSED)
                return LedgerApplyResult(prior[0], duplicate=True)
            history = self._history(fact)
            index = fact_supersession(history)
            candidates = project_candidates(history, index=index)
            moment = ordered_after(moment, history, fact)
            if assertion_is_duplicate(
                candidates, fact, supersedes_fact_ids, absent_fact_ids(history, served, moment),
                stated=stated, index=index, history=history, moment=moment,
            ):
                event = self._insert(
                    event_id=event_id, event_type="asserted", generation_id=generation_id,
                    fact=fact, cards=cards, supersedes_fact_ids=(), request_id=request_id,
                    writer=writer, decision_code=DecisionCode.DUPLICATE, now=moment,
                    policy_version=policy_version,
                )
                self._materialized((event,))
                return LedgerApplyResult(event, duplicate=True)
            event = self._insert(
                event_id=event_id, event_type="asserted", generation_id=generation_id, fact=fact,
                cards=cards, supersedes_fact_ids=supersedes_fact_ids, request_id=request_id,
                writer=writer, decision_code=DecisionCode.APPLIED, now=moment,
                policy_version=policy_version,
            )
            appended = [event]
            if supersedes_fact_ids:
                sup_event_id, sup_request_id = supersession_ids(event_id, request_id)
                appended.append(self._insert(
                    event_id=sup_event_id, event_type="superseded",
                    generation_id=generation_id, fact=None, cards=cards,
                    supersedes_fact_ids=supersedes_fact_ids,
                    request_id=sup_request_id, writer=writer,
                    decision_code=DecisionCode.APPLIED, now=moment,
                    policy_version=policy_version,
                ))
            self._materialized(appended)
            return LedgerApplyResult(event)

    def retract_forgotten(
        self,
        *,
        tenant_id: str,
        touched: Callable[[EvidenceCard], bool],
        gone: Callable[[EvidenceCard], bool],
        writer: str = RETRACTION_WRITER,
        now: datetime | None = None,
        decision_code: str = DecisionCode.SOURCE_FORGOTTEN,
    ) -> tuple[FactEvent, ...]:
        """`PostgresFactLedger.retract_forgotten` on the file: plan and append in one transaction."""
        self._check(tenant_id)
        if decision_code not in RETRACTION_CODES:
            raise ValueError(f"{decision_code!r} is not a retraction code")
        moment = _as_utc(now or datetime.now(UTC))
        written: list[FactEvent] = []
        with self._store._write():
            for assertion in plan_retractions(self._events(), touched=touched, gone=gone):
                event_id, request_id = retraction_ids(self.tenant_id, assertion)
                written.append(self._insert(
                    event_id=event_id, event_type="retracted", generation_id=assertion.generation_id,
                    fact=None, cards=assertion.evidence_cards,
                    supersedes_fact_ids=(str(assertion.fact_id),), request_id=request_id,
                    writer=writer, decision_code=decision_code, now=moment,
                ))
            self._materialized(written)
        return tuple(written)

    def withdraw_forgotten(
        self,
        *,
        tenant_id: str,
        touched: Callable[[EvidenceCard], bool],
        gone: Callable[[EvidenceCard], bool],
        writer: str = WITHDRAWAL_WRITER,
        now: datetime | None = None,
    ) -> tuple[FactEvent, ...]:
        """`PostgresFactLedger.withdraw_forgotten` on the file: plan and append in one transaction."""
        self._check(tenant_id)
        moment = _as_utc(now or datetime.now(UTC))
        written: list[FactEvent] = []
        with self._store._write():
            for assertion in plan_withdrawals(self._events(), touched=touched, gone=gone):
                event_id, request_id = withdrawal_ids(self.tenant_id, assertion)
                written.append(self._insert(
                    event_id=event_id, event_type="withdrawn", generation_id=assertion.generation_id,
                    fact=None, cards=(), supersedes_fact_ids=(str(assertion.fact_id),),
                    request_id=request_id, writer=writer, decision_code=DecisionCode.WITHDRAWN,
                    now=moment,
                ))
            self._materialized(written)
        return tuple(written)

    def record_decision(
        self,
        *,
        tenant_id: str,
        generation_id: str,
        request_id: str,
        writer: str,
        decision_code: str,
        fact: AtomicFact | None,
        cards: tuple[EvidenceCard, ...] = (),
        now: datetime | None = None,
    ) -> FactEvent:
        self._check(tenant_id)
        moment = _as_utc(now or datetime.now(UTC))
        event_type = decision_event_type(decision_code)
        with self._store._write():
            prior = self._events(request_id=request_id)
            if prior:
                return prior[0]
            event = self._insert(
                event_id=decision_event_id(self.tenant_id, request_id, event_type),
                event_type=event_type, generation_id=generation_id, fact=fact, cards=cards,
                supersedes_fact_ids=(), request_id=request_id, writer=writer,
                decision_code=decision_code, now=moment,
            )
            self._materialized((event,))
            return event


class LiteEvidenceCardStore:
    """Immutable evidence cards in the lite file, with `PostgresEvidenceCardStore`'s contract."""

    def __init__(self, store: LiteStore) -> None:
        self._store = store
        self.tenant_id = store.tenant
        ensure_fact_tables(store)

    def put(self, cards: Iterable[EvidenceCard]) -> None:
        """Insert cards idempotently, refusing an identity collision with different content."""
        materialized = tuple(cards)
        if any(card.tenant_id != self.tenant_id for card in materialized):
            raise ValueError("evidence card tenant mismatch")
        if not materialized:
            return
        with self._store._write():
            _sqlite_put_cards(self._store._conn, self.tenant_id, materialized)

    def resolve(self, card_id: str) -> EvidenceCard | None:
        """Resolve and revalidate one card from the file."""
        rows = self._store._metadata_values(
            f"SELECT generation_id, chunk_id, source_digest, indexed_at, card "
            f"FROM {EVIDENCE_CARD_TABLE} WHERE tenant_id = ? AND card_id = ?",
            (self.tenant_id, card_id),
        )
        if not rows:
            return None
        return _sqlite_resolved_card(card_id, self.tenant_id, tuple(rows[0]))


__all__ = ["LiteEvidenceCardStore", "LiteFactLedger", "ensure_fact_tables"]
