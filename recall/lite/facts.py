"""The fact ledger and evidence cards on a lite store: `PostgresFactLedger`'s rules on one SQLite file.

`SQLiteFactLedger` and `SQLiteEvidenceCardStore` (behind `recall provenance --sqlite-path`) open a
connection of their own, read a snapshot taken at open, and follow the in-memory ledger's event
ids. These adapters live on the `LiteStore`'s connection instead: they read the file on every call,
write in one `LiteStore._write()` transaction each, and mint the same event ids and the same
projection as PostgreSQL, so a client of `recall_apply_fact` and `recall_current_facts` cannot tell
the backends apart. Table names and columns are those of the SQLite adapters.
"""
# The SQL identifiers in this module are fixed module constants; values are bound parameters.
# ruff: noqa: S608
from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from recall.fact_ledger import (
    FACT_LEDGER_TABLE,
    LEDGER_COLUMNS,
    assertion_event_id,
    decision_event_id,
    decision_event_type,
    decode_ledger_row,
    project_candidates,
    project_current,
)
from recall.lite.store import LiteStore
from recall.provenance_cards import EVIDENCE_CARD_TABLE
from recall.provenance_controller import (
    CONTROLLER_POLICY_VERSION,
    CONTROLLER_SCHEMA_VERSION,
    DecisionCode,
    FactApplicationPermit,
    FactEvent,
    LedgerApplyResult,
    _as_utc,
    _consume_permit,
    evidence_card_from_payload,
    fact_identity,
    facts_conflict,
)
from recall.types import AtomicFact, EvidenceCard

_SCHEMA = f"""
CREATE TABLE IF NOT EXISTS {FACT_LEDGER_TABLE} (
    event_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    event_type TEXT NOT NULL CHECK (event_type IN ('asserted', 'superseded', 'rejected', 'abstained')),
    fact_id TEXT,
    fact TEXT,
    evidence_cards TEXT NOT NULL,
    supersedes_fact_ids TEXT NOT NULL,
    request_id TEXT NOT NULL,
    writer TEXT NOT NULL,
    decision_code TEXT NOT NULL,
    policy_version TEXT NOT NULL,
    controller_version INTEGER NOT NULL,
    created_at TEXT NOT NULL,
    UNIQUE (tenant_id, request_id)
);
CREATE INDEX IF NOT EXISTS recall_fact_ledger_current_idx
    ON {FACT_LEDGER_TABLE} (tenant_id, fact_id, created_at);
CREATE TRIGGER IF NOT EXISTS recall_fact_ledger_no_update BEFORE UPDATE ON {FACT_LEDGER_TABLE}
BEGIN SELECT RAISE(ABORT, 'recall fact ledger is append-only'); END;
CREATE TRIGGER IF NOT EXISTS recall_fact_ledger_no_delete BEFORE DELETE ON {FACT_LEDGER_TABLE}
BEGIN SELECT RAISE(ABORT, 'recall fact ledger is append-only'); END;
CREATE TABLE IF NOT EXISTS {EVIDENCE_CARD_TABLE} (
    card_id TEXT PRIMARY KEY,
    tenant_id TEXT NOT NULL,
    generation_id TEXT NOT NULL,
    chunk_id TEXT NOT NULL,
    source_digest TEXT NOT NULL,
    card TEXT NOT NULL,
    indexed_at TEXT
);
CREATE INDEX IF NOT EXISTS {EVIDENCE_CARD_TABLE}_lookup_idx
    ON {EVIDENCE_CARD_TABLE} (tenant_id, generation_id, chunk_id);
CREATE TRIGGER IF NOT EXISTS {EVIDENCE_CARD_TABLE}_no_update BEFORE UPDATE ON {EVIDENCE_CARD_TABLE}
BEGIN SELECT RAISE(ABORT, 'recall evidence cards are immutable'); END;
CREATE TRIGGER IF NOT EXISTS {EVIDENCE_CARD_TABLE}_no_delete BEFORE DELETE ON {EVIDENCE_CARD_TABLE}
BEGIN SELECT RAISE(ABORT, 'recall evidence cards are immutable'); END;
"""


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True)


def _stamp(moment: datetime) -> str:
    """Fixed width, so text order is time order (plain `isoformat()` drops zero microseconds)."""
    return _as_utc(moment).isoformat(timespec="microseconds")


def ensure_fact_tables(store: LiteStore) -> None:
    """Create the fact tables in a file made before they existed.

    Outside `_write()` on purpose: `executescript` commits any open transaction first, and every
    statement is `IF NOT EXISTS`, so running it on each adapter construction costs a catalog read.
    """
    with store._lock:
        store._conn.executescript(_SCHEMA)


class LiteFactLedger:
    """The `FactLedger` protocol on a lite store, with `PostgresFactLedger`'s semantics."""

    def __init__(self, store: LiteStore) -> None:
        self._store = store
        self.tenant_id = store.tenant
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

    def current(self, *, tenant_id: str, now: datetime) -> tuple[FactEvent, ...]:
        self._check(tenant_id)
        return project_current(self._events(), now)

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
                CONTROLLER_POLICY_VERSION,
                CONTROLLER_SCHEMA_VERSION,
                _stamp(now),
                fact_id,
            ),
        )
        return FactEvent(
            event_id=event_id, event_type=event_type, tenant_id=self.tenant_id,
            generation_id=generation_id, fact=fact, fact_id=fact_id, evidence_cards=cards,
            supersedes_fact_ids=supersedes_fact_ids, request_id=request_id, writer=writer,
            decision_code=str(decision_code), policy_version=CONTROLLER_POLICY_VERSION,
            controller_version=CONTROLLER_SCHEMA_VERSION, created_at=_as_utc(now),
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
    ) -> LedgerApplyResult:
        del policy_version, controller_version
        self._check(tenant_id)
        moment = _as_utc(now or datetime.now(UTC))
        event_id = assertion_event_id(self.tenant_id, fact, request_id)
        # One BEGIN IMMEDIATE transaction holds the request check, the history read and the
        # insert: what the per-fact advisory lock gives `PostgresFactLedger.apply_assertion`.
        with self._store._write():
            _consume_permit(
                permit, tenant_id=tenant_id, generation_id=generation_id,
                request_id=request_id, fact=fact, cards=cards,
            )
            prior = self._events(request_id=request_id)
            if prior:
                return LedgerApplyResult(prior[0], duplicate=True)
            candidates = project_candidates(self._events())
            if any(event.fact_id == fact_identity(fact) for event in candidates):
                event = self._insert(
                    event_id=event_id, event_type="asserted", generation_id=generation_id,
                    fact=fact, cards=cards, supersedes_fact_ids=(), request_id=request_id,
                    writer=writer, decision_code=DecisionCode.DUPLICATE, now=moment,
                )
                return LedgerApplyResult(event, duplicate=True)
            conflicts = tuple(
                event for event in candidates if event.fact and facts_conflict(event.fact, fact)
            )
            if {event.fact_id for event in conflicts} - set(supersedes_fact_ids):
                raise ValueError(DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION)
            event = self._insert(
                event_id=event_id, event_type="asserted", generation_id=generation_id, fact=fact,
                cards=cards, supersedes_fact_ids=supersedes_fact_ids, request_id=request_id,
                writer=writer, decision_code=DecisionCode.APPLIED, now=moment,
            )
            if supersedes_fact_ids:
                self._insert(
                    event_id=event_id + "_sup", event_type="superseded",
                    generation_id=generation_id, fact=None, cards=cards,
                    supersedes_fact_ids=supersedes_fact_ids,
                    request_id=request_id + ":supersession", writer=writer,
                    decision_code=DecisionCode.APPLIED, now=moment,
                )
            return LedgerApplyResult(event)

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
            return self._insert(
                event_id=decision_event_id(self.tenant_id, request_id, event_type),
                event_type=event_type, generation_id=generation_id, fact=fact, cards=cards,
                supersedes_fact_ids=(), request_id=request_id, writer=writer,
                decision_code=decision_code, now=moment,
            )


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
            for card in materialized:
                self._store._conn.execute(
                    f"INSERT OR IGNORE INTO {EVIDENCE_CARD_TABLE} "
                    "(card_id, tenant_id, generation_id, chunk_id, source_digest, card, indexed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        card.card_id, self.tenant_id, card.generation_id, card.chunk_id,
                        card.source_digest, _json(card.to_payload()),
                        card.indexed_at.isoformat() if card.indexed_at else None,
                    ),
                )
                row = self._store._conn.execute(
                    f"SELECT card FROM {EVIDENCE_CARD_TABLE} WHERE card_id = ?", (card.card_id,)
                ).fetchone()
                existing = evidence_card_from_payload(json.loads(row[0])) if row is not None else None
                if existing != card:
                    raise ValueError(f"evidence card identity collision for {card.card_id}")

    def resolve(self, card_id: str) -> EvidenceCard | None:
        """Resolve and revalidate one card from the file."""
        rows = self._store._metadata_values(
            f"SELECT generation_id, chunk_id, source_digest, indexed_at, card "
            f"FROM {EVIDENCE_CARD_TABLE} WHERE tenant_id = ? AND card_id = ?",
            (self.tenant_id, card_id),
        )
        if not rows:
            return None
        generation_id, chunk_id, digest, indexed_at, encoded = rows[0]
        payload = json.loads(encoded)
        if not isinstance(payload, Mapping):
            raise ValueError("durable evidence card payload is not an object")
        card = evidence_card_from_payload(payload)
        if card.card_id != card_id or card.tenant_id != self.tenant_id:
            raise ValueError("durable evidence card identity mismatch")
        stored_indexed_at = card.indexed_at.isoformat() if card.indexed_at else None
        if (card.generation_id, card.chunk_id, card.source_digest, stored_indexed_at) != (
            generation_id, chunk_id, digest, indexed_at,
        ):
            raise ValueError("durable evidence card projection mismatch")
        return card


__all__ = ["LiteEvidenceCardStore", "LiteFactLedger", "ensure_fact_tables"]
