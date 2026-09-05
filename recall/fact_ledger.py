"""Append-only fact-ledger adapters.

The controller owns the decision. This module owns durable event recording and current-state
projection. The PostgreSQL adapter uses the tenant GUC and row-level security already used by
RE-call's chunk store.
"""
# The SQL identifiers in this module are fixed module constants. Values from requests remain
# bound parameters; the file-level exemption covers only identifier interpolation.
# ruff: noqa: S608
from __future__ import annotations

import hashlib
import json
import sqlite3
import threading
from datetime import UTC, datetime, timedelta
from dataclasses import replace
from typing import Any
from uuid import uuid4

import psycopg
from psycopg.types.json import Jsonb

from recall.provenance_controller import (
    CONTROLLER_POLICY_VERSION,
    CONTROLLER_SCHEMA_VERSION,
    DecisionCode,
    FactEvent,
    InMemoryFactLedger,
    LedgerApplyResult,
    FactApplicationPermit,
    _consume_permit,
    _as_utc,
    canonical_json,
    evidence_card_from_payload,
    fact_conflict_key,
    fact_identity,
    facts_conflict,
)
from recall.schema import FACT_LEDGER_APPEND_FUNCTION, FACT_MATERIALIZATION_APPEND_FUNCTION
from recall.types import AtomicFact, EvidenceCard


FACT_LEDGER_TABLE = "recall_fact_ledger_events"


def _parse_time(value: object) -> datetime | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("ledger timestamp must be an ISO string")
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def _event_from_row(row: tuple[Any, ...]) -> FactEvent:
    (
        event_id, event_type, tenant_id, generation_id, fact_json, cards_json, supersedes_json,
        request_id, writer, decision_code, policy_version, controller_version, created_at,
    ) = row
    fact = AtomicFact.from_payload(fact_json) if fact_json else None
    cards = tuple(evidence_card_from_payload(item) for item in (cards_json or ()))
    created = _parse_time(created_at) if isinstance(created_at, str) else created_at
    if not isinstance(created, datetime):
        raise ValueError("ledger event timestamp is invalid")
    return FactEvent(
        event_id=str(event_id),
        event_type=str(event_type),
        tenant_id=str(tenant_id),
        generation_id=str(generation_id),
        fact=fact,
        fact_id=fact_identity(fact) if fact else None,
        evidence_cards=cards,
        supersedes_fact_ids=tuple(supersedes_json or ()),
        request_id=str(request_id),
        writer=str(writer),
        decision_code=str(decision_code),
        policy_version=str(policy_version),
        controller_version=int(controller_version),
        created_at=_as_utc(created),
    )


def _event_to_payload(event: FactEvent) -> dict[str, Any]:
    return {
        "event_id": event.event_id,
        "event_type": event.event_type,
        "tenant_id": event.tenant_id,
        "generation_id": event.generation_id,
        "fact": event.fact.to_payload() if event.fact else None,
        "evidence_cards": [card.to_payload() for card in event.evidence_cards],
        "supersedes_fact_ids": list(event.supersedes_fact_ids),
        "request_id": event.request_id,
        "writer": event.writer,
        "decision_code": event.decision_code,
        "policy_version": event.policy_version,
        "controller_version": event.controller_version,
        "created_at": _as_utc(event.created_at).isoformat(),
    }


def _event_from_payload(payload: dict[str, Any]) -> FactEvent:
    fact_payload = payload.get("fact")
    fact = AtomicFact.from_payload(fact_payload) if fact_payload else None
    cards = tuple(evidence_card_from_payload(item) for item in payload.get("evidence_cards", ()))
    created = _parse_time(payload.get("created_at"))
    if created is None:
        raise ValueError("materialization event timestamp is invalid")
    return FactEvent(
        event_id=str(payload["event_id"]),
        event_type=str(payload["event_type"]),
        tenant_id=str(payload["tenant_id"]),
        generation_id=str(payload["generation_id"]),
        fact=fact,
        fact_id=fact_identity(fact) if fact else None,
        evidence_cards=cards,
        supersedes_fact_ids=tuple(payload.get("supersedes_fact_ids", ())),
        request_id=str(payload["request_id"]),
        writer=str(payload["writer"]),
        decision_code=str(payload["decision_code"]),
        policy_version=str(payload["policy_version"]),
        controller_version=int(payload["controller_version"]),
        created_at=_as_utc(created),
    )


class SQLiteFactLedger(InMemoryFactLedger):
    """Durable local adapter that preserves the controller's in-memory semantics."""

    def __init__(self, path: str, *, tenant_id: str) -> None:
        if not path or not tenant_id:
            raise ValueError("path and tenant_id must be non-empty")
        super().__init__()
        self.path = path
        self.tenant_id = tenant_id
        self._sqlite_lock = threading.RLock()
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS recall_fact_ledger_events (
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
            )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS recall_fact_ledger_current_idx "
            "ON recall_fact_ledger_events (tenant_id, fact_id, created_at)"
        )
        self._conn.executescript(
            """CREATE TRIGGER IF NOT EXISTS recall_fact_ledger_no_update
               BEFORE UPDATE ON recall_fact_ledger_events
               BEGIN SELECT RAISE(ABORT, 'recall fact ledger is append-only'); END;
               CREATE TRIGGER IF NOT EXISTS recall_fact_ledger_no_delete
               BEFORE DELETE ON recall_fact_ledger_events
               BEGIN SELECT RAISE(ABORT, 'recall fact ledger is append-only'); END;"""
        )
        rows = self._conn.execute(
            "SELECT event_id, event_type, tenant_id, generation_id, fact, evidence_cards, "
            "supersedes_fact_ids, request_id, writer, decision_code, policy_version, "
            "controller_version, created_at FROM recall_fact_ledger_events "
            "WHERE tenant_id = ? ORDER BY created_at, event_id",
            (tenant_id,),
        ).fetchall()
        for row in rows:
            values = list(row)
            values[4] = json.loads(values[4]) if values[4] else None
            values[5] = json.loads(values[5])
            values[6] = json.loads(values[6])
            event = _event_from_row(tuple(values))
            self._events.append(event)
            self._by_request[event.request_id] = event

    def _persist(self, events: tuple[FactEvent, ...]) -> None:
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            self._persist_in_transaction(events)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def _persist_in_transaction(self, events: tuple[FactEvent, ...]) -> None:
        for event in events:
            self._conn.execute(
                "INSERT INTO recall_fact_ledger_events "
                "(event_id, tenant_id, generation_id, event_type, fact_id, fact, evidence_cards, "
                "supersedes_fact_ids, request_id, writer, decision_code, policy_version, "
                "controller_version, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event.event_id, event.tenant_id, event.generation_id, event.event_type,
                    event.fact_id, json.dumps(event.fact.to_payload(), ensure_ascii=False,
                                             separators=(",", ":"), sort_keys=True) if event.fact else None,
                    json.dumps([card.to_payload() for card in event.evidence_cards], ensure_ascii=False,
                               separators=(",", ":"), sort_keys=True),
                    json.dumps(list(event.supersedes_fact_ids), separators=(",", ":")),
                    event.request_id, event.writer, event.decision_code, event.policy_version,
                    event.controller_version, event.created_at.isoformat(),
                ),
            )

    def _reload_from_db(self) -> None:
        rows = self._conn.execute(
            "SELECT event_id, event_type, tenant_id, generation_id, fact, evidence_cards, "
            "supersedes_fact_ids, request_id, writer, decision_code, policy_version, "
            "controller_version, created_at FROM recall_fact_ledger_events "
            "WHERE tenant_id = ? ORDER BY created_at, event_id",
            (self.tenant_id,),
        ).fetchall()
        events: list[FactEvent] = []
        requests: dict[str, FactEvent] = {}
        for row in rows:
            values = list(row)
            values[4] = json.loads(values[4]) if values[4] else None
            values[5] = json.loads(values[5])
            values[6] = json.loads(values[6])
            event = _event_from_row(tuple(values))
            events.append(event)
            requests[event.request_id] = event
        self._events[:] = events
        self._by_request.clear()
        self._by_request.update(requests)

    def apply_assertion(self, **kwargs: Any) -> LedgerApplyResult:
        if kwargs.get("tenant_id") != self.tenant_id:
            raise ValueError("ledger tenant mismatch")
        with self._sqlite_lock:
            old_events = tuple(self._events)
            old_requests = dict(self._by_request)
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._reload_from_db()
                before = len(self._events)
                result = super().apply_assertion(**kwargs)
                self._persist_in_transaction(tuple(self._events[before:]))
                self._conn.execute("COMMIT")
                return result
            except Exception:
                self._conn.execute("ROLLBACK")
                self._events[:] = old_events
                self._by_request.clear()
                self._by_request.update(old_requests)
                raise

    def record_decision(self, **kwargs: Any) -> FactEvent:
        if kwargs.get("tenant_id") != self.tenant_id:
            raise ValueError("ledger tenant mismatch")
        with self._sqlite_lock:
            old_events = tuple(self._events)
            old_requests = dict(self._by_request)
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                self._reload_from_db()
                before = len(self._events)
                event = super().record_decision(**kwargs)
                self._persist_in_transaction(tuple(self._events[before:]))
                self._conn.execute("COMMIT")
                return event
            except Exception:
                self._conn.execute("ROLLBACK")
                self._events[:] = old_events
                self._by_request.clear()
                self._by_request.update(old_requests)
                raise

    def request_event(self, *, tenant_id: str, request_id: str) -> FactEvent | None:
        if tenant_id != self.tenant_id:
            raise ValueError("ledger tenant mismatch")
        with self._sqlite_lock:
            row = self._conn.execute(
                "SELECT event_id, event_type, tenant_id, generation_id, fact, evidence_cards, "
                "supersedes_fact_ids, request_id, writer, decision_code, policy_version, "
                "controller_version, created_at FROM recall_fact_ledger_events "
                "WHERE tenant_id = ? AND request_id = ?",
                (tenant_id, request_id),
            ).fetchone()
            if row is None:
                return None
            values = list(row)
            values[4] = json.loads(values[4]) if values[4] else None
            values[5] = json.loads(values[5])
            values[6] = json.loads(values[6])
            return _event_from_row(tuple(values))

    def close(self) -> None:
        with self._sqlite_lock:
            self._conn.close()

    def __enter__(self) -> "SQLiteFactLedger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class InMemoryMaterializationOutbox:
    """Reference delivery queue used by tests and small local workflows."""

    def __init__(self, *, lease_seconds: int = 300) -> None:
        if lease_seconds < 1:
            raise ValueError("lease_seconds must be positive")
        self.lease_seconds = lease_seconds
        self._events: dict[str, FactEvent] = {}
        self._state: dict[str, tuple[str, int, datetime | None, str | None]] = {}
        self._lease_tokens: dict[str, str] = {}
        self._lock = threading.RLock()

    def enqueue(self, event: FactEvent) -> None:
        with self._lock:
            prior = self._events.get(event.event_id)
            if prior is not None and _event_to_payload(prior) != _event_to_payload(event):
                raise ValueError(f"materialization event collision for {event.event_id}")
            if prior is None:
                self._events[event.event_id] = event
                self._state[event.event_id] = ("pending", 0, None, None)

    def claim(
        self, *, tenant_id: str, event_id: str | None = None, limit: int = 100,
        now: datetime | None = None,
    ) -> tuple[FactEvent, ...]:
        if not isinstance(tenant_id, str) or not tenant_id:
            raise ValueError("outbox tenant must be non-empty")
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        moment = _as_utc(now or datetime.now(UTC))
        with self._lock:
            selected: list[FactEvent] = []
            for key, event in self._events.items():
                if event.tenant_id != tenant_id or (event_id is not None and key != event_id):
                    continue
                status, attempts, lease_until, error = self._state[key]
                if status == "applied":
                    continue
                if status == "processing" and lease_until is not None and moment < lease_until:
                    continue
                self._state[key] = (
                    "processing", attempts + 1, moment + timedelta(seconds=self.lease_seconds), error
                )
                token = uuid4().hex
                self._lease_tokens[key] = token
                selected.append(replace(event, lease_token=token))
                if len(selected) >= limit:
                    break
            return tuple(selected)

    def mark_applied(self, *, tenant_id: str, event_id: str, lease_token: str | None = None) -> None:
        with self._lock:
            event = self._events.get(event_id)
            if event is None or event.tenant_id != tenant_id:
                raise KeyError(event_id)
            if lease_token is not None and self._lease_tokens.get(event_id) != lease_token:
                raise ValueError("outbox lease lost")
            _status, attempts, _lease, error = self._state[event_id]
            self._state[event_id] = ("applied", attempts, None, error)

    def mark_failed(self, *, tenant_id: str, event_id: str, error: str, lease_token: str | None = None) -> None:
        with self._lock:
            event = self._events.get(event_id)
            if event is None or event.tenant_id != tenant_id:
                raise KeyError(event_id)
            if lease_token is not None and self._lease_tokens.get(event_id) != lease_token:
                raise ValueError("outbox lease lost")
            _status, attempts, _lease, _old_error = self._state[event_id]
            self._state[event_id] = ("failed", attempts, None, error[:2000])

    def status(self, event_id: str) -> str:
        with self._lock:
            return self._state[event_id][0]


class SQLiteMaterializationOutbox(InMemoryMaterializationOutbox):
    """Durable local materialization queue with immutable event snapshots."""

    def __init__(self, path: str, *, tenant_id: str, lease_seconds: int = 300) -> None:
        super().__init__(lease_seconds=lease_seconds)
        if not path or not tenant_id:
            raise ValueError("path and tenant_id must be non-empty")
        self.path = path
        self.tenant_id = tenant_id
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.execute(
            """CREATE TABLE IF NOT EXISTS recall_fact_materialization_outbox (
                event_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                event TEXT NOT NULL,
                status TEXT NOT NULL CHECK (status IN ('pending', 'processing', 'failed', 'applied')),
                attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts >= 0),
                lease_until TEXT,
                lease_token TEXT,
                last_error TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )"""
        )
        self._conn.execute(
            "CREATE INDEX IF NOT EXISTS recall_fact_materialization_pending_idx "
            "ON recall_fact_materialization_outbox (tenant_id, status, updated_at)"
        )
        columns = {row[1] for row in self._conn.execute(
            "PRAGMA table_info(recall_fact_materialization_outbox)"
        ).fetchall()}
        if "lease_token" not in columns:
            self._conn.execute(
                "ALTER TABLE recall_fact_materialization_outbox ADD COLUMN lease_token TEXT"
            )
        self._conn.executescript(
            """CREATE TRIGGER IF NOT EXISTS recall_fact_materialization_no_delete
               BEFORE DELETE ON recall_fact_materialization_outbox
               BEGIN SELECT RAISE(ABORT, 'recall materialization outbox is append-only history'); END;
               CREATE TRIGGER IF NOT EXISTS recall_fact_materialization_immutable_event
               BEFORE UPDATE OF event_id, tenant_id, event ON recall_fact_materialization_outbox
               BEGIN SELECT RAISE(ABORT, 'recall materialization event is immutable'); END;"""
        )
        rows = self._conn.execute(
            "SELECT event_id, tenant_id, event, status, attempts, lease_until, lease_token, last_error "
            "FROM recall_fact_materialization_outbox WHERE tenant_id = ? ORDER BY created_at, event_id",
            (tenant_id,),
        ).fetchall()
        for event_id, row_tenant, payload, status, attempts, lease_until, lease_token, error in rows:
            event = _event_from_payload(json.loads(payload))
            self._events[str(event_id)] = event
            self._state[str(event_id)] = (
                str(status), int(attempts), _parse_time(lease_until), error,
            )
            if lease_token:
                self._lease_tokens[str(event_id)] = str(lease_token)

    def enqueue(self, event: FactEvent) -> None:
        if event.tenant_id != self.tenant_id:
            raise ValueError("outbox tenant mismatch")
        now = _as_utc(event.created_at).isoformat()
        payload = json.dumps(_event_to_payload(event), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        with self._lock:
            super().enqueue(event)
            try:
                self._conn.execute(
                    "INSERT INTO recall_fact_materialization_outbox "
                    "(event_id, tenant_id, event, status, created_at, updated_at) VALUES (?, ?, ?, 'pending', ?, ?) "
                    "ON CONFLICT(event_id) DO NOTHING",
                    (event.event_id, event.tenant_id, payload, now, now),
                )
            except Exception:
                self._events.pop(event.event_id, None)
                self._state.pop(event.event_id, None)
                raise

    def claim(self, **kwargs: Any) -> tuple[FactEvent, ...]:
        with self._lock:
            tenant_id = kwargs.get("tenant_id")
            event_id = kwargs.get("event_id")
            limit = kwargs.get("limit", 100)
            if tenant_id != self.tenant_id:
                raise ValueError("outbox tenant mismatch")
            if not isinstance(limit, int) or not 1 <= limit <= 1000:
                raise ValueError("limit must be between 1 and 1000")
            moment = _as_utc(kwargs.get("now") or datetime.now(UTC))
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                params: list[Any] = [self.tenant_id, moment.isoformat()]
                where = (
                    "tenant_id = ? AND (status IN ('pending', 'failed') "
                    "OR (status = 'processing' AND lease_until IS NOT NULL AND lease_until <= ?))"
                )
                if event_id is not None:
                    where += " AND event_id = ?"
                    params.append(event_id)
                params.append(limit)
                rows = self._conn.execute(
                    "SELECT event_id, event, attempts, last_error FROM "
                    "recall_fact_materialization_outbox WHERE " + where +
                    " ORDER BY created_at, event_id LIMIT ?",
                    params,
                ).fetchall()
                lease_until = moment + timedelta(seconds=self.lease_seconds)
                events: list[FactEvent] = []
                for row_event_id, payload, attempts, error in rows:
                    token = uuid4().hex
                    updated = self._conn.execute(
                        "UPDATE recall_fact_materialization_outbox SET status = 'processing', "
                        "attempts = attempts + 1, lease_until = ?, lease_token = ?, updated_at = ? "
                        "WHERE event_id = ? AND tenant_id = ?",
                        (lease_until.isoformat(), token, moment.isoformat(), row_event_id, self.tenant_id),
                    )
                    if updated.rowcount != 1:
                        continue
                    event = replace(_event_from_payload(json.loads(payload)), lease_token=token)
                    self._events[str(row_event_id)] = replace(event, lease_token=None)
                    self._state[str(row_event_id)] = (
                        "processing", int(attempts) + 1, lease_until, error,
                    )
                    self._lease_tokens[str(row_event_id)] = token
                    events.append(event)
                self._conn.execute("COMMIT")
                return tuple(events)
            except Exception:
                self._conn.execute("ROLLBACK")
                raise

    def mark_applied(self, *, tenant_id: str, event_id: str, lease_token: str | None = None) -> None:
        with self._lock:
            _status, attempts, _lease, error = self._state[event_id]
            params: list[Any] = ["applied", attempts, error, datetime.now(UTC).isoformat(), event_id, tenant_id]
            lease_clause = ""
            if lease_token is not None:
                lease_clause = " AND lease_token = ?"
                params.append(lease_token)
            result = self._conn.execute(
                "UPDATE recall_fact_materialization_outbox SET status = ?, attempts = ?, lease_until = NULL, "
                "lease_token = NULL, "
                f"last_error = ? , updated_at = ? WHERE event_id = ? AND tenant_id = ?{lease_clause}",
                params,
            )
            if result.rowcount != 1:
                raise ValueError("outbox lease lost")
            super().mark_applied(tenant_id=tenant_id, event_id=event_id, lease_token=lease_token)

    def mark_failed(self, *, tenant_id: str, event_id: str, error: str, lease_token: str | None = None) -> None:
        with self._lock:
            _status, attempts, _lease, _saved_error = self._state[event_id]
            params: list[Any] = ["failed", attempts, error[:2000], datetime.now(UTC).isoformat(), event_id, tenant_id]
            lease_clause = ""
            if lease_token is not None:
                lease_clause = " AND lease_token = ?"
                params.append(lease_token)
            result = self._conn.execute(
                "UPDATE recall_fact_materialization_outbox SET status = ?, attempts = ?, lease_until = NULL, "
                "lease_token = NULL, "
                f"last_error = ? , updated_at = ? WHERE event_id = ? AND tenant_id = ?{lease_clause}",
                params,
            )
            if result.rowcount != 1:
                raise ValueError("outbox lease lost")
            super().mark_failed(
                tenant_id=tenant_id,
                event_id=event_id,
                error=error,
                lease_token=lease_token,
            )

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "SQLiteMaterializationOutbox":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


class PostgresFactLedger:
    """Production append-only ledger with serializable per-fact application."""

    def __init__(self, dsn: str, *, tenant_id: str, statement_timeout_ms: int = 15000) -> None:
        if not dsn or not tenant_id:
            raise ValueError("dsn and tenant_id must be non-empty")
        if statement_timeout_ms < 1:
            raise ValueError("statement_timeout_ms must be positive")
        self.dsn = dsn
        self.tenant_id = tenant_id
        self.statement_timeout_ms = statement_timeout_ms

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.dsn, autocommit=True, connect_timeout=10)
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (self.tenant_id,))
        conn.execute(f"SET statement_timeout = {int(self.statement_timeout_ms)}")
        return conn

    @staticmethod
    def _select_sql() -> str:
        return (
            f"SELECT event_id, event_type, tenant_id, generation_id, fact, evidence_cards, "
            f"supersedes_fact_ids, request_id, writer, decision_code, policy_version, "
            f"controller_version, created_at FROM {FACT_LEDGER_TABLE}"
        )

    def _events(self, conn: psycopg.Connection, *, request_id: str | None = None) -> list[FactEvent]:
        suffix = " WHERE tenant_id = %s"
        params: list[Any] = [self.tenant_id]
        if request_id is not None:
            suffix += " AND request_id = %s"
            params.append(request_id)
        rows = conn.execute(self._select_sql() + suffix + " ORDER BY created_at, event_id", params).fetchall()
        return [_event_from_row(row) for row in rows]

    @staticmethod
    def _project_current(events: list[FactEvent], now: datetime) -> tuple[FactEvent, ...]:
        superseded = {
            fact_id
            for event in events
            if event.event_type == "superseded"
            for fact_id in event.supersedes_fact_ids
        }
        instant = _as_utc(now)
        return tuple(
            event for event in events
            if event.event_type == "asserted"
            and event.fact_id not in superseded
            and event.fact is not None
            and (event.fact.valid_from is None or instant >= _as_utc(event.fact.valid_from))
            and (event.fact.valid_until is None or instant < _as_utc(event.fact.valid_until))
        )

    @property
    def events(self) -> tuple[FactEvent, ...]:
        """Read the tenant's immutable event stream for recovery reconciliation."""
        with self._connect() as conn:
            return tuple(self._events(conn))

    def current(self, *, tenant_id: str, now: datetime) -> tuple[FactEvent, ...]:
        if tenant_id != self.tenant_id:
            raise ValueError("ledger tenant mismatch")
        with self._connect() as conn:
            events = self._events(conn)
        return self._project_current(events, now)

    @staticmethod
    def _project_candidates(events: list[FactEvent]) -> tuple[FactEvent, ...]:
        superseded = {
            fact_id
            for event in events
            if event.event_type == "superseded"
            for fact_id in event.supersedes_fact_ids
        }
        return tuple(
            event
            for event in events
            if event.event_type == "asserted"
            and event.fact_id not in superseded
            and event.fact is not None
        )

    def request_event(self, *, tenant_id: str, request_id: str) -> FactEvent | None:
        if tenant_id != self.tenant_id:
            raise ValueError("ledger tenant mismatch")
        with self._connect() as conn:
            events = self._events(conn, request_id=request_id)
        return events[0] if events else None

    def _insert(
        self,
        conn: psycopg.Connection,
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
        conn.execute(
            f"SELECT {FACT_LEDGER_APPEND_FUNCTION}(%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)",
            (
                event_id, self.tenant_id, generation_id, event_type,
                fact_identity(fact) if fact else None,
                Jsonb(fact.to_payload()) if fact else None,
                Jsonb([card.to_payload() for card in cards]),
                Jsonb(list(supersedes_fact_ids)), request_id, writer, decision_code,
                CONTROLLER_POLICY_VERSION, CONTROLLER_SCHEMA_VERSION, _as_utc(now),
            ),
        )
        return FactEvent(
            event_id=event_id, event_type=event_type, tenant_id=self.tenant_id,
            generation_id=generation_id, fact=fact, fact_id=fact_identity(fact) if fact else None,
            evidence_cards=cards, supersedes_fact_ids=supersedes_fact_ids,
            request_id=request_id, writer=writer, decision_code=decision_code,
            policy_version=CONTROLLER_POLICY_VERSION, controller_version=CONTROLLER_SCHEMA_VERSION,
            created_at=_as_utc(now),
        )

    def apply_assertion(
        self, *, tenant_id: str, generation_id: str, fact: AtomicFact,
        cards: tuple[EvidenceCard, ...], request_id: str, writer: str,
        permit: FactApplicationPermit | None = None,
        supersedes_fact_ids: tuple[str, ...] = (),
        policy_version: str = CONTROLLER_POLICY_VERSION,
        controller_version: int = CONTROLLER_SCHEMA_VERSION,
        now: datetime | None = None,
        materialization_outbox: Any | None = None,
    ) -> LedgerApplyResult:
        del policy_version, controller_version
        if tenant_id != self.tenant_id:
            raise ValueError("ledger tenant mismatch")
        moment = _as_utc(now or datetime.now(UTC))
        event_id = "evt_" + hashlib.sha256(
            f"{self.tenant_id}\0{fact_identity(fact)}\0{request_id}\0asserted".encode("utf-8")
        ).hexdigest()[:32]
        with self._connect() as conn, conn.transaction():
            _consume_permit(
                permit,
                tenant_id=tenant_id,
                generation_id=generation_id,
                request_id=request_id,
                fact=fact,
                cards=cards,
            )
            conn.execute(
                "SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))",
                (f"fact\x1f{self.tenant_id}\x1f{fact_conflict_key(fact)}",),
            )
            prior = self._events(conn, request_id=request_id)
            if prior:
                if materialization_outbox is not None:
                    self._enqueue_materialization(conn, materialization_outbox, prior[0])
                return LedgerApplyResult(prior[0], duplicate=True)
            history = self._events(conn)
            candidates = self._project_candidates(history)
            if any(event.fact_id == fact_identity(fact) for event in candidates):
                event = self._insert(
                    conn, event_id=event_id, event_type="asserted", generation_id=generation_id,
                    fact=fact, cards=cards, supersedes_fact_ids=(), request_id=request_id,
                    writer=writer, decision_code=DecisionCode.DUPLICATE, now=moment,
                )
                if materialization_outbox is not None:
                    self._enqueue_materialization(conn, materialization_outbox, event)
                return LedgerApplyResult(event, duplicate=True)
            conflicts = tuple(event for event in candidates if event.fact and facts_conflict(event.fact, fact))
            if {event.fact_id for event in conflicts} - set(supersedes_fact_ids):
                raise ValueError(DecisionCode.CONTRADICTION_WITHOUT_SUPERSESSION)
            event = self._insert(
                conn, event_id=event_id, event_type="asserted", generation_id=generation_id,
                fact=fact, cards=cards, supersedes_fact_ids=supersedes_fact_ids,
                request_id=request_id, writer=writer, decision_code=DecisionCode.APPLIED, now=moment,
            )
            if supersedes_fact_ids:
                self._insert(
                    conn, event_id=event_id + "_sup", event_type="superseded",
                    generation_id=generation_id, fact=None, cards=cards,
                    supersedes_fact_ids=supersedes_fact_ids, request_id=request_id + ":supersession",
                    writer=writer, decision_code=DecisionCode.APPLIED, now=moment,
                )
            if materialization_outbox is not None:
                self._enqueue_materialization(conn, materialization_outbox, event)
            return LedgerApplyResult(event)

    @staticmethod
    def _enqueue_materialization(
        conn: psycopg.Connection, outbox: Any, event: FactEvent
    ) -> None:
        enqueue = getattr(outbox, "_enqueue_in_transaction", None)
        if not callable(enqueue):
            raise TypeError("PostgreSQL atomic materialization requires its PostgreSQL outbox adapter")
        enqueue(conn, event)

    def apply_assertion_with_outbox(self, **kwargs: Any) -> LedgerApplyResult:
        """Append the ledger event and outbox snapshot in one PostgreSQL transaction."""
        outbox = kwargs.pop("materialization_outbox", None)
        if outbox is None:
            raise ValueError("materialization_outbox is required")
        return self.apply_assertion(materialization_outbox=outbox, **kwargs)

    def record_decision(
        self, *, tenant_id: str, generation_id: str, request_id: str, writer: str,
        decision_code: str, fact: AtomicFact | None, cards: tuple[EvidenceCard, ...] = (),
        now: datetime | None = None,
    ) -> FactEvent:
        if tenant_id != self.tenant_id:
            raise ValueError("ledger tenant mismatch")
        moment = _as_utc(now or datetime.now(UTC))
        event_type = "abstained" if decision_code == DecisionCode.FRESH_SEARCH_INSUFFICIENT else "rejected"
        event_id = "evt_" + hashlib.sha256(
            f"{self.tenant_id}\0{request_id}\0{event_type}".encode("utf-8")
        ).hexdigest()[:32]
        with self._connect() as conn, conn.transaction():
            prior = self._events(conn, request_id=request_id)
            if prior:
                return prior[0]
            return self._insert(
                conn, event_id=event_id, event_type=event_type, generation_id=generation_id,
                fact=fact, cards=cards, request_id=request_id, writer=writer,
                decision_code=decision_code, supersedes_fact_ids=(), now=moment,
            )


class PostgresMaterializationOutbox:
    """Tenant-scoped durable delivery queue for downstream fact materializers."""

    TABLE = "recall_fact_materialization_outbox"

    def __init__(self, dsn: str, *, tenant_id: str, lease_seconds: int = 300,
                 statement_timeout_ms: int = 15000) -> None:
        if not dsn or not tenant_id:
            raise ValueError("dsn and tenant_id must be non-empty")
        if lease_seconds < 1 or statement_timeout_ms < 1:
            raise ValueError("lease_seconds and statement_timeout_ms must be positive")
        self.dsn = dsn
        self.tenant_id = tenant_id
        self.lease_seconds = lease_seconds
        self.statement_timeout_ms = statement_timeout_ms

    def _connect(self) -> psycopg.Connection:
        conn = psycopg.connect(self.dsn, autocommit=True, connect_timeout=10)
        conn.execute("SELECT set_config('recall.tenant_id', %s, false)", (self.tenant_id,))
        conn.execute(f"SET statement_timeout = {int(self.statement_timeout_ms)}")
        return conn

    def enqueue(self, event: FactEvent) -> None:
        if event.tenant_id != self.tenant_id:
            raise ValueError("outbox tenant mismatch")
        with self._connect() as conn, conn.transaction():
            self._enqueue_in_transaction(conn, event)

    def _enqueue_in_transaction(self, conn: psycopg.Connection, event: FactEvent) -> None:
        if event.tenant_id != self.tenant_id:
            raise ValueError("outbox tenant mismatch")
        payload = Jsonb(_event_to_payload(event))
        prior = conn.execute(
            f"SELECT event FROM {self.TABLE} WHERE event_id = %s AND tenant_id = %s FOR UPDATE",
            (event.event_id, self.tenant_id),
        ).fetchone()
        if prior is not None and canonical_json(prior[0]) != canonical_json(_event_to_payload(event)):
            raise ValueError(f"materialization event collision for {event.event_id}")
        conn.execute(
            f"SELECT {FACT_MATERIALIZATION_APPEND_FUNCTION}(%s, %s, %s)",
            (event.event_id, self.tenant_id, payload),
        )

    def claim(
        self, *, tenant_id: str, event_id: str | None = None, limit: int = 100,
        now: datetime | None = None,
    ) -> tuple[FactEvent, ...]:
        if tenant_id != self.tenant_id:
            raise ValueError("outbox tenant mismatch")
        if not isinstance(limit, int) or not 1 <= limit <= 1000:
            raise ValueError("limit must be between 1 and 1000")
        moment = _as_utc(now or datetime.now(UTC))
        with self._connect() as conn, conn.transaction():
            where = (
                "tenant_id = %s AND (status IN ('pending', 'failed') "
                "OR (status = 'processing' AND lease_until IS NOT NULL AND lease_until <= %s))"
            )
            params: list[Any] = [self.tenant_id, moment]
            if event_id is not None:
                where += " AND event_id = %s"
                params.append(event_id)
            params.append(limit)
            rows = conn.execute(
                f"SELECT event_id, event FROM {self.TABLE} WHERE {where} "
                "ORDER BY created_at, event_id FOR UPDATE SKIP LOCKED LIMIT %s",
                params,
            ).fetchall()
            if not rows:
                return ()
            lease_until = moment + timedelta(seconds=self.lease_seconds)
            events: list[FactEvent] = []
            for row_event_id, payload in rows:
                token = uuid4().hex
                conn.execute(
                    f"UPDATE {self.TABLE} SET status = 'processing', attempts = attempts + 1, "
                    "lease_until = %s, lease_token = %s, updated_at = %s "
                    "WHERE event_id = %s AND tenant_id = %s",
                    (lease_until, token, moment, row_event_id, self.tenant_id),
                )
                events.append(replace(_event_from_payload(payload), lease_token=token))
            return tuple(events)

    def mark_applied(self, *, tenant_id: str, event_id: str, lease_token: str | None = None) -> None:
        if tenant_id != self.tenant_id:
            raise ValueError("outbox tenant mismatch")
        with self._connect() as conn:
            params: list[Any] = [event_id, self.tenant_id]
            lease_clause = ""
            if lease_token is not None:
                lease_clause = " AND lease_token = %s"
                params.append(lease_token)
            result = conn.execute(
                f"UPDATE {self.TABLE} SET status = 'applied', lease_until = NULL, updated_at = clock_timestamp() "
                f"WHERE event_id = %s AND tenant_id = %s{lease_clause}",
                params,
            )
            if result.rowcount != 1:
                raise ValueError("outbox lease lost")

    def mark_failed(self, *, tenant_id: str, event_id: str, error: str, lease_token: str | None = None) -> None:
        if tenant_id != self.tenant_id:
            raise ValueError("outbox tenant mismatch")
        with self._connect() as conn:
            params: list[Any] = [error[:2000], event_id, self.tenant_id]
            lease_clause = ""
            if lease_token is not None:
                lease_clause = " AND lease_token = %s"
                params.append(lease_token)
            result = conn.execute(
                f"UPDATE {self.TABLE} SET status = 'failed', lease_until = NULL, last_error = %s, "
                f"updated_at = clock_timestamp() WHERE event_id = %s AND tenant_id = %s{lease_clause}",
                params,
            )
            if result.rowcount != 1:
                raise ValueError("outbox lease lost")

    def status(self, event_id: str) -> str | None:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT status FROM {self.TABLE} WHERE event_id = %s AND tenant_id = %s",
                (event_id, self.tenant_id),
            ).fetchone()
        return str(row[0]) if row else None


__all__ = [
    "FACT_LEDGER_TABLE", "InMemoryFactLedger", "InMemoryMaterializationOutbox",
    "PostgresFactLedger", "PostgresMaterializationOutbox", "SQLiteFactLedger",
    "SQLiteMaterializationOutbox",
]
