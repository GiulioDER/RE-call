"""Durable evidence-card projection used by the provenance controller.

Cards are immutable server-created projections.  The application request contains only a card
identifier; this store resolves the identifier again from PostgreSQL under the current tenant.
There is deliberately no update or delete API, and the database trigger is the final backstop.
"""
# The SQL identifiers in this module are fixed module constants. Card and tenant values remain
# bound parameters; the file-level exemption covers only identifier interpolation.
# ruff: noqa: S608
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from recall.fact_ledger import SQLITE_REDACTION_SESSION_DDL
from recall.provenance_controller import EvidenceCardStore, evidence_card_from_payload
from recall.store import TENANT_GUC
from recall.types import EvidenceCard, RedactedEvidenceCard


EVIDENCE_CARD_TABLE = "recall_evidence_cards"

# The SQLite card table: `SQLiteEvidenceCardStore` and the lite store's `recall.lite.facts` run it.
# As on PostgreSQL (migration 0029), an UPDATE is admitted only inside a redaction session, and only
# into the redacted-card shape (`recall.types.RedactedEvidenceCard`); a DELETE never is.
SQLITE_EVIDENCE_CARD_DDL = SQLITE_REDACTION_SESSION_DDL + f"""
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
WHEN NOT EXISTS (SELECT 1 FROM recall_redaction_session)
BEGIN SELECT RAISE(ABORT, 'recall evidence cards are immutable'); END;
CREATE TRIGGER IF NOT EXISTS {EVIDENCE_CARD_TABLE}_redaction_shape BEFORE UPDATE ON {EVIDENCE_CARD_TABLE}
WHEN EXISTS (SELECT 1 FROM recall_redaction_session) AND NOT (
    NEW.tenant_id IS OLD.tenant_id AND NEW.generation_id IS OLD.generation_id
    AND NEW.indexed_at IS OLD.indexed_at
    AND NEW.chunk_id = 'redacted' AND NEW.source_digest = 'redacted'
    AND json_extract(NEW.card, '$.card_id') IS NEW.card_id
    AND json_extract(NEW.card, '$.redacted') IS 1
    AND json_extract(NEW.card, '$.redaction_id') IS (SELECT max(redaction_id) FROM recall_redaction_session)
    AND json_extract(NEW.card, '$.tenant_id') IS json_extract(OLD.card, '$.tenant_id')
    AND json_extract(NEW.card, '$.generation_id') IS json_extract(OLD.card, '$.generation_id')
    AND json_extract(NEW.card, '$.source') IS json_extract(OLD.card, '$.source')
    AND (SELECT count(*) FROM json_each(NEW.card)) = 6
)
BEGIN SELECT RAISE(ABORT, 'recall evidence card redaction may change only what it erases'); END;
CREATE TRIGGER IF NOT EXISTS {EVIDENCE_CARD_TABLE}_no_delete BEFORE DELETE ON {EVIDENCE_CARD_TABLE}
BEGIN SELECT RAISE(ABORT, 'recall evidence cards are immutable'); END;
"""


def _put_cards(conn: Any, tenant_id: str, cards: tuple[EvidenceCard, ...]) -> None:
    """Insert `cards` on a caller-held connection whose transaction and tenant are already set.

    The one implementation of `PostgresEvidenceCardStore.put`, shared with the MCP evidence path
    so it can write on the serving store's own connection instead of opening a new one per call.
    Idempotent: a card already present with identical content is accepted, a different card under
    the same identity is refused.
    """
    for card in cards:
        payload = dict(card.to_payload())
        inserted = conn.execute(
            f"INSERT INTO {EVIDENCE_CARD_TABLE} "
            "(card_id, tenant_id, generation_id, chunk_id, source_digest, card, indexed_at) "
            "VALUES (%s, %s, %s, %s, %s, %s, %s) "
            "ON CONFLICT (card_id) DO NOTHING RETURNING card_id",
            (
                card.card_id,
                tenant_id,
                card.generation_id,
                card.chunk_id,
                card.source_digest,
                Jsonb(payload),
                card.indexed_at,
            ),
        ).fetchone()
        if inserted is not None:
            continue
        existing = conn.execute(
            f"SELECT card FROM {EVIDENCE_CARD_TABLE} WHERE card_id = %s",
            (card.card_id,),
        ).fetchone()
        existing_card = (
            evidence_card_from_payload(existing[0])
            if existing is not None and isinstance(existing[0], Mapping)
            else None
        )
        if existing_card != card:
            raise ValueError(f"evidence card identity collision for {card.card_id}")


def _sqlite_put_cards(conn: sqlite3.Connection, tenant_id: str, cards: tuple[EvidenceCard, ...]) -> None:
    """Insert `cards` on a SQLite connection whose transaction the caller holds.

    `SQLiteEvidenceCardStore.put` and the lite store's card store share it: a card already present
    with identical content is accepted, a different card under the same identity is refused.
    """
    for card in cards:
        encoded = json.dumps(card.to_payload(), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        conn.execute(
            f"INSERT OR IGNORE INTO {EVIDENCE_CARD_TABLE} "
            "(card_id, tenant_id, generation_id, chunk_id, source_digest, card, indexed_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                card.card_id,
                tenant_id,
                card.generation_id,
                card.chunk_id,
                card.source_digest,
                encoded,
                card.indexed_at.isoformat() if card.indexed_at else None,
            ),
        )
        existing = conn.execute(
            f"SELECT card FROM {EVIDENCE_CARD_TABLE} WHERE card_id = ?",
            (card.card_id,),
        ).fetchone()
        existing_card = evidence_card_from_payload(json.loads(existing[0])) if existing is not None else None
        if existing_card != card:
            raise ValueError(f"evidence card identity collision for {card.card_id}")


def _sqlite_resolved_card(card_id: str, tenant_id: str, row: tuple[Any, ...]) -> EvidenceCard | None:
    """Revalidate a card read back from SQLite: `(generation_id, chunk_id, digest, indexed_at, card)`.

    None for a card `recall_forget` erased: it supports nothing, as a card never written would not.
    """
    generation_id, chunk_id, source_digest_value, indexed_at, encoded = row
    payload = json.loads(encoded)
    if not isinstance(payload, Mapping):
        raise ValueError("durable evidence card payload is not an object")
    card = evidence_card_from_payload(payload)
    if card.card_id != card_id or card.tenant_id != tenant_id:
        raise ValueError("durable evidence card identity mismatch")
    if isinstance(card, RedactedEvidenceCard):
        return None
    stored_indexed_at = card.indexed_at.isoformat() if card.indexed_at is not None else None
    if (
        card.generation_id != generation_id
        or card.chunk_id != chunk_id
        or card.source_digest != source_digest_value
        or stored_indexed_at != indexed_at
    ):
        raise ValueError("durable evidence card projection mismatch")
    return card


class PostgresEvidenceCardStore:
    """Tenant-scoped immutable evidence-card projection."""

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
        conn.execute(f"SELECT set_config('{TENANT_GUC}', %s, false)", (self.tenant_id,))
        conn.execute(f"SET statement_timeout = {int(self.statement_timeout_ms)}")
        return conn

    def put(self, cards: Iterable[EvidenceCard]) -> None:
        """Insert cards idempotently, rejecting an identity collision with different content."""
        materialized = tuple(cards)
        if any(card.tenant_id != self.tenant_id for card in materialized):
            raise ValueError("evidence card tenant mismatch")
        if not materialized:
            return
        with self._connect() as conn, conn.transaction():
            _put_cards(conn, self.tenant_id, materialized)

    def resolve(self, card_id: str) -> EvidenceCard | None:
        """Resolve and revalidate one card from the authoritative durable projection."""
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT generation_id, chunk_id, source_digest, indexed_at, card "
                f"FROM {EVIDENCE_CARD_TABLE} "
                "WHERE tenant_id = %s AND card_id = %s",
                (self.tenant_id, card_id),
            ).fetchone()
        if row is None:
            return None
        generation_id, chunk_id, source_digest_value, indexed_at, payload = row
        if not isinstance(payload, Mapping):
            raise ValueError("durable evidence card payload is not an object")
        card = evidence_card_from_payload(payload)
        if card.card_id != card_id or card.tenant_id != self.tenant_id:
            raise ValueError("durable evidence card identity mismatch")
        if isinstance(card, RedactedEvidenceCard):
            return None  # erased by `recall_forget`: it supports nothing
        if (
            card.generation_id != generation_id
            or card.chunk_id != chunk_id
            or card.source_digest != source_digest_value
            or card.indexed_at != indexed_at
        ):
            raise ValueError("durable evidence card projection mismatch")
        return card


class SQLiteEvidenceCardStore(EvidenceCardStore):
    """Durable local card projection with the same immutable contract as PostgreSQL."""

    def __init__(self, path: str, *, tenant_id: str) -> None:
        if not path or not tenant_id:
            raise ValueError("path and tenant_id must be non-empty")
        super().__init__()
        self.path = path
        self.tenant_id = tenant_id
        self._conn = sqlite3.connect(path, check_same_thread=False, isolation_level=None)
        self._conn.executescript(SQLITE_EVIDENCE_CARD_DDL)

    def put(self, cards: Iterable[EvidenceCard]) -> None:
        materialized = tuple(cards)
        if any(card.tenant_id != self.tenant_id for card in materialized):
            raise ValueError("evidence card tenant mismatch")
        if not materialized:
            return
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            _sqlite_put_cards(self._conn, self.tenant_id, materialized)
            self._conn.execute("COMMIT")
        except Exception:  # BROAD-CATCH: fail-closed
            self._conn.execute("ROLLBACK")
            raise

    def resolve(self, card_id: str) -> EvidenceCard | None:
        row = self._conn.execute(
            f"SELECT generation_id, chunk_id, source_digest, indexed_at, card "
            f"FROM {EVIDENCE_CARD_TABLE} WHERE tenant_id = ? AND card_id = ?",
            (self.tenant_id, card_id),
        ).fetchone()
        if row is None:
            return None
        card = _sqlite_resolved_card(card_id, self.tenant_id, tuple(row))
        return card

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteEvidenceCardStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["EVIDENCE_CARD_TABLE", "PostgresEvidenceCardStore", "SQLiteEvidenceCardStore"]
