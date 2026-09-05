"""Durable evidence-card projection used by the provenance controller.

Cards are immutable server-created projections.  The application request contains only a card
identifier; this store resolves the identifier again from PostgreSQL under the current tenant.
There is deliberately no update or delete API, and the database trigger is the final backstop.
"""
from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from typing import Any

import psycopg
from psycopg.types.json import Jsonb

from recall.provenance_controller import EvidenceCardStore, evidence_card_from_payload
from recall.store import TENANT_GUC
from recall.types import EvidenceCard


EVIDENCE_CARD_TABLE = "recall_evidence_cards"


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

    @staticmethod
    def _payload(card: EvidenceCard) -> dict[str, Any]:
        return dict(card.to_payload())

    def put(self, cards: Iterable[EvidenceCard]) -> None:
        """Insert cards idempotently, rejecting an identity collision with different content."""
        materialized = tuple(cards)
        if any(card.tenant_id != self.tenant_id for card in materialized):
            raise ValueError("evidence card tenant mismatch")
        if not materialized:
            return
        with self._connect() as conn, conn.transaction():
            for card in materialized:
                payload = self._payload(card)
                inserted = conn.execute(
                    f"INSERT INTO {EVIDENCE_CARD_TABLE} "
                    "(card_id, tenant_id, generation_id, chunk_id, source_digest, card, indexed_at) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s) "
                    "ON CONFLICT (card_id) DO NOTHING RETURNING card_id",
                    (
                        card.card_id,
                        self.tenant_id,
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
        self._conn.execute(
            f"""CREATE TABLE IF NOT EXISTS {EVIDENCE_CARD_TABLE} (
                card_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                generation_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                source_digest TEXT NOT NULL,
                card TEXT NOT NULL,
                indexed_at TEXT
            )"""
        )
        self._conn.execute(
            f"CREATE INDEX IF NOT EXISTS {EVIDENCE_CARD_TABLE}_lookup_idx "
            "ON recall_evidence_cards (tenant_id, generation_id, chunk_id)"
        )
        self._conn.executescript(
            f"""CREATE TRIGGER IF NOT EXISTS {EVIDENCE_CARD_TABLE}_no_update
               BEFORE UPDATE ON {EVIDENCE_CARD_TABLE}
               BEGIN SELECT RAISE(ABORT, 'recall evidence cards are immutable'); END;
               CREATE TRIGGER IF NOT EXISTS {EVIDENCE_CARD_TABLE}_no_delete
               BEFORE DELETE ON {EVIDENCE_CARD_TABLE}
               BEGIN SELECT RAISE(ABORT, 'recall evidence cards are immutable'); END;"""
        )

    def put(self, cards: Iterable[EvidenceCard]) -> None:
        materialized = tuple(cards)
        if any(card.tenant_id != self.tenant_id for card in materialized):
            raise ValueError("evidence card tenant mismatch")
        if not materialized:
            return
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            for card in materialized:
                payload = card.to_payload()
                encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
                self._conn.execute(
                    f"INSERT OR IGNORE INTO {EVIDENCE_CARD_TABLE} "
                    "(card_id, tenant_id, generation_id, chunk_id, source_digest, card, indexed_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (
                        card.card_id,
                        self.tenant_id,
                        card.generation_id,
                        card.chunk_id,
                        card.source_digest,
                        encoded,
                        card.indexed_at.isoformat() if card.indexed_at else None,
                    ),
                )
                existing = self._conn.execute(
                    f"SELECT card FROM {EVIDENCE_CARD_TABLE} WHERE card_id = ?",
                    (card.card_id,),
                ).fetchone()
                existing_card = (
                    evidence_card_from_payload(json.loads(existing[0]))
                    if existing is not None
                    else None
                )
                if existing_card != card:
                    raise ValueError(f"evidence card identity collision for {card.card_id}")
            self._conn.execute("COMMIT")
        except Exception:
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
        generation_id, chunk_id, source_digest_value, indexed_at, encoded = row
        payload = json.loads(encoded)
        if not isinstance(payload, Mapping):
            raise ValueError("durable evidence card payload is not an object")
        card = evidence_card_from_payload(payload)
        if card.card_id != card_id or card.tenant_id != self.tenant_id:
            raise ValueError("durable evidence card identity mismatch")
        if card.indexed_at is not None:
            stored_indexed_at = card.indexed_at.isoformat()
        else:
            stored_indexed_at = None
        if (
            card.generation_id != generation_id
            or card.chunk_id != chunk_id
            or card.source_digest != source_digest_value
            or stored_indexed_at != indexed_at
        ):
            raise ValueError("durable evidence card projection mismatch")
        return card

    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> "SQLiteEvidenceCardStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


__all__ = ["EVIDENCE_CARD_TABLE", "PostgresEvidenceCardStore", "SQLiteEvidenceCardStore"]
