"""Persisting server-created evidence cards: a leaf shared by retrieval and provenance.

Moved out of `recall_mcp.provenance`, which re-exports it, so that `recall_mcp.retrieval` can
register cards without importing provenance, which imports retrieval for its fresh search.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from recall.provenance_cards import PostgresEvidenceCardStore, _put_cards

if TYPE_CHECKING:
    from recall.lite import LiteStore
    from recall.store import PgVectorStore
    from recall.types import EvidenceCard


def register_evidence_cards(
    cards: Sequence[EvidenceCard], *, store: PgVectorStore | LiteStore | None = None
) -> None:
    """Persist server-created cards in the store's own database: PostgreSQL, or a lite file.

    Only the store's database is written. A process-wide in-memory `EvidenceCardStore` was also
    filled here on every `recall_evidence` call and read by nothing (`apply_fact_memory` resolves
    cards from the store's own database), so it grew without bound for the life of the server.
    """
    if store is None:
        return
    from recall.lite import LiteStore

    if isinstance(store, LiteStore):
        # The lite file keeps its own cards. Without this branch every `recall_apply_fact` on lite
        # would answer CARD_NOT_FOUND: the PostgreSQL branches below find no connection and return.
        from recall.lite.facts import LiteEvidenceCardStore

        LiteEvidenceCardStore(store).put(cards)
        return
    tenant = getattr(store, "tenant", None)
    borrow = getattr(store, "_with_retry", None)
    if isinstance(tenant, str) and callable(borrow):
        # On the store's own tenant-bound connection: the separate card store opened a new
        # psycopg connection, set the tenant and a timeout, and closed it, on every
        # `recall_evidence` call. Same DSN, so the same role and the same RLS policy.
        materialized = tuple(cards)
        if any(card.tenant_id != tenant for card in materialized):
            raise ValueError("evidence card tenant mismatch")
        if not materialized:
            return

        def _op(conn: Any) -> None:
            with conn.transaction():
                _put_cards(conn, tenant, materialized)

        borrow(_op)
        return
    dsn = getattr(store, "dsn", None)
    if isinstance(dsn, str) and isinstance(tenant, str):
        PostgresEvidenceCardStore(dsn, tenant_id=tenant).put(cards)
