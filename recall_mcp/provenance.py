"""MCP provenance and fact projection application boundary."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from recall.evidence import cards_from_trusted_result
from recall.fact_ledger import PostgresFactLedger
from recall.frontmatter import validity_bounds
from recall.provenance_cards import PostgresEvidenceCardStore
from recall.provenance_controller import (
    FactApplicationRequest,
    ProvenanceController,
    source_digest,
)
from recall.trust_policy import TrustPolicy
from recall.types import AtomicFact, EvidenceCard

from recall_mcp.retrieval import _retrieve_trusted, register_evidence_cards

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.store import PgVectorStore

FACT_WRITE_DSN_ENV = "RECALL_FACT_WRITE_DSN"


def _fact_write_dsn(store: PgVectorStore) -> str:
    """Resolve the isolated controller DSN, falling back for legacy single-role installs."""
    configured = os.environ.get(FACT_WRITE_DSN_ENV)
    return configured.strip() if configured and configured.strip() else store.dsn


def apply_fact_memory(
    store: PgVectorStore,
    embedder: Embedder,
    *,
    claim: Mapping[str, object],
    evidence_card_ids: Sequence[str],
    request_id: str,
    writer: str,
    policy: TrustPolicy | None = None,
) -> dict[str, object]:
    """Apply one structured fact through the external provenance controller.

    The request accepts only claim fields and opaque card ids. Trust and lineage are loaded from
    the server-owned card registry and the current tenant-bound store.
    """
    fact = AtomicFact.from_payload(dict(claim))
    request = FactApplicationRequest(fact, tuple(evidence_card_ids), request_id)
    ledger = PostgresFactLedger(_fact_write_dsn(store), tenant_id=store.tenant)

    def revalidate_card(card: EvidenceCard) -> EvidenceCard | None:
        """Rebuild source-derived card fields from the currently served generation.

        Retrieval-only fields such as rank and calibrated trust remain bound to the immutable
        card projection. Source identity, validity, structured support, and authored links are
        read again immediately before authorization. A changed projection receives a different
        card id and therefore fails closed, which sends the controller through its one fresh
        search recovery path.
        """
        chunk = store.chunk_by_id(card.chunk_id)
        if chunk is None:
            return None
        metadata = chunk.metadata or {}
        try:
            valid_from, valid_until = validity_bounds(metadata)
        except ValueError:
            return None
        graph = metadata.get("recall_graph", {})
        if not isinstance(graph, Mapping):
            graph = {}
        raw_facts = graph.get("facts", metadata.get("facts", ()))
        structured_facts: list[AtomicFact] = []
        if isinstance(raw_facts, Sequence) and not isinstance(raw_facts, (str, bytes, bytearray)):
            for item in raw_facts:
                if isinstance(item, Mapping):
                    try:
                        structured_facts.append(AtomicFact.from_payload(item))
                    except (TypeError, ValueError, KeyError):
                        return None

        def links(key: str) -> tuple[str, ...]:
            raw_values = [graph.get(key, metadata.get(key, ()))]
            if key == "authored_supersedes":
                raw_values.append(metadata.get("supersedes"))
            values: list[str] = []
            for raw in raw_values:
                if isinstance(raw, str):
                    raw = (raw,)
                if isinstance(raw, Sequence) and not isinstance(raw, (bytes, bytearray)):
                    values.extend(item for item in raw if isinstance(item, str) and item)
            return tuple(dict.fromkeys(values))

        source = metadata.get("file") or chunk.source
        if not isinstance(source, str) or not source:
            return None
        declared_digest = metadata.get("content_hash") or metadata.get("source_digest")
        digest = (
            str(declared_digest)
            if isinstance(declared_digest, str) and declared_digest
            else source_digest(chunk.text)
        )
        return replace(
            card,
            card_id="",
            source=source,
            source_digest=digest,
            valid_from=valid_from,
            valid_until=valid_until,
            structured_facts=tuple(structured_facts),
            supersession_links=links("authored_supersedes"),
            contradiction_links=links("authored_contradicts"),
            support_refs=links("support_refs"),
        )

    def current_digest(card: EvidenceCard) -> str | None:
        chunk = store.chunk_by_id(card.chunk_id)
        if chunk is None:
            return None
        metadata = chunk.metadata or {}
        declared = metadata.get("content_hash") or metadata.get("source_digest")
        return (
            str(declared) if isinstance(declared, str) and declared else source_digest(chunk.text)
        )

    def fresh_search(_fact: AtomicFact, _request: FactApplicationRequest) -> Sequence[str]:
        query = f"{_fact.subject} {_fact.predicate} {json.dumps(_fact.object, ensure_ascii=False)}"
        retrieval = _retrieve_trusted(store, embedder, query, None, 10, None, policy)
        cards = cards_from_trusted_result(retrieval.result)
        register_evidence_cards(cards, store=store)
        return tuple(card.card_id for card in cards)

    card_store = PostgresEvidenceCardStore(store.dsn, tenant_id=store.tenant)
    controller = ProvenanceController(
        tenant_id=store.tenant,
        generation_id=store.generation_id,
        cards=card_store,
        ledger=ledger,
        source_digest_for=current_digest,
        card_revalidator=revalidate_card,
        fresh_search=fresh_search,
        writer=writer,
    )
    decision = controller.apply_fact(request)
    return {
        "allowed": decision.allowed,
        "decision_code": str(decision.code),
        "request_id": decision.request_id,
        "fact_id": decision.fact_id,
        "retried": decision.retried,
        "detail": decision.detail,
        "event_id": decision.event.event_id if decision.event else None,
        "evidence_card_ids": [card.card_id for card in decision.cards],
    }


def current_facts_memory(
    store: PgVectorStore, *, as_of: datetime | None = None
) -> dict[str, object]:
    """Return the ledger's deterministic current fact projection for this tenant."""
    instant = as_of or datetime.now(UTC)
    events = PostgresFactLedger(_fact_write_dsn(store), tenant_id=store.tenant).current(
        tenant_id=store.tenant, now=instant
    )
    return {
        "tenant_id": store.tenant,
        "generation_id": store.generation_id,
        "as_of": instant.isoformat(),
        "facts": [
            {
                "event_id": event.event_id,
                "fact_id": event.fact_id,
                "fact": event.fact.to_payload() if event.fact else None,
                "evidence_card_ids": [card.card_id for card in event.evidence_cards],
                "generation_id": event.generation_id,
                "writer": event.writer,
                "asserted_at": event.created_at.isoformat(),
            }
            for event in events
        ],
    }
