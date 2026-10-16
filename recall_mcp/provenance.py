"""MCP provenance and fact projection application boundary."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import UTC, datetime
from typing import TYPE_CHECKING

from recall import eval_clock
from recall.evidence import cards_from_trusted_result
from recall.fact_ledger import PostgresFactLedger, project_current
from recall.fact_retraction import served_sources
from recall.frontmatter import validity_bounds
from recall.generations import NoActiveGeneration
from recall.provenance_cards import PostgresEvidenceCardStore
from recall_mcp.evidence_cards import register_evidence_cards  # re-exported
from recall.provenance_controller import (
    EvidenceCardResolver,
    FactApplicationRequest,
    FactEvent,
    FactLedger,
    ProvenanceController,
    live_sources,
    source_digest,
    split_contested,
)
from recall.trust_policy import TrustPolicy
from recall.types import AtomicFact, EvidenceCard
from recall_mcp.settings import runtime_environment

if TYPE_CHECKING:
    from recall.lite import LiteStore
    from recall.embeddings import Embedder
    from recall.security_policy import AccessContext, SourceSecurityPolicy
    from recall.store import PgVectorStore

FACT_WRITE_DSN_ENV = "RECALL_FACT_WRITE_DSN"


def _fact_write_dsn(store: PgVectorStore) -> str:
    """Resolve the isolated controller DSN, falling back for legacy single-role installs."""
    configured = runtime_environment().get(FACT_WRITE_DSN_ENV)
    return configured.strip() if configured and configured.strip() else store.dsn


def _fact_backends(store: PgVectorStore | LiteStore) -> tuple[FactLedger, EvidenceCardResolver]:
    """The ledger and card store for `store`: the lite file's own, or the PostgreSQL ones.

    A lite file serves one local client, so `RECALL_FACT_WRITE_DSN` (a separate controller role
    on PostgreSQL) has nothing to name there and is not read.
    """
    from recall.lite import LiteStore

    if isinstance(store, LiteStore):
        from recall.lite.facts import LiteEvidenceCardStore, LiteFactLedger

        return LiteFactLedger(store), LiteEvidenceCardStore(store)
    return (
        PostgresFactLedger(_fact_write_dsn(store), tenant_id=store.tenant),
        PostgresEvidenceCardStore(store.dsn, tenant_id=store.tenant),
    )


def apply_fact_memory(
    store: PgVectorStore,
    embedder: Embedder,
    *,
    claim: Mapping[str, object],
    evidence_card_ids: Sequence[str],
    request_id: str,
    writer: str,
    policy: TrustPolicy | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> dict[str, object]:
    """Apply one structured fact through the external provenance controller."""
    fact = AtomicFact.from_payload(dict(claim))
    request = FactApplicationRequest(fact, tuple(evidence_card_ids), request_id)
    ledger, card_store = _fact_backends(store)

    def revalidate_card(card: EvidenceCard) -> EvidenceCard | None:
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
        from recall_mcp.retrieval import _retrieve_trusted

        query = f"{_fact.subject} {_fact.predicate} {json.dumps(_fact.object, ensure_ascii=False)}"
        retrieval = _retrieve_trusted(
            store,
            embedder,
            query,
            None,
            10,
            None,
            policy,
            security_policy=security_policy,
            access_context=access_context,
        )
        cards = cards_from_trusted_result(retrieval.result)
        register_evidence_cards(cards, store=store)
        return tuple(card.card_id for card in cards)

    controller = ProvenanceController(
        tenant_id=store.tenant,
        generation_id=store.generation_id,
        cards=card_store,
        ledger=ledger,
        source_digest_for=current_digest,
        card_revalidator=revalidate_card,
        fresh_search=fresh_search,
        writer=writer,
        served_sources=served_sources(store),
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
    store: PgVectorStore, *, as_of: datetime | None = None,
    security_policy: SourceSecurityPolicy | None = None,
    access_context: AccessContext | None = None,
) -> dict[str, object]:
    """Return the ledger's deterministic current fact projection for this tenant.

    A fact none of whose memos the store serves now (deleted, pruned, or left out of the active
    generation) is not listed, and is listed again when a memo returns
    (`recall.fact_retraction`). A fact lists its memos in `sources`. Facts that conflict, with no
    supersession between their memos, are listed under `contested` and not under `facts`
    (`recall.provenance_controller.split_contested`).
    """
    if security_policy is not None:
        if access_context is None or access_context.tenant != store.tenant:
            raise PermissionError("a matching source access context is required for current facts")
    instant = as_of or eval_clock.now() or datetime.now(UTC)
    served = served_sources(store)
    ledger = _fact_backends(store)[0]
    # One history snapshot serves both the projection and its supporting source names.
    history = getattr(ledger, "events", None)
    if history is not None:
        history = tuple(event for event in history if event.tenant_id == store.tenant)
        events = project_current(history, instant, served)
    else:
        events = (
            ledger.current(tenant_id=store.tenant, now=instant)
            if served is None
            else ledger.current(tenant_id=store.tenant, now=instant, served=served)
        )
        history = events
    sources = live_sources(history, served)
    if security_policy is not None:
        assert access_context is not None

        def permitted(name: str) -> bool:
            decision = security_policy.decide(name, access_context)
            # Structured values cannot safely be projected through a text redaction rule.
            return decision.allowed and not decision.redactions

        sources = {fact_id: tuple(name for name in names if permitted(name))
                   for fact_id, names in sources.items()}
        events = tuple(event for event in events if sources.get(event.fact_id or ""))

    def card_ids(event: FactEvent) -> list[str]:
        allowed = sources.get(event.fact_id or "", ())
        return [card.card_id for card in event.evidence_cards
                if security_policy is None or card.source in allowed]

    standing, contested = split_contested(events)
    try:
        generation_id: str | None = store.generation_id
    except NoActiveGeneration:
        # The ledger is tenant scoped and answers before any generation is promoted; say there
        # is no serving generation rather than raise or name a generation that does not exist.
        generation_id = None

    def names(event: FactEvent) -> list[str]:
        return list(sources.get(event.fact_id or "", ()))

    return {
        "tenant_id": store.tenant,
        "generation_id": generation_id,
        "as_of": instant.isoformat(),
        "facts": [
            {
                "event_id": event.event_id,
                "fact_id": event.fact_id,
                "fact": event.fact.to_payload() if event.fact else None,
                "evidence_card_ids": card_ids(event),
                "generation_id": event.generation_id,
                "writer": event.writer,
                "asserted_at": event.created_at.isoformat(),
                "sources": names(event),
            }
            for event in standing
        ],
        "contested": [
            {
                "namespace": group[0].fact.namespace if group[0].fact else None,
                "subject": group[0].fact.subject if group[0].fact else None,
                "predicate": group[0].fact.predicate if group[0].fact else None,
                "context": group[0].fact.context if group[0].fact else {},
                "values": [
                    {
                        "fact": event.fact.to_payload() if event.fact else None,
                        "fact_id": event.fact_id,
                        "sources": names(event),
                        "evidence_card_ids": card_ids(event),
                    }
                    for event in group
                ],
            }
            for group in contested
        ],
    }
