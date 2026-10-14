"""Facts and evidence cards the fact-ledger tests share."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from recall.types import AtomicFact, EvidenceCard

NOW = datetime(2026, 10, 12, 12, tzinfo=UTC)
TEA = AtomicFact(namespace="memory", subject="user", predicate="drink", object="tea")
COFFEE = AtomicFact(namespace="memory", subject="user", predicate="drink", object="coffee")


def evidence_card(
    fact: AtomicFact,
    *,
    tenant: str,
    generation: str,
    chunk: str,
    source: str,
    links: tuple[str, ...] = (),
) -> EvidenceCard:
    """A trusted card for `fact`; its id is derived from its content when it is built."""
    return EvidenceCard(
        card_id="", chunk_id=chunk, source=source, source_digest="digest", valid_from=None,
        valid_until=None, first_indexed_at=NOW - timedelta(days=1), indexed_at=NOW,
        tenant_id=tenant, generation_id=generation, pipeline_fingerprint="p" * 64,
        corpus_fingerprint="c" * 64, calibration_id="cal", calibration_status="certified",
        trust_state="trusted", verdict="ok", confidence=0.99, rank=1,
        supersession_links=links, structured_facts=(fact,),
    )
