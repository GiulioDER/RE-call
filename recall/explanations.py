"""Stable structured explanations for retrieval decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from collections.abc import Collection, Sequence
from typing import Any

from recall.types import TrustedHit


@dataclass(frozen=True)
class RetrievalExplanation:
    """Machine readable retrieval provenance with corpus text kept out of instruction fields."""

    query_class: str
    routing_profile: str
    routing_policy_version: str
    routing_mode: str = "shadow"
    matched_rules: tuple[str, ...] = ()
    expansion_mode: str | None = None
    candidate_pool_size: int = 0
    stage_names: tuple[str, ...] = ()
    selection_reason: str | None = None
    trust_reason: str | None = None
    abstention_reason: str | None = None
    related_seed_chunk_id: str | None = None
    related_relation: str | None = None
    generation_id: str = "legacy"
    details: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "query_class": self.query_class,
            "routing_profile": self.routing_profile,
            "routing_policy_version": self.routing_policy_version,
            "routing_mode": self.routing_mode,
            "matched_rules": list(self.matched_rules),
            "expansion_mode": self.expansion_mode,
            "candidate_pool_size": self.candidate_pool_size,
            "stage_names": list(self.stage_names),
            "selection_reason": self.selection_reason,
            "trust_reason": self.trust_reason,
            "abstention_reason": self.abstention_reason,
            "related_seed_chunk_id": self.related_seed_chunk_id,
            "related_relation": self.related_relation,
            "generation_id": self.generation_id,
            "details": dict(self.details),
        }


def memory_audit(
    hits: Sequence[TrustedHit], *, context_chunk_ids: Collection[str] | None = None
) -> dict[str, Any]:
    """Return bounded, deterministic observability for one memory decision.

    This is deliberately a retrieval audit, not a quality score. Counts that the retrieval layer
    can observe are reported directly. Retrieval precision, forgetting, and downstream action
    influence remain ``not_measured`` because they require labels or feedback that this call does
    not possess. No corpus text or identifiers are copied into the explanation.

    ``context_chunk_ids`` identifies the passages admitted to an evidence context. It is optional
    because ``recall_search`` returns the retrieval set itself, while ``recall_evidence`` has a
    second bounded selection step. The selection ratio describes boundary utilization, not whether
    a generator actually used a passage in its answer.
    """
    verdict_counts = Counter(hit.verdict for hit in hits)
    source_keys = {
        hit.provenance.file or hit.provenance.source
        for hit in hits
        if hit.provenance.file or hit.provenance.source
    }
    validity_declared = sum(
        hit.validity.valid_from is not None or hit.validity.valid_until is not None
        for hit in hits
    )
    supersession_declared = sum(hit.validity.superseded_by is not None for hit in hits)
    # A future memory and a memory absent at a historical instant are not stale. They are
    # temporal boundary results, so only an expired or superseded hit contributes here.
    stale_verdicts = {"superseded", "expired"}
    stale_count = sum(hit.verdict in stale_verdicts for hit in hits)
    retrieved_count = len(hits)
    audit: dict[str, Any] = {
        "retrieved_count": retrieved_count,
        "trusted_count": verdict_counts.get("ok", 0),
        "untrusted_count": retrieved_count - verdict_counts.get("ok", 0),
        "verdict_counts": dict(sorted(verdict_counts.items())),
        "distinct_source_count": len(source_keys),
        "source_diversity": round(len(source_keys) / retrieved_count, 4)
        if retrieved_count
        else None,
        "validity_declared_count": validity_declared,
        "validity_coverage": round(validity_declared / retrieved_count, 4)
        if retrieved_count
        else None,
        "supersession_declared_count": supersession_declared,
        "stale_count": stale_count,
        "stale_rate": round(stale_count / retrieved_count, 4) if retrieved_count else None,
        "quality": {
            "retrieval_precision": None,
            "status": "not_measured",
            "reason": "requires labelled queries or downstream judgement",
        },
        "forgetting": {
            "forget_rate": None,
            "status": "not_measured",
            "reason": "forgetting events are outside retrieval scope",
        },
        "influence": {
            "action_influenced": None,
            "status": "not_measured",
            "reason": "the consumer action is not supplied to retrieval",
        },
    }
    if context_chunk_ids is None:
        audit["context"] = {
            "selected_count": None,
            "selection_ratio": None,
            "status": "not_applicable",
        }
    else:
        selected_ids = set(context_chunk_ids)
        selected = [hit for hit in hits if hit.chunk.id in selected_ids]
        audit["context"] = {
            "selected_count": len(selected),
            "selection_ratio": round(len(selected) / retrieved_count, 4)
            if retrieved_count
            else None,
            "distinct_source_count": len(
                {
                    hit.provenance.file or hit.provenance.source
                    for hit in selected
                    if hit.provenance.file or hit.provenance.source
                }
            ),
            "status": "selected",
        }
    return audit


__all__ = ["RetrievalExplanation", "memory_audit"]
