"""Stable structured explanations for retrieval decisions."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


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


__all__ = ["RetrievalExplanation"]
