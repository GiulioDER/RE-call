"""Public inference proposal protocol types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal, Protocol

from recall.reasoning_graph import ReasoningGraphNode, ReasoningGraphProjection

PROPOSAL_SCHEMA_VERSION = 1
DETERMINISTIC_PROVIDER_ID = "recall.deterministic"
DETERMINISTIC_MODEL_ID = "rules"
DETERMINISTIC_PROVIDER_REVISION = "session3-v1"

ProposalStatus = Literal["candidate", "rejected", "requires_review"]
ProposedRelation = Literal["supersedes", "contradicts", "same_entity", "references"]
ProviderFailureKind = Literal["timeout", "malformed_output", "wrong_cardinality", "provider_error"]
PROVIDER_FAILURE_KINDS: tuple[ProviderFailureKind, ...] = (
    "timeout",
    "malformed_output",
    "wrong_cardinality",
    "provider_error",
)


def _freeze_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                key: _freeze_value(item)
                for key, item in sorted(value.items(), key=lambda entry: str(entry[0]))
            }
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_value(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((_freeze_value(item) for item in value), key=repr))
    return value


@dataclass(frozen=True)
class ProposalContext:
    tenant_id: str
    generation_id: str
    pipeline_id: str
    provider_id: str
    model_id: str
    provider_revision: str


@dataclass(frozen=True)
class EvidenceClaim:
    id: str
    evidence_id: str
    text: str
    entity_id: str | None = None
    subject: str | None = None
    valid_from: datetime | None = None
    valid_until: datetime | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))


@dataclass(frozen=True)
class EntityResolution:
    id: str
    evidence_ids: tuple[str, ...]
    canonical_entity: str
    aliases: tuple[str, ...] = ()
    confidence: float | None = None
    uncertainty: tuple[str, ...] = ()


@dataclass(frozen=True)
class InferenceProposal:
    id: str
    source_evidence_ids: tuple[str, ...]
    proposed_relation: ProposedRelation
    subject_id: str
    object_id: str
    explanation: str
    model_id: str
    pipeline_id: str
    provider_id: str
    provider_revision: str
    confidence: float | None
    uncertainty: tuple[str, ...]
    generation_id: str
    status: ProposalStatus = "candidate"
    rule_id: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "source_evidence_ids", tuple(self.source_evidence_ids))
        object.__setattr__(self, "uncertainty", tuple(self.uncertainty))
        object.__setattr__(self, "metadata", _freeze_value(self.metadata))


@dataclass(frozen=True)
class ProviderFailure:
    kind: ProviderFailureKind
    provider_id: str
    model_id: str
    provider_revision: str
    message: str


@dataclass(frozen=True)
class ProposalProtocolReport:
    schema_version: int
    generation_id: str
    pipeline_id: str
    proposals: tuple[InferenceProposal, ...]
    rejected_proposals: tuple[InferenceProposal, ...]
    provider_failures: tuple[ProviderFailure, ...]

    @property
    def failure_matrix(self) -> Mapping[str, int]:
        counts: dict[str, int] = {kind: 0 for kind in PROVIDER_FAILURE_KINDS}
        for failure in self.provider_failures:
            counts[failure.kind] += 1
        return MappingProxyType(counts)


class ClaimExtractor(Protocol):
    def extract_claims(
        self, evidence: Sequence[ReasoningGraphNode], context: ProposalContext
    ) -> Sequence[EvidenceClaim]:
        ...


class EntityResolver(Protocol):
    def resolve_entities(
        self, claims: Sequence[EvidenceClaim], context: ProposalContext
    ) -> Sequence[EntityResolution]:
        ...


class RelationProposer(Protocol):
    def propose_relations(
        self,
        claims: Sequence[EvidenceClaim],
        entities: Sequence[EntityResolution],
        context: ProposalContext,
    ) -> Sequence[InferenceProposal]:
        ...


class ContradictionDetector(Protocol):
    def detect_contradictions(
        self, claims: Sequence[EvidenceClaim], context: ProposalContext
    ) -> Sequence[InferenceProposal]:
        ...


class ModelBackedProposalProvider(Protocol):
    """Optional provider port. Implementations may call a model, this module never does."""

    provider_id: str
    model_id: str
    provider_revision: str
    max_proposals: int

    def propose(
        self, graph: ReasoningGraphProjection, context: ProposalContext
    ) -> Sequence[InferenceProposal | Mapping[str, Any]]:
        ...
