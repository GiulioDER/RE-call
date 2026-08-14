"""Public inference proposal protocol types."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from types import MappingProxyType
from typing import Any, Literal, Protocol

from recall.reasoning_graph import ReasoningGraphNode, ReasoningGraphProjection

#: Version 2 adds `declares_validity` and `declares_status` to `ProposedRelation`. Every
#: proposal id is a hash over this constant, so the bump changes every id in existence — that
#: is the point: an id minted under a vocabulary that could not express validity must not be
#: mistaken for one minted under a vocabulary that can.
PROPOSAL_SCHEMA_VERSION = 2
DETERMINISTIC_PROVIDER_ID = "recall.deterministic"
DETERMINISTIC_MODEL_ID = "rules"
DETERMINISTIC_PROVIDER_REVISION = "session3-v1"

ProposalStatus = Literal["candidate", "rejected", "requires_review"]
#: `declares_validity` and `declares_status` are NOT relations between two documents: they
#: are a document asserting something about itself. Forcing them into `references` would put
#: a false relation into an audit record, which is worse than having no record of them.
ProposedRelation = Literal[
    "supersedes",
    "contradicts",
    "same_entity",
    "references",
    "declares_validity",
    "declares_status",
]
#: Single source of truth for the vocabulary. `_providers._checked_relation` validates against
#: this rather than its own literal set, so the two cannot drift apart.
PROPOSED_RELATIONS: tuple[ProposedRelation, ...] = (
    "supersedes",
    "contradicts",
    "same_entity",
    "references",
    "declares_validity",
    "declares_status",
)
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
    """Stable execution identity supplied to deterministic and model proposal providers."""

    tenant_id: str
    generation_id: str
    pipeline_id: str
    provider_id: str
    model_id: str
    provider_revision: str


@dataclass(frozen=True)
class EvidenceClaim:
    """A provider-neutral claim extracted from one evidence item."""

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
    """A proposed canonical entity grouping for extracted evidence claims."""

    id: str
    evidence_ids: tuple[str, ...]
    canonical_entity: str
    aliases: tuple[str, ...] = ()
    confidence: float | None = None
    uncertainty: tuple[str, ...] = ()


@dataclass(frozen=True)
class InferenceProposal:
    """A side effect free candidate relationship between two evidence-backed subjects."""

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
    """A sanitized failure record for an optional model-backed proposal provider."""

    kind: ProviderFailureKind
    provider_id: str
    model_id: str
    provider_revision: str
    message: str


@dataclass(frozen=True)
class ProposalProtocolReport:
    """Complete proposal output for one graph generation and pipeline identity."""

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
    """Port for extracting claims from projected reasoning graph evidence."""

    def extract_claims(
        self, evidence: Sequence[ReasoningGraphNode], context: ProposalContext
    ) -> Sequence[EvidenceClaim]:
        ...


class EntityResolver(Protocol):
    """Port for clustering extracted claims into canonical entity candidates."""

    def resolve_entities(
        self, claims: Sequence[EvidenceClaim], context: ProposalContext
    ) -> Sequence[EntityResolution]:
        ...


class RelationProposer(Protocol):
    """Port for proposing relationships between claims and resolved entities."""

    def propose_relations(
        self,
        claims: Sequence[EvidenceClaim],
        entities: Sequence[EntityResolution],
        context: ProposalContext,
    ) -> Sequence[InferenceProposal]:
        ...


class ContradictionDetector(Protocol):
    """Port for proposing contradictions among extracted claims."""

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
