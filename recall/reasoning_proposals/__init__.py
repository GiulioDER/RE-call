"""Side effect free inference proposal APIs for reasoning over evidence graphs."""

from recall.reasoning_proposals._deterministic import deterministic_inference_proposals
from recall.reasoning_proposals._graph import proposal_to_graph_edge
from recall.reasoning_proposals._metrics import proposal_precision_recall
from recall.reasoning_proposals._providers import proposal_report
from recall.reasoning_proposals.types import (
    ClaimExtractor,
    ContradictionDetector,
    DETERMINISTIC_MODEL_ID,
    DETERMINISTIC_PROVIDER_ID,
    DETERMINISTIC_PROVIDER_REVISION,
    EntityResolution,
    EntityResolver,
    EvidenceClaim,
    InferenceProposal,
    ModelBackedProposalProvider,
    PROPOSAL_SCHEMA_VERSION,
    PROVIDER_FAILURE_KINDS,
    ProposalContext,
    ProposalProtocolReport,
    ProposalStatus,
    ProposedRelation,
    ProviderFailure,
    ProviderFailureKind,
    RelationProposer,
)

__all__ = [
    "ClaimExtractor",
    "ContradictionDetector",
    "DETERMINISTIC_MODEL_ID",
    "DETERMINISTIC_PROVIDER_ID",
    "DETERMINISTIC_PROVIDER_REVISION",
    "EntityResolution",
    "EntityResolver",
    "EvidenceClaim",
    "InferenceProposal",
    "ModelBackedProposalProvider",
    "PROPOSAL_SCHEMA_VERSION",
    "PROVIDER_FAILURE_KINDS",
    "ProposalContext",
    "ProposalProtocolReport",
    "ProposalStatus",
    "ProposedRelation",
    "ProviderFailure",
    "ProviderFailureKind",
    "RelationProposer",
    "deterministic_inference_proposals",
    "proposal_precision_recall",
    "proposal_report",
    "proposal_to_graph_edge",
]
