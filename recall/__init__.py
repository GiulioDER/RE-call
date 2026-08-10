"""recall — Retrieval-Augmented Self-Recall for long-running agents."""

from recall.calibration_v2 import CalibrationArtifactV2, CalibrationStatus

# The generator-neutral evidence boundary. Exported here because a guarantee reachable only by
# importing a module nothing references is a guarantee nobody applies: `recall/evidence.py` was
# complete and correct, and its only importer in the whole repository was its own test.
# `recall.evidence` imports `recall.types` and the standard library, so this adds no dependency
# and no import-time work to the package.
from recall.evidence import (
    AnswerEnvelope,
    EvidenceBundle,
    EvidenceItem,
    EvidencePolicy,
    EvidenceValidationError,
    GenerationResult,
    Tokenizer,
    ValidationResult,
    build_evidence_bundle,
    generate_from_evidence,
    normalize_citations,
    parse_answer_envelope,
    render_evidence_prompt,
    validate_answer,
)
from recall.lineage import (
    ChunkerIdentity,
    EmbedderIdentity,
    GenerationState,
    IndexManifestV1,
    PipelineIdentity,
)
from recall.reasoning_graph import (
    ReasoningGraphDiagnostic,
    ReasoningGraphEdge,
    ReasoningGraphNode,
    ReasoningGraphProjection,
    build_reasoning_graph,
    project_store_graph,
)
from recall.reasoning_proposals import (
    ClaimExtractor,
    ContradictionDetector,
    EntityResolution,
    EntityResolver,
    EvidenceClaim,
    InferenceProposal,
    ModelBackedProposalProvider,
    ProposalContext,
    ProposalProtocolReport,
    RelationProposer,
    deterministic_inference_proposals,
    proposal_precision_recall,
    proposal_report,
    proposal_to_graph_edge,
)

__version__ = "0.9.2"

__all__ = [
    "AnswerEnvelope",
    "CalibrationArtifactV2",
    "CalibrationStatus",
    "ChunkerIdentity",
    "EmbedderIdentity",
    "EvidenceBundle",
    "EvidenceItem",
    "EvidencePolicy",
    "EvidenceValidationError",
    "GenerationResult",
    "GenerationState",
    "IndexManifestV1",
    "ClaimExtractor",
    "ContradictionDetector",
    "EntityResolution",
    "EntityResolver",
    "EvidenceClaim",
    "InferenceProposal",
    "ModelBackedProposalProvider",
    "PipelineIdentity",
    "ProposalContext",
    "ProposalProtocolReport",
    "ReasoningGraphDiagnostic",
    "ReasoningGraphEdge",
    "ReasoningGraphNode",
    "ReasoningGraphProjection",
    "RelationProposer",
    "Tokenizer",
    "ValidationResult",
    "build_evidence_bundle",
    "build_reasoning_graph",
    "deterministic_inference_proposals",
    "generate_from_evidence",
    "normalize_citations",
    "parse_answer_envelope",
    "proposal_precision_recall",
    "proposal_report",
    "proposal_to_graph_edge",
    "project_store_graph",
    "render_evidence_prompt",
    "validate_answer",
]
