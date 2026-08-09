"""recall — Retrieval-Augmented Self-Recall for long-running agents."""

from recall.calibration_v2 import CalibrationArtifactV2, CalibrationStatus
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

__version__ = "0.9.0"

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
    "PipelineIdentity",
    "Tokenizer",
    "ValidationResult",
    "build_evidence_bundle",
    "generate_from_evidence",
    "normalize_citations",
    "parse_answer_envelope",
    "render_evidence_prompt",
    "validate_answer",
]
