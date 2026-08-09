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
