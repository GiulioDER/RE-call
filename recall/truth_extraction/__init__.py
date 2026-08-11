"""Model backed extraction of structured truth claims from memo prose.

Retrieval acts on authored frontmatter. Prose that states the same relation is invisible to
it. This package turns prose into structured, quoted claims so that a human can review them
and, separately, declare them. It never writes corpus metadata and never runs on the query
path: extraction is an ingest concern.
"""

# Imported here rather than re-exported from `recall.reasoning_proposals`: the query
# planner imports that package, and pulling extraction in through it would put a model
# backed component on the query path. The dependency runs one way only.
from recall.reasoning_proposals._extracted import ExtractedClaimProposalProvider
from recall.truth_extraction._engine import (
    DETERMINISTIC_EXTRACTION_ENGINE_ID,
    DETERMINISTIC_EXTRACTION_MODEL_ID,
    DETERMINISTIC_EXTRACTION_REVISION,
    DeterministicExtractionEngine,
    ExtractionEngine,
    resolve_extraction_engine,
)
from recall.truth_extraction._normalize import human_body_of, normalize_extraction
from recall.truth_extraction._prompt import (
    PROMPT_REVISION,
    ExtractionPrompt,
    build_extraction_prompt,
)
from recall.truth_extraction.types import (
    BATCH_RUNGS,
    CLAIM_KINDS,
    CLAIM_RUNGS,
    ClaimKind,
    ClaimRejection,
    ExtractedClaim,
    ExtractionBatchRejected,
    FileExtraction,
    IdentityClaim,
    MAX_CLAIMS_PER_FILE,
    STATUS_VOCABULARY,
    StatusClaim,
    SupersessionClaim,
    VALIDITY_CLAIM_KEYS,
    ValidityClaim,
    ValidityKey,
)

__all__ = [
    "BATCH_RUNGS",
    "DETERMINISTIC_EXTRACTION_ENGINE_ID",
    "DETERMINISTIC_EXTRACTION_MODEL_ID",
    "DETERMINISTIC_EXTRACTION_REVISION",
    "DeterministicExtractionEngine",
    "ExtractedClaimProposalProvider",
    "ExtractionEngine",
    "ExtractionPrompt",
    "PROMPT_REVISION",
    "CLAIM_KINDS",
    "CLAIM_RUNGS",
    "ClaimKind",
    "ClaimRejection",
    "ExtractedClaim",
    "ExtractionBatchRejected",
    "FileExtraction",
    "IdentityClaim",
    "MAX_CLAIMS_PER_FILE",
    "STATUS_VOCABULARY",
    "StatusClaim",
    "SupersessionClaim",
    "VALIDITY_CLAIM_KEYS",
    "ValidityClaim",
    "ValidityKey",
    "build_extraction_prompt",
    "human_body_of",
    "normalize_extraction",
    "resolve_extraction_engine",
]
