"""recall — Retrieval-Augmented Self-Recall for long-running agents."""

from recall.calibration_v2 import CalibrationArtifactV2, CalibrationStatus
from recall.lineage import (
    ChunkerIdentity,
    EmbedderIdentity,
    GenerationState,
    IndexManifestV1,
    PipelineIdentity,
)

__version__ = "0.8.0"

__all__ = [
    "CalibrationArtifactV2",
    "CalibrationStatus",
    "ChunkerIdentity",
    "EmbedderIdentity",
    "GenerationState",
    "IndexManifestV1",
    "PipelineIdentity",
]
