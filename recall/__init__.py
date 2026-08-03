"""recall — Retrieval-Augmented Self-Recall for long-running agents."""

from recall.lineage import (
    ChunkerIdentity,
    EmbedderIdentity,
    GenerationState,
    IndexManifestV1,
    PipelineIdentity,
)

__version__ = "0.8.0"

__all__ = [
    "ChunkerIdentity",
    "EmbedderIdentity",
    "GenerationState",
    "IndexManifestV1",
    "PipelineIdentity",
]
