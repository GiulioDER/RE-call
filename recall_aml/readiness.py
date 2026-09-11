"""Live startup probes for mandatory hosted model dependencies."""

from __future__ import annotations

from recall.embeddings import Embedder, embed_query
from recall.rerank import Reranker
from recall.types import Chunk, ScoredChunk
from recall_aml.compiler import Compiler
from recall_aml.variants import HostedVariant


def verify_model_readiness(
    *,
    embedder: Embedder,
    compiler: Compiler | None,
    reranker: Reranker,
    behavior: HostedVariant,
) -> dict[str, bool]:
    """Make one bounded live call to every provider stage used by the active variant."""
    vector = embed_query(embedder, "RE-call Hosted readiness probe")
    if len(vector) != embedder.dim:
        raise RuntimeError("embedding readiness probe returned the wrong dimension")
    status = {"embedder_ready": True, "compiler_ready": False, "reranker_ready": False}

    if behavior.compiler or behavior.facets:
        if compiler is None:
            raise RuntimeError(f"{behavior.name} has no compiler client")
        compiler.facets("RE-call Hosted readiness probe", {"choices": []})
        status["compiler_ready"] = True

    if behavior.reranker:
        probe = ScoredChunk(
            Chunk(
                id="readiness-probe",
                source="readiness://probe",
                text="RE-call Hosted readiness probe",
                metadata={"record_type": "readiness"},
            ),
            1.0,
        )
        result = reranker.rerank("RE-call Hosted readiness probe", [probe])
        if len(result) != 1 or result[0].chunk.id != probe.chunk.id:
            raise RuntimeError("reranker readiness probe returned an invalid result")
        status["reranker_ready"] = True
    return status
