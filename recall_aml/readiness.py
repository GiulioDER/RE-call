"""Live startup probes for mandatory hosted model dependencies."""

from __future__ import annotations

from recall.embeddings import Embedder, embed_query
from recall.rerank import Reranker
from recall.sparse import SparseEncoderProtocol
from recall.types import Chunk, ScoredChunk
from recall_aml.compiler import Compiler
from recall_aml.models import Message
from recall_aml.variants import HostedVariant


def verify_model_readiness(
    *,
    embedder: Embedder,
    compiler: Compiler | None,
    reranker: Reranker,
    sparse_encoder: SparseEncoderProtocol | None = None,
    behavior: HostedVariant,
) -> dict[str, bool]:
    """Make one bounded live call to every provider stage used by the active variant."""
    vector = embed_query(embedder, "RE-call Hosted readiness probe")
    if len(vector) != embedder.dim:
        raise RuntimeError("embedding readiness probe returned the wrong dimension")
    status = {"embedder_ready": True, "compiler_ready": False, "reranker_ready": False}

    if behavior.learned_sparse:
        if sparse_encoder is None:
            raise RuntimeError(f"{behavior.name} has no learned sparse encoder")
        sparse_vectors = sparse_encoder.encode(["RE-call Hosted readiness probe"])
        if len(sparse_vectors) != 1 or not sparse_vectors[0]:
            raise RuntimeError("learned sparse readiness probe returned an invalid vector")
        status["sparse_ready"] = True

    if behavior.compiler or behavior.facets:
        if compiler is None:
            raise RuntimeError(f"{behavior.name} has no compiler client")

    if behavior.compiler:
        assert compiler is not None
        compiler.compile(
            [
                Message(
                    role="user",
                    content="Readiness evidence: pytest validated the WidgetError repair.",
                )
            ],
            "readiness-probe",
            [],
        )

    if behavior.facets:
        assert compiler is not None
        if behavior.task_conditioned:
            compiler.plan("RE-call Hosted readiness probe", {"choices": []})
        else:
            compiler.facets("RE-call Hosted readiness probe", {"choices": []})

    if behavior.compiler or behavior.facets:
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
