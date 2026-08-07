from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from recall.embeddings import verify_artifact
from recall.types import ScoredChunk

if TYPE_CHECKING:  # pragma: no cover - numpy arrives with the fastembed extra
    import numpy as np


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]: ...


class NoOpReranker:
    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        return hits


DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
#: Pinned Hub commit of the DEFAULT reranker. An unpinned Hub reference is mutable — the repo
#: owner (or a compromise) can swap the weights and every consumer silently picks them up on the
#: next cold cache. Pinning makes the resolved artifact immutable (mirrors DEFAULT_QNLI_REVISION).
DEFAULT_RERANKER_REVISION = "c5ee24cb16019beea0893ab7796b1df96625c6b8"

#: The reranker the QUALITY retrieval profile is pinned to, as an identity rather than a name.
#:
#: `RECALL_RERANK_PATH` stays an operator variable because a filesystem layout is deployment
#: specific. The DIGEST is not: a different artifact tree under the same model name is a different
#: model, and the quality profile's whole claim — same candidate pool as fast, one extra stage,
#: a 1500 ms budget — is a claim about a specific set of weights. So the digest is pinned here and
#: the environment must AGREE with it rather than define it, mirroring how
#: `recall.embedding_registry` pins each embedding profile's artifact.
#:
#: Value: the tree digest of the provisioned `ms-marco-MiniLM-L-6-v2` artifact, recomputed
#: independently on VPS2 on 2026-08-05 and equal to the digest recorded in
#: `/opt/recall-enterprise/manifest.json` on 2026-08-03.
PINNED_RERANKER_MODEL = DEFAULT_RERANKER_MODEL
PINNED_RERANKER_REVISION = DEFAULT_RERANKER_REVISION
PINNED_RERANKER_SHA256 = "db6ad87969c7dc78320152e68a16118aeb4b2a6f7d8cc979c57f61ddb5e2ab2a"


class CrossEncoderReranker:
    """Reorder hits by cross-encoder relevance. Requires `pip install recall[rerank]`.

    The default model is pinned to a Hub revision; if you supply your own `model`, pin your own
    `revision` too (the default pin belongs to the default model only)."""

    def __init__(
        self,
        model: str = DEFAULT_RERANKER_MODEL,
        revision: str | None = DEFAULT_RERANKER_REVISION,
        local_files_only: bool = False,
        artifact_sha256: str | None = None,
        inference_threads: int | None = None,
    ) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("CrossEncoderReranker requires: pip install recall[rerank]") from exc
        if local_files_only:
            if artifact_sha256 is None:
                raise ValueError("offline reranking requires an artifact_sha256")
            model = str(verify_artifact(Path(model), artifact_sha256))
            revision = None
        if model != DEFAULT_RERANKER_MODEL and revision == DEFAULT_RERANKER_REVISION:
            revision = None  # the default pin belongs to the default model only
        if inference_threads is not None:
            if inference_threads < 1:
                raise ValueError("inference_threads must be positive")
            try:
                import torch

                torch.set_num_threads(inference_threads)
            except ImportError:  # pragma: no cover
                pass
        self._model = CrossEncoder(model, revision=revision, local_files_only=local_files_only)

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        if not hits:
            return hits
        scores = self._model.predict([(query, h.chunk.text) for h in hits])
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        # Reorder ONLY — each hit keeps its dense cosine `score` and `indexed_at`. The
        # cross-encoder logit is an unbounded relevance score in different units; leaking it
        # into `score` would corrupt every downstream consumer that reads it as a cosine
        # (the trust layer's thresholds and calibrated confidence in particular).
        return [hits[i] for i in order]


def maxsim(query_tokens: "np.ndarray", doc_tokens: "np.ndarray") -> float:
    """ColBERT late-interaction score: sum over query tokens of the best-matching doc token.

    Both arrays are `(n_tokens, dim)` and L2-normalised by the encoder, so a dot product is a
    cosine. The `max` is the whole point: it keeps per-token evidence instead of pooling the
    pair into one representation, which is the deficiency this experiment exists to test.

    Empty inputs RAISE rather than scoring 0.0, for the reason `rerank_order` raises on a
    missing score: a zero is not a neutral value in a ranking, it silently places the item
    mid-pool.
    """
    if query_tokens.shape[0] == 0:
        raise ValueError("query has no tokens")
    if doc_tokens.shape[0] == 0:
        raise ValueError("document has no tokens")
    if query_tokens.shape[1] != doc_tokens.shape[1]:
        raise ValueError(
            f"dimension mismatch: query is {query_tokens.shape[1]}-d, "
            f"document is {doc_tokens.shape[1]}-d"
        )
    return float((query_tokens @ doc_tokens.T).max(axis=1).sum())
