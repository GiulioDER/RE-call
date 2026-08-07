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


#: Licence per late-interaction checkpoint, mirroring `recall.sparse.KNOWN_MODELS`.
#:
#: An unrecorded checkpoint RAISES rather than defaulting to permissive, an unrecorded licence is
#: exactly what this check exists to prevent.
LATE_INTERACTION_MODELS: dict[str, str] = {
    "colbert-ir/colbertv2.0": "mit",
    "answerdotai/answerai-colbert-small-v1": "apache-2.0",
    # Capacity diagnostic ONLY (~560M against the 110M default). Non-commercial, so it is refused
    # without an explicit opt-in and it may never contribute to a shipping decision. See the
    # preregistration's monotonicity rule.
    "jinaai/jina-colbert-v2": "cc-by-nc-4.0",
}

#: Licences compatible with RE-call's own MIT distribution for commercial use.
#:
#: A SET, not an equality test. `recall/sparse.py:195` gates on `license_id != "apache-2.0"`,
#: which would refuse an MIT checkpoint. That is latent there (no MIT entry in `KNOWN_MODELS`) and
#: would be fatal here, because the DEFAULT model below is MIT and would be refused by its own
#: guard. `sparse.py` is deliberately left alone, its defect cannot fire.
PERMISSIVE_LICENCES = frozenset({"mit", "apache-2.0"})

DEFAULT_LATE_INTERACTION_MODEL = "colbert-ir/colbertv2.0"


def late_interaction_licence(
    model_name: str, *, accept_noncommercial_license: bool = False
) -> str:
    """The checkpoint's licence, refusing unknown or non-permissive ones.

    The opt-in waives the LICENCE check only. An unrecorded checkpoint raises either way, because
    the point of the registry is that no licence goes unrecorded.
    """
    licence = LATE_INTERACTION_MODELS.get(model_name)
    if licence is None:
        raise ValueError(
            f"unknown late-interaction model {model_name!r}; known models are "
            f"{sorted(LATE_INTERACTION_MODELS)}. Record it in LATE_INTERACTION_MODELS with its "
            f"licence first — an unrecorded licence is exactly what this check exists to prevent."
        )
    if licence not in PERMISSIVE_LICENCES and not accept_noncommercial_license:
        raise ValueError(
            f"{model_name} is licensed {licence}, which is not compatible with RE-call's MIT "
            f"distribution for commercial use. Pass accept_noncommercial_license=True to use it "
            f"anyway (benchmark reproduction only — it may not contribute to a shipping "
            f"decision), or keep the default {DEFAULT_LATE_INTERACTION_MODEL}."
        )
    return licence


class LateInteractionReranker:
    """Reorder hits by ColBERT style MaxSim. Requires `pip install recall[fastembed]`.

    The encoder is INJECTED rather than loaded in `__init__`, mirroring `SpladeEncoder`, so the
    scoring path is testable against fake token matrices without a 0.44 GB download.
    `from_pretrained` is the production constructor.

    Queries go through `query_embed` and documents through `passage_embed`. ColBERT prepends
    distinct `[Q]`/`[D]` markers and pads the query side with `[MASK]` tokens, so using one method
    for both sides yields wrong scores that still look like plausible numbers.
    """

    def __init__(self, encoder: object, *, model_name: str) -> None:
        # Validates the checkpoint even on the injected path: a test or a benchmark that fakes the
        # encoder must not be able to fake its way past the licence registry.
        self.licence = late_interaction_licence(model_name, accept_noncommercial_license=True)
        self.model_name = model_name
        self._encoder = encoder

    @classmethod
    def from_pretrained(
        cls,
        model_name: str = DEFAULT_LATE_INTERACTION_MODEL,
        *,
        accept_noncommercial_license: bool = False,
        cache_dir: str | None = None,
        threads: int | None = None,
    ) -> "LateInteractionReranker":
        """Load `model_name`, refusing an unknown or non-permissive checkpoint."""
        late_interaction_licence(
            model_name, accept_noncommercial_license=accept_noncommercial_license
        )
        try:
            from fastembed import LateInteractionTextEmbedding
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError(
                "LateInteractionReranker requires: pip install recall[fastembed]"
            ) from exc
        encoder = LateInteractionTextEmbedding(
            model_name=model_name, cache_dir=cache_dir, threads=threads
        )
        return cls(encoder, model_name=model_name)

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        if not hits:
            return hits
        qtokens = list(self._encoder.query_embed([query]))[0]  # type: ignore[attr-defined]
        # Validated UP FRONT, not left to `maxsim`. `maxsim` only runs for documents that have
        # tokens, so a batch in which every document is unscoreable would skip the query check
        # entirely and return an unranked order for a query carrying no evidence at all.
        if qtokens.shape[0] == 0:
            raise ValueError("query has no tokens")
        texts = [h.chunk.text for h in hits]
        dtokens = list(self._encoder.passage_embed(texts))  # type: ignore[attr-defined]
        # A document that encodes to zero tokens cannot be scored, and `maxsim` refuses it. That
        # refusal is right for ONE document, because 0.0 is not a neutral score, it lands the item
        # mid-pool. It is wrong for a BATCH: raising here would abort reranking for every hit in
        # the request over one malformed chunk, which is worse than the outcome the refusal exists
        # to prevent. Ranking such a document LAST satisfies the original objection (last is not
        # mid) without letting one bad chunk break every query that retrieves it.
        #
        # The QUERY side deliberately still raises, in the guard above: if the query encodes to
        # nothing there is no evidence to rank ANY document by, so no ordering is salvageable.
        scores = [
            float("-inf") if d.shape[0] == 0 else maxsim(qtokens, d) for d in dtokens
        ]
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        # Reorder ONLY, each hit keeps its dense cosine `score`, `indexed_at` and
        # `first_indexed_at`. Identical to CrossEncoderReranker.rerank and for the identical
        # reason: `recall.trust` reads `score` as a cosine.
        return [hits[i] for i in order]
