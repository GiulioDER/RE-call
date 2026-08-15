from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

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
    """Reorder hits by cross-encoder relevance. Requires `pip install "recall-rag[rerank]"`.

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
            raise ImportError(
                'CrossEncoderReranker requires: pip install "recall-rag[rerank]"'
            ) from exc
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


def maxsim_or_last(query_tokens: "np.ndarray", doc_tokens: "np.ndarray") -> float:
    """`maxsim`, or `-inf` for a document that encodes to no tokens so it sorts LAST.

    THE single definition of that rule. It was written out three times (the live reranker, the
    offloaded scorer, the validate gate) and those three MUST agree: the gate compares a live
    ranking against offloaded scores, and `rerank_order` refuses a candidate with no score, so a
    divergence there does not degrade the comparison, it crashes it. Three copies kept in step by
    comment are a convention. One function is a guarantee.

    `maxsim` itself still refuses an empty document, and still refuses an empty QUERY through
    this path too: a query with no tokens carries no evidence to rank anything by, so unlike one
    bad document there is no partial ordering worth returning.
    """
    if doc_tokens.shape[0] == 0:
        return float("-inf")
    return maxsim(query_tokens, doc_tokens)


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
    """Reorder hits by ColBERT style MaxSim. Requires `pip install "recall-rag[fastembed]"`.

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
                'LateInteractionReranker requires: pip install "recall-rag[fastembed]"'
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
        scores = [maxsim_or_last(qtokens, d) for d in dtokens]
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
        # Reorder ONLY, each hit keeps its dense cosine `score`, `indexed_at` and
        # `first_indexed_at`. Identical to CrossEncoderReranker.rerank and for the identical
        # reason: `recall.trust` reads `score` as a cosine.
        return [hits[i] for i in order]


#: Voyage's cross-encoder. Named here rather than inlined so the wizard, the docs and
#: `reranker_from_name` cannot drift apart on which model "voyage" means.
DEFAULT_VOYAGE_RERANK_MODEL = "rerank-2.5"


class VoyageReranker:
    """Reorder hits by Voyage cross-encoder relevance. Reorders, never rescores.

    Promoted into the shipped package from `benchmarks/voyage_rerank.py`, where it has served the
    EnterpriseRAG work since it was written. It could not be *served* from there:
    `pyproject.toml` builds `packages = ["recall", "recall_mcp"]`, so `benchmarks/` is absent from
    the wheel, and nothing installed from PyPI could import it.

    `client` is injectable (any object exposing
    ``rerank(query, documents, model, top_k) -> results``) so the reordering is unit-testable
    without the network or the `voyage` extra. When it is None the real client is built lazily on
    first use, under a lock.
    """

    def __init__(
        self,
        model: str = DEFAULT_VOYAGE_RERANK_MODEL,
        api_key: str | None = None,
        top_k: int | None = None,
        max_document_chars: int | None = None,
        client: "Any | None" = None,
    ) -> None:
        import os
        import threading

        self.model = model
        self.top_k = top_k
        if max_document_chars is not None and max_document_chars < 1:
            raise ValueError("max_document_chars must be positive when set")
        self.max_document_chars = max_document_chars
        self._client = client
        #: Guards the lazy build. One instance is reached concurrently: the retrieval profile
        #: resolves a single reranker and every in-flight query calls it, so `if x is None: x = ...`
        #: is a check-then-set that without this builds a client per racing thread and drops all
        #: but one, each still holding an HTTP connection pool nobody closes.
        self._client_lock = threading.Lock()
        # Resolve the key eagerly, mirroring VoyageEmbedder, so a missing key fails at construction
        # rather than on the first query. Skipped when a client was injected, which needs no key.
        self._api_key = api_key or os.environ.get("VOYAGE_API_KEY")
        if self._client is None and not self._api_key:
            raise RuntimeError("VoyageReranker needs VOYAGE_API_KEY (env) or an explicit api_key")

    def _voyage_client(self) -> "Any":
        # Taken unconditionally rather than as the second half of a double-check: the unlocked fast
        # path would save an uncontended acquire off a call that crosses the network, and would buy
        # a publication argument with no settled answer on a free-threaded build.
        with self._client_lock:
            if self._client is None:
                import voyageai

                self._client = voyageai.Client(api_key=self._api_key)
            return self._client

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        if not hits:
            return hits
        limit = self.top_k if self.top_k is not None else len(hits)
        documents = [
            h.chunk.text[: self.max_document_chars] if self.max_document_chars else h.chunk.text
            for h in hits
        ]
        result = self._voyage_client().rerank(query, documents, model=self.model, top_k=limit)
        # `result.results` is sorted by descending relevance and each item's `.index` points back
        # into `documents`. Reorder the ORIGINAL objects: identity preserved, scores intact, for
        # the same reason as CrossEncoderReranker — `recall.trust` reads `score` as a cosine.
        return [hits[item.index] for item in result.results]


class FallbackReranker:
    """Run `primary`; on failure run `fallback`, and REFUSE to hide that it did.

    Keeping retrieval alive through a Voyage outage is the point. The danger is the other half: a
    run that silently alternates between two rerankers has measured neither, and that confound is
    named explicitly in the pre-registration for the bge-large corpus. So every fallback is counted
    and logged, and `served_by` reports which reranker actually answered the last call.

    A failing FALLBACK propagates. Returning the unranked pool instead would be indistinguishable
    from a reranked result to every caller, which is the failure this class exists to prevent.
    """

    def __init__(self, primary: Reranker, fallback: Reranker) -> None:
        self.primary = primary
        self.fallback = fallback
        self.fallback_count = 0
        self.served_by: str | None = None

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        try:
            out = self.primary.rerank(query, hits)
            self.served_by = "primary"
            return out
        except Exception as exc:
            from recall.observability import get_logger

            self.fallback_count += 1
            self.served_by = "fallback"
            get_logger("rerank").warning(
                "primary reranker failed, falling back (%d so far this process): %s",
                self.fallback_count,
                exc,
            )
            return self.fallback.rerank(query, hits)


def reranker_kind(name: str) -> tuple[str, str]:
    """Route a `RECALL_RERANK_MODEL` spelling to `(kind, model)` WITHOUT constructing anything.

    Separate from `reranker_from_name` so the routing can be asserted without downloading
    cross-encoder weights. An earlier draft folded both into one function behind a `build` flag,
    which made its return type a union of "a reranker" and "a description of one" — untypeable at
    the call site, and mypy said so.
    """
    if name == "voyage" or name.startswith("voyage:"):
        model = name[len("voyage:") :] if name.startswith("voyage:") else DEFAULT_VOYAGE_RERANK_MODEL
        return ("voyage", model)
    return ("cross-encoder", name)


def reranker_from_name(name: str, *, api_key: str | None = None) -> Reranker:
    """Build the reranker a `RECALL_RERANK_MODEL` spelling names.

    Spellings: ``voyage``, ``voyage:<model>``, or anything else, which stays a LOCAL cross-encoder
    model name. The unprefixed form is deliberately unchanged: this is additive, and an operator
    with `RECALL_RERANK_MODEL=BAAI/bge-reranker-base` set must keep getting exactly what they had.
    """
    kind, model = reranker_kind(name)
    if kind == "voyage":
        return VoyageReranker(model=model, api_key=api_key)
    return CrossEncoderReranker(model=model)
