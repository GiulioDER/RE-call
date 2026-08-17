from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast, runtime_checkable

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
COREB_CODE_RERANKER_MODEL = "hq-bench/coreb-code-reranker"
COREB_CODE_RERANKER_REVISION = "24d2ad50bb4a53149cfd3c42c0e966e954cdbcf1"
RERANKER_MODEL_ALIASES = {
    "coreb-code": COREB_CODE_RERANKER_MODEL,
}
KNOWN_RERANKER_REVISIONS = {
    DEFAULT_RERANKER_MODEL: DEFAULT_RERANKER_REVISION,
    COREB_CODE_RERANKER_MODEL: COREB_CODE_RERANKER_REVISION,
}

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


class QwenYesNoReranker:
    """Reorder hits with a Qwen-style yes/no causal-LM reranker.

    CoREB's code reranker is fine-tuned from Qwen3-Reranker-4B and is not a
    `sentence-transformers.CrossEncoder`. Its published score is the logit difference between the
    next-token "yes" and "no" answers, so it needs a dedicated loader instead of the cross-encoder
    path above.
    """

    _PREFIX = (
        "<|im_start|>system\n"
        "Judge whether the Document meets the requirements based on the Query and the Instruct "
        'provided. Note that the answer can only be "yes" or "no".<|im_end|>\n'
        "<|im_start|>user\n"
    )
    _SUFFIX = "<|im_end|>\n<|im_start|>assistant\n"
    _INSTRUCT = "Given a code search query, does the following code snippet match the query intent?"

    def __init__(
        self,
        model: str = COREB_CODE_RERANKER_MODEL,
        revision: str | None = COREB_CODE_RERANKER_REVISION,
        *,
        max_length: int = 8192,
        batch_size: int = 4,
        inference_threads: int | None = None,
        trust_remote_code: bool = False,
        tokenizer: object | None = None,
        causal_lm: object | None = None,
    ) -> None:
        if inference_threads is not None:
            if inference_threads < 1:
                raise ValueError("inference_threads must be positive")
            try:
                import torch

                torch.set_num_threads(inference_threads)
            except ImportError:  # pragma: no cover
                pass
        if tokenizer is None or causal_lm is None:
            try:
                from transformers import AutoModelForCausalLM, AutoTokenizer
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    'QwenYesNoReranker requires: pip install "transformers" "torch"'
                ) from exc
            tokenizer = AutoTokenizer.from_pretrained(
                model, revision=revision, trust_remote_code=trust_remote_code
            )
            causal_lm = AutoModelForCausalLM.from_pretrained(
                model,
                revision=revision,
                trust_remote_code=trust_remote_code,
                torch_dtype="auto",
            )
        self._tokenizer: Any = tokenizer
        self._model: Any = causal_lm
        self._max_length = max_length
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        self._batch_size = batch_size
        eval_method = getattr(self._model, "eval", None)
        if callable(eval_method):
            eval_method()
        tokenizer_for_ids = cast(Any, self._tokenizer)
        self._yes_id = tokenizer_for_ids.convert_tokens_to_ids("yes")
        self._no_id = tokenizer_for_ids.convert_tokens_to_ids("no")
        if self._yes_id is None or self._no_id is None:
            raise ValueError("reranker tokenizer must expose yes/no token ids")

    def _prompt(self, query: str, document: str) -> str:
        return (
            f"{self._PREFIX}<Instruct>: {self._INSTRUCT}\n"
            f"<Query>: {query}\n<Document>: {document}{self._SUFFIX}"
        )

    def _device(self) -> object | None:
        device = getattr(self._model, "device", None)
        if device is not None:
            return cast(object, device)
        parameters = getattr(self._model, "parameters", None)
        if callable(parameters):
            try:
                return cast(object, next(parameters()).device)
            except (StopIteration, AttributeError, TypeError):
                return None
        return None

    def _score_batch(self, query: str, documents: list[str]) -> list[float]:
        prompts = [self._prompt(query, document) for document in documents]
        inputs = self._tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=self._max_length,
        )
        device = self._device()
        if device is not None:
            move = getattr(inputs, "to", None)
            if callable(move):
                inputs = move(device)
            else:
                inputs = {
                    key: value.to(device) if hasattr(value, "to") else value
                    for key, value in inputs.items()
                }
        try:
            import torch

            guard = torch.inference_mode()
        except ImportError:  # pragma: no cover - used by fake-model unit tests
            guard = nullcontext()
        with guard:
            outputs = self._model(**inputs)
        logits = outputs.logits
        attention_mask = inputs.get("attention_mask") if hasattr(inputs, "get") else None
        if attention_mask is None:
            scores = logits[:, -1, self._yes_id] - logits[:, -1, self._no_id]
        else:
            positions = self._last_nonpad_positions(attention_mask)
            scores = [
                logits[row, position, self._yes_id] - logits[row, position, self._no_id]
                for row, position in enumerate(positions)
            ]
        detach = getattr(scores, "detach", None)
        if callable(detach):
            scores = detach()
        cpu = getattr(scores, "cpu", None)
        if callable(cpu):
            scores = cpu()
        values = scores.tolist() if hasattr(scores, "tolist") else list(scores)
        return [float(value) for value in values]

    def _last_nonpad_positions(self, attention_mask: object) -> list[int]:
        rows = attention_mask.tolist() if hasattr(attention_mask, "tolist") else attention_mask
        positions: list[int] = []
        for row in rows:  # type: ignore[union-attr]
            last: int | None = None
            for index, value in enumerate(row):
                if bool(value):
                    last = index
            if last is None:
                raise ValueError("reranker tokenizer returned an empty attention row")
            positions.append(last)
        return positions

    def rerank(self, query: str, hits: list[ScoredChunk]) -> list[ScoredChunk]:
        if not hits:
            return hits
        scores: list[float] = []
        for i in range(0, len(hits), self._batch_size):
            batch = [hit.chunk.text for hit in hits[i : i + self._batch_size]]
            scores.extend(self._score_batch(query, batch))
        order = sorted(range(len(hits)), key=lambda i: scores[i], reverse=True)
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
