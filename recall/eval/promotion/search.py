"""Turn a live RE-call store into the `SearchFn` the runner consumes.

Kept apart from `run.py` on purpose: the runner must stay executable with a callable that touches
no database, because that is what makes the harness's own behaviour testable without one. This
module is the only place in the package that imports `recall.trust` or `recall.store`.

`label_kind` decides what a retrieved hit is compared against, and it is the single most dangerous
thing in this package to get wrong. Every corpus in this repository names its relevant documents in
its OWN id space:

===============  ==========================================================================
`file_ord`       ``"cache_decision.md:0"`` — the labelled eval set. This is the repository's
                 own key (`recall/eval/harness.py:83` and `:419`), NOT the chunk id, which is
                 a content hash and matches nothing any label file contains.
`source`         ``"pep-0690.rst"`` — the PEPs set, and LongMemEval's converted sessions.
`chunk_id`       the store's own chunk id. MT-RAG indexes its corpus documents under their
                 own ids (`benchmarks/mtrag/run.py:164`), so its qrels line up directly.
`locomo_dia`     ``"D1:3"`` — LOCOMO turn ids, recovered from the filename the way
                 `recall/eval/locomo.py::_retrieved_dia_ids` recovers them.
`ladder_doc`     ``"c1/D1:3"`` — Answerability Ladder doc ids, recovered the way
                 `benchmarks/ladder/systems/recall_system.py::_filename_to_doc_id` does.
===============  ==========================================================================

A harness that assumed one shape would score an entire corpus as a total miss and report a
successful run. That is not hypothetical: the first end-to-end run of this harness compared
`file.md:0` labels against content-hash chunk ids and scored 0/14, which is why
`aggregate.build_outcomes` now refuses an arm that hits nothing rather than reporting it.
"""
from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from recall.calibration import Calibration
from recall.embeddings import Embedder
from recall.eval.promotion.run import SearchOutcome
from recall.rerank import Reranker
from recall.store import PgVectorStore
from recall.trust import trusted_search
from recall.types import TrustedHit
from recall.trust_policy import TrustPolicy


def _file_ord(hit: TrustedHit) -> str:
    """`recall/eval/harness.py`'s own key, re-derived rather than restated as a convention."""
    metadata = hit.chunk.metadata
    return f"{metadata.get('file')}:{metadata.get('ord')}"


def _locomo_dia(hit: TrustedHit) -> str:
    from recall.eval.locomo import _filename_to_dia_id

    name = hit.chunk.metadata.get("file")
    return _filename_to_dia_id(name) if name else ""


def _ladder_doc(hit: TrustedHit) -> str:
    from benchmarks.ladder.systems.recall_system import _filename_to_doc_id

    name = hit.chunk.metadata.get("file")
    return _filename_to_doc_id(name) if name else ""


#: Every id space a label can live in. Adding a corpus means adding a row here, deliberately:
#: there is NO default, so a new adapter cannot inherit a silently-wrong comparison.
LABEL_KEYS: dict[str, Callable[[TrustedHit], str]] = {
    "chunk_id": lambda hit: hit.chunk.id,
    "source": lambda hit: Path(hit.chunk.source).name,
    "file_ord": _file_ord,
    "locomo_dia": _locomo_dia,
    "ladder_doc": _ladder_doc,
}


@dataclass(frozen=True)
class StoreSearch:
    """A `SearchFn` over one store, at fixed k and candidate pool."""

    store: PgVectorStore
    embedder: Embedder
    k: int
    candidate_k: int
    retrieval_profile: str
    #: A key of `LABEL_KEYS`. Required, with no default: see the module docstring.
    label_kind: str
    #: Passed to `trusted_search` as `index_generation`, not merely recorded. An earlier version
    #: stamped `ArmConfig.generation` onto every row while retrieval silently used the default
    #: "legacy", so the evidence named a generation the search never used.
    generation: str = "legacy"
    calibration: Calibration | None = None
    reranker: Reranker | None = None
    #: Left as `None` so `trusted_search` applies its own default, which is STRICT. A harness that
    #: defaulted to development mode would silently score a degraded system and publish the
    #: numbers as if the trust gate had run. Under strict mode a store with no generation-bound
    #: certified calibration raises `TrustRefusal`, which is the correct answer and not a bug in
    #: this module.
    policy: TrustPolicy | None = None

    def __post_init__(self) -> None:
        if self.label_kind not in LABEL_KEYS:
            raise ValueError(
                f"label_kind must be one of {sorted(LABEL_KEYS)}, got {self.label_kind!r}. "
                f"There is no default: a wrong one scores an entire corpus as a total miss and "
                f"reports a successful run."
            )

    def __call__(self, query: str) -> SearchOutcome:
        result = trusted_search(
            self.store,
            self.embedder,
            query,
            k=self.k,
            calibration=self.calibration,
            reranker=self.reranker,
            candidate_k=self.candidate_k,
            retrieval_profile=self.retrieval_profile,
            index_generation=self.generation,
            policy=self.policy,
        )
        key = LABEL_KEYS[self.label_kind]
        ids = tuple(key(hit) for hit in result.hits)
        top = result.hits[0] if result.hits else None
        # A `TrustedResult` with `abstained=True` vouched for nothing, so the record's verdict is
        # the absence of one. Otherwise it is the verdict of the hit the layer served first, which
        # is what `superseded_trust_rate` is measured over — see `aggregate._STALE`.
        verdict = "abstained" if result.abstained or top is None else top.verdict
        if self.reranker is None:
            reranking_status = "not_configured"
        else:
            # ⚠️ `"failed"` is currently UNREACHABLE from this call site, and saying so is the
            # point. `HybridRetriever` sets `reranking_ran = self._reranker is not None` BEFORE it
            # calls the reranker (`recall/retriever.py`), so a reranker that was configured and
            # then threw is indistinguishable here from one that worked. The three-state field is
            # kept because the distinction is real and the record schema should be able to carry
            # it; making it observable needs a change in the retriever, which is outside this
            # module. Until then this line reports what the diagnostics can actually tell it.
            reranking_status = "ran" if result.diagnostics.reranking_ran else "failed"
        return SearchOutcome(
            retrieved_chunk_ids=ids,
            dense_cosine=top.cosine if top is not None else math.nan,
            confidence=top.confidence if top is not None else math.nan,
            trust_verdict=verdict,
            reranking_status=reranking_status,
            stage_timings=dict(result.diagnostics.stage_ms),
        )


SearchFactory = Callable[[], StoreSearch]
