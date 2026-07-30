"""The TUNED RE-call arm: shipped defaults replaced by the published best free configuration.

Prior work: `results/FINDINGS.md` §11-§12 and
[[project-recall-nearmiss-signal-exhaustion-2026-07-29]] §8f/§8g/§8i -- reranking is the largest
retrieval gain measured in this project AND does not move the abstention signal. This arm is
pre-registered in `benchmarks/PREREGISTRATION-ladder-v4.md` to produce a null, on a different
signal (`top_cosine`) than the prior measurement used.

**Separate file, separate `name`, on purpose.** `recall_system.RecallSystem`'s docstring says it is
the headline arm and "must not quietly become the tuned one" (SUITE-DESIGN rule 4). Subclassing it
and overriding `query` keeps that literally true: the headline arm's own code path is unchanged and
un-imported-into, and the two arms are distinguishable on the board by `name` and by config, not by
a flag someone has to remember was set.

The three overrides are exactly RE-call's published best configuration
([[reference-recall-best-configuration-2026-07-28]]) minus its embedder, which is a paid API and is
deliberately not run. Holding the embedder at the headline arm's `BAAI/bge-small-en-v1.5` is what
makes v4 a single-variable comparison against v2 rather than two things changing at once.

⚠️ `candidate_k` MUST exceed `k` or the reranker is inert: it would be handed exactly the `k`
documents that are already the answer, and reordering a set you are about to return whole changes
nothing. 250 > 45 is the published pair. `_assert_reranker_can_bite` refuses to run otherwise --
this project has shipped a `candidate_k == k` inert-reranker defect before.
"""
from __future__ import annotations

from recall.rerank import CrossEncoderReranker
from recall.trust import trusted_search

from benchmarks.ladder.adapter import Response
from benchmarks.ladder.systems.recall_system import RecallSystem, _filename_to_doc_id

#: RE-call's published best free configuration. Not tuned here, not swept here -- read off
#: [[reference-recall-best-configuration-2026-07-28]] and held fixed, so this arm cannot become an
#: accidental search over configurations scored on their own results.
TUNED_K = 45
TUNED_CANDIDATE_K = 250


def _assert_reranker_can_bite(k: int, candidate_k: int) -> None:
    if candidate_k <= k:
        raise ValueError(
            f"candidate_k={candidate_k} <= k={k}: the reranker would be handed exactly the "
            f"documents about to be returned, so reordering them cannot change the result. That "
            f"is an INERT mechanism wearing a config's clothes -- refuse rather than spend hours "
            f"measuring a no-op."
        )


class RecallTunedSystem(RecallSystem):
    """`RecallSystem` with `k`, `candidate_k` and a local cross-encoder reranker supplied.

    Everything else -- ingest, the replace contract, `indexed_doc_ids`, the embedding cache -- is
    inherited unchanged, which is the point: invariant 1 and the corpus-state batching are the
    same code that produced the v2 and v3 arms.
    """

    name = "recall-tuned"

    def __init__(self, *args, k: int = TUNED_K, candidate_k: int = TUNED_CANDIDATE_K, **kwargs):
        _assert_reranker_can_bite(k, candidate_k)
        super().__init__(*args, **kwargs)
        self._k = k
        self._candidate_k = candidate_k
        # Constructed once, not per query: the cross-encoder loads ~90 MB of weights and this arm
        # issues 1200 queries.
        self._reranker = CrossEncoderReranker()

    def query(self, question: str) -> Response:
        """Same shape as the headline arm's `query`, with the tuned retrieval parameters.

        `top_cosine` is deliberately still `max(h.cosine for h in hits)`. The cross-encoder
        REORDERS ONLY -- every hit keeps its dense cosine (`recall/rerank.py`) -- so this records
        "the highest dense cosine among the documents the reranker selected", which is precisely
        the quantity `PREREGISTRATION-ladder-v4.md` P3 is about. Substituting the cross-encoder's
        own logit here would silently change the axis's units mid-benchmark and make v4
        incomparable to v2, which is the opposite of the point.
        """
        result = trusted_search(
            self._store,
            self._embedder,
            question,
            k=self._k,
            candidate_k=self._candidate_k,
            reranker=self._reranker,
        )
        top_cosine = max((h.cosine for h in result.hits), default=None)
        if result.abstained or not result.hits:
            return Response(answer=None, top_cosine=top_cosine)
        ok_hits = [h for h in result.hits if h.verdict == "ok"]
        top = ok_hits[0] if ok_hits else result.hits[0]
        cited: list[str] = []
        for hit in ok_hits:
            filename = hit.chunk.metadata.get("file")
            if not filename:
                continue
            doc_id = _filename_to_doc_id(filename)
            if doc_id not in cited:
                cited.append(doc_id)
        answer = top.chunk.text
        tokens = len(answer.split())
        return Response(answer=answer, cited_ids=tuple(cited), tokens=tokens, top_cosine=top_cosine)
