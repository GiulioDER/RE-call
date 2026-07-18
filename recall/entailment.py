"""Entailment-based abstention: a DECISION on top of the trust layer, not another score.

The calibrated cosine threshold (FINDINGS §2) catches *far* gaps — queries whose best match is
semantically distant. It cannot catch the **near-miss**: a memory adjacent to the query that does
not answer it, whose similarity clears any threshold *by construction*. The abstention signal for
that class cannot come from the retriever's own score; it needs a separate judgment that the
retrieved memory actually ENTAILS an answer to the query. Proximity is a candidate; entailment is
the evidence.

Design mirrors `recall.trust`: a pure post-processing stage over `TrustedResult`. Only verdict-ok
hits are judged (judging an already-untrusted hit would waste a model call and could resurrect a
superseded memory); an ok hit whose text does not entail the query is demoted to ``not_entailed``,
and when no entailed hit remains the result becomes an explicit abstention.

The judge emits a per-candidate DECISION at its own model-fixed boundary — unlike the cosine
threshold there is no per-embedder constant left to recalibrate, which is the transfer property
the evaluation measures. Cost is honest and real: one cross-encoder pass per ok candidate.

OFF by default: nothing calls this unless an ``EntailmentJudge`` is explicitly passed.
"""
from __future__ import annotations

from dataclasses import replace
from typing import Protocol, runtime_checkable

from recall.types import TrustedHit, TrustedResult


@runtime_checkable
class EntailmentJudge(Protocol):
    """Decides, per candidate text, whether it entails an answer to the query."""

    def judge(self, query: str, texts: list[str]) -> list[bool]: ...


class QnliEntailmentJudge:
    """QNLI cross-encoder judge — "does this sentence answer this question?" as a binary decision.

    The decision boundary (sigmoid 0.5) is the model's own trained boundary: fixed per judge
    model, independent of whichever embedder retrieved the candidates. Requires
    `pip install recall[entail]` (sentence-transformers).
    """

    def __init__(self, model: str = "cross-encoder/qnli-distilroberta-base",
                 threshold: float = 0.5) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("QnliEntailmentJudge requires: pip install recall[entail]") from exc
        self._model = CrossEncoder(model)
        self._threshold = threshold

    def judge(self, query: str, texts: list[str]) -> list[bool]:
        if not texts:
            return []
        scores = self._model.predict([(query, t) for t in texts])
        return [float(s) >= self._threshold for s in scores]


def _abstain_reason(hits: list[TrustedHit]) -> str:
    best = max(hits, key=lambda h: h.cosine)
    if best.verdict == "not_entailed":
        return (
            f"best candidate ({best.provenance.file}) is semantically close but does not "
            f"entail an answer to the query (near-miss)"
        )
    # non-entailment abstentions (all ok hits were consumed by earlier verdicts) keep the
    # trust layer's wording
    from recall.trust import _abstain_reason as trust_reason

    return trust_reason(hits)


def apply_entailment(result: TrustedResult, judge: EntailmentJudge) -> TrustedResult:
    """Re-judge the verdict-ok hits; demote non-entailed ones and recompute abstention.

    Pure function: no DB access, no clock reads, no model state beyond the judge itself.
    A judge returning the wrong number of decisions fails CLOSED (raises) — silently zipping
    short would let unjudged hits keep verdict ``ok``.
    """
    ok = [h for h in result.hits if h.verdict == "ok"]
    if not ok:
        return result  # already abstained (or nothing trusted) — no model cost
    decisions = judge.judge(result.query, [h.chunk.text for h in ok])
    if len(decisions) != len(ok):
        raise ValueError(
            f"entailment judge returned {len(decisions)} decisions for {len(ok)} candidates"
        )
    entailed_ids = {id(h) for h, d in zip(ok, decisions) if d}
    rejudged = [
        replace(h, verdict="not_entailed")
        if h.verdict == "ok" and id(h) not in entailed_ids
        else h
        for h in result.hits
    ]
    still_ok = [h for h in rejudged if h.verdict == "ok"]
    rest = [h for h in rejudged if h.verdict != "ok"]
    abstained = not still_ok
    return replace(
        result,
        hits=still_ok + rest,
        abstained=abstained,
        reason=_abstain_reason(rest) if abstained else "",
    )
