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

# safe one-way import: trust never imports entailment at runtime (TYPE_CHECKING + lazy only)
from recall.trust import abstain_reason as _trust_abstain_reason
from recall.types import TrustedHit, TrustedResult


@runtime_checkable
class EntailmentJudge(Protocol):
    """Decides, per candidate text, whether it entails an answer to the query."""

    def judge(self, query: str, texts: list[str]) -> list[bool]: ...


DEFAULT_QNLI_MODEL = "cross-encoder/qnli-distilroberta-base"
#: Pinned Hub commit of the DEFAULT model. An unpinned Hub reference is mutable — the repo
#: owner (or a compromise) can swap the weights and every consumer silently picks them up on
#: the next cold cache. Pinning makes the resolved artifact immutable.
DEFAULT_QNLI_REVISION = "7dd04ee0a6040c06fb381ad7edcb8585f4d937fd"


class QnliEntailmentJudge:
    """QNLI cross-encoder judge — "does this sentence answer this question?" as a binary decision.

    The decision boundary (sigmoid 0.5) is the model's own trained boundary: fixed per judge
    model, independent of whichever embedder retrieved the candidates. Requires
    `pip install recall[entail]` (sentence-transformers). The default model is pinned to a
    Hub revision; if you supply your own `model`, pin your own `revision` too.
    """

    def __init__(self, model: str = DEFAULT_QNLI_MODEL, threshold: float = 0.5,
                 revision: str | None = DEFAULT_QNLI_REVISION) -> None:
        try:
            from sentence_transformers import CrossEncoder
        except ImportError as exc:  # pragma: no cover - exercised only without the extra
            raise ImportError("QnliEntailmentJudge requires: pip install recall[entail]") from exc
        if model != DEFAULT_QNLI_MODEL and revision == DEFAULT_QNLI_REVISION:
            revision = None  # the default pin belongs to the default model only
        self._model = CrossEncoder(model, revision=revision)
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
    return _trust_abstain_reason(hits)


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
