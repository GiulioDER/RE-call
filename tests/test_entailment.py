"""The entailment stage: a DECISION on top of the trust layer, not another score.

A near-miss — a memory semantically adjacent to the query that does NOT answer it — clears any
calibrated cosine threshold by construction. `apply_entailment` re-judges the verdict-ok hits
with an EntailmentJudge and demotes the ones that do not entail an answer, so abstention can
fire on high-similarity noise. Pure post-processing: no DB, no clock.
"""
from __future__ import annotations

import pytest

from recall.entailment import EntailmentJudge, apply_entailment
from recall.types import (
    Chunk,
    Provenance,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)


class FakeJudge:
    """Deterministic judge: entails iff the text contains 'ANSWER'. Records its inputs."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, list[str]]] = []

    def judge(self, query: str, texts: list[str]) -> list[bool]:
        self.calls.append((query, list(texts)))
        return ["ANSWER" in t for t in texts]


def _hit(cid: str, text: str, verdict: str, cosine: float = 0.8) -> TrustedHit:
    return TrustedHit(
        chunk=Chunk(id=cid, source="f", text=text, metadata={"file": f"{cid}.md", "ord": 0}),
        cosine=cosine,
        confidence=0.9,
        verdict=verdict,  # type: ignore[arg-type]
        provenance=Provenance(source="f", file=f"{cid}.md", ord=0, indexed_at=None),
        validity=Validity(valid_from=None, valid_until=None, superseded_by=None),
    )


def _result(hits: list[TrustedHit], abstained: bool = False, reason: str = "") -> TrustedResult:
    return TrustedResult(
        query="what is the retry backoff multiplier",
        hits=hits,
        abstained=abstained,
        reason=reason,
        calibrated=True,
        gap_warning=False,
        staleness=StalenessReport(stale=False, newest_indexed_at=None, age=None,
                                  max_age=__import__("datetime").timedelta(days=2)),
    )


def test_judge_protocol_runtime_checkable():
    assert isinstance(FakeJudge(), EntailmentJudge)


def test_non_entailed_ok_hit_is_demoted_and_entailed_kept():
    res = _result([_hit("near", "adjacent but no answer", "ok"),
                   _hit("gold", "the ANSWER is three", "ok")])
    out = apply_entailment(res, FakeJudge())
    verdicts = {h.chunk.id: h.verdict for h in out.hits}
    assert verdicts["gold"] == "ok"
    assert verdicts["near"] == "not_entailed"
    # entailed ok hits are reordered first
    assert out.hits[0].chunk.id == "gold"
    assert out.abstained is False


def test_abstains_when_no_ok_hit_entails():
    res = _result([_hit("near1", "close but wrong", "ok"),
                   _hit("near2", "also close, also wrong", "ok")])
    out = apply_entailment(res, FakeJudge())
    assert out.abstained is True
    assert all(h.verdict == "not_entailed" for h in out.hits)
    assert "entail" in out.reason  # the reason names the new failure mode


def test_only_ok_hits_are_judged_and_non_ok_verdicts_survive():
    judge = FakeJudge()
    res = _result([_hit("gold", "the ANSWER", "ok"),
                   _hit("stale", "an ANSWER but superseded", "superseded"),
                   _hit("weak", "an ANSWER but low", "low_confidence")])
    out = apply_entailment(res, judge)
    # the judge saw ONLY the ok hit's text — judging untrusted hits would waste the model call
    # and could resurrect a superseded memory
    assert judge.calls == [(res.query, ["the ANSWER"])]
    verdicts = {h.chunk.id: h.verdict for h in out.hits}
    assert verdicts["stale"] == "superseded"
    assert verdicts["weak"] == "low_confidence"


def test_preserves_cosine_confidence_and_provenance():
    res = _result([_hit("gold", "the ANSWER", "ok", cosine=0.73)])
    out = apply_entailment(res, FakeJudge())
    h = out.hits[0]
    assert h.cosine == 0.73
    assert h.confidence == 0.9
    assert h.provenance.file == "gold.md"


def test_already_abstained_result_passes_through_unchanged():
    res = _result([_hit("stale", "text", "superseded")], abstained=True, reason="superseded")
    judge = FakeJudge()
    out = apply_entailment(res, judge)
    assert out == res
    assert judge.calls == []  # nothing to judge — no model cost


def test_wrong_arity_from_judge_fails_closed():
    class BrokenJudge:
        def judge(self, query: str, texts: list[str]) -> list[bool]:
            return []  # wrong arity

    res = _result([_hit("gold", "the ANSWER", "ok")])
    with pytest.raises(ValueError):
        apply_entailment(res, BrokenJudge())


try:
    import sentence_transformers  # noqa: F401

    _HAS_ST = True
except ImportError:
    _HAS_ST = False


@pytest.mark.skipif(not _HAS_ST, reason="sentence-transformers not installed (recall[entail])")
def test_qnli_judge_separates_answering_from_adjacent_text():
    from recall.entailment import QnliEntailmentJudge

    judge = QnliEntailmentJudge()
    assert isinstance(judge, EntailmentJudge)
    query = "how many attempts does the retry policy allow"
    answering = "Outbound HTTP calls use exponential backoff with a hard cap of three attempts."
    adjacent = "Outbound HTTP calls require idempotency keys on all POST endpoints."
    decisions = judge.judge(query, [answering, adjacent])
    assert decisions == [True, False]
