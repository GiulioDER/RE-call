"""`recall_search` carries the citable part of `recall_evidence`'s decision.

Measured basis (2026-10-10, 1,440 tool conversations over three models): given a search result, no
model called `recall_evidence` before answering in more than 10% of conversations, even with the
tool's description and the routing guide in front of it. So search now says what may be cited.

Invariants, each with the mutation that turns its test red (recorded in the test's docstring):
- S1 search's `evidence.citable` is exactly the chunk ids `recall_evidence` would admit for the same
  trusted result: the `ok` hits, in retrieval order, never a demoted one; and the advice tells the
  agent to cite them.
- S2 an abstaining search cites nothing, carries the evidence tool's reason code, and its advice
  does not invite an answer.
- S3 `RECALL_SEARCH_EVIDENCE=off` removes the field and the advice sentence.
- H1, H2 a relevant `superseded` or `expired` hit adds, last, an exception for questions about an
  earlier time; an irrelevant one, or any other verdict, does not.

The trusted search is a stand-in; `_retrieve_trusted`, `search_memory` and `evidence_memory` are the
real ones.
"""

from __future__ import annotations

import contextlib
import functools
from collections.abc import Callable
from dataclasses import replace
from datetime import UTC, datetime, timedelta

import pytest

from recall.types import (
    Chunk,
    Provenance,
    RetrievalDiagnostics,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)
from recall_mcp import retrieval

_AT = datetime(2026, 1, 1, tzinfo=UTC)
_STALE = StalenessReport(False, None, None, timedelta(days=2))
_DIAG = RetrievalDiagnostics("bge-small-symmetric-v1", "legacy", "gen-7", 20, False, {})


def _hit(i: int, verdict: str, demoted_confidence: float = 0.2) -> TrustedHit:
    return TrustedHit(
        chunk=Chunk(f"c{i}", f"note{i}.md", f"text {i}", {"file": f"note{i}.md"}),
        cosine=0.9 - i * 0.01,
        confidence=0.9 if verdict == "ok" else demoted_confidence,
        verdict=verdict,
        provenance=Provenance(source=f"note{i}.md", file=f"note{i}.md", ord=0, indexed_at=_AT),
        validity=Validity(valid_from=_AT, valid_until=None,
                          superseded_by="note9.md" if verdict == "superseded" else None),
    )


class _FakeSearch:
    """`trusted_search` with fixed verdicts: valid hits first, abstained when none is `ok`."""

    def __init__(self, verdicts: list[str], demoted_confidence: float = 0.2) -> None:
        self.verdicts = verdicts
        self.demoted_confidence = demoted_confidence

    def __call__(self, store, embedder, query, *, k, pre_trust_transform=None, **_kw):  # type: ignore[no-untyped-def]
        hits = [_hit(i, v, self.demoted_confidence) for i, v in enumerate(self.verdicts)][:k]
        ok = [h for h in hits if h.verdict == "ok"]
        rest = [h for h in hits if h.verdict != "ok"]
        return TrustedResult(
            query=query, hits=ok + rest, abstained=not ok,
            reason="" if ok else "no hit above the calibrated confidence threshold (probable corpus gap)",
            gap_warning=False, staleness=_STALE, diagnostics=_DIAG,
            calibration_id="cal-fixture", calibration_status="certified",
        )


class _Store:
    generation_id = "gen-7"


class _Embedder:
    dim = 2
    name = "search-evidence-fixture"

    def embed(self, texts):  # type: ignore[no-untyped-def]
        return [[1.0, 0.0] for _ in texts]


def _both(verdicts: list[str], env: dict[str, str] | None = None,  # type: ignore[no-untyped-def]
          demoted_confidence: float = 0.2):
    retrieve = functools.partial(
        retrieval._retrieve_trusted,
        reranker_builder=lambda *_a, **_k: None,
        admission_factory=lambda _profile: contextlib.nullcontext(),
        trusted_search_fn=_FakeSearch(verdicts, demoted_confidence),
    )
    env = {} if env is None else env
    search = retrieval.search_memory(_Store(), _Embedder(), "q", k=5, env=env,  # type: ignore[arg-type]
                                     _retrieve_trusted_fn=retrieve)
    evidence = retrieval.evidence_memory(_Store(), _Embedder(), "q", k=5, env=env,  # type: ignore[arg-type]
                                         _retrieve_trusted_fn=retrieve,
                                         register_evidence_cards_fn=lambda *_a, **_k: None)
    return search, evidence


def test_search_cites_exactly_what_the_evidence_tool_would() -> None:
    """S1. Red proof: `_search_evidence` taking every hit (`citable=[h.chunk.id for h in
    result.hits]`) fails `assert ['c0', 'c2', 'c1', 'c3'] == ['c0', 'c2']`; dropping
    `evidence=evidence` from the `SearchResult` call fails "search carried no evidence"."""
    search, evidence = _both(["ok", "superseded", "ok", "low_confidence"])
    assert search.evidence is not None, "search carried no evidence"
    assert search.evidence.decision == "answer" == evidence.decision
    assert search.evidence.citable == [item.chunk_id for item in evidence.items]
    assert search.evidence.citable == ["c0", "c2"]
    assert search.advice.endswith(retrieval.SEARCH_EVIDENCE_NOTE), "the advice does not say what to cite"


def test_an_abstaining_search_cites_nothing() -> None:
    """S2. Red proof: the advice condition `evidence.decision == "answer"` weakened to
    `evidence is not None` fails "an abstaining search invited an answer"."""
    search, evidence = _both(["superseded", "low_confidence"])
    assert search.abstained and search.evidence is not None
    assert search.evidence.decision == "abstain" == evidence.decision
    assert search.evidence.citable == []
    assert search.evidence.reason_code == evidence.reason_code is not None
    assert retrieval.SEARCH_EVIDENCE_NOTE not in search.advice, "an abstaining search invited an answer"


def test_the_switch_turns_search_evidence_off() -> None:
    """S3. Red proof: `search_evidence_enabled` returning True always fails "evidence despite the
    switch"."""
    search, _ = _both(["ok"], env={"RECALL_SEARCH_EVIDENCE": "off"})
    assert search.evidence is None, "evidence despite the switch"
    assert retrieval.SEARCH_EVIDENCE_NOTE not in search.advice
    on, _ = _both(["ok"])
    assert on.evidence is not None and on.evidence.citable == ["c0"]


def test_a_relevant_superseded_or_expired_hit_is_a_record_of_the_past() -> None:
    """H1. Every rule in search's advice is about the present ("say you don't know", "rely only on
    the current version", "cite only `evidence.citable`"), so a question about an earlier time was
    told to refuse the very memory that answers it. The exception must come LAST, after the
    citation rule it qualifies, on the abstaining and the answering path alike.
    Red proof: `_has_historical_record` returning False fails "no historical exception"; appending
    the note before `SEARCH_EVIDENCE_NOTE` fails "the citation rule overrides the exception"."""
    for verdicts in (["superseded", "low_confidence"], ["expired"], ["ok", "superseded"]):
        search, _ = _both(verdicts, demoted_confidence=0.7)
        assert retrieval.HISTORICAL_RECORD_NOTE in search.advice, f"no historical exception: {verdicts}"
        assert search.advice.endswith(retrieval.HISTORICAL_RECORD_NOTE), (
            f"the citation rule overrides the exception: {verdicts}")


def test_the_historical_exception_needs_a_relevant_record() -> None:
    """H2. The gate assigns `superseded` and `expired` before it reads the score, so an irrelevant
    superseded memory must not invite an answer to a question it does not match.
    Red proof: dropping `and hit.confidence >= 0.5` from `_has_historical_record` fails
    "irrelevant record invited an answer"; widening `_HISTORICAL_VERDICTS` with `low_confidence`
    fails "exception without a superseded or expired hit"."""
    search, _ = _both(["superseded", "expired"], demoted_confidence=0.2)
    assert retrieval.HISTORICAL_RECORD_NOTE not in search.advice, "irrelevant record invited an answer"
    search, _ = _both(["ok", "low_confidence", "not_yet_valid"], demoted_confidence=0.7)
    assert retrieval.HISTORICAL_RECORD_NOTE not in search.advice, (
        "exception without a superseded or expired hit")


def _transform_fake_result(
    monkeypatch: pytest.MonkeyPatch, transform: Callable[[TrustedResult], TrustedResult]
) -> None:
    original = _FakeSearch.__call__

    def transformed(*args, **kwargs):  # type: ignore[no-untyped-def]
        return transform(original(*args, **kwargs))

    monkeypatch.setattr(_FakeSearch, "__call__", transformed)


@pytest.mark.parametrize("verdicts", [["expired"], ["ok", "superseded"]])
def test_historical_candidates_are_separate_from_current_evidence(verdicts: list[str]) -> None:
    """DAT-001: current abstention/citations and conditional historical eligibility must agree.

    Red proof: PR HEAD 0367a335, search_memory's historical advice had no structured candidates;
    this exact node fails the assertion 'historical guidance has no structured eligibility'.
    """
    search, evidence = _both(verdicts, demoted_confidence=0.7)
    historical = search.model_dump().get("historical")
    assert historical is not None, "historical guidance has no structured eligibility"
    assert historical["candidates"] == ["c0" if verdicts == ["expired"] else "c1"]
    assert historical["requires_validity_check"] is True
    assert "historical.candidates" in search.advice
    assert search.evidence is not None
    assert search.evidence.citable == [item.chunk_id for item in evidence.items]
    assert search.evidence.citable == ([] if verdicts == ["expired"] else ["c0"])


def test_historical_candidates_exclude_a_low_confidence_neighbor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STAKES-001: one eligible record must not authorize another below the threshold.

    Red proof: PR HEAD 0367a335's result-wide _has_historical_record switch omits individual
    eligibility, failing 'one relevant record must not authorize its irrelevant neighbor'.
    """
    _transform_fake_result(
        monkeypatch,
        lambda result: replace(result, hits=[result.hits[0], replace(result.hits[1], confidence=0.2)]),
    )
    search, _ = _both(["superseded", "expired"], demoted_confidence=0.7)
    historical = search.model_dump().get("historical")
    assert historical is not None and historical["candidates"] == ["c0"], (
        "one relevant record must not authorize its irrelevant neighbor")


@pytest.mark.parametrize(
    "changes", [{"calibration_id": None}, {"calibration_status": "uncertified"},
                {"trust_state": "degraded"}],
)
def test_historical_eligibility_needs_certified_trust(
    monkeypatch: pytest.MonkeyPatch, changes: dict[str, object],
) -> None:
    """The historical exception cannot use an uncertified confidence threshold.

    Red proof: PR HEAD 0367a335's _has_historical_record ignores result.calibrated;
    every parameter fails 'uncertified historical confidence invited an answer'.
    """
    _transform_fake_result(monkeypatch, lambda result: replace(result, **changes))
    search, _ = _both(["superseded"], demoted_confidence=0.7)
    assert search.model_dump().get("historical") is None
    assert retrieval.HISTORICAL_RECORD_NOTE not in search.advice, (
        "uncertified historical confidence invited an answer")


def test_historical_candidates_require_the_requested_validity_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """STAKES-002: archival eligibility is not proof of a fact at the requested past instant.

    Red proof: PR HEAD 0367a335's unconditional historical note fails 'a superseded future
    assertion was presented as an established past fact'; no date parser is claimed here.
    """
    _transform_fake_result(
        monkeypatch,
        lambda result: replace(result, hits=[replace(
            result.hits[0], validity=Validity(
                valid_from=datetime(2050, 1, 1, tzinfo=UTC),
                valid_until=datetime(2100, 1, 1, tzinfo=UTC), superseded_by="note9.md"))]),
    )
    search, _ = _both(["superseded"], demoted_confidence=0.7)
    historical = search.model_dump().get("historical")
    assert historical is not None and historical["requires_validity_check"] is True, (
        "a superseded future assertion was presented as an established past fact")
    assert "valid_from" in search.advice and "valid_until" in search.advice
    assert "do not claim it held then" in search.advice


def test_disabling_search_evidence_disables_historical_eligibility() -> None:
    """DAT-001: disabling the evidence surface must not leave historical citation advice.

    Red proof: PR HEAD 0367a335 appends its historical exception despite the off switch;
    this node fails 'historical answer guidance survived disabled evidence'.
    """
    search, _ = _both(["expired"], env={"RECALL_SEARCH_EVIDENCE": "off"}, demoted_confidence=0.7)
    assert search.evidence is None and search.model_dump().get("historical") is None
    assert retrieval.HISTORICAL_RECORD_NOTE not in search.advice, (
        "historical answer guidance survived disabled evidence")


def test_supported_historical_time_explicitly_permits_an_answer() -> None:
    """FIX-003, L6 revision: temporal support must preserve the authorized historical answer.

    Red proof: the first repair's HISTORICAL_RECORD_NOTE (tree 48857418) prohibited asserting
    that a record ever held. This node fails 'supported historical time is not allowed an answer'
    on that wording at the public search_memory boundary; the unsupported-time control above
    still requires the prohibition when neither interval nor text establishes the requested time.
    """
    search, _ = _both(["expired"], demoted_confidence=0.7)
    assert "If they establish it, answer for that time" in search.advice, (
        "supported historical time is not allowed an answer")
