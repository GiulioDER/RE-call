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
- S4 the citation sentence names no other tool (a named tool primes small models to call it).

The trusted search is a stand-in; `_retrieve_trusted`, `search_memory` and `evidence_memory` are the
real ones.
"""

from __future__ import annotations

import contextlib
import functools
from datetime import UTC, datetime, timedelta

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


def _hit(i: int, verdict: str) -> TrustedHit:
    return TrustedHit(
        chunk=Chunk(f"c{i}", f"note{i}.md", f"text {i}", {"file": f"note{i}.md"}),
        cosine=0.9 - i * 0.01,
        confidence=0.9 if verdict == "ok" else 0.2,
        verdict=verdict,
        provenance=Provenance(source=f"note{i}.md", file=f"note{i}.md", ord=0, indexed_at=_AT),
        validity=Validity(valid_from=_AT, valid_until=None,
                          superseded_by="note9.md" if verdict == "superseded" else None),
    )


class _FakeSearch:
    """`trusted_search` with fixed verdicts: valid hits first, abstained when none is `ok`."""

    def __init__(self, verdicts: list[str]) -> None:
        self.verdicts = verdicts

    def __call__(self, store, embedder, query, *, k, pre_trust_transform=None, **_kw):  # type: ignore[no-untyped-def]
        hits = [_hit(i, v) for i, v in enumerate(self.verdicts)][:k]
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


def _both(verdicts: list[str], env: dict[str, str] | None = None):  # type: ignore[no-untyped-def]
    retrieve = functools.partial(
        retrieval._retrieve_trusted,
        reranker_builder=lambda *_a, **_k: None,
        admission_factory=lambda _profile: contextlib.nullcontext(),
        trusted_search_fn=_FakeSearch(verdicts),
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


def test_the_citation_advice_names_no_other_tool() -> None:
    """S4. Measured 2026-10-10: a citation sentence that named `recall_evidence` made a 3B model call
    that tool in 60 of 60 conversations, after abstentions too. Red proof: the wording merged in
    #906 ("...; `recall_evidence` adds card warrants, a rendered answer prompt, and related or paged
    passages these hits do not include.") fails "the advice names another tool"."""
    assert "recall_" not in retrieval.SEARCH_EVIDENCE_NOTE, "the advice names another tool"
