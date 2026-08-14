"""The probe compares what a nearest-match retriever serves against what RE-call serves."""
from __future__ import annotations

from datetime import timedelta
from typing import Any

import pytest

from recall.types import (
    Chunk,
    Provenance,
    StalenessReport,
    TrustedHit,
    TrustedResult,
    Validity,
)
from recall_consistency.stale_probe import probe


def _hit(file: str, cosine: float, verdict: str, superseded_by: str | None) -> TrustedHit:
    return TrustedHit(
        chunk=Chunk(id=f"{file}:0", source=file, text="body"),
        cosine=cosine,
        confidence=cosine,
        verdict=verdict,  # type: ignore[arg-type]
        provenance=Provenance(source=file, file=file, ord=0, indexed_at=None),
        validity=Validity(valid_from=None, valid_until=None, superseded_by=superseded_by),
    )


def _result(query: str, hits: list[TrustedHit], abstained: bool = False) -> TrustedResult:
    return TrustedResult(
        query=query,
        hits=hits,
        abstained=abstained,
        reason="no valid hit cleared the threshold" if abstained else "",
        gap_warning=False,
        staleness=StalenessReport(
            stale=False, newest_indexed_at=None, age=None, max_age=timedelta(days=30)
        ),
    )


def test_a_superseded_nearest_match_is_reported() -> None:
    old = _hit("old.md", cosine=0.91, verdict="superseded", superseded_by="new.md")
    new = _hit("new.md", cosine=0.80, verdict="ok", superseded_by=None)

    def fake_search(store: Any, embedder: Any, query: str, **kw: Any) -> TrustedResult:
        return _result(query, [new, old])  # verdict-ok first, as trusted_search orders them

    found = probe(None, None, ["what does the SPLADE leg buy?"], search=fake_search)

    assert len(found) == 1
    assert found[0].question == "what does the SPLADE leg buy?"
    assert found[0].plain_top_file == "old.md"
    assert found[0].plain_top_superseded_by == "new.md"
    assert found[0].trusted_verdict == "ok"
    assert found[0].trusted_abstained is False


def test_a_current_nearest_match_is_not_reported() -> None:
    new = _hit("new.md", cosine=0.91, verdict="ok", superseded_by=None)

    def fake_search(store: Any, embedder: Any, query: str, **kw: Any) -> TrustedResult:
        return _result(query, [new])

    assert probe(None, None, ["q"], search=fake_search) == []


def test_an_abstention_over_a_confident_neighbour_is_reported() -> None:
    near = _hit("near.md", cosine=0.88, verdict="low_confidence", superseded_by=None)

    def fake_search(store: Any, embedder: Any, query: str, **kw: Any) -> TrustedResult:
        return _result(query, [near], abstained=True)

    found = probe(None, None, ["q"], search=fake_search)

    assert len(found) == 1
    assert found[0].trusted_abstained is True
    assert found[0].plain_top_file == "near.md"
    assert found[0].plain_top_superseded_by == ""


def test_a_question_with_no_hits_at_all_is_skipped() -> None:
    def fake_search(store: Any, embedder: Any, query: str, **kw: Any) -> TrustedResult:
        return _result(query, [], abstained=True)

    assert probe(None, None, ["q"], search=fake_search) == []


def test_a_tie_on_cosine_reports_the_superseded_hit() -> None:
    """`hits` arrives ok-first and `max` keeps the first maximum, so a tie hides the stale hit.

    A tie means a plain retriever could return either, so the superseded revision is a nearest
    match and the operator needs to see it. Drop the second element of the sort key and this
    goes red, silently reporting nothing.
    """
    old = _hit("old.md", cosine=0.90, verdict="superseded", superseded_by="new.md")
    new = _hit("new.md", cosine=0.90, verdict="ok", superseded_by=None)

    def fake_search(store: Any, embedder: Any, query: str, **kw: Any) -> TrustedResult:
        return _result(query, [new, old])

    found = probe(None, None, ["q"], search=fake_search)

    assert len(found) == 1
    assert found[0].plain_top_file == "old.md"


def test_a_failing_search_names_the_position_and_never_the_question() -> None:
    """`TrustRefusal` excludes the query by construction; this wrapper must not put it back.

    The position and the exception type are enough to drop a question and re-run. The cause is
    deliberately not chained, because the default excepthook prints a chained cause's message in
    full, which would reintroduce whatever text that exception carried. The exception is not even
    raised inside the handler: Python fills `__context__` in implicitly at the point of a `raise`,
    regardless of any `from` clause, so only raising after the handler has already exited leaves
    both `__cause__` and `__context__` empty.

    Three mutants: interpolating the full exception text in place of the type name turns this red;
    moving the `raise` back inside the except block, keeping `from None`, turns the `__context__`
    assertion red while `__cause__` stays clear; and moving it back inside with `from exc` turns
    the `__cause__` assertion red.
    """

    def fake_search(store: Any, embedder: Any, query: str, **kw: Any) -> TrustedResult:
        raise ValueError("calibration missing for what is the rate limit?")

    with pytest.raises(RuntimeError) as excinfo:
        probe(None, None, ["what is the rate limit?"], search=fake_search)

    message = str(excinfo.value)
    assert "question 1 of 1" in message
    assert "ValueError" in message
    assert "rate limit" not in message
    assert "calibration missing" not in message
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__context__ is None


def test_a_three_way_tie_still_reports_the_superseded_hit() -> None:
    """A key that only asks "not ok" hands the tie to whichever non-ok verdict came first.

    All three tie on cosine and the low-confidence hit precedes the superseded one. Revert the
    key to `(hit.cosine, hit.verdict != "ok")` and this goes red, reporting nothing at all.
    """
    ok = _hit("new.md", cosine=0.90, verdict="ok", superseded_by=None)
    weak = _hit("weak.md", cosine=0.90, verdict="low_confidence", superseded_by=None)
    old = _hit("old.md", cosine=0.90, verdict="superseded", superseded_by="new.md")

    def fake_search(store: Any, embedder: Any, query: str, **kw: Any) -> TrustedResult:
        return _result(query, [ok, weak, old])

    found = probe(None, None, ["q"], search=fake_search)

    assert len(found) == 1
    assert found[0].plain_top_file == "old.md"
    assert found[0].plain_top_superseded_by == "new.md"
