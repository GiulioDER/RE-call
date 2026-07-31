"""Bi-temporal retrieval: `known_as_of` (transaction time) beside `now` (valid time).

Prior work: `docs_search(source_type="memory")` on temporal/validity/point-in-time. Two memos are
load-bearing: [[project-recall-entailment-supersession-phase0-done-2026-07-18]] (supersession
shipped) and [[project-recall-finance-market-nogo-2026-07-25]], which recorded that Zep/Graphiti
ship bi-temporal point-in-time while "RE-call has validity time only". That memo is what this
closes, and it turned out to be about QUERYING rather than storage: `indexed_at` has been a real
indexed column all along, reaching every hit as `ScoredChunk.indexed_at`, with no way to ask
about it.

Built for users rather than for a benchmark score
([[feedback-user-value-over-benchmark-scores-2026-07-31]]). The success criterion is that an agent
can replay what it knew at a past instant and get an honest answer, not that any harness number
moves. `docs/REFERENCE_TIME_DESIGN.md` measured that it will not move `temporal_reasoning`, and
that is a caveat on marketing, not a reason to withhold the capability.

Every assertion here fails against the pre-change code: `known_as_of` did not exist, so
`not_yet_known` could never be returned and a past-instant query silently returned present-day
memories.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from recall.trust import _verdict, abstain_reason
from recall.types import Chunk, ScoredChunk, TrustedHit, Provenance, Validity

MON = datetime(2026, 3, 2, tzinfo=timezone.utc)
TUE = datetime(2026, 3, 3, tzinfo=timezone.utc)
WED = datetime(2026, 3, 4, tzinfo=timezone.utc)


def _hit(*, indexed_at=None, score=0.9, meta=None):
    return ScoredChunk(
        chunk=Chunk(id="c1", source="s", text="t", metadata=meta or {"file": "a.md"}),
        score=score,
        indexed_at=indexed_at,
    )


def _v(hit, *, now=WED, known_as_of=None, supersession=None):
    return _verdict(hit, supersession or {}, 0.5, now, frozenset(), known_as_of)[0]


def test_a_memory_written_after_the_as_of_instant_is_not_yet_known():
    """The capability itself: replaying Tuesday must not surface Wednesday's memory."""
    assert _v(_hit(indexed_at=WED), known_as_of=TUE) == "not_yet_known"


def test_a_memory_written_before_it_is_visible():
    assert _v(_hit(indexed_at=MON), known_as_of=TUE) == "ok"


def test_written_exactly_at_the_instant_is_visible():
    """Boundary is inclusive: a memory written AT the as-of instant existed at that instant."""
    assert _v(_hit(indexed_at=TUE), known_as_of=TUE) == "ok"


def test_without_known_as_of_nothing_changes():
    """The feature is opt-in. Every existing caller passes nothing and must be unaffected."""
    assert _v(_hit(indexed_at=WED)) == "ok"


def test_a_hit_with_no_indexed_at_is_left_alone_not_hidden():
    """Defaulting an unknown write time to "after the as-of" would silently empty a result set for
    any store predating the column. Left visible, deliberately."""
    assert _v(_hit(indexed_at=None), known_as_of=TUE) == "ok"


def test_the_two_axes_are_independent():
    """Valid time and transaction time answer different questions and must not collapse.

    The memory was WRITTEN on Monday (so it was known by Tuesday) but is only VALID from
    Wednesday. Asked as of Tuesday about Tuesday, it is `not_yet_valid`: we had it, it did not
    apply yet. That is a different answer from `not_yet_known`, and conflating them would tell a
    caller replaying a decision that it had no such memory when it did.
    """
    meta = {"file": "a.md", "valid_from": "2026-03-04"}
    assert _v(_hit(indexed_at=MON, meta=meta), now=TUE, known_as_of=TUE) == "not_yet_valid"


def test_transaction_time_is_checked_before_supersession():
    """A memory that did not exist yet cannot meaningfully be reported as superseded, and its
    successor is a document the caller cannot see."""
    meta = {"file": "a.md"}
    verdict = _v(
        _hit(indexed_at=WED, meta=meta), known_as_of=TUE, supersession={"a.md": "b.md"}
    )
    assert verdict == "not_yet_known"


def test_abstain_reason_distinguishes_not_yet_known_from_not_yet_valid():
    """Only the first exonerates a past decision, so the two must not read alike."""
    def _trusted(verdict):
        return TrustedHit(
            chunk=Chunk(id="c", source="s", text="t", metadata={"file": "a.md"}),
            cosine=0.9, confidence=0.9, verdict=verdict,
            provenance=Provenance(source="s", file="a.md", ord=0, indexed_at=None),
            validity=Validity(valid_from=None, valid_until=None, superseded_by=None),
        )

    known = abstain_reason([_trusted("not_yet_known")])
    valid = abstain_reason([_trusted("not_yet_valid")])

    assert known != valid
    assert "did not exist" in known
    assert "not yet valid" in valid


@pytest.mark.parametrize("delta", [timedelta(seconds=1), timedelta(days=365)])
def test_any_amount_of_lateness_hides_it(delta):
    assert _v(_hit(indexed_at=TUE + delta), known_as_of=TUE) == "not_yet_known"
