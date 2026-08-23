"""A LOCOMO turn must carry its session date as `valid_from`, so the trust layer can fire on it.

Prior work: searched, `docs_search(source_type="memory", ...)` on the temporal/validity question.
Supersession and validity windows are already shipped
([[project-recall-entailment-supersession-phase0-done-2026-07-18]],
[[project-recall-finance-market-nogo-2026-07-25]]); what was missing is that benchmark data never
populated a window, so the machinery could not run. See `benchmarks/check_temporal_inert.py` for
the measurement, with its positive controls.

Every test here was run against the PRE-CHANGE `_turn_document` first and observed to fail. That
matters more than usual in this repo: the defect being fixed is a capability that could not fire,
and a test that passes before and after would be the same class of defect wearing a test's
clothes.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from recall.eval.locomo import _turn_document, parse_session_date
from recall.frontmatter import parse_frontmatter, validity_bounds
from recall.trust import _verdict

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness

TURN = {"speaker": "Caroline", "text": "The Sprint 1 deadline is February 15, 2024."}


class _Chunk:
    def __init__(self, metadata):
        self.metadata = metadata
        self.text = "x"


class _Hit:
    def __init__(self, metadata, score=0.9):
        self.chunk = _Chunk(metadata)
        self.score = score


def _verdict_for(document, now):
    meta, _ = parse_frontmatter(document)
    return _verdict(_Hit(meta), {}, 0.5, now)[0]


@pytest.mark.parametrize("stamp,expected", [
    ("1:56 pm on 8 May, 2023", "2023-05-08"),
    ("7:55 pm on 9 June, 2023", "2023-06-09"),
    ("11:00 am on 25 May, 2023", "2023-05-25"),
    ("1:14 pm on 1 December, 2024", "2024-12-01"),
])
def test_real_locomo_stamps_parse(stamp, expected):
    assert parse_session_date(stamp) == expected


@pytest.mark.parametrize("stamp", ["unknown date", "", "sometime in May", "31 February, 2023"])
def test_unparseable_stamps_return_none_rather_than_a_guess(stamp):
    """A wrong `valid_from` is worse than none: it would let the trust layer demote a turn on a
    date nobody asserted, which is the silent wrong answer the layer exists to prevent."""
    assert parse_session_date(stamp) is None


def test_turn_document_carries_its_session_date_as_valid_from():
    doc = _turn_document(TURN, "1:56 pm on 8 May, 2023")
    meta, _ = parse_frontmatter(doc)
    start, end = validity_bounds(meta)

    assert start is not None, "no valid_from: the trust layer cannot fire on this turn"
    assert start.date().isoformat() == "2023-05-08"
    # Open-ended on purpose: closing the interval needs supersession extraction, which this
    # change does not attempt. "Said then, never explicitly revoked" is the honest encoding.
    assert end is None


def test_the_body_still_shows_speaker_and_date():
    """Additive, not a move. `_turn_document`'s docstring explains that the speaker and date are
    in the BODY because they are frequently the answer, and LOCOMO's temporal and adversarial
    questions turn on exactly those two fields. Relocating them into metadata would fix the trust
    layer by handicapping the retriever."""
    doc = _turn_document(TURN, "1:56 pm on 8 May, 2023")
    _meta, body = parse_frontmatter(doc)

    assert "Caroline" in body
    assert "8 May, 2023" in body
    assert TURN["text"] in body


def test_a_turn_is_not_yet_valid_when_asked_about_an_earlier_time():
    """The mechanism §9l needed and never had.

    A turn recorded in a LATER session must not read as trustworthy for a question whose reference
    time precedes it. This is what would demote the "updated deadline of 15 Apr" turn for a
    question about 25 March, the first row of the §9l failure table.
    """
    later_turn = _turn_document(TURN, "1:56 pm on 15 April, 2024")

    asked_before = _verdict_for(later_turn, datetime(2024, 3, 25, tzinfo=timezone.utc))
    asked_after = _verdict_for(later_turn, datetime(2024, 5, 1, tzinfo=timezone.utc))

    assert asked_before == "not_yet_valid"
    assert asked_after == "ok", "must stay usable once the reference time reaches it"


def test_wall_clock_still_reads_ok_which_is_why_this_was_invisible():
    """The benchmark adapter passes no reference time, so it uses wall clock, which is later than
    every LOCOMO session. Under wall clock this turn reads `ok` both before and after the change,
    which is exactly why the missing window never showed up as a failure anywhere."""
    doc = _turn_document(TURN, "1:56 pm on 15 April, 2024")
    assert _verdict_for(doc, datetime.now(timezone.utc)) == "ok"


def test_an_unparseable_stamp_degrades_to_the_old_behaviour_not_an_error():
    """Corpus turns with `"unknown date"` must still index. Failing them closed would trade a
    silent gap for a loud one and take out turns that were previously fine."""
    doc = _turn_document(TURN, "unknown date")
    meta, body = parse_frontmatter(doc)

    assert validity_bounds(meta) == (None, None)
    assert TURN["text"] in body
    assert _verdict_for(doc, datetime(2020, 1, 1, tzinfo=timezone.utc)) == "ok"
