"""T-2 ``query_time_range``: the calendar range a question names.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-query-time-leg.md, apparatus check 1.
The anchor is Wednesday 2023-05-10, so weekday and week rules cannot pass by coincidence of a
Monday anchor (the lesson of the T-1 tests).

Red proofs, each run on 2026-09-25 against a deliberate mutation of the production line named,
failing at the range equality, then green after restoring it:

* ``[last Wednesday]``: the ``or 7`` in ``_last_weekday`` removed (resolves to the anchor itself).
* ``[in August]``: ``_month_only`` always using the anchor's year (a month not yet begun must be
  last year's).
* ``[In June]``: the ``(?i:...)`` scope removed from the month-only prefix (capitalised "In" no
  longer matches).
* ``[3 days ago]``: ``_ago`` returning the bare centre day, without the +-1 widening.
* ``test_the_earliest_expression_wins``: the position comparison in ``query_time_range`` reversed.
* ``test_the_verb_may_is_not_the_month``: ``re.IGNORECASE`` added to the month-only rule.
* ``[Sept. 2021]``: the ``\\.?`` removed from the month-only rule's no-day-or-year guard. This case
  found a real bug while the tests were written: "in Sept. 2021" matched the month-only rule at
  "in", before the month-and-year rule, and resolved to the anchor's September.
"""

from __future__ import annotations

from datetime import date

import pytest

from recall_aml.temporal_query import query_time_range

ANCHOR = date(2023, 5, 10)  # a Wednesday


def d(text: str) -> date:
    return date.fromisoformat(text)


@pytest.mark.parametrize(
    ("question", "start", "end"),
    [
        ("What did I do on 2023-04-02?", "2023-04-02", "2023-04-02"),
        ("What happened on 7 May 2023?", "2023-05-07", "2023-05-07"),
        ("What happened on the 21st March 2022?", "2022-03-21", "2022-03-21"),
        ("What happened on May 7th, 2023?", "2023-05-07", "2023-05-07"),
        ("What did Melanie do in October 2022?", "2022-10-01", "2022-10-31"),
        ("Which course did I take in Sept. 2021?", "2021-09-01", "2021-09-30"),
        ("What did I buy in March?", "2023-03-01", "2023-03-31"),
        ("What did I plan in August?", "2022-08-01", "2022-08-31"),
        ("In June, where did we go?", "2022-06-01", "2022-06-30"),
        ("What did I do in mid-March?", "2023-03-01", "2023-03-31"),
        ("What happened in 2021?", "2021-01-01", "2021-12-31"),
        ("What did I eat yesterday?", "2023-05-09", "2023-05-09"),
        ("Where did I go last night?", "2023-05-09", "2023-05-09"),
        ("Where did I go last Saturday?", "2023-05-06", "2023-05-06"),
        ("Who did I see last Wednesday?", "2023-05-03", "2023-05-03"),
        ("What did I do this past Monday?", "2023-05-08", "2023-05-08"),
        ("What did I do last weekend?", "2023-05-06", "2023-05-07"),
        ("What happened last week?", "2023-05-01", "2023-05-07"),
        ("Who called last month?", "2023-04-01", "2023-04-30"),
        ("What did I start last year?", "2022-01-01", "2022-12-31"),
        ("What did I do 3 days ago?", "2023-05-06", "2023-05-08"),
        ("What did I do two weeks ago?", "2023-04-19", "2023-05-03"),
        ("What happened a couple of months ago?", "2023-02-01", "2023-04-30"),
        ("What changed five years ago?", "2017-01-01", "2019-12-31"),
    ],
)
def test_each_registered_expression_resolves(question: str, start: str, end: str) -> None:
    assert query_time_range(question, ANCHOR) == (d(start), d(end))


@pytest.mark.parametrize(
    "question",
    [
        "How long did the trip take?",
        "When did Caroline go to the support group?",
        "What is my favourite colour?",
        "Did I finish reading the book?",
    ],
)
def test_a_question_without_a_time_gets_no_range(question: str) -> None:
    assert query_time_range(question, ANCHOR) is None


def test_the_verb_may_is_not_the_month() -> None:
    assert query_time_range("Should I sort them before may be too late?", ANCHOR) is None


def test_the_earliest_expression_wins() -> None:
    question = "Before the trip in May 2023, and also last week, what did I pack?"
    assert query_time_range(question, ANCHOR) == (d("2023-05-01"), d("2023-05-31"))
