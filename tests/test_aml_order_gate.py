"""E-1's ordering-question gate (``recall_aml.order_gate``).

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-gated-chronological-order.md,
apparatus check 1.

Red proof, 2026-09-25: each mutation applied to ``recall_aml/order_gate.py`` alone, the named
test run, then restored and run green.

- Deleting ``r"which came first"`` from ``ORDER_PATTERNS``:
  ``test_every_registered_phrase_fires[Which came first, the move or the new job?]`` failed on
  ``assert asks_for_order(question)``.
- Widening ``r"what order"`` to ``r"order"``:
  ``test_near_misses_do_not_fire[In order to save money, what did I stop buying?]`` failed on
  ``assert not asks_for_order(question)``.
- Dropping ``re.IGNORECASE``: ``test_the_match_ignores_case`` failed on its assertion.
"""

from __future__ import annotations

import pytest

from recall_aml.order_gate import ORDER_PATTERNS, asks_for_order, matched_patterns

FIRES = (
    "In what order did I visit Rome, Oslo and Lima?",
    "In which order were the three offers made?",
    "What order should these steps go in?",
    "List the trips chronologically.",
    "Give a chronological account of the renovation.",
    "What was the sequence of events after the fire?",
    "Describe the order in which I adopted my pets.",
    "What order did the promotions come in?",
    "Build a timeline of my job changes.",
    "Which came first, the move or the new job?",
    "Which happened first: the wedding or the trip?",
    "What happened next after the interview?",
    "List my certifications in the order I earned them.",
    "Rank the projects by date.",
    "Put the meetings from earliest to latest.",
    "Sort the photos oldest to newest.",
    "Go through the steps from first to last.",
)
NEAR_MISSES = (
    "In order to save money, what did I stop buying?",
    "What did I order for dinner on Friday?",
    "Did I order a pizza last week?",
    "What is my order number for the shoes?",
    "When did I first meet Sara?",
    "What was the last book I finished?",
)


@pytest.mark.parametrize("question", FIRES)
def test_every_registered_phrase_fires(question: str) -> None:
    assert asks_for_order(question)


def test_every_pattern_is_exercised() -> None:
    hit = {pattern for question in FIRES for pattern in matched_patterns(question)}
    assert hit == set(ORDER_PATTERNS)


@pytest.mark.parametrize("question", NEAR_MISSES)
def test_near_misses_do_not_fire(question: str) -> None:
    assert not asks_for_order(question)


def test_the_match_ignores_case() -> None:
    assert asks_for_order("WHICH CAME FIRST, THE MOVE OR THE JOB?")


def test_a_list_pattern_does_not_cross_a_line_break() -> None:
    assert not asks_for_order("List the options.\nA. Keep them in order\nB. Drop them")
