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


# E-2: the revised list (docs/preregistrations/2026-09-25-aml-c9-revised-ordering-gate.md,
# apparatus check 1). Red proof, 2026-09-26, each mutation to ``ORDER_PATTERNS_V2`` alone:
# - deleting ``r"correct order"``:
#   ``test_v2_fires_on_every_added_or_changed_phrase[What is the correct order of the three moves?]``
#   failed on ``assert asks_for_order_v2(question)``;
# - reverting ``r"list (.* )?in (the )?order"`` to ``r"list .* in (the )?order"``:
#   ``test_v2_fires_on_every_added_or_changed_phrase[Can you list in order how my plans changed?]``
#   failed on the same assertion;
# - restoring bare ``r"timeline"`` for ``r"timeline of"``:
#   ``test_v2_ignores_timeline_as_a_noun[What is the development timeline for the app?]`` failed on
#   ``assert not asks_for_order_v2(question)``.

from recall_aml.order_gate import ORDER_PATTERNS_V2, asks_for_order_v2, matched_patterns_v2  # noqa: E402

V2_ADDED = (
    "Can you list in order how my plans changed?",
    "Give me a timeline of my job changes.",
    "What is the correct order of the three moves?",
    "Put the trips in the right order.",
    "Order the causes from nearest to farthest.",
    "Go from farthest to nearest.",
    "Sort the photos newest to oldest.",
    "List them from latest to earliest.",
    "Go through the steps from last to first.",
)
TIMELINE_AS_NOUN = (
    "Have I worked with Michael on editing timelines before?",
    "What is the development timeline for the app?",
    "Is my project timeline still realistic?",
)


@pytest.mark.parametrize("question", V2_ADDED)
def test_v2_fires_on_every_added_or_changed_phrase(question: str) -> None:
    assert asks_for_order_v2(question)


@pytest.mark.parametrize("question", TIMELINE_AS_NOUN)
def test_v2_ignores_timeline_as_a_noun(question: str) -> None:
    assert not asks_for_order_v2(question)


@pytest.mark.parametrize("question", NEAR_MISSES)
def test_v2_still_rejects_e1_near_misses(question: str) -> None:
    assert not asks_for_order_v2(question)


def test_every_v2_pattern_is_exercised() -> None:
    hit = {pattern for question in FIRES + V2_ADDED for pattern in matched_patterns_v2(question)}
    assert hit == set(ORDER_PATTERNS_V2)


def test_e1_is_unchanged_by_e2() -> None:
    assert asks_for_order("Is my project timeline still realistic?")
    assert not asks_for_order("What is the correct order of the three moves?")
