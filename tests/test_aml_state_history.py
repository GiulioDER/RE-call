"""W3: keyed fact histories (``recall_aml.state_history``). Each test names its red proof."""

from __future__ import annotations

from datetime import UTC, datetime

from recall_aml.conversation_records import ConversationFact
from recall_aml.state_history import classify, key_histories


def _fact(value: str, relation: str, day: int, key: str = "repo|commits", speaker: str = "user") -> ConversationFact:
    subject, _, attribute = key.partition("|")
    return ConversationFact(
        key=key, subject=subject, attribute=attribute, value=value, relation=relation, speaker=speaker,
        event_date=None, mention_time=datetime(2024, 3, day, tzinfo=UTC), anchor_ids=("a000_x",),
        message_ordinals=(0,), sensitive=False,
    )


def test_an_update_orders_values_by_when_they_were_said_and_marks_the_latest() -> None:
    """Invariant: the latest statement is the current value, whatever order the facts arrive in,
    and earlier values are kept and labelled with when they were replaced.

    Red proof: sorting by the facts' arrival position alone (ignoring ``mention_time``) makes the
    150 statement latest and fails the ``latest`` assertion.
    """
    facts = [_fact("165 commits", "state", 20), _fact("150 commits", "state", 5)]
    history = key_histories(facts)["repo|commits"]
    assert history.kind == "update"
    assert history.latest is not None and history.latest.value == "165 commits"
    assert history.rendered() == (
        '[history · repo / commits · oldest first] 2024-03-05 user: "150 commits" '
        '(no longer current: replaced 2024-03-20); 2024-03-20 user: "165 commits" (latest on record)'
    )


def test_a_never_statement_against_a_happened_one_is_a_conflict_with_no_winner() -> None:
    """Invariant: "never did X" and "did X" on one key are both kept, with no latest value.

    Red proof: removing the ``never`` branch from `classify` leaves the pair unclassified
    (``none``) and fails the kind assertion.
    """
    facts = [_fact("I have never written any Flask routes", "never", 5, key="flask|routes"),
             _fact("implemented a basic homepage route", "state", 2, key="flask|routes")]
    history = key_histories(facts)["flask|routes"]
    assert history.kind == "conflict" and history.latest is None
    assert history.rendered().startswith("[two statements on record, both kept · flask / routes]")


def test_repeats_and_single_values_are_not_updates() -> None:
    """Invariant: one value said twice, or a single fact, is not an update; a repeated value keeps
    its latest statement.

    Red proof: counting facts instead of distinct values in `classify` makes the repeated value an
    update and fails the first assertion.
    """
    assert classify([_fact("165 commits", "state", 5), _fact("165 Commits", "state", 9)]) == "none"
    assert classify([_fact("went to Lisbon", "event", 5)]) == "none"
    chain = key_histories([_fact("150", "state", 1), _fact("165", "state", 5), _fact("150", "state", 9)])["repo|commits"]
    assert [f.value for f in chain.facts] == ["165", "150"]


def test_a_caller_order_breaks_same_day_ties() -> None:
    """Invariant: facts said on the same day are ordered by the caller's key (a turn id), so BEAM's
    batch-level dates cannot scramble an update.

    Red proof: ignoring ``order`` in `key_histories` keeps the arrival order and fails the latest
    assertion.
    """
    same_day = [_fact("165 commits", "state", 5), _fact("150 commits", "state", 5)]
    turn = {"165 commits": 114, "150 commits": 86}
    history = key_histories(same_day, order=lambda f: turn[f.value])["repo|commits"]
    assert history.latest is not None and history.latest.value == "165 commits"
