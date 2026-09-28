"""W1 and W3 v3: event updates in the history classifier, the same-Add width rule, and selection
under the width bar. Each test names its red proof."""

from __future__ import annotations

from datetime import UTC, datetime
import sys
from pathlib import Path

from recall_aml.conversation_records import ConversationFact
from recall_aml.key_resolution import KeyResolver, ResolverConfig
from recall_aml.state_history import classify, key_histories

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _fact(value: str, relation: str, day: int, key: str = "zoom call|date and time") -> ConversationFact:
    subject, _, attribute = key.partition("|")
    return ConversationFact(
        key=key, subject=subject, attribute=attribute, value=value, relation=relation, speaker="user",
        event_date=None, mention_time=datetime(2024, 4, day, tzinfo=UTC), anchor_ids=("a",),
        message_ordinals=(0,), sensitive=False,
    )


def test_an_event_then_a_state_is_an_update_only_with_event_updates() -> None:
    """Invariant: with ``event_updates`` a scheduled event followed by a different stated value on
    one key is an update whose latest fact is the new value; without it, the recorded behaviour
    (``none``) is unchanged. The pair is from BEAM 100K conversation 7.

    Red proof: making `_counted` ignore ``event_updates`` (always `CHANGING`) classifies the pair
    ``none`` and fails the first equality.
    """
    facts = [_fact("April 21 at 3 PM", "event", 1), _fact("April 22 at 11 AM", "state", 2)]
    assert classify(facts, event_updates=True) == "update"
    assert classify(facts) == "none"
    history = key_histories(facts, event_updates=True)["zoom call|date and time"]
    assert history.latest is not None and history.latest.value == "April 22 at 11 AM"


def test_events_alone_are_never_an_update() -> None:
    """Invariant: several events on one key (two trips) are a record of events, not a changing
    state, even with ``event_updates``.

    Red proof: dropping the ``relations & CHANGING`` condition in `classify` makes the two trips an
    update and fails the equality.
    """
    trips = [_fact("Paris", "event", 1, "user|trip"), _fact("Rome", "event", 5, "user|trip")]
    assert classify(trips, event_updates=True) == "none"


def test_two_values_stated_in_one_add_do_not_share_a_cluster() -> None:
    """Invariant: under ``same_add_values`` a key cannot join a cluster that already holds a
    different value from the same Add; across Adds it still can. The keys are from BEAM 100K
    conversation 15, where v2 folded every budget into "budget".

    Red proof, two mutations: making `KeyResolver._clashes` always return False lets "gift budget"
    join the budget cluster in the same Add and fails the second equality; scoring every
    containment flat (ignoring ``specific_containment`` in `KeyResolver._score`) makes "holiday gift
    budget" tie with the older generic cluster, which takes it, and fails the third.
    """
    config = ResolverConfig(0.5, 1.0, True, 1, True, same_add_values=True, specific_containment=True)
    resolver = KeyResolver(config)
    resolver.resolve("budget|total amount", add="c15:a3", value="$2,000")
    assert resolver.resolve("dining out budget|total amount", add="c15:a9", value="$300") == "budget|total amount"
    assert resolver.resolve("gift budget|total amount", add="c15:a9", value="$200") == "gift budget|total amount"
    assert resolver.resolve("holiday gift budget|total amount", add="c15:a12", value="$450") == "gift budget|total amount"


def test_selection_only_considers_configurations_under_the_width_bar() -> None:
    """Invariant: tuning picks the best current-value recall AMONG configurations whose largest
    cluster is at most the bar, so a wide configuration cannot win on recall alone.

    Red proof: selecting over all configurations (dropping the eligibility filter in `select`)
    picks the 12-key configuration and fails the equality.
    """
    from aml_w1_key_resolution_v3 import select

    wide, narrow = ResolverConfig(0.5, 1.0, True), ResolverConfig(0.67, 1.0, True, 2, True, True)
    tuning = [
        (wide, {"largest_cluster": 12, "keys_merged": 300, "current_value_recall_given_coverage": 0.3}),
        (narrow, {"largest_cluster": 5, "keys_merged": 120, "current_value_recall_given_coverage": 0.2}),
    ]
    assert select(tuning) == (narrow, True)
