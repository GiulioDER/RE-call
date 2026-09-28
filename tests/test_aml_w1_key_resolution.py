"""W1 key resolution: drifted names of one slot merge, other slots stay apart. Each test names its
red proof."""

from __future__ import annotations

import sys
from pathlib import Path

from recall_aml.key_resolution import KeyResolver, ResolverConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

STRICT = ResolverConfig(subject_jaccard=0.67, attribute_jaccard=0.5, synonyms=True)


def test_drifted_names_of_one_slot_resolve_to_the_first_key() -> None:
    """Invariant: a later key whose subject contains (or is contained in) an earlier one, with an
    attribute naming the same slot, maps onto the earlier key. Both pairs are from the W1 run.

    Red proof, two mutations of `recall_aml.key_resolution`: dropping the containment rule in
    `KeyResolver._score` leaves the zoom call apart (Jaccard 0.5 is under 0.67); ignoring
    ``synonyms`` in `key_words` leaves "date and time" and "scheduled time" apart (Jaccard 1/3).
    """
    resolver = KeyResolver(STRICT)
    assert resolver.resolve("zoom call|date and time") == "zoom call|date and time"
    assert resolver.resolve("zoom call with the creative director|scheduled time") == "zoom call|date and time"
    assert resolver.resolve("api integration module|test coverage") == "api integration module|test coverage"
    assert resolver.resolve("api integration|test coverage") == "api integration module|test coverage"
    assert resolver.aliases["zoom call|date and time"] == [
        "zoom call|date and time", "zoom call with the creative director|scheduled time",
    ]


def test_other_slots_of_one_subject_stay_apart() -> None:
    """Invariant: the same subject with a different attribute is a different slot, and an unrelated
    subject never merges, whatever the attribute.

    Red proof: turning the attribute gate off in `KeyResolver._score` (always passing) merges
    "zoom call|attendees" into the date slot and fails the first equality.
    """
    resolver = KeyResolver(STRICT)
    resolver.resolve("zoom call|date and time")
    assert resolver.resolve("zoom call|attendees") == "zoom call|attendees"
    assert resolver.resolve("budget tracker project|testing") == "budget tracker project|testing"
    assert resolver.resolve("dentist appointment|date") == "dentist appointment|date"
    assert resolver.largest_cluster() == 1


def test_the_replay_follows_add_order_not_file_order() -> None:
    """Invariant: facts are replayed in Add order within a conversation, so the canonical key is
    the one stated first even when the file lists a later Add first (the run wrote conversations
    in parallel).

    Red proof: replaying in file order (dropping the sort in `resolved_keys`) makes the second
    Add's name canonical and fails the equality.
    """
    from aml_w1_key_resolution import resolved_keys

    records = [
        {"add": "c0:a1", "conversation": 0, "facts": [{"key": "api integration|test coverage", "value": "78%", "turns": [9]}]},
        {"add": "c0:a0", "conversation": 0, "facts": [{"key": "api integration module|test coverage", "value": "65%", "turns": [2]}]},
    ]
    keys, largest = resolved_keys(records, STRICT)
    assert keys[0][9] == {"api integration module|test coverage"}
    assert largest[0] == (2, 1)


def test_distractors_are_other_pairs_with_both_sides_extracted_and_disjoint() -> None:
    """Invariant: a distractor combination is one pair's first side against a DIFFERENT pair's
    side; combinations with an unextracted side or overlapping turns are not counted.

    Red proof: counting a pair against itself (`itertools.product` for `permutations` in
    `distractor_counts`) adds the pair's own sides and fails the count equality.
    """
    from aml_w1_key_resolution import distractor_counts

    pairs = [
        {"type": "knowledge_update", "left": [1], "right": [5]},
        {"type": "knowledge_update", "left": [2], "right": [6]},
        {"type": "contradiction_resolution", "left": [3], "right": [1]},
    ]
    keys = {1: {"a|x"}, 5: {"a|x"}, 2: {"b|y"}, 6: {"a|x"}, 3: {"c|z"}}
    # pair0.left [1] vs pair1 sides [2] {b|y}: no, [6] {a|x}: yes; vs pair2 [3]: no, [1]: overlap, skipped.
    # pair1.left [2] vs pair0 [1] no, [5] no; vs pair2 [3] no, [1] no.
    # pair2.left [3] vs pair0 [1] no, [5] no; vs pair1 [2] no, [6] no.
    assert distractor_counts(pairs, keys) == (1, 11)
