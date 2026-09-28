"""W1 key resolution v2: the generic-subject guard, the ``ies`` stem, and the W3-based selection
metric. Each test names its red proof."""

from __future__ import annotations

import sys
from pathlib import Path

from recall_aml.key_resolution import KeyResolver, ResolverConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

V2 = ResolverConfig(subject_jaccard=0.5, attribute_jaccard=1.0, synonyms=True, min_subject_words=2, ies_stem=True)


def test_a_bare_generic_subject_no_longer_absorbs_specific_ones() -> None:
    """Invariant: under the guard a one-word subject matches only itself, so "budget" keeps its
    own slot and each specific budget keeps its own; two-word subjects still merge by containment.
    All keys are from conversation 15 of the W1 run, where v1 put 12 of them in one slot.

    Red proof: deleting the guard's early return in `KeyResolver._score` merges "dining out
    budget|total amount" into "budget|total amount" and fails the second equality.
    """
    resolver = KeyResolver(V2)
    assert resolver.resolve("budget|total amount") == "budget|total amount"
    assert resolver.resolve("dining out budget|total amount") == "dining out budget|total amount"
    assert resolver.resolve("budget|amount") == "budget|total amount"
    assert resolver.resolve("gift budget|total amount") == "gift budget|total amount"
    assert resolver.resolve("holiday gift budget|total amount") == "gift budget|total amount"


def test_the_ies_stem_joins_groceries_and_grocery() -> None:
    """Invariant: with ``ies_stem`` "joint budget for groceries" contains "grocery budget", so the
    W1 run's grocery update ($500 to $550) shares a slot without any generic cluster.

    Red proof: removing the ``ie`` branch from `_stem` leaves "grocerie" and fails the first
    equality; mapping only a literal "ies" to "y" (the first version) splits "movie" from "movies"
    and fails the second.
    """
    resolver = KeyResolver(V2)
    resolver.resolve("joint budget for groceries|amount")
    assert resolver.resolve("grocery budget|amount") == "joint budget for groceries|amount"
    # One word each, so under the guard only an exact stem can match (with more words, Jaccard
    # would merge them anyway and hide a broken stem).
    resolver.resolve("movies|rating")
    assert resolver.resolve("movie|rating") == "movies|rating"


def test_the_default_configuration_is_still_v1() -> None:
    """Invariant: the defaults reproduce v1 exactly (no guard, v1's stem), so v1's recorded result
    can be re-derived for comparison.

    Red proof: defaulting `ResolverConfig.min_subject_words` to 2 keeps "dining out budget" apart
    and fails the equality.
    """
    resolver = KeyResolver(ResolverConfig(0.5, 1.0, True))
    resolver.resolve("budget|total amount")
    assert resolver.resolve("dining out budget|total amount") == "budget|total amount"


def test_current_value_recall_counts_over_covered_pairs() -> None:
    """Invariant: the selection metric is correct update histories over COVERED update pairs, so a
    configuration cannot win by linking few pairs cleanly.

    Red proof: dividing by linked pairs instead of covered in `current_value_recall` gives 0.75
    and fails the equality.
    """
    from aml_w1_key_resolution_v2 import current_value_recall

    resolution = {"pairs": {"knowledge_update": {"covered": 12, "linked": 8}}}
    history = {"knowledge_update": {"linked": 8, "update_correct": 0.75}}
    assert current_value_recall(resolution, history) == 0.5


def test_resolved_records_replay_in_add_order() -> None:
    """Invariant: the records handed to the W3 check carry canonical keys decided in Add order.

    Red proof: replaying in file order (dropping the sort in `resolved_records`) makes the later
    Add's name canonical and fails the equality.
    """
    from aml_w1_key_resolution_v2 import resolved_records

    records = [
        {"add": "c0:a1", "conversation": 0, "facts": [{"key": "grocery budget|amount", "turns": [9]}]},
        {"add": "c0:a0", "conversation": 0, "facts": [{"key": "joint budget for groceries|amount", "turns": [2]}]},
    ]
    out = resolved_records(records, V2)
    assert [f["key"] for r in out for f in r["facts"]] == ["joint budget for groceries|amount"] * 2
