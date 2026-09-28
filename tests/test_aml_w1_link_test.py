"""`scripts/aml_w1_link_test.py`: the Add cutting and pair metrics that decide W1's go/no-go.

Each test names the mutation of the harness it was watched to fail on (the red proof).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aml_w1_link_test as w1  # noqa: E402


def _turn(i: int, words: int = 5) -> dict:
    return {"id": i, "role": "user", "content": " ".join(["w"] * words), "date": "March-15-2024"}


def test_adds_hold_at_most_twenty_messages_and_two_thousand_words_in_order() -> None:
    """Invariant: an Add closes at 20 messages or before it would pass 2,000 words, as AML cuts.

    Red proof: dropping the word bound from `chunk_adds` keeps the long turns in one Add and fails
    the second assertion.
    """
    assert [len(add) for add in w1.chunk_adds([_turn(i) for i in range(45)])] == [20, 20, 5]
    long = [_turn(0, 1500), _turn(1, 600), _turn(2, 100)]
    assert [[t["id"] for t in add] for add in w1.chunk_adds(long)] == [[0], [1, 2]]


def _row(batches: list[list[dict]], pairs: list[tuple[list[int], list[int]]]) -> dict:
    return {
        "chat": batches,
        "probing_questions": {"knowledge_update": [
            {"source_chat_ids": {"original_info": left, "updated_info": right}} for left, right in pairs
        ]},
    }


def test_a_prefix_keeps_whole_turns_up_to_the_word_budget() -> None:
    """Invariant: the cut keeps turns in order while the running total stays within the budget,
    drops the turn that would pass it and everything after, and keeps batch boundaries.

    Red proof: keeping the crossing turn (appending before the budget check in `cut_to_words`)
    keeps turn 3 and fails the equality.
    """
    row = _row([[_turn(0, 40), _turn(1, 40)], [_turn(2, 15), _turn(3, 10), _turn(4, 1)]], [])
    cut = w1.cut_to_words(row, 100)
    assert [[t["id"] for t in batch] for batch in cut["chat"]] == [[0, 1], [2]]


def test_a_pair_cut_away_is_dropped_and_the_others_keep_their_index() -> None:
    """Invariant: a pair with a source turn no longer in the chat is dropped rather than counted
    as uncovered, and surviving pairs keep their index among their type (the W3 check and the
    audits look pairs up by it).

    Red proof: removing the presence filter from `pairs_of` returns the cut-away pair and fails
    the equality.
    """
    row = _row([[_turn(0), _turn(1), _turn(2)]], [([0], [9]), ([1], [2])])
    assert [(p["index"], p["left"], p["right"]) for p in w1.pairs_of(row)] == [(1, [1], [2])]


def test_a_pair_is_linked_only_when_both_sides_share_a_key() -> None:
    """Invariant: coverage needs a fact on both sides; a link needs one key on both sides.

    Red proof: counting a pair as linked when both sides merely have facts (``bool(left) and
    bool(right)``) fails the linked count.
    """
    pairs = [
        {"type": "knowledge_update", "left": [1], "right": [2]},
        {"type": "knowledge_update", "left": [3], "right": [4]},
        {"type": "knowledge_update", "left": [5], "right": [6]},
    ]
    keys = {1: {"repo|commits"}, 2: {"repo|commits", "user|city"}, 3: {"api|latency"}, 4: {"api|speed"}}
    result = w1.pair_metrics(pairs, keys)["knowledge_update"]
    assert (result["covered"], result["linked"]) == (2, 1)
    assert result["link_recall"] == round(1 / 3, 3) and result["link_recall_given_coverage"] == 0.5
