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
