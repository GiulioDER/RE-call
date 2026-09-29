"""W0 R1 stage 1: local BM25, weighted fusion, turn reciprocal rank, weight choice and the
recovery rule. Each test names its red proof."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aml_w0_r1_reorder as r1  # noqa: E402


def test_bm25_puts_the_item_holding_the_rare_query_terms_first() -> None:
    """Invariant: BM25 over the given items ranks the item holding both query terms first and
    breaks ties in the given order.

    Red proof: sorting by ascending score in `bm25_order` puts an item with no query term first and
    fails the equality.
    """
    texts = ["we talked about the weather", "my golden retriever Max", "a retriever", "nothing here"]
    order = r1.bm25_order("golden retriever", texts)
    assert order[:2] == [1, 2]
    assert order[2:] == [0, 3]


def test_fusion_keeps_the_stored_order_at_zero_weight_and_follows_lexical_when_heavy() -> None:
    """Invariant: with weight 0 the stored order is kept; with a large lexical weight the lexical
    first item rises to the top.

    Red proof: fusing the stored rank with itself (ignoring ``lexical``) in `fuse` leaves item 3
    at the bottom and fails the second equality.
    """
    assert r1.fuse(4, [3, 2, 1, 0], 0.0) == [0, 1, 2, 3]
    assert r1.fuse(4, [3, 2, 1, 0], 50.0)[0] == 3


def test_turn_rr_is_one_over_the_first_gold_turn_rank() -> None:
    """Invariant: the reciprocal rank counts from 1 at the first item holding a gold turn; no hit
    gives 0.

    Red proof: counting ranks from 0 in `turn_rr` gives 1/2 for a hit at position 3 and fails the
    first equality.
    """
    from aml_w0_embedding_compare import Question

    gold = "I adopted a golden retriever named Max last week"
    question = Question("q", "1", "u", "dog?", frozenset({"s"}), (gold,))
    items = [{"content": "a"}, {"content": "b"}, {"content": gold}]
    assert r1.turn_rr(items, question) == 1 / 3
    assert r1.turn_rr([{"content": "a"}], question) == 0.0


def test_the_weight_is_chosen_on_tuning_with_ties_to_the_smaller() -> None:
    """Invariant: the weight with the best tuning turn_hit@10 wins; an exact tie goes to the
    smaller weight.

    Red proof: breaking ties toward the larger weight in `choose_weight` picks 2.0 and fails.
    """
    assert r1.choose_weight({0.5: 90.0, 1.0: 91.0, 2.0: 91.0}) == 1.0
    assert r1.choose_weight({0.5: 92.0, 1.0: 91.0, 2.0: 90.0}) == 0.5


def test_each_arm_reads_the_collect_it_names() -> None:
    """Invariant: "A" and "R(A)" re-order A's items; "P", "R(P)" and "F(w,P)" re-order P's.

    Red proof: the first version's ``arm.endswith("A")`` sends "R(A)" to P's items (its name ends
    in a parenthesis) and fails the second equality. Found in review before any run.
    """
    assert r1.arm_base("A") == "A"
    assert r1.arm_base("R(A)") == "A"
    assert [r1.arm_base(a) for a in ("P", "R(P)", "F(1.0,P)")] == ["P", "P", "P"]


def test_recovery_needs_one_and_a_half_points_at_ten_and_no_loss_at_five() -> None:
    """Invariant: an arm recovers enough when turn_hit@10 rises by at least 1.5 points AND
    turn_hit@5 does not fall.

    Red proof: dropping the turn_hit@5 condition in `recovers` accepts the second case and fails.
    """
    base = {"turn_hit@10": 88.0, "turn_hit@5": 80.0}
    assert r1.recovers({"turn_hit@10": 89.5, "turn_hit@5": 80.0}, base)
    assert not r1.recovers({"turn_hit@10": 90.0, "turn_hit@5": 79.9}, base)
    assert not r1.recovers({"turn_hit@10": 89.4, "turn_hit@5": 85.0}, base)
