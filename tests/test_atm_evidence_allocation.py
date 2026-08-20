"""The evidence packer must not drop a retrieved item to feed the one ranked above it.

Measured on the 1,013 saved retrievals of the Voyage 4 / DeepSeek run, the greedy packer it
replaces presented a mean of 6.13 items of 10 whole and hid a retrieved GOLD item entirely on 61
questions. Those questions scored 0.4067 against 0.6892 overall. These tests pin the properties
that make that impossible, not the exact allowances, which are free to change.
"""

from __future__ import annotations

import pytest

from benchmarks.atm_full_run import (
    DEFAULT_EVIDENCE_FLOOR,
    _allocate,
    _evidence_text,
    _truncate_on_boundary,
)


def _hits(*lengths: int) -> list[dict[str, object]]:
    return [
        {"id": f"item{i:02d}", "text": " ".join(["word"] * (length // 5))}
        for i, length in enumerate(lengths)
    ]


def test_everything_fits_is_left_exactly_alone() -> None:
    hits = _hits(100, 100, 100)
    out = _evidence_text(hits, 8192)
    assert out.count("\n\n") == 2
    for hit in hits:
        assert f"[{hit['id']}] {hit['text']}" in out


def test_no_item_is_dropped_when_one_is_enormous() -> None:
    """The regression this function exists for: item 0 alone used to eat the whole budget."""
    hits = _hits(9000, 700, 700, 700, 700, 700, 700, 700, 700, 700)
    out = _evidence_text(hits, 8192)
    for hit in hits:
        assert f"[{hit['id']}]" in out, f"{hit['id']} was dropped"


def test_the_budget_is_never_exceeded() -> None:
    for budget in (500, 1000, 4096, 8192):
        out = _evidence_text(_hits(*([3000] * 10)), budget)
        assert len(out) <= budget, f"overran at {budget}: {len(out)}"


def test_the_floor_is_large_enough_to_carry_an_identifying_header() -> None:
    """100 characters is chosen for what it renders, and the test says so rather than pinning 100.

    A usable Timestamp survives in 100% of the corpus's rendered blocks at this floor and in 0% at
    60. If the floor is ever lowered, this is the property that breaks first.
    """
    block = "[email202409200010] ID: email202409200010 Timestamp: 2024-09-20 23:12:00 Summary: a long tail"
    kept = _truncate_on_boundary(block, DEFAULT_EVIDENCE_FLOOR)
    assert "Timestamp: 2024-09-20 23:12:00" in kept


def test_the_lead_item_keeps_a_larger_share_than_the_tail() -> None:
    allowance = _allocate([5000] * 10, 8192, floor=DEFAULT_EVIDENCE_FLOOR)
    assert allowance[0] > allowance[-1]
    assert min(allowance) >= min(DEFAULT_EVIDENCE_FLOOR, 8192 // 10)


def test_a_short_item_releases_its_surplus_to_a_long_one() -> None:
    """Rank greedy, not equal shares: nine short items must not each reserve a tenth."""
    equal = 8192 // 10
    allowance = _allocate([9000] + [50] * 9, 8192, floor=100)
    assert allowance[0] > equal * 3
    assert allowance[1:] == [50] * 9


def test_truncation_prefers_a_word_boundary() -> None:
    assert _truncate_on_boundary("alpha beta gamma", 12) == "alpha beta"
    # A boundary too early to be worth honouring falls back to a hard cut, rather than throwing
    # away most of an allowance that was paid for.
    assert _truncate_on_boundary("a supercalifragilistic", 20) == "a supercalifragilist"


def test_an_allowance_never_promises_more_than_the_item_holds() -> None:
    allowance = _allocate([100, 9000], 8192, floor=400)
    assert allowance[0] == 100
    assert sum(allowance) <= 8192


@pytest.mark.parametrize("count", [1, 2, 5, 10, 25])
def test_every_item_survives_at_any_plausible_k(count: int) -> None:
    out = _evidence_text(_hits(*([4000] * count)), 8192)
    assert out.count("[item") == count


@pytest.mark.parametrize("budget", [0, 1, 5, 17, 40, 99])
def test_a_degenerate_budget_does_not_overrun_with_separators(budget: int) -> None:
    """Found by reviewing the diff, not by the corpus: a budget below the separator cost used to
    return a page of blank lines that was itself longer than the budget."""
    out = _evidence_text(_hits(*([500] * 10)), budget)
    assert len(out) <= budget
    assert out.strip() == out


def test_the_greedy_packer_stays_runnable_as_the_control() -> None:
    """The 68.92 answer file was produced by the greedy packer, so it has to stay reachable, WITH
    its defects: repairing them would leave the H7 experiment without a baseline."""
    hits = _hits(9000, 700, 700)
    greedy = _evidence_text(hits, 8192, packer="greedy")
    allocated = _evidence_text(hits, 8192, packer="allocated")
    assert "[item01]" not in greedy, "the control must still drop the item it dropped"
    assert "[item01]" in allocated

    # The overrun needs more than one block, because what is not counted is the separators. Three
    # blocks make the control pay four characters it never budgeted for.
    spill = _hits(4000, 4000, 700)
    assert len(_evidence_text(spill, 8192, packer="greedy")) > 8192
    assert len(_evidence_text(spill, 8192, packer="allocated")) <= 8192


def test_an_unknown_packer_is_refused() -> None:
    with pytest.raises(ValueError, match="unknown evidence packer"):
        _evidence_text(_hits(100), 8192, packer="whatever")
