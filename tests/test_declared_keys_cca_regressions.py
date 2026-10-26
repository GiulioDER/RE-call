"""Receipt state and work bounds found during the PR 990 CCA review."""

from __future__ import annotations

import json
from typing import Any

from recall.declared_keys import key_receipt
from tests.test_assistant_surface import TEA, Call, _serve_both
from tests.test_assistant_surface import envs as envs
from tests.test_declared_keys import _contested, _fact, _listing, _standing


def test_future_write_is_not_in_force_among_existing_contested_values(
    envs: tuple[dict[str, str], dict[str, str]],
) -> None:
    async def body(a: Call, p: Call, state: dict[str, Any]) -> None:
        await a("recall_remember", text="The user drink tea every morning.", kind="pref", fact=TEA)
        await a("recall_remember", text="The user drink coffee every morning.", kind="pref",
                fact={**TEA, "object": "coffee"})
        answer = json.loads(await a(
            "recall_remember", text="The user will drink cocoa after 2099.", kind="pref",
            fact={**TEA, "object": "cocoa"}, valid_from="2099-01-01", include_receipt=True,
        ))
        receipt = answer["receipt"]["fact_key"]
        assert receipt["state"] == "not_in_force", receipt
        assert {item["value"] for item in receipt["in_force"]} == {"tea", "coffee"}, receipt
        assert receipt["proposals"] == [], receipt

    _serve_both(envs, body)


def test_absent_source_is_not_in_force_even_if_its_key_is_contested() -> None:
    tea, coffee = _fact("user", "drink", "tea"), _fact("user", "drink", "coffee")
    listing = _listing([], [_contested((tea, "tea.md"), (coffee, "coffee.md"))])
    receipt = key_receipt(listing, "future.md", _fact("user", "drink", "cocoa", 20))
    assert receipt["state"] == "not_in_force", receipt


def test_unchecked_listing_cannot_be_promoted_to_a_current_receipt() -> None:
    tea = _fact("user", "drink", "tea")
    receipt = key_receipt(_listing([_standing(tea, "tea.md")], served="unchecked"), "tea.md", tea)
    assert receipt["state"] == "unchecked", receipt
    assert receipt["in_force"] == [] and receipt["proposals"] == [], receipt


def test_receipt_does_not_materialize_unrelated_successor_pairs() -> None:
    class CountedName(str):
        hashes = 0

        def __hash__(self) -> int:
            type(self).hashes += 1
            return super().__hash__()

    size = 40
    earlier, later = _fact("user", "drink", "tea", 1), _fact("user", "drink", "coffee", 2)
    older_names = [CountedName(f"old-{i:02}.md") for i in range(size)]
    newer_names = [CountedName(f"new-{i:02}.md") for i in range(size)]
    listing = _listing([_standing(earlier, *older_names), _standing(later, *newer_names)])
    receipt = key_receipt(listing, newer_names[0], later)
    assert {(item["superseded"], item["superseder"]) for item in receipt["proposals"]} == {
        (name, newer_names[0]) for name in older_names
    }, receipt
    assert CountedName.hashes < 20 * size, "the receipt allocated pairs unrelated to its source"
