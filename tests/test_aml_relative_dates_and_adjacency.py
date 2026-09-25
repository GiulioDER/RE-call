"""T-1 (resolved relative dates) and K-2 (same-subject adjacency) behaviour proofs.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-relative-dates-and-conflict-adjacency.md.
Apparatus check 1 of that record: every row of the T-1 table on a fixed anchor, the render-only
property of both transforms, and the service wiring behind default-off flags.

Red proofs, each run on 2026-09-25 against a deliberate mutation of the production line named,
failing at the assertion named, then green after restoring it:

* ``test_each_registered_expression_resolves_against_the_anchor[last Monday...]``: in
  ``temporal_render._resolve`` the ``or 7`` of the "last <weekday>" branch removed, so "last
  Monday" on a Monday resolves to the anchor itself; fails at the resolution equality.
* ``test_week_based_expressions_resolve_from_a_midweek_anchor``: ``_monday`` returning its
  argument unchanged; fails at the resolution equality. (The Monday-anchored table cannot see this
  mutation, since a Monday is its own week's Monday; the first attempt at this proof stayed green.)
* ``test_removing_the_inserted_brackets_restores_the_original_text``: the replacement dropping the
  phrase and keeping only its resolution; fails at the restored-text equality. (Inserting the
  resolution before the phrase is not caught there, since removal undoes it too; the second-pass
  assertion in the same test catches that one.)
* ``test_items_on_the_same_day_are_never_grouped``: the different-day condition removed from
  ``same_subject_adjacent``; fails at the order equality.
* ``test_a_group_moves_to_its_best_rank_newest_first``: ``newest_first`` sorted oldest first;
  fails at the order equality.
* ``test_words_shared_by_half_the_window_do_not_link_items``: the ``common`` word filter removed;
  fails at the order equality.
* ``test_a_group_holds_at_most_four_items``: ``MAX_GROUP`` raised to 10; fails at the group-size
  assertion.
* ``test_the_service_resolves_relative_dates_only_when_the_flag_is_on`` and
  ``test_the_service_reorders_same_subject_items_only_when_the_flag_is_on``: the matching call
  removed from ``HostedService.search``; each fails at its flag-on assertion.
"""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
import re

import pytest

from recall_aml.conflict_order import same_subject_adjacent
from recall_aml.models import AddRequest, SearchItem, SearchRequest
from recall_aml.temporal_render import resolve_relative_times, resolve_text
from tests.test_aml_specialist_fusion import _service

ANCHOR = date(2023, 5, 8)  # a Monday
BRACKET = re.compile(r" \[(?:=|≈|week of|weekend of) [0-9-]+\]")


@pytest.mark.parametrize(
    ("phrase", "resolution"),
    [
        ("today", "[= 2023-05-08]"),
        ("tonight", "[= 2023-05-08]"),
        ("this evening", "[= 2023-05-08]"),
        ("yesterday", "[= 2023-05-07]"),
        ("last night", "[= 2023-05-07]"),
        ("the day before yesterday", "[= 2023-05-06]"),
        ("tomorrow", "[= 2023-05-09]"),
        ("3 days ago", "[≈ 2023-05-05]"),
        ("two weeks ago", "[≈ 2023-04-24]"),
        ("a couple of months ago", "[≈ 2023-03]"),
        ("a few years ago", "[≈ 2020]"),
        ("last week", "[week of 2023-05-01]"),
        ("this week", "[week of 2023-05-08]"),
        ("next week", "[week of 2023-05-15]"),
        ("last weekend", "[weekend of 2023-05-06]"),
        ("next weekend", "[weekend of 2023-05-20]"),
        ("last month", "[= 2023-04]"),
        ("next month", "[= 2023-06]"),
        ("last year", "[= 2022]"),
        ("last Friday", "[= 2023-05-05]"),
        ("next Friday", "[= 2023-05-12]"),
        ("this Friday", "[= 2023-05-12]"),
        ("last Monday", "[= 2023-05-01]"),
        ("next Monday", "[= 2023-05-15]"),
    ],
)
def test_each_registered_expression_resolves_against_the_anchor(phrase: str, resolution: str) -> None:
    assert resolve_text(f"We met {phrase} at noon.", ANCHOR) == f"We met {phrase} {resolution} at noon."


@pytest.mark.parametrize(
    ("phrase", "resolution"),
    [
        ("last week", "[week of 2023-05-01]"),
        ("this week", "[week of 2023-05-08]"),
        ("this weekend", "[weekend of 2023-05-13]"),
        ("last weekend", "[weekend of 2023-05-06]"),
        ("this Monday", "[= 2023-05-08]"),
        ("last Monday", "[= 2023-05-08]"),
        ("last Thursday", "[= 2023-05-04]"),
        ("next Thursday", "[= 2023-05-18]"),
        ("this Sunday", "[= 2023-05-14]"),
    ],
)
def test_week_based_expressions_resolve_from_a_midweek_anchor(phrase: str, resolution: str) -> None:
    """The Monday anchor above hides any error in finding a week's Monday; a Thursday does not."""
    thursday = date(2023, 5, 11)
    assert resolve_text(f"We met {phrase}.", thursday) == f"We met {phrase} {resolution}."


def test_removing_the_inserted_brackets_restores_the_original_text() -> None:
    text = (
        "Caroline: I went to a support group yesterday. [shared image: a dog] Melanie: Last "
        "weekend we camped, and two weeks ago I painted. Next Friday is the recital."
    )
    resolved = resolve_text(text, ANCHOR)
    assert resolved.count("[") == text.count("[") + 4
    assert BRACKET.sub("", resolved) == text
    assert resolve_text(resolved, ANCHOR) == resolved, "a second pass must add nothing"


def test_words_that_only_contain_a_pattern_are_left_alone() -> None:
    text = "The weekend newsletter comes weekly; yesterdays are gone; Todayville is a town."
    assert resolve_text(text, ANCHOR) == text


def _item(item_id: str, content, day: int | None, score: float = 0.5) -> SearchItem:
    created = None if day is None else datetime(2023, 5, 1, 12, tzinfo=timezone.utc) + timedelta(days=day)
    return SearchItem.model_validate(
        {
            "id": item_id,
            "content": content,
            "created_at": created,
            "source": "s",
            "session_id": f"d{day}",
            "kind": "raw",
            "score": score,
        }
    )


def test_relative_dates_use_each_items_own_date_and_skip_the_rest() -> None:
    items = [
        _item("a", "I adopted a puppy yesterday.", 1),
        _item("b", "I adopted a puppy yesterday.", 5),
        _item("c", "No anchor, yesterday stays.", None),
        _item("d", [{"type": "text", "text": "yesterday"}], 3),
    ]
    resolved = resolve_relative_times(items)
    assert [item.id for item in resolved] == ["a", "b", "c", "d"]
    assert resolved[0].content == "I adopted a puppy yesterday [= 2023-05-01]."
    assert resolved[1].content == "I adopted a puppy yesterday [= 2023-05-05]."
    assert resolved[2] == items[2]
    assert resolved[3] == items[3]
    assert [item.score for item in resolved] == [item.score for item in items]


PUPPY_OLD = "Melanie adopted golden retriever puppy named Luna from animal shelter downtown"
PUPPY_NEW = "Melanie's golden retriever puppy Luna now sleeps beside animal shelter volunteers downtown"
UNRELATED = [
    "Caroline painted sunset landscape canvas using watercolour brushes",
    "Harbour ferry schedule changed because storms flooded northern piers",
    "Grandfather repaired antique clock mechanism with brass gears",
    "Chemistry exam covered titration, molar volumes and buffer solutions",
]


def test_a_group_moves_to_its_best_rank_newest_first() -> None:
    items = [
        _item("old", PUPPY_OLD, 1),
        _item("u0", UNRELATED[0], 2),
        _item("u1", UNRELATED[1], 3),
        _item("new", PUPPY_NEW, 9),
        _item("u2", UNRELATED[2], 4),
    ]
    assert [item.id for item in same_subject_adjacent(items)] == ["new", "old", "u0", "u1", "u2"]


def test_items_on_the_same_day_are_never_grouped() -> None:
    items = [
        _item("old", PUPPY_OLD, 1),
        _item("u0", UNRELATED[0], 2),
        _item("u1", UNRELATED[1], 3),
        _item("new", PUPPY_NEW, 1),
        _item("u2", UNRELATED[2], 4),
    ]
    assert [item.id for item in same_subject_adjacent(items)] == ["old", "u0", "u1", "new", "u2"]


def test_words_shared_by_half_the_window_do_not_link_items() -> None:
    """Speaker names recur in every LoCoMo window; they must not make everything one subject."""
    # Four shared names and one own word each: Jaccard 4/6 would link every pair if names counted.
    names = "Caroline Melanie Jonathan Beatrice"
    items = [
        _item("a", f"{names} painting", 1),
        _item("b", f"{names} ferries", 2),
        _item("c", f"{names} clocks", 3),
        _item("d", f"{names} chemistry", 4),
    ]
    assert [item.id for item in same_subject_adjacent(items)] == ["a", "b", "c", "d"]


def test_a_group_holds_at_most_four_items() -> None:
    items = [_item(f"p{day}", PUPPY_OLD, day) for day in range(6)]
    items += [_item(f"u{index}", text, 10 + index) for index, text in enumerate(UNRELATED * 2)]
    ordered = [item.id for item in same_subject_adjacent(items)]
    # The best-ranked item plus its three strongest links (all tie, so by rank), newest first;
    # members cut by the cap keep their own ranks and are not regrouped.
    grouped = ordered[:4]
    assert len(grouped) == 4 and grouped == ["p3", "p2", "p1", "p0"]
    assert ordered[4:6] == ["p4", "p5"]


def test_only_the_top_thirty_move_and_nothing_is_added_or_lost() -> None:
    items = [_item(f"u{index}", f"{UNRELATED[index % 4]} item {index}", index % 3) for index in range(29)]
    items += [_item("old", PUPPY_OLD, 1), _item("far", PUPPY_NEW, 9)]
    ordered = same_subject_adjacent(items)
    assert [item.id for item in ordered][-1] == "far", "rank 31 is outside the window"
    assert sorted(item.model_dump_json() for item in ordered) == sorted(
        item.model_dump_json() for item in items
    )


def _add(service, user_id: str, request_id: str, text: str, day: int) -> None:
    stamp = int((datetime(2023, 5, 1, 12, tzinfo=timezone.utc) + timedelta(days=day)).timestamp() * 1000)
    asyncio.run(
        service.add(
            AddRequest.model_validate(
                {
                    "request_id": request_id,
                    "user_id": user_id,
                    "session_id": request_id,
                    "messages": [{"role": "user", "content": text, "timestamp": stamp}],
                }
            )
        )
    )


def _search(service, user_id: str, query: str):
    return asyncio.run(
        service.search(SearchRequest.model_validate({"query": query, "user_id": user_id}))
    )


def test_the_service_resolves_relative_dates_only_when_the_flag_is_on(monkeypatch) -> None:
    for flag, expected in (("0", "yesterday."), ("1", "yesterday [= 2023-05-02].")):
        monkeypatch.setenv("RECALL_AML_RESOLVE_RELATIVE_TIMES", flag)
        service, _, _, _, _ = _service("C7_routed_specialists")
        _add(service, "t1-user", "t1-a", "I adopted a puppy yesterday.", 2)
        response = _search(service, "t1-user", "puppy adoption")
        assert any(str(item.content).endswith(expected) for item in response.data), flag


def test_the_service_reorders_same_subject_items_only_when_the_flag_is_on(monkeypatch) -> None:
    def ids(flag: str) -> list[str]:
        monkeypatch.setenv("RECALL_AML_SAME_SUBJECT_ORDER", flag)
        service, _, _, _, _ = _service("C7_routed_specialists")
        _add(service, "k2-user", "k2-old", PUPPY_OLD, 1)
        for index, text in enumerate(UNRELATED):
            _add(service, "k2-user", f"k2-u{index}", text, 3 + index)
        _add(service, "k2-user", "k2-new", PUPPY_NEW, 9)
        response = _search(service, "k2-user", "golden retriever puppy")
        return [item.session_id for item in response.data]

    off, on = ids("0"), ids("1")
    old_off, new_off = off.index("k2-old"), off.index("k2-new")
    # Precondition that makes the red proof possible: served order does not already do K-2.
    assert not (new_off + 1 == old_off), off
    first = min(on.index("k2-old"), on.index("k2-new"))
    assert on[first : first + 2] == ["k2-new", "k2-old"], on
    assert sorted(on) == sorted(off)


def test_a_bad_flag_value_stops_service_startup(monkeypatch) -> None:
    monkeypatch.setenv("RECALL_AML_SAME_SUBJECT_ORDER", "yes")
    with pytest.raises(ValueError, match="RECALL_AML_SAME_SUBJECT_ORDER must be 1 or 0"):
        _service("C7_routed_specialists")
