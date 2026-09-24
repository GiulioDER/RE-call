"""Rules of the C9 aggregation view experiment (scripts/aml_locomo_aggregation_view.py).

Red proof, 2026-09-24, each by one mutation of the named function (scratchpad mutate_agg.py):

* ``test_grounding_drops_foreign_turns_and_non_speakers``: removing ``turn is None or`` from the
  ``grounded_items`` filter kept the entry citing a turn from another session.
* ``test_a_bracketed_turn_id_is_the_same_turn``: removing ``.strip("[]")`` from
  ``grounded_items`` dropped the entry, as the first extraction run did to 57% of entries.
* ``test_records_are_oldest_first_and_deduplicated``: dropping the ``seen`` check in
  ``build_records`` listed "chess" twice.
* ``test_selection_is_limited_to_the_named_person``: making ``select_records`` ignore ``named``
  let Melanie's record outrank Caroline's.
* ``test_records_go_after_the_protected_top_five_and_the_list_stays_at_100``: inserting at the top
  instead of after ``INSERT_AFTER_RANK`` moved raw item 0 off rank 1.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from aml_locomo_aggregation_view import (  # noqa: E402
    build_records,
    grounded_items,
    select_records,
    with_records,
)

SESSION = {
    "id": "conv-1:session_1",
    "sample_id": "conv-1",
    "date": "1:56 pm on 8 May, 2023",
    "timestamp_ms": 1_683_554_160_000,
    "speakers": ("Caroline", "Melanie"),
    "turns": [
        {"id": "D1:1", "speaker": "Caroline", "text": "I play chess on weekends."},
        {"id": "D1:2", "speaker": "Melanie", "text": "I paint sunsets."},
    ],
}


def test_grounding_drops_foreign_turns_and_non_speakers() -> None:
    reply = (
        '{"items": ['
        '{"person": "Caroline", "category": "games", "item": "chess", "turn_id": "D1:1"},'
        '{"person": "Caroline", "category": "games", "item": "go", "turn_id": "D9:9"},'
        '{"person": "Bob", "category": "games", "item": "poker", "turn_id": "D1:1"},'
        '{"person": "melanie", "category": "painting", "item": "sunsets", "turn_id": "D1:2"}'
        "]}"
    )
    kept, proposed = grounded_items(SESSION, reply)
    assert proposed == 4
    assert [(k["person"], k["item"], k["category"]) for k in kept] == [
        ("Caroline", "chess", "games"),
        ("Melanie", "sunsets", "other"),
    ]


def test_a_bracketed_turn_id_is_the_same_turn() -> None:
    reply = '{"items": [{"person": "Caroline", "category": "games", "item": "chess", "turn_id": "[D1:1]"}]}'
    kept, _ = grounded_items(SESSION, reply)
    assert [k["turn_id"] for k in kept] == ["D1:1"]


def test_records_are_oldest_first_and_deduplicated() -> None:
    later = {**SESSION, "id": "conv-1:session_2", "date": "2:00 pm on 9 June, 2023",
             "timestamp_ms": SESSION["timestamp_ms"] + 1}
    item = {"person": "Caroline", "category": "games", "turn_text": "t"}
    extracted = {
        "conv-1:session_2": {"items": [{**item, "item": "Chess"}, {**item, "item": "go"}]},
        "conv-1:session_1": {"items": [{**item, "item": "chess"}]},
    }
    records = build_records([SESSION, later], extracted)["conv-1"]
    assert len(records) == 1 and records[0]["items"] == 2
    lines = records[0]["content"].splitlines()[1:]
    assert lines[0].startswith("- 1:56 pm on 8 May, 2023: chess")
    assert lines[1].startswith("- 2:00 pm on 9 June, 2023: go")


def test_selection_is_limited_to_the_named_person() -> None:
    records = [
        {"id": "a", "person": "Melanie", "content": "games games games chess tournaments"},
        {"id": "b", "person": "Caroline", "content": "games chess"},
        {"id": "c", "person": "Caroline", "content": "pottery"},
    ]
    chosen = select_records("What games does Caroline play?", records, ("Caroline", "Melanie"))
    assert [r["id"] for r in chosen] == ["b"]


def test_records_go_after_the_protected_top_five_and_the_list_stays_at_100() -> None:
    raw = [{"id": str(i)} for i in range(100)]
    served = with_records(raw, [{"id": "x"}, {"id": "y"}])
    assert [r["id"] for r in served[:7]] == ["0", "1", "2", "3", "4", "x", "y"]
    assert len(served) == 100 and served[-1]["id"] == "97"
