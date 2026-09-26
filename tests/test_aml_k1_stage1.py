"""The K-1 Stage 1 arms render stored items exactly as registered.

Invariant: R and R2 are the stored items unchanged; K1 inserts role marks only into a window that
spans both roles and can be placed uniquely in its session, and keeps the date header in front.

Red proof, 2026-09-26, each against ``scripts/aml_k1_stage1.py`` with this file unchanged
(``PYTHONDONTWRITEBYTECODE=1``, so no stale bytecode can mask a mutant):
- ``marked`` returning ``rendered`` without the header:
  ``test_k1_marks_a_mixed_window_and_keeps_its_date_header`` failed on the ``startswith`` assertion.
- ``marked`` placing an unplaceable window at offset 0 instead of returning it:
  ``test_a_window_that_cannot_be_placed_is_left_as_stored`` failed on ``is item``.
- ``view`` marking R2 as well as K1: ``test_r_and_r2_are_the_stored_items`` failed on ``is items``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

SPEC = importlib.util.spec_from_file_location(
    "aml_k1_stage1", Path(__file__).parents[1] / "scripts" / "aml_k1_stage1.py"
)
assert SPEC and SPEC.loader
stage1 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(stage1)

QUESTION = {
    "haystack_session_ids": ["s1"],
    "haystack_sessions": [[
        {"role": "user", "content": "I moved to Lisbon last spring and I love it"},
        {"role": "assistant", "content": "That sounds wonderful, how is the weather there"},
    ]],
}
SESSIONS = stage1.sessions_of(QUESTION)
HEADER = "[2023-05-22 21:18 UTC] "


def test_k1_marks_a_mixed_window_and_keeps_its_date_header() -> None:
    item = {"id": "w1", "session_id": "s1", "content": HEADER + "and I love it That sounds wonderful,"}
    out = stage1.marked(item, SESSIONS)
    assert out["content"].startswith(HEADER)
    assert out["content"] == HEADER + "[user] and I love it [assistant] That sounds wonderful,"
    assert item["content"] == HEADER + "and I love it That sounds wonderful,"  # stored item untouched


def test_a_one_role_window_is_left_as_stored() -> None:
    item = {"id": "w2", "session_id": "s1", "content": HEADER + "moved to Lisbon last spring"}
    assert stage1.marked(item, SESSIONS) is item


def test_a_window_that_cannot_be_placed_is_left_as_stored() -> None:
    # Longer than the first message (10 words), so a window wrongly placed at offset 0 would span
    # both roles and be marked; a shorter one would pass even under that defect.
    item = {"id": "w3", "session_id": "s1",
            "content": HEADER + "these twelve words are nowhere in the session at all so skip"}
    assert stage1.marked(item, SESSIONS) is item


def test_an_item_from_an_unknown_session_is_left_as_stored() -> None:
    item = {"id": "w4", "session_id": "elsewhere", "content": HEADER + "and I love it That sounds wonderful,"}
    assert stage1.marked(item, SESSIONS) is item


def test_r_and_r2_are_the_stored_items() -> None:
    items = [{"id": "w1", "session_id": "s1", "content": HEADER + "and I love it That sounds wonderful,"}]
    assert stage1.view("R", items, SESSIONS) is items
    assert stage1.view("R2", items, SESSIONS) is items
    assert stage1.view("K1", items, SESSIONS)[0]["content"] != items[0]["content"]
