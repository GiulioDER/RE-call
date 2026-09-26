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


def test_the_longmemeval_pipeline_is_loaded_not_the_locomo_one(tmp_path: Path, monkeypatch) -> None:
    """Found before Stage 1 ran: the first harness reused ``load_aml_pipeline``, which loads AML's
    LoCoMo pipeline, so every answer would have used the wrong template. Red proof: ``LME_PIPELINE``
    pointed back at ``data/locomo-refined/pipeline.py`` failed on ``== "longmemeval-s"``."""
    for name in ("longmemeval-s", "locomo-refined"):
        (tmp_path / "data" / name).mkdir(parents=True)
        (tmp_path / "data" / name / "pipeline.py").write_text(f"WHICH = {name!r}\n", encoding="utf-8")
    monkeypatch.setattr(
        stage1.subprocess, "run", lambda *a, **k: type("R", (), {"stdout": stage1.AML_COMMIT + "\n"})()
    )
    assert stage1.load_lme_pipeline(tmp_path).WHICH == "longmemeval-s"


def test_r_and_r2_are_the_stored_items() -> None:
    items = [{"id": "w1", "session_id": "s1", "content": HEADER + "and I love it That sounds wonderful,"}]
    assert stage1.view("R", items, SESSIONS) is items
    assert stage1.view("R2", items, SESSIONS) is items
    assert stage1.view("K1", items, SESSIONS)[0]["content"] != items[0]["content"]


def test_t1_resolves_a_relative_date_against_the_items_own_created_at() -> None:
    """T-1 amendment 6. Red proof: ``resolved`` anchoring every item on ``datetime.now`` instead of
    its ``created_at`` failed on ``"2023-05-21" in``."""
    dated = {"id": "t1", "session_id": "s1", "created_at": "2023-05-22T21:18:00Z", "kind": "raw",
             "score": 0.4, "source": "aml://session/x", "content": HEADER + "I went hiking yesterday"}
    plain = {**dated, "id": "t2", "content": HEADER + "no relative words here"}
    out = stage1.view("T1", [dated, plain], SESSIONS)
    assert "2023-05-21" in out[0]["content"] and out[0]["id"] == "t1"
    assert out[1] is plain
    assert dated["content"] == HEADER + "I went hiking yesterday"  # stored item untouched


def test_score_refuses_an_arm_answered_twice_across_files(tmp_path: Path) -> None:
    """Red proof: the duplicate check removed from ``score`` let the second file's label win
    silently, and the test failed on ``DID NOT RAISE SystemExit``."""
    import json

    import pytest

    row = {"id": "q1", "arm": "R", "label": "CORRECT", "type": "temporal-reasoning"}
    first, second = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    first.write_text(json.dumps(row) + "\n", encoding="utf-8")
    second.write_text(json.dumps({**row, "label": "WRONG"}) + "\n", encoding="utf-8")
    args = type("A", (), {"answers": [first, second], "out": tmp_path / "s.json"})()
    with pytest.raises(SystemExit, match="twice"):
        stage1.score(args)
