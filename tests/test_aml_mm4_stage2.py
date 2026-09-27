"""MM-4 Stage 2 answers S, S2, M4r and M4s, with M4s rendered from the census's sidecar texts.

Invariants: every S row yields S and S2 and every M4r row yields M4r and M4s; a parent image
message's sidecar texts come back in image order (``_imgtext_2`` before ``_imgtext_10``) and are
kept per scenario; M4s is the production ``shown_items`` render of those texts.

Red proof, 2026-09-27, each against ``scripts/aml_mm_scope_stage2.py`` with this file unchanged
(``PYTHONDONTWRITEBYTECODE=1``):
- ``plan_mm4`` without the S2 job: ``test_every_arm_is_planned`` fails at the arm list.
- ``load_sidecars`` sorting by the text, not the numeric image index:
  ``test_sidecars_come_back_in_image_order`` fails at the list equality.
- ``load_sidecars`` keyed by parent alone: ``test_sidecars_stay_per_scenario`` fails at the
  first equality.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from recall_aml.image_text import SHOWN_LABEL, shown_items
from recall_aml.models import SearchItem

SPEC = importlib.util.spec_from_file_location(
    "aml_mm_scope_stage2", Path(__file__).parents[1] / "scripts" / "aml_mm_scope_stage2.py"
)
assert SPEC and SPEC.loader
stage2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stage2
SPEC.loader.exec_module(stage2)


def _census(tmp_path: Path, sidecars: dict) -> Path:
    path = tmp_path / "census.json"
    path.write_text(json.dumps({"sidecars": sidecars}), encoding="utf-8")
    return path


def test_every_arm_is_planned(tmp_path: Path) -> None:
    results = tmp_path / "s1.jsonl"
    rows = [{"scenario": "Brand", "question_id": "q1", "rotation": 0, "arm": arm, "items": []}
            for arm in ("S", "M4r")]
    results.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert [arm for _, arm in stage2.plan_mm4(results)] == ["S", "S2", "M4r", "M4s"]


def test_sidecars_come_back_in_image_order(tmp_path: Path) -> None:
    sidecars = {"Brand": [{"id": f"raw_p_imgtext_{i}", "primary_id": "raw_p", "text": f"image {i}"}
                          for i in (10, 2, 0)]}
    loaded = stage2.load_sidecars(_census(tmp_path, sidecars))
    assert loaded[("Brand", "raw_p")] == ["image 0", "image 2", "image 10"]


def test_sidecars_stay_per_scenario(tmp_path: Path) -> None:
    sidecars = {"Brand": [{"id": "raw_p_imgtext_0", "primary_id": "raw_p", "text": "a logo"}],
                "Card": [{"id": "raw_p_imgtext_0", "primary_id": "raw_p", "text": "a card"}]}
    loaded = stage2.load_sidecars(_census(tmp_path, sidecars))
    assert loaded[("Brand", "raw_p")] == ["a logo"]
    assert loaded[("Card", "raw_p")] == ["a card"]


def test_m4s_appends_the_sidecar_text_to_image_items_only() -> None:
    image = SearchItem.model_validate({"id": "raw_p", "content": [{"type": "text", "text": "photo"}],
                                       "source": "s", "session_id": "s1", "kind": "raw", "score": 0.5})
    text = SearchItem.model_validate({"id": "raw_t", "content": "plain words", "source": "s",
                                      "session_id": "s1", "kind": "raw", "score": 0.4})
    shown = shown_items([image, text], {"raw_p": ["a red logo"]})
    assert shown[1] == text
    assert shown[0].content[-1].text == f"{SHOWN_LABEL}\na red logo"
