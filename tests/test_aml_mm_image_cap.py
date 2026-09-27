"""MM-1/MM-3 amendment 4: the 30-image cap on Stage 2's answer packer.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-multimodal-scope-and-dates.md,
amendment 4. The cap keeps the longest ranked prefix of the returned items carrying at most the
limit's images; it never reorders, never drops a text item from inside the prefix, and does
nothing when the limit is not reached.

Red proof, 2026-09-25: each mutation applied to ``scripts/aml_mm_scope_stage2.py`` alone, the
named test run, then restored and run green.

- ``if total > limit`` changed to ``if total >= limit``:
  ``test_a_prefix_exactly_at_the_limit_is_kept_whole`` failed on its equality assertion.
- ``return items[:index], True`` changed to ``return [i for i in items if not image_count(i)], True``
  (keep every text item, drop images, which is not a ranked prefix):
  ``test_the_cap_keeps_a_ranked_prefix`` failed on its equality assertion.
- ``if row.get("error")`` in ``failed_keys`` changed to ``if not row.get("valid")``:
  ``test_only_rows_with_an_error_are_retried`` failed on its equality assertion.
"""

from __future__ import annotations

import json
from pathlib import Path

from scripts.aml_mm_scope_stage2 import cap_images, failed_keys, image_count

IMAGE = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}


def _item(name: str, images: int) -> dict:
    if not images:
        return {"session_id": name, "content": f"text of {name}"}
    return {"session_id": name, "content": [{"type": "text", "text": name}, *([IMAGE] * images)]}


def test_image_count_counts_image_parts_only() -> None:
    assert image_count(_item("a", 0)) == 0
    assert image_count(_item("b", 3)) == 3


def test_the_cap_keeps_a_ranked_prefix() -> None:
    items = [_item("a", 10), _item("t1", 0), _item("b", 15), _item("t2", 0), _item("c", 10), _item("t3", 0)]
    capped, cut = cap_images(items, 30)
    assert [i["session_id"] for i in capped] == ["a", "t1", "b", "t2"]
    assert cut


def test_a_prefix_exactly_at_the_limit_is_kept_whole() -> None:
    items = [_item("a", 20), _item("b", 10)]
    capped, cut = cap_images(items, 30)
    assert [i["session_id"] for i in capped] == ["a", "b"]
    assert not cut


def test_no_limit_changes_nothing() -> None:
    items = [_item("a", 40)]
    assert cap_images(items, None) == (items, False)


def test_only_rows_with_an_error_are_retried(tmp_path: Path) -> None:
    rows = [
        {"scenario": "s", "question_id": "1", "rotation": 0, "arm": "D", "valid": False, "error": "HTTP 400"},
        {"scenario": "s", "question_id": "2", "rotation": 0, "arm": "D", "valid": False, "error": ""},
        {"scenario": "s", "question_id": "3", "rotation": 1, "arm": "P", "valid": True},
    ]
    path = tmp_path / "s2.jsonl"
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
    assert failed_keys(path) == {("s", "1", 0, "D")}
