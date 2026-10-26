"""Deterministic work-budget coverage for PR988 CCA PERF-001."""

from __future__ import annotations

import json
from collections.abc import Iterator, Set
from datetime import UTC, datetime
from pathlib import Path

from recall.records_files import load_memo_records, match_memo_records
from recall_hooks import auditor_brief


class _CountedNames(Set[str]):
    """An immutable set with observable enumeration and ordinary constant-time membership.

    Counting element visits measures rebuilding work without prescribing whether the renderer
    reuses the original set or builds one reusable membership index.
    """

    def __init__(self, names: frozenset[str]) -> None:
        self._names = names
        self.element_visits = 0

    def __contains__(self, value: object) -> bool:
        return value in self._names

    def __len__(self) -> int:
        return len(self._names)

    def __iter__(self) -> Iterator[str]:
        for name in self._names:
            self.element_visits += 1
            yield name


def test_retracted_candidates_share_one_filename_membership_index(tmp_path: Path) -> None:
    """PERF-001: rendering may enumerate the filename set once, regardless of hidden candidates.

    Node: tests/test_auditor_brief_cca_perf.py::test_retracted_candidates_share_one_filename_membership_index.
    Baseline: PR988 HEAD 00a124ba211ea79b72427e51f61ed94ac62ec950. Production target:
    recall_hooks.auditor_brief.render, the per-candidate set(unnamed_files) allocation.
    Expected RED: the final element-visits assertion fails because 32 retracted lessons and one
    visible bug each enumerate all 64 filenames. Retraction and filename-suppression assertions
    must pass first. The fixture uses admitted memos and the production matcher; no timing claim
    or new production symbol is involved. The primary records actual the testbench RED/GREEN results.
    """
    fixtures = Path(__file__).parent / "fixtures" / "audit_memory" / "expected_records.json"
    golden = list(json.loads(fixtures.read_text(encoding="utf-8")).values())
    lesson = next(record for record in golden if record["kind"] == "lesson")
    bug = next(record for record in golden if record["kind"] == "bug")
    false_positive = next(record for record in golden if record["kind"] == "false_positive")
    memos = tmp_path / "memos"
    memos.mkdir()

    def write(stem: str, record: dict, supersedes: str | None = None) -> None:
        claim = f"supersedes: {supersedes}\n" if supersedes is not None else ""
        (memos / f"{stem}.md").write_text(
            "---\nvalid_from: 2026-08-04\n" + claim
            + "recall_record: " + json.dumps(record, ensure_ascii=True) + "\n---\n\nbody\n",
            encoding="utf-8",
        )

    hidden_rule = "RETRACTED-CONTEXT-MUST-NOT-APPEAR"
    for index in range(32):
        stem = f"lesson-{index:03}"
        ident = f"example:lesson:{index:03}"
        write(stem, {**lesson, "id": ident, "rule": hidden_rule})
        write(f"retraction-{index:03}", {
            "v": 1, "id": f"{ident}:retracted", "kind": "lesson_retraction",
            "project": lesson["project"], "repo": lesson["repo"], "date": "2026-08-04",
            "target_id": ident, "reason": "Invented retired lesson.", "retracted_by": "owner",
        }, supersedes=stem)
    for index in range(64):
        write(
            f"false-positive-{index:03}", {**false_positive, "id": f"example:fp:{index:03}"},
            supersedes="visible-bug" if index == 0 else None,
        )
    # The first false-positive successor stays unnamed while its predecessor remains visible.
    write("visible-bug", {**bug, "summary": "VISIBLE-CONTROL"})

    records = load_memo_records(memos)
    assert not records.skipped, "the work-budget fixture must contain admitted records"
    now = datetime(2026, 10, 26, tzinfo=UTC)
    entries = match_memo_records(
        records, repo=lesson["repo"], anchors=lesson["anchors"][:1],
        kinds=["bug", "lesson"], now=now,
    )
    retractions = {memo.file for memo in records if memo.record["kind"] == "lesson_retraction"}
    names = frozenset(memo.file for memo in records if memo.record["kind"] == "false_positive")
    assert len(entries) == 33
    assert sum(entry["superseded_by"] in retractions for entry in entries) == 32
    assert len(names) == 64
    counted = _CountedNames(names)

    block = auditor_brief.render(
        entries, now=now, retraction_files=retractions,
        unnamed_files=counted,  # type: ignore[arg-type]  # immutable set proxy counts enumeration
    )

    assert "VISIBLE-CONTROL" in block, "the visible predecessor control disappeared"
    assert hidden_rule not in block, "a retracted lesson reached the auditor"
    assert "(superseded)" in block, "the false-positive successor lost its suppressed-name note"
    assert not any(name in block for name in names), "a false-positive filename reached the auditor"
    assert counted.element_visits <= len(names), (
        f"render enumerated {counted.element_visits} filename elements; "
        f"one reusable membership index needs at most {len(names)}"
    )
