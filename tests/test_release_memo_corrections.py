"""Release regressions at the memo writer boundary.

The new behavior nodes are proved red on ac67de930 before repair.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from recall.lite.assistant_memos import ASSISTANT_AUTHOR, MemoFolder
from recall.lite.memo_requests import write_memo


@pytest.mark.parametrize("prepared", [False, True])
def test_a_release_baseline_correction_replays_after_upgrade(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, prepared: bool) -> None:
    """BUG-001. Red returns request_mismatch for the same baseline-serialized request."""
    from recall.lite import memo_requests

    folder = MemoFolder(tmp_path / "agent-a", index_root=tmp_path)
    first = folder.write("The workshop used to meet on Friday.")
    target = folder.path_of(first["memo_id"])
    assert target is not None
    expected = hashlib.sha256(target.read_bytes()).hexdigest()
    body = "---\nsupersedes: note-0001.md\n---\nThe workshop now meets on Monday.\n"
    original = memo_requests._create
    if prepared:
        def interrupted(*args: object, **kwargs: object) -> None:
            raise OSError("synthetic interrupted publication")
        monkeypatch.setattr(memo_requests, "_create", interrupted)
    try:
        baseline = write_memo(
            folder.folder, folder.ledger_path, body, kind="note", author=ASSISTANT_AUTHOR,
            supersedes="note-0001", request_id="baseline-correction", expected_revision=expected,
            include_receipt=True, target=(target, "agent-a/note-0001.md"),
        )
    except OSError:
        assert prepared
        baseline = {"memo_id": "note-0002"}
    monkeypatch.setattr(memo_requests, "_create", original)
    result = folder.write("The workshop now meets on Monday.", supersedes="note-0001",
                          request_id="baseline-correction", expected_revision=expected)
    assert result.get("memo_id") == baseline["memo_id"], result
    assert result["receipt"]["replayed"] is True
    assert (folder.folder / "note-0002.md").read_text(encoding="utf-8") == body
    assert result["receipt"]["revision"] == hashlib.sha256(body.encode()).hexdigest()
    changed = folder.write("Different caller intent.", supersedes="note-0001",
                           request_id="baseline-correction", expected_revision=expected)
    assert changed.get("conflict") == "request_mismatch", changed


def test_a_durable_third_folder_successor_blocks_another_publication(tmp_path: Path) -> None:
    """BUG-004. Red publishes a second correction when the third folder is unindexed."""
    target = MemoFolder(tmp_path / "agent-b", index_root=tmp_path)
    first_writer = MemoFolder(tmp_path / "agent-c", index_root=tmp_path)
    second_writer = MemoFolder(tmp_path / "agent-a", index_root=tmp_path)
    assert target.write("The workshop used to meet on Friday.") == {"memo_id": "note-0001"}
    assert first_writer.write("The workshop now meets on Monday.", supersedes="agent-b/note-0001") == {"memo_id": "note-0001"}
    result = second_writer.write("The workshop now meets on Tuesday.", supersedes="agent-b/note-0001")
    assert result.get("conflict") == "stale_target", result
    assert not list(second_writer.folder.glob("*.md"))
