"""Root correction preflight uses the shared supersession matcher."""
from __future__ import annotations

from pathlib import Path

from recall.lite.assistant_memos import MemoFolder


def test_an_ambiguous_explicit_root_target_is_refused(tmp_path: Path) -> None:
    """STAKES-002. Red publishes an edge that the shared resolver reads as ambiguous."""
    root = MemoFolder(tmp_path, index_root=tmp_path)
    sibling = MemoFolder(tmp_path / "agent-b", index_root=tmp_path)
    writer = MemoFolder(tmp_path / "agent-a", index_root=tmp_path)
    root.write("The workshop used to meet on Friday.")
    sibling.write("A distinct workshop meets on Thursday.")
    result = writer.write("The workshop now meets on Monday.", supersedes="./note-0001")
    assert result.get("conflict") == "ambiguous_target", result
    assert not list(writer.folder.glob("*.md"))


def test_a_unique_explicit_root_target_still_writes_and_replays(tmp_path: Path) -> None:
    """Compatibility control for an unambiguous root target and a completed replay."""
    root = MemoFolder(tmp_path, index_root=tmp_path)
    writer = MemoFolder(tmp_path / "agent-a", index_root=tmp_path)
    root.write("The workshop used to meet on Friday.")
    first = writer.write("The workshop now meets on Monday.", supersedes="./note-0001",
                         request_id="root-correction")
    second = writer.write("The workshop now meets on Monday.", supersedes="./note-0001",
                          request_id="root-correction")
    assert first.get("memo_id") == second.get("memo_id") == "note-0001"
    assert second["receipt"]["replayed"] is True
