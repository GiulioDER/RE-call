"""Cross-folder erasure journals, through the file and generic erasure consumers.

STAKES-001 red proof uses the pinned pre-fix implementation ac67de930. All behavior
nodes below assert removal of a target-derived identity, rollback, or the root boundary.
The compatibility adapter omits only the new root keyword on that implementation,
so the red run reaches existing cleanup instead of failing on its signature.
The assistant consumer cases have separate continuation proof after helper v2 integration.
"""

from __future__ import annotations

import errno
import inspect
import shutil
from pathlib import Path
from typing import Any

import pytest

from recall.lite import memo_requests
from recall.lite.assistant_memos import MemoFolder, forget_memo_requests
from recall.lite.memo_requests import load_ledger, revision


@pytest.fixture(autouse=True)
def _exclusive_create_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the supported exclusive-create fallback where the sandbox denies hard links."""
    def unsupported(*args: Any, **kwargs: Any) -> None:
        raise OSError(errno.ENOTSUP, "synthetic filesystem without hard links")

    monkeypatch.setattr(memo_requests.os, "link", unsupported)
    monkeypatch.setattr(memo_requests, "_NO_LINK_FOLDERS", set())


def _correction(root: Path, target_folder: str = "agent-b", writer_folder: str = "agent-a") -> tuple[MemoFolder, MemoFolder, dict[str, Any], dict[str, Any]]:
    target = MemoFolder(root / target_folder, index_root=root)
    writer = MemoFolder(root / writer_folder, index_root=root)
    first = target.write("The workshop met on Friday.", request_id="original")
    target_id = first["memo_id"]
    intent = {
        "supersedes": f"{target_folder}/../{target_folder}/{target_id}",
        "request_id": "correction",
    }
    result = writer.write(
        "The workshop meets on Monday.", expected_revision=revision(target.path_of(target_id)),
        **intent,
    )
    assert "memo_id" in result, result
    return target, writer, intent, result


def _forget(sources: list[str], roots: dict[str, Any]) -> int:
    if "index_roots" in inspect.signature(forget_memo_requests).parameters:
        if "Sequence" not in str(inspect.signature(forget_memo_requests)):
            roots = {source: value[0] if isinstance(value, list) else value for source, value in roots.items()}
        return forget_memo_requests(sources, index_roots=roots)
    return forget_memo_requests(sources)


def test_delete_scrubs_sibling_revision_and_replays_without_target_bytes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ac67de930 red: MemoFolder.delete leaves identity in the sibling journal."""
    target, writer, intent, written = _correction(tmp_path)
    path = target.path_of("note-0001")
    before = load_ledger(writer.ledger_path)["requests"]["correction"]
    target.delete("note-0001")
    after = load_ledger(writer.ledger_path)["requests"]["correction"]
    assert "identity" not in after and after.get("target_forgotten"), "sibling target identity survived deletion"
    assert after["content"] == before["content"]
    assert target.authors()["note-0001"] == "assistant"
    real_read = Path.read_bytes

    def read(candidate: Path) -> bytes:
        assert candidate != path, "completed replay read the erased target"
        return real_read(candidate)

    monkeypatch.setattr(Path, "read_bytes", read)
    replay = writer.write("The workshop meets on Monday.", **intent)
    assert replay["memo_id"] == written["memo_id"] and replay["receipt"]["replayed"], replay
    assert replay["receipt"]["revision"] == written["receipt"]["revision"]


def test_generic_cleanup_uses_root_and_preserves_unrelated_namesakes(tmp_path: Path) -> None:
    """Red at ac67de930 and with root list ignored: only one entry changes instead of two."""
    root = tmp_path / "shared"
    target, writer, _, _ = _correction(root)
    namesake, other_writer, _, _ = _correction(root, "agent-c", "agent-d")
    _, foreign_writer, _, _ = _correction(tmp_path / "foreign")
    kept = {folder.ledger_path: folder.ledger_path.read_bytes() for folder in (namesake, other_writer, foreign_writer)}
    path = target.path_of("note-0001")
    changed = _forget([str(path)], {str(path): root})
    entry = load_ledger(writer.ledger_path)["requests"]["correction"]
    assert changed == 2 and "identity" not in entry, "generic cleanup missed a same-root correction"
    assert load_ledger(target.ledger_path)["requests"]["original"]["erased"]
    assert all(path.read_bytes() == raw for path, raw in kept.items())


def test_later_journal_save_failure_restores_earlier_journals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ac67de930 red: cleanup never reaches the later sibling save, so no failure is raised."""
    target, writer, _, _ = _correction(tmp_path, "a-target", "b-writer")
    last = MemoFolder(tmp_path / "c-writer", index_root=tmp_path)
    # Persist a pre-upgrade second correction, which new publications now refuse.
    shutil.copytree(writer.folder, last.folder)
    before = {folder.ledger_path: folder.ledger_path.read_bytes() for folder in (target, writer, last)}
    real_save = memo_requests._save
    failed = False

    def save(path: Path, ledger: dict[str, Any]) -> None:
        nonlocal failed
        if path == last.ledger_path and not failed:
            failed = True
            raise OSError("synthetic journal save failure")
        real_save(path, ledger)

    monkeypatch.setattr(memo_requests, "_save", save)
    with pytest.raises(OSError, match="synthetic journal"):
        target.delete("note-0001")
    assert all(path.read_bytes() == raw for path, raw in before.items()), "earlier journal cancellation survived a failed batch"
    assert target.path_of("note-0001").is_file()


def test_unlink_failure_restores_every_changed_journal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """ac67de930 red: the sibling identity is still present when unlink is attempted."""
    target, writer, _, _ = _correction(tmp_path)
    path = target.path_of("note-0001")
    before = {folder.ledger_path: folder.ledger_path.read_bytes() for folder in (target, writer)}
    real_unlink = Path.unlink
    saw_scrubbed = False

    def unlink(candidate: Path, *args: Any, **kwargs: Any) -> None:
        nonlocal saw_scrubbed
        if candidate == path:
            saw_scrubbed = "identity" not in load_ledger(writer.ledger_path)["requests"]["correction"]
            raise OSError("synthetic unlink failure")
        real_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    with pytest.raises(OSError, match="synthetic unlink"):
        target.delete("note-0001")
    assert saw_scrubbed, "unlink began before sibling request hashes were scrubbed"
    assert all(path.read_bytes() == raw for path, raw in before.items()), "unlink failure changed a journal"
    assert path.is_file()


def test_root_map_refuses_a_source_outside_its_boundary(tmp_path: Path) -> None:
    """Red at ac67de930 and with root list ignored: the boundary refusal assertion fails."""
    target, _, _, _ = _correction(tmp_path / "owned")
    path = target.path_of("note-0001")
    before = target.ledger_path.read_bytes()
    with pytest.raises(ValueError, match="root"):
        _forget([str(path)], {str(path): tmp_path / "elsewhere"})
    assert target.ledger_path.read_bytes() == before


def test_sources_only_cleanup_keeps_local_compatibility(tmp_path: Path) -> None:
    """Legacy control, no red claim. A sources-only caller still cancels its local memo."""
    folder = MemoFolder(tmp_path)
    folder.write("A local fact.", request_id="local")
    assert forget_memo_requests([str(folder.path_of("note-0001"))]) == 1
    assert load_ledger(folder.ledger_path)["requests"]["local"]["erased"]


def test_assistant_forget_scrubs_same_root_correction_journals(tmp_path: Path) -> None:
    """Helper consumer RED: the old assistant caller left the sibling target identity intact."""
    from recall.lite import LiteStore
    from recall.types import Chunk
    from recall_mcp.assistant import AssistantMemos
    from tests.test_lite_calibration import ContentWordEmbedder

    target, writer, _, _ = _correction(tmp_path)
    memos = AssistantMemos(target.folder, index_root=tmp_path)
    path = memos.path_of("note-0001")
    assert path is not None
    entry = load_ledger(writer.ledger_path)["requests"]["correction"]
    assert "identity" in entry, "negative control: the correction never recorded its target identity"
    embedder = ContentWordEmbedder()
    text = path.read_text(encoding="utf-8")
    with LiteStore(tmp_path / "memory.db", dim=embedder.dim) as store:
        store.upsert([Chunk("target", str(path), text, {"file": "agent-b/note-0001.md"})], embedder.embed([text]))
        result = memos.forget(store, embedder, "note-0001")
        assert result["deleted"] == 1 and store.count() == 0
    after = load_ledger(writer.ledger_path)["requests"]["correction"]
    assert "identity" not in after and after.get("target_forgotten"), "assistant forget left the sibling target identity"
    assert not path.exists()


def test_assistant_forget_unlink_failure_restores_the_cancellation_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Helper consumer RED: an old-format cancellation batch masks unlink failure during rollback."""
    from recall.lite import LiteStore
    from recall.types import Chunk
    from recall_mcp.assistant import AssistantMemos
    from tests.test_lite_calibration import ContentWordEmbedder

    target, writer, _, _ = _correction(tmp_path)
    memos = AssistantMemos(target.folder, index_root=tmp_path)
    path = memos.path_of("note-0001")
    assert path is not None
    before = {folder.ledger_path: folder.ledger_path.read_bytes() for folder in (target, writer)}
    real_unlink = Path.unlink

    def unlink(candidate: Path, *args: Any, **kwargs: Any) -> None:
        if candidate == path:
            raise OSError("synthetic assistant unlink failure")
        real_unlink(candidate, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", unlink)
    embedder = ContentWordEmbedder()
    text = path.read_text(encoding="utf-8")
    caught: Exception | None = None
    with LiteStore(tmp_path / "memory.db", dim=embedder.dim) as store:
        store.upsert([Chunk("target", str(path), text, {"file": "agent-b/note-0001.md"})], embedder.embed([text]))
        try:
            memos.forget(store, embedder, "note-0001")
        except Exception as exc:  # The regression asserts the incidental rollback exception, rather than erroring on it.
            caught = exc
    assert isinstance(caught, OSError) and "synthetic assistant unlink" in str(caught), "rollback masked the unlink failure"
    assert path.is_file()
    assert all(ledger.read_bytes() == raw for ledger, raw in before.items()), "assistant unlink failure changed a journal"


def test_one_source_scrubs_corrections_from_every_verified_root(tmp_path: Path) -> None:
    """Red with first-root-only and ignored-roots mutations: a correction identity survives."""
    outer = tmp_path / "outer"
    inner = outer / "shared"
    target = MemoFolder(inner / "agent-b", index_root=inner)
    target.write("The workshop met on Friday.", request_id="original")
    path = target.path_of("note-0001")
    expected = revision(path)
    first = MemoFolder(outer / "agent-c", index_root=outer)
    second = MemoFolder(inner / "agent-a", index_root=inner)
    first.write(
        "The workshop meets on Monday.", supersedes="shared/agent-b/note-0001",
        request_id="outer", expected_revision=expected,
    )
    second.write(
        "The workshop meets on Tuesday.", supersedes="agent-b/note-0001",
        request_id="inner", expected_revision=expected,
    )
    changed = _forget([str(path)], {str(path): [outer, inner]})
    entries = [
        load_ledger(first.ledger_path)["requests"]["outer"],
        load_ledger(second.ledger_path)["requests"]["inner"],
    ]
    assert changed == 3 and all("identity" not in entry for entry in entries), "a verified root's correction retained the target identity"

@pytest.mark.parametrize("operation", ["delete", "forget"])
def test_a_linked_authorship_ledger_refuses_before_erasure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str,
) -> None:
    """L5.5 STAKES-001 regression. The root wrapper first failed the refusal assertion.

    The baseline's direct ledger reader refused links. The initial root-aware repair
    classified them as non-assistant sources, allowing deletion without journal cleanup.
    """
    from types import SimpleNamespace

    from recall.lite import LiteStore
    from recall_mcp import lifecycle
    from recall_mcp.assistant import AssistantMemos

    memos = AssistantMemos(tmp_path / "memos", index_root=tmp_path)
    memos.write("A synthetic workshop schedule.", request_id="original")
    path = memos.path_of("note-0001")
    before = memos.ledger_path.read_bytes()
    is_link = Path.is_symlink
    monkeypatch.setattr(Path, "is_symlink", lambda candidate: candidate == memos.ledger_path or is_link(candidate))
    calls: list[str] = []
    monkeypatch.setattr(lifecycle, "forget_memory", lambda *args, **kwargs: (
        calls.append("erased") or SimpleNamespace(facts_retracted=0, facts_redacted=0, sources_not_found=[])
    ))
    monkeypatch.setattr(memos, "refresh", lambda *args, **kwargs: None)
    store = LiteStore(tmp_path / "memory.db", dim=8)
    try:
        with pytest.raises(ValueError, match="symbolic link"):
            if operation == "delete":
                memos.delete("note-0001")
            else:
                memos.forget(store, None, "note-0001")
        assert path.is_file()
        assert memos.ledger_path.read_bytes() == before
        assert calls == []
    finally:
        store.close()
