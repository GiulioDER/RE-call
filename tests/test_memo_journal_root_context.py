"""Store-owned context passed to the cross-folder journal erasure helper."""

import hashlib
import json
from pathlib import Path

import pytest

from recall.index import Indexer
from recall.lite import LiteStore
from recall.lite import assistant_memos
from recall.lite import memo_requests
from recall.lite.assistant_memos import MemoFolder
from recall.lite.memo_requests import load_ledger
from recall.types import Chunk
from recall_mcp.lifecycle import forget_memory
from tests.test_lite_calibration import ContentWordEmbedder


def _memo(store: LiteStore, root: Path) -> Path:
    folder = root / "agent-b"
    folder.mkdir(parents=True)
    (folder / ".recall-authors.json").write_text("{}", encoding="utf-8")
    path = folder / "note-0001.md"
    path.write_text("Synthetic retained target.", encoding="utf-8")
    store.upsert(
        [Chunk("target", str(path), "Synthetic retained target.", {"file": path.relative_to(root).as_posix()})],
        [[1.0, 0.0, 0.0, 0.0]],
    )
    return path


@pytest.mark.parametrize(
    ("retry", "multiple_roots"), [(False, False), (True, False), (False, True)],
    ids=["forget", "retry-after-rows-vanish", "multiple-roots"],
)
def test_generic_forget_supplies_recorded_root_context(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, retry: bool, multiple_roots: bool
) -> None:
    with LiteStore(tmp_path / "memory.db", dim=4) as store:
        root = tmp_path / "indexed"
        path = _memo(store, root)
        roots = [str(root)]
        if multiple_roots:
            store.upsert([Chunk("alias", str(path), "Synthetic retained target.", {"file": path.name})],
                         [[1.0, 0.0, 0.0, 0.0]])
            roots.append(str(path.parent))
        observed = []

        def erase(sources, *, index_roots=None):  # noqa: ANN001, ANN202
            observed.append((set(sources), index_roots))
            assert store.count() == 0, "negative control: cleanup did not follow deletion"
            if retry and len(observed) == 1:
                raise OSError("synthetic journal failure")
            return 0

        monkeypatch.setattr(assistant_memos, "forget_memo_requests", erase)
        result = forget_memory(store, [str(path)])
        assert result.chunks_removed == (2 if multiple_roots else 1)
        if retry:
            assert "WARNING" in result.message
            result = forget_memory(store, [str(path)])
            assert result.chunks_removed == 0
        assert observed[-1] == ({str(path)}, {str(path): sorted(roots)}), (
            "journal cleanup lost the store-owned root context"
        )


def test_prune_supplies_the_indexers_key_root(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with LiteStore(tmp_path / "memory.db", dim=4) as store:
        root = tmp_path / "indexed"
        path = _memo(store, root)
        path.unlink()
        observed = []

        def erase(sources, *, index_roots=None):  # noqa: ANN001, ANN202
            observed.append((set(sources), index_roots))
            assert store.count() == 1, "negative control: prune deleted rows before journal cleanup"
            return 0

        monkeypatch.setattr(assistant_memos, "forget_memo_requests", erase)
        Indexer(store, ContentWordEmbedder()).index_path(path.parent, key_root=root)
        assert store.count() == 0, "negative control: the missing source was not pruned"
        assert observed == [({str(path)}, {str(path): [str(root)]})], (
            "prune lost the configured indexing key root"
        )


def test_an_empty_legacy_receipt_does_not_discard_root_context(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with LiteStore(tmp_path / "memory.db", dim=4) as store:
        root = tmp_path / "indexed"
        path = _memo(store, root)
        identity = json.dumps([store.tenant, str(path)], ensure_ascii=False)
        store.append_audit_event(
            "forget_memo_journal", {}, source_uri=str(path),
            event_id="forget_memo_journal_" + hashlib.sha256(identity.encode()).hexdigest(),
        )
        observed = []

        def erase(sources, *, index_roots=None):  # noqa: ANN001, ANN202
            observed.append(index_roots)
            return 0

        monkeypatch.setattr(assistant_memos, "forget_memo_requests", erase)
        forget_memory(store, [str(path)])
        assert observed == [{str(path): [str(root)]}], "a legacy receipt discarded newly verified roots"


def test_reindexing_retains_each_verified_root_for_retry(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with LiteStore(tmp_path / "memory.db", dim=4) as store:
        root = tmp_path / "indexed"
        path = _memo(store, root)
        observed = []

        def erase(sources, *, index_roots=None):  # noqa: ANN001, ANN202
            observed.append(index_roots)
            return 0

        monkeypatch.setattr(assistant_memos, "forget_memo_requests", erase)
        forget_memory(store, [str(path)])
        store.upsert([Chunk("new-index", str(path), "new source revision", {"file": path.name})],
                     [[1.0, 0.0, 0.0, 0.0]])
        forget_memory(store, [str(path)])
        assert observed[-1] == {str(path): sorted([str(root), str(path.parent)])}, (
            "a prior receipt discarded the root of a later indexing"
        )


def test_a_source_file_is_not_recorded_as_its_own_index_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy file='.' metadata still permits the memo's local journal cleanup."""
    folder = MemoFolder(tmp_path / "memos")
    monkeypatch.setattr(memo_requests, "_NO_LINK_FOLDERS", {str(folder.folder)})
    first = folder.write("Synthetic legacy memo.", request_id="original")
    path = folder.path_of(first["memo_id"])
    with LiteStore(tmp_path / "memory.db", dim=4) as store:
        store.upsert(
            [Chunk("target", str(path), path.read_text(encoding="utf-8"), {"file": "."})],
            [[1.0, 0.0, 0.0, 0.0]],
        )
        forget_memory(store, [str(path)])
        assert store.count() == 0
    entry = load_ledger(folder.ledger_path)["requests"]["original"]
    assert entry.get("erased") and "identity" not in entry, "the source file was used as an index root"


@pytest.mark.parametrize("operation", ["forget", "retry", "prune"])
def test_store_erasure_scrubs_cross_folder_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, operation: str
) -> None:
    """Real consumers use verified roots, preserve namesakes and replay content after erasure."""
    root = tmp_path / "indexed"
    target = MemoFolder(root / "agent-b", index_root=root)
    writer = MemoFolder(root / "agent-a", index_root=root)
    namesake = MemoFolder(root / "agent-c", index_root=root)
    foreign = MemoFolder(tmp_path / "foreign", index_root=tmp_path / "foreign")
    # Exercise the supported exclusive-create transport on a filesystem without hard links.
    monkeypatch.setattr(
        memo_requests, "_NO_LINK_FOLDERS",
        memo_requests._NO_LINK_FOLDERS | {str(folder.folder) for folder in (target, writer, namesake, foreign)},
    )
    first = target.write("The workshop met on Friday.", request_id="original")
    reference = f"agent-b/{first['memo_id']}"
    corrected = writer.write(
        "The workshop meets on Monday.", supersedes=reference, request_id="correction",
        expected_revision=first["receipt"]["revision"],
    )
    assert "memo_id" in corrected, corrected
    namesake.write("An unrelated namesake.", request_id="namesake")
    foreign.write("An unrelated source.", request_id="foreign")
    kept = {folder.ledger_path: folder.ledger_path.read_bytes() for folder in (namesake, foreign)}
    path = target.path_of(first["memo_id"])
    assert "identity" in load_ledger(writer.ledger_path)["requests"]["correction"], "negative control"
    with LiteStore(tmp_path / "memory.db", dim=4) as store:
        store.upsert(
            [Chunk("target", str(path), path.read_text(encoding="utf-8"), {"file": path.relative_to(root).as_posix()})],
            [[1.0, 0.0, 0.0, 0.0]],
        )
        if operation == "prune":
            path.unlink()
            Indexer(store, ContentWordEmbedder()).index_path(target.folder, key_root=root)
        else:
            if operation == "retry":
                def fail(sources, *, index_roots=None):  # noqa: ANN001, ANN202
                    raise OSError("synthetic journal failure")

                with monkeypatch.context() as patch:
                    patch.setattr(assistant_memos, "forget_memo_requests", fail)
                    result = forget_memory(store, [str(path)])
                assert store.count() == 0 and "WARNING" in result.message
            forget_memory(store, [str(path)])
        assert store.count() == 0
    entry = load_ledger(writer.ledger_path)["requests"]["correction"]
    assert "identity" not in entry and entry.get("target_forgotten"), "a sibling retained the erased target identity"
    assert all(ledger.read_bytes() == raw for ledger, raw in kept.items()), "an unrelated journal changed"
    replay = writer.write("The workshop meets on Monday.", supersedes=reference, request_id="correction")
    assert replay.get("memo_id") == corrected["memo_id"] and replay.get("receipt", {}).get("replayed"), replay
