"""Correction invariants exercised through the existing write and MCP consumers.

Red proof against b7f5d9c032791477f046ebc374eb77a2233b052f: the original nineteen cases plus
test_assistant_surface.py::test_the_published_tools_are_the_contract and
test_assistant_surface.py::test_remember_then_search_answers_in_the_compact_view produced
20 intended assertion failures and one passing legacy control, zero errors. The five existing
production files were restored byte for byte from a persistent journal before the green run.
New-node prefix: tests/test_memo_corrections.py::. Baseline failure targets:
receipt/index/readiness nodes: missing receipt; idempotency/recovery nodes: duplicate memo or
missing durable reservation; revision/stale/concurrent nodes: missing conflict; MCP node:
missing optional parameters. The two existing contract nodes reject the old tool schema and
missing revision projection respectively. The compatibility adapter discards
only parameters absent from the baseline signature: old code reaches the real write path and
fails behavioral assertions, never collection or unexpected-keyword errors.

Audit regressions BUG-001 and DAT-002: seven cases failed at intended assertions against
139b138b6152c09a2c18082c8b3806a9bf3ec68a, with zero setup or collection errors. They exercise
explicit forget after interrupted publication, cancellation-save failure, real search revisions
for CRLF/BOM/NUL targets, and receipt parity with the actual stored hash for BOM/NUL text.
The two raw-byte replay controls also failed their intended memo_changed assertions with the
plausible mutation replacing the raw replay digest by revision(path); a persistent journal
restored the fixed production bytes and verified their SHA-256 before the final green run.
"""
from __future__ import annotations

import errno
import hashlib
import inspect
import sys
import time
import json
from concurrent.futures import ThreadPoolExecutor

import pytest

from recall.lite import LiteStore
from recall.lite.calibration import auto_calibrate
from recall_mcp.assistant import AssistantMemos
from tests.test_lite_calibration import ContentWordEmbedder, _store
from tests.test_assistant_surface import _published, _serve_both, envs as base_envs


@pytest.fixture
def serving(tmp_path):
    store = _store(tmp_path, 48)
    assert auto_calibrate(store, ContentWordEmbedder()).artifact.certified
    yield store, AssistantMemos(tmp_path / "memos", index_root=tmp_path)
    store.close()


def remember(store, memos, text="The user drinks tea.", **kwargs):
    offered = inspect.signature(memos.remember).parameters
    return memos.remember(store, ContentWordEmbedder(), text,
                          **{k: v for k, v in kwargs.items() if k in offered})


def initial(store, memos):
    result = remember(store, memos)
    path = memos.path_of(result["memo_id"])
    return result["memo_id"], hashlib.sha256(path.read_bytes()).hexdigest()


def test_legacy_reply_is_unchanged(serving):
    """Compatibility control: existing clients still receive exactly the old success shape."""
    store, memos = serving
    assert remember(store, memos) == {"memo_id": "note-0001"}


def test_write_receipt_distinguishes_persistence_indexing_and_readiness(serving):
    """Red: baseline b7f5d9c (no request journal, MemoFolder.write allocating ids itself) failed `write receipt missing`: the reply had no receipt.
    Target: write_memo's receipt and AssistantMemos.remember's index and readiness checks.
    """
    store, memos = serving
    result = remember(store, memos, include_receipt=True)
    receipt = result.get("receipt", {})
    assert receipt.get("write_state") == "written", "write receipt missing"
    assert receipt.get("index_state") == "indexed", "written memo was not verified in the index"
    assert receipt.get("calibration_status") == "certified"
    assert receipt.get("retrieval_ready") is True
    assert receipt["revision"] == hashlib.sha256(memos.path_of(result["memo_id"]).read_bytes()).hexdigest()


def test_idempotent_retry_uses_the_original_memo(serving):
    """Red: baseline b7f5d9c (no request journal, MemoFolder.write allocating ids itself) failed `retry allocated another memo` (note-0002 for note-0001).
    Target: the request journal lookup in write_memo.
    """
    store, memos = serving
    first = remember(store, memos, request_id="write-1")
    second = remember(store, memos, request_id="write-1")
    assert second["memo_id"] == first["memo_id"], "retry allocated another memo"
    assert second["receipt"]["replayed"] is True
    assert len(list(memos.folder.glob("*.md"))) == 1


def test_legacy_writes_preserve_request_identities(serving):
    """Red: baseline b7f5d9c (no request journal, MemoFolder.write allocating ids itself) failed `legacy write discarded the request journal` (note-0003 for
    note-0001). Target: write_memo keeping `requests` when a legacy write saves the ledger.
    """
    store, memos = serving
    first = remember(store, memos, request_id="write-1")
    remember(store, memos, "Another independent fact.")
    retry = remember(store, memos, request_id="write-1")
    assert retry["memo_id"] == first["memo_id"], "legacy write discarded the request journal"
    assert len(list(memos.folder.glob("*.md"))) == 2


@pytest.mark.parametrize("stage", ["before_publish", "after_publish"])
def test_prepared_request_survives_interrupted_publication(serving, monkeypatch, stage):
    """Red: baseline b7f5d9c (no request journal, MemoFolder.write allocating ids itself) failed `interrupted write lost its durable identity` for both stages.
    Target: the prepared reservation that _publish saves before _create.
    """
    from recall.lite import memo_requests
    store, memos = serving
    create, save = memo_requests._create, memo_requests._save
    def broken_create(*args):
        raise OSError("injected interruption before memo publication")
    def broken_save(path, ledger):
        if ledger.get("requests", {}).get("interrupted", {}).get("state") == "written":
            raise OSError("injected interruption after memo publication")
        return save(path, ledger)
    monkeypatch.setattr(memo_requests, "_create", broken_create if stage == "before_publish" else create)
    monkeypatch.setattr(memo_requests, "_save", broken_save if stage == "after_publish" else save)
    try:
        remember(store, memos, request_id="interrupted")
    except OSError:
        pass
    ledger = json.loads(memos.ledger_path.read_text())
    entry = ledger.get("requests", {}).get("interrupted")
    assert entry is not None and entry["state"] == "prepared", "interrupted write lost its durable identity"
    monkeypatch.setattr(memo_requests, "_create", create)
    monkeypatch.setattr(memo_requests, "_save", save)
    result = remember(store, memos, request_id="interrupted")
    assert result["memo_id"] == entry["memo_id"]
    assert result["receipt"]["index_state"] == "indexed"
    assert len(list(memos.folder.glob("*.md"))) == 1


def test_calibration_inspection_failure_does_not_erase_indexing_success(serving, monkeypatch):
    """Red: baseline b7f5d9c (no request journal, MemoFolder.write allocating ids itself) failed `calibration inspection erased successful indexing` (no
    index_state). Target: the separate calibration try block in AssistantMemos.remember.
    """
    store, memos = serving
    def broken():
        raise RuntimeError("injected calibration inspection failure")
    monkeypatch.setattr(store, "resolve_calibration", broken)
    result = remember(store, memos, include_receipt=True)
    receipt = result.get("receipt", {})
    assert receipt.get("index_state") == "indexed", "calibration inspection erased successful indexing"
    assert receipt["calibration_status"] == "unavailable" and receipt["retrieval_ready"] is False


@pytest.mark.parametrize("change", ["text", "target", "revision"])
def test_request_identity_cannot_be_reused_for_another_write(serving, change):
    """Red: baseline b7f5d9c (no request journal, MemoFolder.write allocating ids itself) failed `request identity accepted different intent` for all three
    changes. Target: the identity comparison in write_memo.
    """
    store, memos = serving
    target, revision = initial(store, memos)
    other, _ = initial(store, memos)
    kwargs = dict(request_id="correction-1", supersedes=target, expected_revision=revision)
    remember(store, memos, "The user drinks coffee.", **kwargs)
    text = "The user drinks coffee."
    if change == "text":
        text = "The user drinks water."
    elif change == "target":
        kwargs["supersedes"] = other
    else:
        kwargs["expected_revision"] = "0" * 64
    result = remember(store, memos, text, **kwargs)
    assert result.get("conflict") == "request_mismatch", "request identity accepted different intent"
    assert len(list(memos.folder.glob("*.md"))) == 3


def test_revision_precondition_refuses_an_edited_target(serving):
    """Red: baseline b7f5d9c (no request journal, MemoFolder.write allocating ids itself) failed `an edited target was replaced` (no conflict).
    Target: the expected_revision check in _target_conflict.
    """
    store, memos = serving
    target, revision = initial(store, memos)
    memos.path_of(target).write_text("The user drinks water.\n", encoding="utf-8")
    result = remember(store, memos, "The user drinks coffee.", supersedes=target, expected_revision=revision)
    assert result.get("conflict") == "revision_mismatch", "an edited target was replaced"
    assert len(list(memos.folder.glob("*.md"))) == 1


def test_a_second_correction_cannot_fork_an_already_replaced_target(serving):
    """Red: baseline b7f5d9c (no request journal, MemoFolder.write allocating ids itself) failed `a stale target acquired competing successors` (no
    conflict). Target: the successor scan in _target_conflict.
    """
    store, memos = serving
    target, _ = initial(store, memos)
    remember(store, memos, "The user drinks coffee.", supersedes=target)
    result = remember(store, memos, "The user drinks water.", supersedes=target)
    assert result.get("conflict") == "stale_target", "a stale target acquired competing successors"
    assert len(list(memos.folder.glob("*.md"))) == 2


def test_correction_retry_remains_idempotent_after_another_successor(serving):
    """Red: baseline b7f5d9c (no request journal, MemoFolder.write allocating ids itself) failed `retry lost the original correction identity` (note-0004
    for note-0002). Target: write_memo replaying a written entry before any target check.
    """
    store, memos = serving
    target, revision = initial(store, memos)
    first = remember(store, memos, "The user drinks coffee.", supersedes=target,
                     expected_revision=revision, request_id="correct-1")
    remember(store, memos, "The user drinks water.", supersedes=first["memo_id"])
    retry = remember(store, memos, "The user drinks coffee.", supersedes=target,
                     expected_revision=revision, request_id="correct-1")
    assert retry.get("memo_id") == first["memo_id"], "retry lost the original correction identity"
    assert len(list(memos.folder.glob("*.md"))) == 3


def test_refresh_failure_can_be_retried_without_a_duplicate(serving, monkeypatch):
    """Red: baseline b7f5d9c (no request journal, MemoFolder.write allocating ids itself) failed `refresh retry duplicated the stored memo` (2 files for 1).
    Target: the receipt-enabled refresh path of AssistantMemos.remember plus the journal.
    """
    store, memos = serving
    refresh = memos.refresh
    def broken(*args, **kwargs):
        raise RuntimeError("injected refresh failure")
    monkeypatch.setattr(memos, "refresh", broken)
    try:
        first = remember(store, memos, request_id="recover-1")
    except RuntimeError:
        first = None  # baseline reaches the real, legacy raising refresh boundary
    monkeypatch.setattr(memos, "refresh", refresh)
    second = remember(store, memos, request_id="recover-1")
    assert len(list(memos.folder.glob("*.md"))) == 1, "refresh retry duplicated the stored memo"
    assert first["receipt"]["index_state"] == "failed"
    assert first["receipt"]["retrieval_ready"] is False
    assert second["receipt"]["index_state"] == "indexed"


@pytest.mark.parametrize("change", ["edit", "forget"])
def test_retry_never_overwrites_or_resurrects_a_changed_memo(serving, change):
    """Red: baseline b7f5d9c (no request journal, MemoFolder.write allocating ids itself) failed `retry ignored a later edit or deletion` for edit and forget.
    Target: the memo_changed and memo_missing branches of write_memo.
    """
    store, memos = serving
    first = remember(store, memos, request_id="durable-1")
    path = memos.path_of(first["memo_id"])
    if change == "edit":
        path.write_text("Human changed this memo.\n", encoding="utf-8")
    else:
        memos.forget(store, ContentWordEmbedder(), first["memo_id"])
    result = remember(store, memos, request_id="durable-1")
    assert result.get("conflict") == ("memo_changed" if change == "edit" else "memo_missing"), "retry ignored a later edit or deletion"
    assert len(list(memos.folder.glob("*.md"))) == (1 if change == "edit" else 0)


def test_two_clients_cannot_install_competing_corrections(serving):
    """Red: baseline b7f5d9c (no request journal, MemoFolder.write allocating ids itself) failed `two clients installed competing corrections` (2 for 1).
    Target: _target_conflict run under the store's write lane.
    """
    store, memos = serving
    target, revision = initial(store, memos)
    def correct(drink):
        with LiteStore(store.path, dim=ContentWordEmbedder.dim) as client:
            return remember(client, memos, f"The user drinks {drink}.", supersedes=target,
                            expected_revision=revision, request_id=f"concurrent-{drink}")
    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(correct, ["coffee", "water"]))
    assert sum("memo_id" in result for result in results) == 1, "two clients installed competing corrections"
    assert len(list(memos.folder.glob("*.md"))) == 2


def test_cold_store_receipt_does_not_claim_trusted_readiness(tmp_path):
    """Red: baseline b7f5d9c (no request journal, MemoFolder.write allocating ids itself) failed `cold-store write hid its indexing result` (no index_state).
    Target: AssistantMemos.remember's receipt on an uncalibrated store.
    """
    with LiteStore(tmp_path / "memory.db", dim=ContentWordEmbedder.dim) as store:
        memos = AssistantMemos(tmp_path / "memos", index_root=tmp_path)
        result = remember(store, memos, include_receipt=True)
        receipt = result.get("receipt", {})
        assert receipt.get("index_state") == "indexed", "cold-store write hid its indexing result"
        assert receipt.get("calibration_status") == "missing"
        assert receipt.get("retrieval_ready") is False, "uncalibrated write claimed trusted readiness"


def test_mcp_exposes_optional_guards_and_indexed_revision(tmp_path, monkeypatch):
    """Red: baseline b7f5d9c (no request journal, MemoFolder.write allocating ids itself) failed `MCP omitted optional correction guards` (request_id,
    expected_revision, include_receipt absent). Target: the recall_remember registration in
    recall_mcp/server.py and SearchHit.revision.
    """
    envs = base_envs.__wrapped__(tmp_path, monkeypatch)
    published = _published(envs[0])
    fields = published["recall_remember"]["parameters"]["properties"]
    assert {"request_id", "expected_revision", "include_receipt"} <= set(fields), "MCP omitted optional correction guards"
    async def body(a, p, state):
        first = json.loads(await a("recall_remember", text="The user drink tea, every morning."))
        full = json.loads(await p("recall_search", query="user drink tea"))
        compact = json.loads(await a("recall_search", query="user drink tea"))
        expected = hashlib.sha256((tmp_path / "memos" / f"{first['memo_id']}.md").read_bytes()).hexdigest()
        assert full["hits"][0].get("revision") == expected, "search did not expose the indexed revision"
        assert compact["hits"][0].get("revision") == expected, "compact search lost the correction precondition"
    _serve_both(envs, body)


def test_forget_cancels_a_published_but_uncommitted_request(serving, monkeypatch):
    """BUG-001: baseline 139b138b recreates the memo after a successful explicit forget."""
    from recall.lite import memo_requests
    store, memos = serving
    save = memo_requests._save
    def interrupted(path, ledger):
        if ledger.get("requests", {}).get("interrupted", {}).get("state") == "written":
            raise OSError("injected final journal save failure")
        return save(path, ledger)
    monkeypatch.setattr(memo_requests, "_save", interrupted)
    with pytest.raises(OSError, match="final journal save"):
        remember(store, memos, request_id="interrupted")
    monkeypatch.setattr(memo_requests, "_save", save)
    entry = json.loads(memos.ledger_path.read_text())["requests"]["interrupted"]
    assert entry["state"] == "prepared" and memos.path_of(entry["memo_id"]).exists()
    memos.refresh(store, ContentWordEmbedder())
    assert memos.forget(store, ContentWordEmbedder(), entry["memo_id"])["deleted"] == 1
    result = remember(store, memos, request_id="interrupted")
    assert result.get("conflict") == "memo_missing", "retry resurrected an explicitly forgotten prepared memo"
    assert not memos.path_of(entry["memo_id"]).exists()
    assert str(memos.path_of(entry["memo_id"])) not in store.source_raw_hashes()


def test_forget_persists_cancellation_before_unlinking(serving, monkeypatch):
    """BUG-001: baseline 139b138b deletes without durable cancellation, so no error is raised."""
    from recall.lite import memo_requests
    store, memos = serving
    first = remember(store, memos, request_id="cancel-save")
    save = memo_requests._save
    def unavailable(path, ledger):
        if ledger.get("requests", {}).get("cancel-save", {}).get("state") == "cancelled":
            raise OSError("injected cancellation save failure")
        return save(path, ledger)
    monkeypatch.setattr(memo_requests, "_save", unavailable)
    with pytest.raises(OSError, match="cancellation save"):
        memos.forget(store, ContentWordEmbedder(), first["memo_id"])
    assert memos.path_of(first["memo_id"]).exists(), "failed cancellation unlinked the memo"


@pytest.mark.parametrize("encoding", ["crlf", "bom", "nul"])
def test_search_revision_can_correct_the_same_indexed_markdown(tmp_path, monkeypatch, encoding):
    """DAT-002: baseline 139b138b rejects the real search revision as revision_mismatch."""
    envs = base_envs.__wrapped__(tmp_path, monkeypatch)
    async def body(a, p, state):
        first = json.loads(await a("recall_remember", text="The user drink tea, every morning."))
        path = tmp_path / "memos" / f"{first['memo_id']}.md"
        raw = path.read_bytes()
        changed = {"crlf": raw.replace(b"\n", b"\r\n"), "bom": b"\xef\xbb\xbf" + raw,
                   "nul": raw.replace(b"tea", b"t\x00ea")}[encoding]
        path.write_bytes(changed)
        AssistantMemos(path.parent, index_root=tmp_path).refresh(state["store"], state["embedder"])
        full = json.loads(await p("recall_search", query="user drink tea"))
        compact = json.loads(await a("recall_search", query="user drink tea"))
        matching = [h for h in full["hits"] if h["source"].endswith(f"memos/{path.name}")]
        assert matching, [h["source"] for h in full["hits"]]
        hit = matching[0]
        small = next(h for h in compact["hits"] if h["memo_id"] == first["memo_id"])
        assert small["revision"] == hit["revision"]
        corrected = json.loads(await a("recall_remember", text="The user drinks coffee.",
                                       supersedes=first["memo_id"], expected_revision=hit["revision"]))
        assert "memo_id" in corrected, "unchanged indexed Markdown rejected its published revision"
    _serve_both(envs, body)


@pytest.mark.parametrize("text", ["\ufeffThe user drinks tea.", "The user drinks t\x00ea."])
def test_receipt_revision_matches_the_indexed_markdown_representation(serving, text):
    """DAT-002: baseline 139b138b reports failed indexing for successfully indexed BOM/NUL text."""
    store, memos = serving
    result = remember(store, memos, text, request_id="normalized-receipt")
    receipt = result["receipt"]
    indexed = store.source_raw_hashes()[str(memos.path_of(result["memo_id"]))]
    assert receipt["revision"] == indexed, "write receipt differs from the indexed Markdown revision"
    assert receipt["index_state"] == "indexed" and receipt["retrieval_ready"] is True
    replay = remember(store, memos, text, request_id="normalized-receipt")
    assert replay["memo_id"] == result["memo_id"] and replay["receipt"]["replayed"]


@pytest.mark.parametrize("change", ["crlf", "bom"])
def test_replay_protects_raw_bytes_even_when_indexed_revision_is_unchanged(serving, change):
    """DAT-002 control: mutating the replay raw digest to revision(path) must expose an edit."""
    store, memos = serving
    first = remember(store, memos, request_id="raw-byte-guard")
    path = memos.path_of(first["memo_id"])
    raw = path.read_bytes()
    edited = raw.replace(b"\n", b"\r\n") if change == "crlf" else b"\xef\xbb\xbf" + raw
    path.write_bytes(edited)
    result = remember(store, memos, request_id="raw-byte-guard")
    assert result.get("conflict") == "memo_changed", "replay ignored a raw-byte edit with the same indexed revision"
    assert path.read_bytes() == edited


# Review fixes on PR 947. Each docstring names the baseline (561ba05ae, the PR head) or the
# mutation, the production symbol, and the failure seen; the runs are recorded in the PR.


def _attempt(store, memos, text="The user drinks tea.", **kwargs):
    """`remember`, with an exception returned as the reply so the assertion is what fails."""
    try:
        return remember(store, memos, text, **kwargs)
    except OSError as exc:
        return {"raised": repr(exc)}


@pytest.mark.parametrize("code", [errno.EPERM, errno.EINVAL])
@pytest.mark.parametrize("request_id", [None, "no-links"])
def test_publication_falls_back_where_the_filesystem_has_no_hard_links(serving, monkeypatch, code, request_id):
    """Finding 1. Red: baseline 561ba05ae, memo_requests._create calls os.link with no fallback,
    so the reply is the raised OSError and `a filesystem without hard links refused the memo`
    fails. Target: the `_link_unsupported` fallback to `_create_exclusive` in `_create`.
    """
    from recall.lite import memo_requests
    store, memos = serving
    def no_links(src, dst, *args, **kwargs):
        raise OSError(code, "hard links are not supported here")
    monkeypatch.setattr(memo_requests.os, "link", no_links)
    result = _attempt(store, memos, request_id=request_id)
    assert "memo_id" in result, f"a filesystem without hard links refused the memo: {result}"
    assert memos.path_of(result["memo_id"]).read_text(encoding="utf-8") == "The user drinks tea.\n"
    assert not list(memos.folder.glob(".recall-memo-*")), "the staged copy was left behind"


def test_a_real_link_failure_is_not_taken_for_a_missing_feature(serving, monkeypatch):
    """Finding 1 control. Red: mutation of `_link_unsupported` to `return True` (fall back on
    every OSError) writes the memo, and `pytest.raises` fails with DID NOT RAISE OSError.
    Target: `_link_unsupported` in recall/lite/memo_requests.py.
    """
    from recall.lite import memo_requests
    store, memos = serving
    def broken_disk(src, dst, *args, **kwargs):
        raise OSError(errno.EIO, "injected I/O error")
    monkeypatch.setattr(memo_requests.os, "link", broken_disk)
    with pytest.raises(OSError, match="injected I/O error"):
        remember(store, memos)
    assert not list(memos.folder.glob("*.md")), "an I/O error still published a memo"


def _take_the_name_first(monkeypatch, folder, name="note-0001.md"):
    """Make a person's file appear under `name` between the folder scan and the create."""
    from recall.lite import memo_requests
    create = memo_requests._create
    def racing(path, body):
        if path.name == name and not path.exists():
            path.write_bytes(b"A person's own note.\n")
        return create(path, body)
    monkeypatch.setattr(memo_requests, "_create", racing)


@pytest.mark.parametrize("request_id", [None, "race-1"])
def test_a_name_taken_during_the_write_moves_to_the_next_id_without_claiming_it(serving, monkeypatch, request_id):
    """Findings 2 and 3. Red: baseline 561ba05ae lets the FileExistsError out of write_memo (the
    old retry loop was gone), so the reply is the exception and `the write did not move past a
    taken name` fails, for the legacy and the guarded path alike. Target: the FileExistsError
    retry in `_publish`, and `authors` being written only after `_create` returns.
    """
    store, memos = serving
    _take_the_name_first(monkeypatch, memos.folder)
    try:
        result = remember(store, memos, request_id=request_id)
    except FileExistsError as exc:
        result = {"raised": repr(exc)}
    assert result.get("memo_id") == "note-0002", f"the write did not move past a taken name: {result}"
    assert (memos.folder / "note-0001.md").read_bytes() == b"A person's own note.\n"
    assert "note-0001" not in memos.authors(), "the assistant was recorded as author of a person's file"
    if request_id is not None:
        entry = json.loads(memos.ledger_path.read_text())["requests"][request_id]
        assert (entry["memo_id"], entry["state"]) == ("note-0002", "written")
        assert remember(store, memos, request_id=request_id)["memo_id"] == "note-0002"


def test_a_reservation_is_not_authorship_until_the_memo_exists(serving, monkeypatch):
    """Finding 2. Red: baseline 561ba05ae saves `authors[memo_id]` with the prepared reservation,
    so `a reservation claimed authorship` fails; after a person takes the name, its retry
    returned memo_changed for ever. Target: write_memo's prepared-replay branch and `_publish`.
    """
    from recall.lite import memo_requests
    store, memos = serving
    create = memo_requests._create
    def interrupted(path, body):
        raise OSError("injected interruption before memo publication")
    monkeypatch.setattr(memo_requests, "_create", interrupted)
    with pytest.raises(OSError, match="before memo publication"):
        remember(store, memos, request_id="reserved")
    entry = json.loads(memos.ledger_path.read_text())["requests"]["reserved"]
    assert entry["memo_id"] not in memos.authors(), "a reservation claimed authorship"
    monkeypatch.setattr(memo_requests, "_create", create)
    taken = memos.path_of(entry["memo_id"])
    taken.write_bytes(b"A person's own note.\n")
    result = remember(store, memos, request_id="reserved")
    assert result.get("memo_id") not in (None, entry["memo_id"]), f"the retry is stuck on a taken name: {result}"
    assert taken.read_bytes() == b"A person's own note.\n"
    assert entry["memo_id"] not in memos.authors()


@pytest.mark.parametrize("unrelated", ["latin1", "directory"])
def test_an_unreadable_file_does_not_block_a_correction(serving, unrelated):
    """Finding 4. Red: baseline 561ba05ae reads every memo with read_text(encoding="utf-8"), so
    a Latin-1 file raises UnicodeDecodeError and a directory named `*.md` raises
    IsADirectoryError, and `an unrelated file blocked the correction` fails. Target: the
    tolerant read in `_target_conflict`.
    """
    store, memos = serving
    target, revision = initial(store, memos)
    if unrelated == "latin1":
        (memos.folder / "caf.md").write_bytes(b"Caf\xe9 au lait, a person's own note.\n")
    else:
        (memos.folder / "folder.md").mkdir()
    try:
        result = remember(store, memos, "The user drinks coffee.", supersedes=target, expected_revision=revision)
    except (OSError, UnicodeDecodeError) as exc:
        result = {"raised": repr(exc)}
    assert "memo_id" in result, f"an unrelated file blocked the correction: {result}"


def test_a_successor_in_a_non_utf8_file_still_makes_the_target_stale(serving):
    """Finding 4 control. Red: mutation of `_target_conflict` to skip any file that is not valid
    UTF-8 (decode with errors="strict" inside a try that continues) lets the correction through,
    and `a declared successor was missed` fails. Target: `errors="replace"` in `_target_conflict`.
    """
    store, memos = serving
    target, _ = initial(store, memos)
    (memos.folder / "caf.md").write_bytes(f"---\nsupersedes: {target}.md\n---\nCaf\xe9.\n".encode("latin-1"))
    result = remember(store, memos, "The user drinks coffee.", supersedes=target)
    assert result.get("conflict") == "stale_target", f"a declared successor was missed: {result}"


def test_a_correction_parses_only_files_that_can_declare_a_successor(serving, monkeypatch):
    """Finding 7. Red: baseline 561ba05ae calls parse_document on every memo, so with five plain
    memos and one correction `the successor scan parsed files that declare nothing` fails (7 for
    1). Target: the `b"supersedes"` prefilter in `_target_conflict`.
    """
    from recall.lite import memo_requests
    store, memos = serving
    target, _ = initial(store, memos)
    for n in range(4):
        remember(store, memos, f"Unrelated fact number {n}.")
    remember(store, memos, "The user drinks coffee.", supersedes=target)
    parsed = []
    parse = memo_requests.parse_document
    monkeypatch.setattr(memo_requests, "parse_document", lambda text: parsed.append(text) or parse(text))
    other = remember(store, memos, "Another unrelated fact.")
    assert "memo_id" in other
    target2 = other["memo_id"]
    remember(store, memos, "A correction of the unrelated fact.", supersedes=target2)
    assert len(parsed) == 1, f"the successor scan parsed files that declare nothing: {len(parsed)}"


def test_completed_requests_are_pruned_past_their_retention(serving, monkeypatch):
    """Finding 6. Red: baseline 561ba05ae never prunes the journal, so both `an expired request
    stayed in the journal` and the count bound fail. Target: `_prune` in write_memo's final save.
    """
    from recall.lite import memo_requests
    store, memos = serving
    for n in range(3):
        remember(store, memos, f"Fact number {n}.", request_id=f"old-{n}")
    ledger = json.loads(memos.ledger_path.read_text())
    now = time.time()
    ledger["requests"]["old-0"]["at"] = 0  # completed in 1970: far past any retention
    ledger["requests"]["old-1"]["at"] = now - 100
    ledger["requests"]["old-2"]["at"] = now - 50
    ledger["requests"]["held"] = {"identity": "0" * 64, "memo_id": "note-0099", "state": "prepared"}
    memos.ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    monkeypatch.setattr(memo_requests, "REQUEST_RETENTION_COUNT", 2, raising=False)
    remember(store, memos, "A newer fact.", request_id="new")
    requests = json.loads(memos.ledger_path.read_text())["requests"]
    assert "old-0" not in requests, "an expired request stayed in the journal"
    assert set(requests) == {"old-2", "new", "held"}, f"the journal kept more than its bound: {sorted(requests)}"
    assert memos.authors().keys() >= {"note-0001", "note-0002", "note-0003", "note-0100"}


def test_a_pure_replay_rewrites_nothing(serving, monkeypatch):
    """Finding 6. Red: baseline 561ba05ae rewrites the whole journal on every replay, so
    `a replay of a completed write rewrote the journal` fails (1 save for 0). Target: the
    `changed = False` branch for a written entry in write_memo.
    """
    from recall.lite import memo_requests
    store, memos = serving
    first = remember(store, memos, request_id="once")
    saves = []
    save = memo_requests._save
    monkeypatch.setattr(memo_requests, "_save", lambda path, ledger: saves.append(path) or save(path, ledger))
    assert remember(store, memos, request_id="once")["memo_id"] == first["memo_id"]
    assert saves == [], "a replay of a completed write rewrote the journal"


def test_a_failed_unlink_leaves_the_request_finishable(serving, monkeypatch):
    """Finding 8. Red: baseline 561ba05ae saves the cancellation and then fails to unlink, so the
    retry returns memo_missing and `a memo still on disk was read as forgotten` fails. Target:
    `restore_requests` in MemoFolder.delete.
    """
    import pathlib
    store, memos = serving
    first = remember(store, memos, request_id="held-open")
    path = memos.path_of(first["memo_id"])
    unlink = pathlib.Path.unlink
    def held(self, *args, **kwargs):
        if self == path:
            raise PermissionError("injected: another program holds the file")
        return unlink(self, *args, **kwargs)
    monkeypatch.setattr(pathlib.Path, "unlink", held)
    with pytest.raises(PermissionError, match="holds the file"):
        memos.forget(store, ContentWordEmbedder(), first["memo_id"])
    monkeypatch.setattr(pathlib.Path, "unlink", unlink)
    assert path.exists()
    retry = remember(store, memos, request_id="held-open")
    assert retry.get("memo_id") == first["memo_id"], f"a memo still on disk was read as forgotten: {retry}"


def test_the_receipt_looks_up_only_its_own_source(serving, monkeypatch):
    """Finding 10. Red: baseline 561ba05ae reads `store.source_raw_hashes()` for every source, so
    with that call refused the receipt reports `failed` and `the receipt read every source`
    fails. Target: `LiteStore.source_raw_hash` in AssistantMemos.remember.
    """
    store, memos = serving
    whole = store.source_raw_hashes
    def every_source():
        if sys._getframe(1).f_code.co_filename.endswith("assistant.py"):
            raise AssertionError("injected: the receipt read the whole-store hash map")
        return whole()
    monkeypatch.setattr(store, "source_raw_hashes", every_source)
    result = remember(store, memos, include_receipt=True)
    assert result["receipt"].get("index_state") == "indexed", f"the receipt read every source: {result}"
