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

import hashlib
import inspect
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
    store, memos = serving
    result = remember(store, memos, include_receipt=True)
    receipt = result.get("receipt", {})
    assert receipt.get("write_state") == "written", "write receipt missing"
    assert receipt.get("index_state") == "indexed", "written memo was not verified in the index"
    assert receipt.get("calibration_status") == "certified"
    assert receipt.get("retrieval_ready") is True
    assert receipt["revision"] == hashlib.sha256(memos.path_of(result["memo_id"]).read_bytes()).hexdigest()


def test_idempotent_retry_uses_the_original_memo(serving):
    store, memos = serving
    first = remember(store, memos, request_id="write-1")
    second = remember(store, memos, request_id="write-1")
    assert second["memo_id"] == first["memo_id"], "retry allocated another memo"
    assert second["receipt"]["replayed"] is True
    assert len(list(memos.folder.glob("*.md"))) == 1


def test_legacy_writes_preserve_request_identities(serving):
    store, memos = serving
    first = remember(store, memos, request_id="write-1")
    remember(store, memos, "Another independent fact.")
    retry = remember(store, memos, request_id="write-1")
    assert retry["memo_id"] == first["memo_id"], "legacy write discarded the request journal"
    assert len(list(memos.folder.glob("*.md"))) == 2


@pytest.mark.parametrize("stage", ["before_publish", "after_publish"])
def test_prepared_request_survives_interrupted_publication(serving, monkeypatch, stage):
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
    store, memos = serving
    target, revision = initial(store, memos)
    memos.path_of(target).write_text("The user drinks water.\n", encoding="utf-8")
    result = remember(store, memos, "The user drinks coffee.", supersedes=target, expected_revision=revision)
    assert result.get("conflict") == "revision_mismatch", "an edited target was replaced"
    assert len(list(memos.folder.glob("*.md"))) == 1


def test_a_second_correction_cannot_fork_an_already_replaced_target(serving):
    store, memos = serving
    target, _ = initial(store, memos)
    remember(store, memos, "The user drinks coffee.", supersedes=target)
    result = remember(store, memos, "The user drinks water.", supersedes=target)
    assert result.get("conflict") == "stale_target", "a stale target acquired competing successors"
    assert len(list(memos.folder.glob("*.md"))) == 2


def test_correction_retry_remains_idempotent_after_another_successor(serving):
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
    with LiteStore(tmp_path / "memory.db", dim=ContentWordEmbedder.dim) as store:
        memos = AssistantMemos(tmp_path / "memos", index_root=tmp_path)
        result = remember(store, memos, include_receipt=True)
        receipt = result.get("receipt", {})
        assert receipt.get("index_state") == "indexed", "cold-store write hid its indexing result"
        assert receipt.get("calibration_status") == "missing"
        assert receipt.get("retrieval_ready") is False, "uncalibrated write claimed trusted readiness"


def test_mcp_exposes_optional_guards_and_indexed_revision(tmp_path, monkeypatch):
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
