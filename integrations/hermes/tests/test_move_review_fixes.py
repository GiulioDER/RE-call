"""Fixes from the 2026-11-20 review of the shared Lite store commits, made when the provider moved
into the RE-call repository (integrations/hermes, RE-call PR 1471)."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

from conftest import wait_for
from test_shared_store import NOTE, _bound_on_the_old_store, _memo, _shared

from re_call_hermes import provider as provider_module
from re_call_hermes.config import load_settings
from re_call_hermes.provider import _digest


def test_a_switch_to_the_shared_store_keeps_mirroring(harness, private_shared_store):
    """Invariant (BUG-001): a profile bound to its old store that switches to the shared store when
    that appears (the old store still in place) keeps mirroring: the old journal is set aside under
    `previous`, instead of every later built-in memory write being refused and dropped. A dsn or
    memo_folder the user points elsewhere stays refused (tests/test_pr1_regressions.py).

    Red proof (2026-11-20, `python tests/mutations.py SWITCH-FRESH-MAPPING`): in
    re_call_hermes/provider.py, `if found is None and self._switched_to_shared(recorded):` became
    `if False:`; the test failed at `assert False` (no recall_remember call within 8 s). Restored, green.
    """
    _bound_on_the_old_store(harness)  # the old store and its journal stay where they are
    _memo(_shared(private_shared_store) / "memos" / "hermes", NOTE, "Has a dog")  # another agent's memo
    harness.scenario({"recall_remember": [{"memo_id": "note-0002.md"}]})
    provider = harness.start(dsn="")
    assert provider._settings.store_path == private_shared_store / "memory.db"
    provider.on_memory_write("add", "memory", "Likes jazz", {})
    assert wait_for(lambda: harness.events("call"), timeout=8)
    assert harness.events("call")[0]["tool"] == "recall_remember"
    journal = json.loads((harness.data / "mirror.json").read_text(encoding="utf-8"))
    assert journal["previous"][0]["memos"] == {_digest("memory", "Prefers tea"): NOTE}


def test_an_entry_removed_before_its_memo_is_in_place_is_forgotten_once_it_is(
        harness, private_shared_store, tmp_path):
    """Invariant (BUG-002): a remove of an entry whose memo was not yet readable at the rebind (moved
    in pieces) forgets that memo once it is in place holding the entry's text, so the removed text is
    not served as current.

    Red proof (2026-11-20, `python tests/mutations.py RETIRE-WHEN-READ`): in
    re_call_hermes/provider.py, `state["retire_when_read"][old_key] = candidate` became
    `state["unconfirmed"][old_key] = candidate`, the earlier behaviour; the test failed at
    `assert False` (no forget call within 8 s). Restored, green.
    """
    old = (harness.data / "memory.db", harness.data / "memos" / "hermes")
    _memo(old[1], NOTE, "Prefers tea")
    _memo(old[1], "note-0002", "Has a cat")
    cat = _digest("memory", "Has a cat")
    path = harness.data / "mirror.json"
    path.write_text(json.dumps({
        "destination": [os.path.normcase(str(p.resolve())) for p in old],
        "memos": {_digest("memory", "Prefers tea"): NOTE, cat: "note-0002"},
        "pending": [], "submitted": []}), encoding="utf-8")
    private_shared_store.mkdir(parents=True)
    shutil.move(harness.data / "memory.db", private_shared_store / "memory.db")
    shutil.move(old[1] / "note-0002.md", tmp_path / "note-0002.md")  # not in place yet
    shutil.move(harness.data / "memos", private_shared_store / "memos")
    harness.scenario({"recall_forget_memo": [{"deleted": True}], "recall_remember": [{"memo_id": "note-0003.md"}]})
    provider = harness.start(dsn="")
    provider.on_memory_write("remove", "memory", "", {"previous_content": "Has a cat"})
    assert wait_for(lambda: cat in json.loads(path.read_text(encoding="utf-8")).get("retire_when_read", {}),
                    timeout=8)
    assert harness.events("call") == []  # nothing forgotten while the memo is not in place
    shutil.move(tmp_path / "note-0002.md", private_shared_store / "memos" / "hermes" / "note-0002.md")
    provider.on_memory_write("add", "memory", "Likes jazz", {})  # the next write drains the journal
    assert wait_for(lambda: any(c["tool"] == "recall_forget_memo" for c in harness.events("call")), timeout=8)
    forgets = [c for c in harness.events("call") if c["tool"] == "recall_forget_memo"]
    assert forgets[0]["arguments"] == {"memo_id": "note-0002"}


def test_no_home_directory_is_a_reason_not_an_exception(tmp_path, monkeypatch):
    """Invariant (BUG-004): where no home directory can be found, the default configuration reports
    why it cannot start, and a PostgreSQL configuration does not depend on a home directory at all.

    Red proof (2026-11-20, `python tests/mutations.py NO-HOME-REASON`): in re_call_hermes/config.py,
    `except RuntimeError:\\n        return None` became `except ValueError:\\n        return None`; the
    test failed at `AssertionError: assert 'no home directory' in 'raised: Could not determine home
    directory'`. Restored, green.
    """
    def no_home() -> Path:
        raise RuntimeError("Could not determine home directory")

    monkeypatch.delenv("XDG_CONFIG_HOME", raising=False)
    monkeypatch.setattr(Path, "home", staticmethod(no_home))
    home = tmp_path / "hermes-home"
    (home / "re-call").mkdir(parents=True)

    def problem(**config) -> str:
        (home / "re-call" / "config.json").write_text(json.dumps(config), encoding="utf-8")
        try:
            return load_settings(home).problem()
        except RuntimeError as exc:
            return f"raised: {exc}"

    assert "no home directory" in problem()
    assert "raised" not in problem(dsn="postgresql://recall@localhost/recall")


def test_a_shutdown_while_the_server_starts_leaves_no_writer(harness, monkeypatch):
    """Invariant: a shutdown() that arrives while initialize() waits for the server leaves no mirror
    writer: none is started for a closed provider, so none is left blocked on its queue, and none
    joined before it started (the teardown RuntimeError seen on 2026-11-14 and on 2026-11-20).

    Red proof (2026-11-20, `python tests/mutations.py CLOSED-NO-WRITER`): in
    re_call_hermes/provider.py, `if self._closed:\\n                    return` was deleted; the test
    failed at `AssertionError: assert ['re-call-writer'] == []`. Restored, green.
    """
    real_spawn = provider_module.spawn_context_thread
    writers: list[str] = []

    def spawn(target, *, name, **kwargs):
        if name == "re-call-writer":
            writers.append(name)
        if name != "re-call-start":
            return real_spawn(target, name=name, **kwargs)

        def start_then_shut_down():
            target()
            harness.provider.shutdown()  # Hermes shutting down while initialize() waits for the server

        return real_spawn(start_then_shut_down, name=name, **kwargs)

    monkeypatch.setattr(provider_module, "spawn_context_thread", spawn)
    harness.scenario({})
    provider = harness.start(ready=False, dsn="")
    assert wait_for(lambda: provider._closed, timeout=10)
    wait_for(lambda: writers, timeout=3)  # long enough for initialize() to reach the writer
    assert writers == []
    assert provider._writer is None


def _bound_journal(harness, shared: Path, **fields) -> Path:
    """A journal bound to the shared store, with `fields` merged in."""
    _shared(shared)
    store, folder = shared / "memory.db", shared / "memos" / "hermes"
    path = harness.data / "mirror.json"
    path.write_text(json.dumps({"destination": provider_module._destination((store, folder)),
                                "memos": {}, "pending": [], "submitted": [], **fields}), encoding="utf-8")
    return path


def test_a_memo_the_journal_maps_now_is_never_retired(harness, private_shared_store):
    """Invariant (review of the fixes, BUG-002): a retire entry whose memo id the journal now maps (the
    id given again to the same text, added back) is dropped, never forgotten.

    Red proof (2026-11-20, `python tests/mutations.py RETIRE-SKIPS-LIVE`): in
    re_call_hermes/provider.py, `if v in state["memos"].values()}` became `if False}`; the test failed
    at `AssertionError: assert [{'event': 'c...'note-0002'}}] == []`. Restored, green.
    """
    cat = _digest("memory", "Has a cat")
    path = _bound_journal(harness, private_shared_store, memos={cat: "note-0002"},
                          retire_when_read={cat: "note-0002"})
    _memo(private_shared_store / "memos" / "hermes", "note-0002", "Has a cat")
    harness.scenario({"recall_forget_memo": [{"deleted": True}]})
    provider = harness.start(dsn="")
    assert wait_for(lambda: not json.loads(path.read_text(encoding="utf-8")).get("retire_when_read"), timeout=8)
    provider.shutdown()
    assert [c for c in harness.events("call") if c["tool"] == "recall_forget_memo"] == []


def test_a_failed_retire_keeps_its_entry_and_the_drain_goes_on(harness, private_shared_store, monkeypatch):
    """Invariant (review of the fixes, BUG-005): a forget that fails without a refusal (a timeout, a
    lost server) keeps its retire entry for the next drain, and the writes queued behind it are still
    mirrored in this drain.

    Red proof (2026-11-20, `python tests/mutations.py RETIRE-ANY-MCP-ERROR`): in
    re_call_hermes/provider.py, `except McpError:  # not sent, a timeout` became
    `except NotSent:  # not sent, a timeout`; the test failed at `AssertionError: assert 'raised:
    timed out' == 'drained'`. Restored, green.
    """
    from re_call_hermes.mcp_client import McpError
    from re_call_hermes.provider import ReCallMemoryProvider

    cat = _digest("memory", "Has a cat")
    path = _bound_journal(harness, private_shared_store, retire_when_read={cat: "note-0002"},
                          pending=[["r1", "add", "memory", "Likes jazz", None]])
    _memo(private_shared_store / "memos" / "hermes", "note-0002", "Has a cat")
    (harness.data / "config.json").write_text(json.dumps({**harness.config, "dsn": ""}), encoding="utf-8")
    calls = []

    def call(tool, arguments, timeout):
        calls.append(tool)
        if tool == "recall_forget_memo":
            raise McpError("timed out")
        return {"memo_id": "note-0003.md"}

    provider = ReCallMemoryProvider()
    provider._settings = load_settings(harness.home)
    monkeypatch.setattr(provider, "_call", call)
    try:
        provider._drain_pending()
        outcome = "drained"
    except McpError as exc:
        outcome = f"raised: {exc}"
    assert outcome == "drained"
    state = json.loads(path.read_text(encoding="utf-8"))
    assert state["retire_when_read"] == {cat: "note-0002"}
    assert state["pending"] == [] and calls == ["recall_forget_memo", "recall_remember"]

