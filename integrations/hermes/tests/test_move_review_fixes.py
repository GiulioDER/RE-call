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


def test_a_shutdown_before_the_writer_starts_does_not_join_it(harness, monkeypatch):
    """Invariant: a shutdown() that arrives while initialize() is creating the mirror writer neither
    joins an unstarted thread (RuntimeError) nor leaves the provider half shut down. Seen once in the
    suite on 2026-11-14 and again on the remote host on 2026-11-20, as a teardown error of
    test_a_start_that_cannot_make_its_folders_says_why.

    Red proof (2026-11-20, `python tests/mutations.py WRITER-STARTED-FIRST`): in
    re_call_hermes/provider.py, `writer.start()\\n            self._writer = writer` became
    `self._writer = writer\\n            writer.start()`; the test failed at `AssertionError: assert
    [RuntimeError('cannot join thread before it is started')] == []`. Restored, green.
    """
    real_spawn = provider_module.spawn_context_thread
    errors: list[Exception] = []

    def spawn(target, *, name, **kwargs):
        thread = real_spawn(target, name=name, **kwargs)
        if name == "re-call-writer":
            real_start = thread.start

            def start():
                try:
                    harness.provider.shutdown()  # Hermes shutting down while initialize() runs
                except RuntimeError as exc:
                    errors.append(exc)
                real_start()

            thread.start = start
        return thread

    monkeypatch.setattr(provider_module, "spawn_context_thread", spawn)
    harness.scenario({})
    provider = harness.start(ready=False, dsn="")
    assert wait_for(lambda: provider._writer is not None or errors, timeout=10)
    assert errors == []
