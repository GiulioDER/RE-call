"""The shared per-user Lite store (owner decision 2026-11-13, "One shared Lite store"), and the
profile's old store, which is kept rather than stranded."""

from __future__ import annotations

import json
import logging
import os
import shutil
import sys
from pathlib import Path

import pytest
from conftest import wait_for

from re_call_hermes.config import load_settings, save_settings
from re_call_hermes.provider import _digest


def _profile(tmp_path: Path, *, old_store: bool, **config) -> Path:
    """A Hermes home whose config.json holds `config`, with or without the profile's old store."""
    home = tmp_path / "hermes-home"
    data = home / "re-call"
    data.mkdir(parents=True)
    (data / "config.json").write_text(json.dumps({"recall_command": sys.executable, **config}), encoding="utf-8")
    if old_store:
        (data / "memory.db").touch()
    return home


def _shared(directory: Path) -> Path:
    directory.mkdir(parents=True)
    (directory / "memory.db").touch()
    return directory


def _notices(caplog) -> list[str]:
    return [r.getMessage() for r in caplog.records if r.name == "re_call_hermes.config"]


def test_the_shared_store_is_chosen_when_present(tmp_path, private_shared_store, caplog):
    """Invariant: with no dsn, a store in the shared directory is used even when the profile still
    has its old one, and one line names the old store as no longer read and how to keep it.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py SHARED-PRESENT`): in
    re_call_hermes/config.py, `if (shared / STORE_FILE).is_file():` became `if False:`;
    the test failed at `AssertionError: assert WindowsPath('C:/Users/<user>/.../hermes-home/re-call/me...`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py SHARED-UNUSED-NOTICE`): in
    re_call_hermes/config.py, `_notify(old, f"re-call: using the shared store` became `(old, f"re-call: using the shared store`;
    the test failed at `assert 0 == 1`. Restored, green.
    """
    caplog.set_level(logging.WARNING, logger="re_call_hermes.config")
    shared = _shared(private_shared_store)
    home = _profile(tmp_path, old_store=True)
    for _ in range(3):
        settings = load_settings(home)
    assert settings.store_path == shared / "memory.db"
    assert settings.memo_folder == shared / "memos"
    assert settings.problem() == ""
    old = home / "re-call" / "memory.db"
    assert len(_notices(caplog)) == 1
    assert str(old) in _notices(caplog)[0] and f"dsn to sqlite:///{old.as_posix()}" in _notices(caplog)[0]


def test_the_old_store_is_kept_and_the_notice_is_logged_once(tmp_path, private_shared_store, caplog):
    """Invariant: a profile whose store predates the shared one keeps using it, and the provider says
    once, per process, where the shared store is and how to move the old one there safely.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py OLD-KEPT`): in
    re_call_hermes/config.py, `return data_dir` became `return shared`;
    the test failed at `AssertionError: assert WindowsPath('C:/Users/<user>/.../xdg/re-call/memory.db'`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py NOTICE-ONCE`): in
    re_call_hermes/config.py, `if old not in _NOTIFIED:` became `if True:`;
    the test failed at `AssertionError: ["re-call: using this profile's old store ...", ...]` (three lines, not one`. Restored, green.
    """
    caplog.set_level(logging.WARNING, logger="re_call_hermes.config")
    home = _profile(tmp_path, old_store=True)
    data = home / "re-call"
    for _ in range(3):  # probes, initialize and setup each load the settings
        settings = load_settings(home)
    assert settings.store_path == data / "memory.db"
    assert settings.memo_folder == data / "memos"
    notices = _notices(caplog)
    assert len(notices) == 1, notices
    line = notices[0]
    for part in (str(data / "memory.db"), str(private_shared_store / "memory.db"), "close every agent",
                 "memory.db-wal", "memory.db-shm", "memos/", f"to {private_shared_store}"):
        assert part in line, part
    assert "\n" not in line


def test_the_shared_default_applies_when_neither_exists(tmp_path, private_shared_store, monkeypatch, caplog):
    """Invariant: with no store anywhere, the default is the shared directory, under
    XDG_CONFIG_HOME or else ~/.config, and the "no store" message gives the setup command for it.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py NEITHER-SHARED`): in
    re_call_hermes/config.py, `if not old.is_file(): return shared` became `if not old.is_file(): return data_dir`;
    the test failed at `AssertionError: assert WindowsPath('C:/Users/<user>/.../hermes-home/re-call/me...`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py SHARED-MEMOS`): in
    re_call_hermes/config.py, `memos or store_dir / MEMO_DIR` became `memos or data_dir / MEMO_DIR`;
    the test failed at `AssertionError: assert WindowsPath('C:/Users/<user>/.../hermes-home/re-call/me...`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py XDG-FALLBACK`): in
    re_call_hermes/config.py, `Path.home() / ".config") / DATA_DIR` became `Path.home()) / DATA_DIR`;
    the test failed at `AssertionError: assert WindowsPath('C:/Users/<user>/.../user/re-call/memory.db...`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py XDG-RELATIVE`): in
    re_call_hermes/config.py, `(Path(xdg) if os.path.isabs(xdg) else` became `(Path(xdg) if xdg else`;
    the test failed at `AssertionError: assert WindowsPath('relative/config/re-call/memory.db') == ...`. Restored, green.
    """
    caplog.set_level(logging.WARNING, logger="re_call_hermes.config")
    home = _profile(tmp_path, old_store=False)
    settings = load_settings(home)
    store, memos = private_shared_store / "memory.db", private_shared_store / "memos"
    assert settings.store_path == store
    assert settings.memo_folder == memos
    assert f"setup --lite --memory {memos} --store {store} --no-register" in settings.problem()
    assert _notices(caplog) == []

    monkeypatch.delenv("XDG_CONFIG_HOME")
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: tmp_path / "user"))
    assert load_settings(home).store_path == tmp_path / "user" / ".config" / "re-call" / "memory.db"
    monkeypatch.setenv("XDG_CONFIG_HOME", "relative/config")  # ignored, as the XDG specification says
    assert load_settings(home).store_path == tmp_path / "user" / ".config" / "re-call" / "memory.db"


@pytest.mark.parametrize("shared_store,old_store", [(True, True), (False, True), (False, False)],
                         ids=["shared", "old-kept", "neither"])
def test_the_store_and_the_memo_root_resolve_together(tmp_path, private_shared_store, shared_store, old_store):
    """Invariant: the default memo root is always `memos/` beside the default store, so the child
    indexes the whole memo root of the store it serves and writes only to the `hermes` subfolder.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py MEMO-WITH-STORE`): in
    re_call_hermes/config.py, `memos or store_dir / MEMO_DIR` became `memos or shared_dir() / MEMO_DIR`;
    the test failed at `AssertionError: assert WindowsPath('C:/Users/<user>/.../xdg/re-call/memos') ==` in the old-kept case`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py ASSISTANT-SUBFOLDER`): in
    re_call_hermes/config.py, `ASSISTANT_SUBFOLDER = "hermes"` became `ASSISTANT_SUBFOLDER = "assistant"`;
    the test failed at `AssertionError: assert 'C:\\Users\\<user>...os\\assistant' == 'C:\\Users\\<user>...memos\\hermes'`. Restored, green.
    """
    if shared_store:
        _shared(private_shared_store)
    home = _profile(tmp_path, old_store=old_store)
    settings = load_settings(home)
    memos = settings.store_path.parent / "memos"
    assert settings.memo_folder == memos
    env = settings.child_env({})
    assert env["RECALL_INDEX_ROOT"] == str(memos)
    assert env["RECALL_ASSISTANT_MEMOS"] == str(memos / "hermes")


def test_an_explicit_dsn_wins(tmp_path, private_shared_store, caplog):
    """Invariant: a dsn in config.json is used whatever the shared and old stores are, and its memo
    folder stays under the profile, as before the shared store.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py DSN-WINS`): in
    re_call_hermes/config.py, `store = _absolute(...) if dsn else store_dir / STORE_FILE` became `store = store_dir / STORE_FILE`;
    the test failed at `AssertionError: assert WindowsPath('C:/Users/<user>/.../hermes-home/re-call/memory....`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py DSN-MEMOS`): in
    re_call_hermes/config.py, `store_dir = data_dir if dsn or custom_memos else _store_dir(data_dir)`
    became `store_dir = _store_dir(data_dir)`;
    the test failed at `AssertionError: assert WindowsPath('C:/Users/<user>/.../xdg/re-call/memos') == ...`. Restored, green.
    """
    caplog.set_level(logging.WARNING, logger="re_call_hermes.config")
    _shared(private_shared_store)
    elsewhere = tmp_path / "elsewhere" / "mine.db"
    home = _profile(tmp_path, old_store=True, dsn="sqlite:///" + elsewhere.as_posix())
    settings = load_settings(home)
    assert settings.store_path == elsewhere
    assert settings.memo_folder == home / "re-call" / "memos"
    assert _notices(caplog) == []


def test_config_and_logs_stay_per_profile(harness, private_shared_store):
    """Invariant: with the shared store in use, the profile's own config.json is what is read and
    saved, and the server log, the embedding cache and the child's working directory stay in
    `<hermes_home>/re-call/`.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py PER-PROFILE-DATA`): in
    re_call_hermes/config.py, `data_dir=data_dir,` became `data_dir=store_dir,`;
    the test failed at `AssertionError: assert WindowsPath('C:/Users/<user>/.../xdg/re-call') == ...`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py PER-PROFILE-CONFIG`): in
    re_call_hermes/config.py, `read_json(data_dir / CONFIG_FILE)` became `read_json(shared_dir() / CONFIG_FILE)`;
    the test failed at `AssertionError: assert 3 == 7`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py PER-PROFILE-SAVE`): in
    re_call_hermes/config.py, `path = Path(hermes_home) / DATA_DIR / CONFIG_FILE` became `path = shared_dir() / CONFIG_FILE`;
    the test failed at `assert 7 == 9`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py PER-PROFILE-LOG`): in
    re_call_hermes/provider.py, `stderr_path=settings.data_dir / "server.log",` became `stderr_path=settings.store_path.parent / "server.log",`;
    the test failed at `AssertionError: assert False`. Restored, green.
    """
    shared = _shared(private_shared_store)
    (harness.data / "memory.db").unlink()  # this profile never had a store of its own
    # Another agent's settings beside the shared store, which this profile must not read.
    (shared / "config.json").write_text(json.dumps({**harness.config, "dsn": "", "k": 3}), encoding="utf-8")
    harness.scenario({})
    provider = harness.start(dsn="", k=7)
    settings = provider._settings
    assert settings.store_path == shared / "memory.db"
    assert settings.k == 7
    started = harness.events("start")[0]
    assert started["env"]["RECALL_SERVING_DSN"] == "sqlite:///" + (shared / "memory.db").as_posix()
    assert Path(started["cwd"]) == harness.data
    assert started["env"]["RECALL_EMBED_CACHE"] == str(harness.data / "embeddings.sqlite")
    assert (harness.data / "server.log").is_file()
    assert not (shared / "server.log").exists()
    save_settings({"k": 9}, harness.home)
    assert json.loads((harness.data / "config.json").read_text(encoding="utf-8"))["k"] == 9
    assert json.loads((shared / "config.json").read_text(encoding="utf-8"))["k"] == 3


def test_a_custom_memo_folder_keeps_a_store_of_its_own(tmp_path, private_shared_store):
    """Invariant: the shared store indexes only `<shared>/memos`, so a profile with its own memo
    folder and no dsn keeps a store under the profile, even when the shared store exists.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py MEMO-OWN-STORE`): in
    re_call_hermes/config.py, `store_dir = data_dir if dsn or custom_memos else _store_dir(data_dir)` became `store_dir = data_dir if dsn else _store_dir(data_dir)`;
    the test failed at `AssertionError: assert WindowsPath('C:/Users/<user>/.../xdg/re-call/memory.db'`. Restored, green.
    """
    _shared(private_shared_store)
    notes = tmp_path / "notes"
    home = _profile(tmp_path, old_store=False, memo_folder=str(notes))
    settings = load_settings(home)
    store = home / "re-call" / "memory.db"
    assert settings.store_path == store
    assert settings.memo_folder == notes
    assert f"--memory {notes} --store {store} --no-register" in settings.problem()


def test_the_shared_store_and_the_shared_memo_folder_go_together(tmp_path, private_shared_store):
    """Invariant: the shared store is served only with `<shared>/memos`, and `<shared>/memos` only
    with the shared store. A dsn naming the shared store gets that memo folder, a memo_folder naming
    it with no dsn gets the shared store, and any other pairing with either is refused.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py DSN-SHARED-MEMOS`): in
    re_call_hermes/config.py, `store_dir = store.parent` became `pass`;
    the test failed at `AssertionError: assert WindowsPath('C:/Users/<user>/.../a/hermes-home/re-call/...`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py SHARED-MEMOS-AS-BLANK`): in
    re_call_hermes/config.py, `custom_memos = memos is not None and not _same_path(memos, shared_dir() / MEMO_DIR)` became `custom_memos = memos is not None`;
    the test failed at `AssertionError: assert WindowsPath('C:/Users/<user>/.../b/hermes-home/re-call/...`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py SHARED-MEMO-ROOT-GUARD`): in
    re_call_hermes/config.py, `if store is not None and _same_path(store, shared / STORE_FILE) != _same_path(` became `if False and _same_path(store, shared / STORE_FILE) != _same_path(`;
    the test failed at `AssertionError: shared store, own memos`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py GUARD-BOTH-WAYS`): in
    re_call_hermes/config.py, `_same_path(store, shared / STORE_FILE) != _same_path(` became `_same_path(store, shared / STORE_FILE) > _same_path(`;
    the test failed at `AssertionError: own store, shared memos`. Restored, green.
    """
    shared = _shared(private_shared_store)
    store, memos, notes = shared / "memory.db", shared / "memos", tmp_path / "notes"
    dsn = "sqlite:///" + store.as_posix()
    settings = load_settings(_profile(tmp_path / "a", old_store=False, dsn=dsn))
    assert settings.memo_folder == memos
    assert settings.problem() == ""
    settings = load_settings(_profile(tmp_path / "b", old_store=False, memo_folder=str(memos)))
    assert settings.store_path == store
    assert settings.problem() == ""
    own = "sqlite:///" + (tmp_path / "own.db").as_posix()
    for name, config in (("shared store, own memos", {"dsn": dsn, "memo_folder": str(notes)}),
                         ("own store, shared memos", {"dsn": own, "memo_folder": str(memos)})):
        home = _profile(tmp_path / name.replace(" ", "-").replace(",", ""), old_store=False, **config)
        assert "go together" in load_settings(home).problem(), name


NOTE = "note-0001"


def _memo(folder: Path, memo_id: str, text: str) -> None:
    """A memo file as RE-call 0.15.0 writes one: `<memo_id>.md`, frontmatter, then the text."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{memo_id}.md").write_text(f"---\nvalid_from: 2026-11-01\n---\n{text}\n", encoding="utf-8")


def _bound_on_the_old_store(harness) -> None:
    """The profile's old store, whose memo note-0001 holds the built-in entry 'Prefers tea', a
    journal mapping the entry to it, and one start on that store, which records the store's identity."""
    old = (harness.data / "memory.db", harness.data / "memos" / "hermes")
    _memo(old[1], NOTE, "Prefers tea")
    journal = {"destination": [os.path.normcase(str(p.resolve())) for p in old],
               "memos": {_digest("memory", "Prefers tea"): NOTE}, "pending": [], "submitted": []}
    path = harness.data / "mirror.json"
    path.write_text(json.dumps(journal), encoding="utf-8")
    harness.scenario({})
    provider = harness.start(dsn="")
    wait_for(lambda: "store" in json.loads(path.read_text(encoding="utf-8")), timeout=5)
    provider.shutdown()
    harness.scenario({"recall_forget_memo": [{"deleted": True}]})


def _remove_the_entry(harness):
    provider = harness.start(dsn="")
    provider.on_memory_write("remove", "memory", "", {"previous_content": "Prefers tea"})
    return provider


def _another_store_in_the_shared_dir(harness, shared: Path) -> None:
    """The old store and memos deleted, and another store in the shared directory whose own
    note-0001 holds another memo, as RE-call's sequential ids make likely."""
    (harness.data / "memory.db").unlink()
    shutil.rmtree(harness.data / "memos")
    _shared(shared)
    _memo(shared / "memos" / "hermes", NOTE, "Has a dog")


def test_the_mirror_journal_follows_the_old_store_moved_as_the_notice_says(harness, private_shared_store):
    """Invariant: after the old store and memos/ are moved to the shared directory, the profile's
    mirror journal follows them, because its memo ids still name its entries' memos there.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py MIRROR-REBIND`): in
    re_call_hermes/provider.py, `if found is None: raise ToolError("mirror journal names...` became `if True: raise ToolError("mirror journal names...`;
    the test failed at `assert False` (no forget call within 8 s`. Restored, green.
    """
    _bound_on_the_old_store(harness)
    private_shared_store.mkdir(parents=True)
    shutil.move(harness.data / "memory.db", private_shared_store / "memory.db")
    shutil.move(harness.data / "memos", private_shared_store / "memos")
    _remove_the_entry(harness)
    assert wait_for(lambda: harness.events("call"), timeout=8)
    assert harness.events("call")[0]["arguments"] == {"memo_id": NOTE}


def test_a_journal_does_not_follow_a_deleted_store_to_another_one(harness, private_shared_store):
    """Invariant: vanished old paths are no evidence of a move. With the old store deleted and
    another store in the shared directory, whose note-0001 holds other text, the journal is refused.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py MAPPING-CONTRADICTION`): in
    re_call_hermes/provider.py, `if _digest(key.rpartition(":")[0], text) != key: return None` became `if False: return None`;
    the test failed at `AssertionError: assert [{'event': 'c...'note-0001'}}] == []`. Restored, green.
    """
    _bound_on_the_old_store(harness)
    _another_store_in_the_shared_dir(harness, private_shared_store)
    provider = _remove_the_entry(harness)
    assert provider._settings.store_path == private_shared_store / "memory.db"
    provider.shutdown()
    assert harness.events("call") == []


def test_a_reused_file_number_is_no_evidence_of_a_move(harness, private_shared_store, monkeypatch):
    """Invariant: a store file whose volume and file number match the recorded ones (a freed inode
    reused, as ext4 and tmpfs do) is still checked by its memos, and refused when they differ.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py IDENTITY-TRUSTED`): in
    re_call_hermes/provider.py, `changed = recorded != destination or state["store"] not in (None, identity)` became `changed = state["store"] not in (None, identity)`;
    the test failed at `AssertionError: assert [{'event': 'c...'note-0001'}}] == []`. Restored, green.
    """
    from re_call_hermes import provider as module

    monkeypatch.setattr(module, "_store_identity", lambda store: "2049:1311")
    _bound_on_the_old_store(harness)
    _another_store_in_the_shared_dir(harness, private_shared_store)
    provider = _remove_the_entry(harness)
    provider.shutdown()
    assert harness.events("call") == []


def test_another_store_at_the_same_path_is_checked(harness, monkeypatch):
    """Invariant: when the store file at the journal's own path is another one (its recorded
    identity changed) and its note-0001 holds other text, the journal is refused, not applied.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py MIRROR-STAMP`): in
    re_call_hermes/provider.py, `atomic_json_write(path, state)  # records the identity a later move is recognised by` became `pass`;
    the test failed at `AssertionError: assert [{'event': 'c...'note-0001'}}] == []`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py SAME-PATH`): in
    re_call_hermes/provider.py, `changed = recorded != destination or state["store"] not in (None, identity)` became `changed = recorded != destination`;
    the test failed at `AssertionError: assert [{'event': 'c...'note-0001'}}] == []`. Restored, green.
    """
    from re_call_hermes import provider as module

    identity = {"value": "2049:1311"}
    monkeypatch.setattr(module, "_store_identity", lambda store: identity["value"])
    _bound_on_the_old_store(harness)
    identity["value"] = "2049:4242"  # the store deleted and set up again at the same path
    _memo(harness.data / "memos" / "hermes", NOTE, "Has a dog")
    provider = _remove_the_entry(harness)
    assert provider._settings.store_path == harness.data / "memory.db"
    provider.shutdown()
    assert harness.events("call") == []


def test_a_store_moved_without_its_memo_folder_is_not_followed(harness, private_shared_store):
    """Invariant: the journal follows a store only when at least one of its memos is found there; a
    store moved without its memos is refused rather than mirrored into a folder that lacks them.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py MAPPING-FOUND`): in
    re_call_hermes/provider.py, `return (held, unread) if held or not memos else None` became `return (held, unread)`;
    the test failed at `AssertionError: assert {} == {'memory:dbe4...: 'note-0001'}`. Restored, green.
    """
    _bound_on_the_old_store(harness)
    private_shared_store.mkdir(parents=True)
    shutil.move(harness.data / "memory.db", private_shared_store / "memory.db")
    provider = _remove_the_entry(harness)
    assert provider._settings.store_path == private_shared_store / "memory.db"
    provider.shutdown()
    assert harness.events("call") == []
    journal = json.loads((harness.data / "mirror.json").read_text(encoding="utf-8"))
    assert journal["memos"] == {_digest("memory", "Prefers tea"): NOTE}  # kept for reconciliation


def test_recording_the_identity_keeps_the_journal_as_read(harness):
    """Invariant: recording the store identity at a start rewrites only the binding fields, so a
    malformed pending entry, which the normalized journal drops, stays on disk for reconciliation.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py STAMP-KEEPS-RAW`): in
    re_call_hermes/provider.py, `atomic_json_write(path, {**raw, **binding})` became `atomic_json_write(path, {**state, **binding})`;
    the test failed at `AssertionError: assert [] == [['malformed']]`. Restored, green.
    """
    path = harness.data / "mirror.json"
    here = (harness.data / "memory.db", harness.data / "memos" / "hermes")
    path.write_text(json.dumps({"destination": [os.path.normcase(str(p.resolve())) for p in here],
                                "memos": {}, "pending": [["malformed"]], "submitted": []}), encoding="utf-8")
    harness.scenario({})
    harness.start()
    assert wait_for(lambda: "store" in json.loads(path.read_text(encoding="utf-8")), timeout=5)
    assert json.loads(path.read_text(encoding="utf-8"))["pending"] == [["malformed"]]


def test_a_move_needs_no_earlier_start_on_the_old_store(harness, private_shared_store):
    """Invariant: a journal that never recorded a store identity (no start on this version before the
    move) still follows a moved store, because the move is checked by the journal's memos.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py REBIND-NEEDS-NO-STAMP`): in
    re_call_hermes/provider.py, `if found is None:` became `if found is None or state["store"] is None:`;
    the test failed at `assert False` (no forget call within 8 s`. Restored, green.
    """
    old = (harness.data / "memory.db", harness.data / "memos" / "hermes")
    _memo(old[1], NOTE, "Prefers tea")
    (harness.data / "mirror.json").write_text(json.dumps({
        "destination": [os.path.normcase(str(p.resolve())) for p in old],
        "memos": {_digest("memory", "Prefers tea"): NOTE}, "pending": [], "submitted": []}), encoding="utf-8")
    private_shared_store.mkdir(parents=True)
    shutil.move(harness.data / "memory.db", private_shared_store / "memory.db")
    shutil.move(harness.data / "memos", private_shared_store / "memos")
    harness.scenario({"recall_forget_memo": [{"deleted": True}]})
    _remove_the_entry(harness)
    assert wait_for(lambda: harness.events("call"), timeout=8)
    assert harness.events("call")[0]["arguments"] == {"memo_id": NOTE}


def test_a_missing_identity_keeps_the_recorded_one(harness, monkeypatch):
    """Invariant: a start that cannot read the store file's identity (a failed stat, a filesystem
    without file numbers) leaves the recorded identity in the journal, so a store replaced later at
    the same path is still checked.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py NONE-KEEPS-IDENTITY`): in
    re_call_hermes/provider.py, `state["store"] = identity if identity is not None else state["store"]` became `state["store"] = identity`;
    the test failed at `AssertionError: assert None == '2049:1311'`. Restored, green.
    """
    from re_call_hermes import provider as module

    identity = {"value": "2049:1311"}
    monkeypatch.setattr(module, "_store_identity", lambda store: identity["value"])
    _bound_on_the_old_store(harness)
    identity["value"] = None
    harness.start(dsn="").shutdown()
    journal = json.loads((harness.data / "mirror.json").read_text(encoding="utf-8"))
    assert journal["store"] == "2049:1311"


def _journal_with_an_absent_memo(harness, shared: Path) -> tuple[Path, str]:
    """A journal bound to the deleted old store, mapping 'Prefers tea' to note-0001 and 'Has a cat' to
    note-0002, and a shared store whose note-0001 holds the same entry (another profile mirrored it)
    and which has no note-0002 yet."""
    old = (harness.data / "memory.db", harness.data / "memos" / "hermes")
    cat = _digest("memory", "Has a cat")
    path = harness.data / "mirror.json"
    path.write_text(json.dumps({
        "destination": [os.path.normcase(str(p.resolve())) for p in old],
        "memos": {_digest("memory", "Prefers tea"): NOTE, cat: "note-0002"},
        "pending": [], "submitted": []}), encoding="utf-8")
    (harness.data / "memory.db").unlink()  # the old store deleted, not moved
    _memo(_shared(shared) / "memos" / "hermes", NOTE, "Prefers tea")
    return path, cat


def test_a_rebound_journal_drops_mappings_whose_memo_is_absent(harness, private_shared_store):
    """Invariant: when a journal is rebound to a store where some mapped memos are absent, those
    mappings are dropped and the drop is saved, since RE-call gives a later memo the next free id and
    a stale mapping would then forget it.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py PRUNE-PERSIST`): in
    re_call_hermes/provider.py, `binding["memos"] = held` became `pass`;
    the test failed at `AssertionError: assert [{'event': 'c...'note-0002'}}] == []`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py UNCONFIRMED-CHECKED`): in
    re_call_hermes/provider.py, `elif _digest(target, text) == old_key: memos[old_key] = candidate` became `else: memos[old_key] = candidate`;
    the test failed at `AssertionError: assert [{'event': 'c...'note-0002'}}] == []`. Restored, green.
    """
    path, cat = _journal_with_an_absent_memo(harness, private_shared_store)
    harness.scenario({"recall_forget_memo": [{"deleted": True}]})
    provider = harness.start(dsn="")
    wait_for(lambda: cat not in json.loads(path.read_text(encoding="utf-8"))["memos"], timeout=5)
    _memo(private_shared_store / "memos" / "hermes", "note-0002", "Has a dog")  # the next memo written
    provider.on_memory_write("remove", "memory", "", {"previous_content": "Has a cat"})
    provider.shutdown()
    assert harness.events("call") == []


def test_a_journaled_write_that_rebinds_keeps_the_drop(harness, private_shared_store):
    """Invariant: when the first bind after a move is a journaled write (the server was not ready,
    so no drain ran first), the write saves the journal without the dropped mappings.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py PRUNE-ABSENT`): in
    re_call_hermes/provider.py, `state["memos"] = held  # so only ...` became `pass  # so only ...`;
    the test failed at `AssertionError: assert 'memory:5ab331c2...' not in {'memory:5ab331c2...: 'note-0002', ...}`. Restored, green.
    """
    from re_call_hermes.provider import ReCallMemoryProvider

    path, cat = _journal_with_an_absent_memo(harness, private_shared_store)
    (harness.data / "config.json").write_text(json.dumps({**harness.config, "dsn": ""}), encoding="utf-8")
    provider = ReCallMemoryProvider()
    provider._settings = load_settings(harness.home)
    assert provider._journal("add", "memory", "Likes jazz", {"_request_id": "r1"})
    assert cat not in json.loads(path.read_text(encoding="utf-8"))["memos"]


def test_a_memo_in_place_after_the_rebind_is_still_managed(harness, private_shared_store, tmp_path):
    """Invariant: a mapping whose memo could not be read at the rebind (moved in pieces, held open)
    is kept unconfirmed, and a later remove of its entry still forgets the memo once it is in place
    holding the entry's text.

    Red proof (2026-11-14, local Windows, `python tests/mutations.py UNCONFIRMED-USED`): in
    re_call_hermes/provider.py, `memos[old_key] = candidate  # confirmed ...` became `pass  # confirmed ...`;
    the test failed at `assert False` (no forget call within 8 s`. Restored, green.
    Red proof (2026-11-14, local Windows, `python tests/mutations.py UNCONFIRMED-PERSIST`): in
    re_call_hermes/provider.py, `binding["unconfirmed"] = state["unconfirmed"]` became `(deleted)`;
    the test failed at `assert False` (no forget call within 8 s`. Restored, green.
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
    harness.scenario({"recall_forget_memo": [{"deleted": True}]})
    provider = harness.start(dsn="")
    wait_for(lambda: cat in json.loads(path.read_text(encoding="utf-8")).get("unconfirmed", {}), timeout=5)
    shutil.move(tmp_path / "note-0002.md", private_shared_store / "memos" / "hermes" / "note-0002.md")
    provider.on_memory_write("remove", "memory", "", {"previous_content": "Has a cat"})
    assert wait_for(lambda: harness.events("call"), timeout=8)
    assert harness.events("call")[0]["arguments"] == {"memo_id": "note-0002"}


def test_drive_rooted_xdg_is_ignored_on_older_windows_python(monkeypatch):
    """The shared default uses the same absolute path rule as OpenCode on every Python version."""
    from pathlib import PureWindowsPath
    from types import SimpleNamespace

    from re_call_hermes import config

    class WindowsPath(PureWindowsPath):
        @classmethod
        def home(cls):
            return cls("C:/Users/sample")

    monkeypatch.setattr(config, "Path", WindowsPath)
    monkeypatch.setattr(config, "os", SimpleNamespace(
        environ={"XDG_CONFIG_HOME": "/drive-rooted"}, path=SimpleNamespace(isabs=lambda _: True),
    ))
    assert config.shared_dir() == WindowsPath("C:/Users/sample/.config/re-call")
