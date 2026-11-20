"""Tests for the CCA run 20261108T171507Z fixes, one per CONFIRMED finding (ids in each docstring)."""

from __future__ import annotations

import json
import os
import stat
import sys
import threading
import tomllib
from pathlib import Path

from conftest import ROOT, wait_for

from re_call_hermes import provider as provider_module
from re_call_hermes.config import load_settings
from re_call_hermes.mcp_client import McpError


def _write_config(home: Path, **values) -> None:
    (home / "re-call").mkdir(parents=True, exist_ok=True)
    (home / "re-call" / "config.json").write_text(json.dumps(values), encoding="utf-8")


def _read_mirror(path: Path) -> dict:
    """mirror.json as the test sees it. On Windows a read can meet the provider's os.replace."""
    for _ in range(50):
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (PermissionError, ValueError):
            threading.Event().wait(0.02)
    return {}


def _journal(settings, **values):
    # Fixture provenance for a journal created under this exact destination.
    return {"destination": [os.path.normcase(str(p.resolve())) for p in
                            (settings.store_path, settings.assistant_folder)],
            "memos": {}, "pending": [], "submitted": [], **values}


def _plant(directory: Path, name: str) -> Path:
    path = directory / (name + (".exe" if os.name == "nt" else ""))
    path.write_bytes(b"planted")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_the_server_command_is_never_taken_from_the_working_directory(tmp_path, monkeypatch):
    """Invariant (SEC-001, P1): a `recall-mcp` in the directory Hermes runs in is never chosen;
    only absolute PATH folders are searched.

    The cwd hit is reached two ways the fix closes: Windows' own search, which tries the current
    directory first unless NoDefaultCurrentDirectoryInExePath is set, and a relative PATH entry.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    SEC-001-CWD`): in re_call_hermes/config.py, `found = _which_on_path(argv[0])` became `found =
    __import__('shutil').which(argv[0])`; the test failed at `AssertionError: assert (['.//recall-mcp.EXE']
    is not None and False)`. Restored, green.
    """
    project, real_bin = tmp_path / "project", tmp_path / "bin"
    project.mkdir()
    real_bin.mkdir()
    _plant(project, "recall-mcp")
    real = _plant(real_bin, "recall-mcp")
    monkeypatch.delenv("NoDefaultCurrentDirectoryInExePath", raising=False)
    monkeypatch.setenv("PATH", os.pathsep.join([".", str(real_bin)]))
    monkeypatch.chdir(project)
    home = tmp_path / "home"
    _write_config(home)
    argv = load_settings(home).resolve_argv()
    assert argv is not None and Path(argv[0]).is_absolute()
    assert Path(argv[0]).resolve() == real.resolve()


def test_a_relative_server_path_is_refused(tmp_path, monkeypatch):
    """Invariant (SEC-001, P1): a recall_command naming a file by a relative path is not resolved
    against Hermes's working directory.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    SEC-001-REL`): in re_call_hermes/config.py, `if whole.is_absolute() and whole.is_file():` became `if
    whole.is_file():`; the test failed at `AssertionError: assert ['tools//server.exe'] is None`. Restored,
    green.
    """
    project = tmp_path / "project"
    (project / "tools").mkdir(parents=True)
    _plant(project / "tools", "server")
    monkeypatch.chdir(project)
    home = tmp_path / "home"
    name = "server.exe" if os.name == "nt" else "server"
    _write_config(home, recall_command=str(Path("tools") / name))
    assert load_settings(home).resolve_argv() is None


def test_a_password_in_any_dsn_form_is_refused(tmp_path):
    """Invariant (SEC-002): a password in a URL query or a libpq keyword DSN is refused like one
    in the URL user info.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py SEC-002`): in
    re_call_hermes/config.py, `(urlsplit(self.dsn).password or _DSN_PASSWORD.search(self...` became
    `(urlsplit(self.dsn).password)`; the test failed at `AssertionError: host=db user=u password=x
    dbname=recall`. Restored, green.
    """
    for dsn in ("host=db user=u password=x dbname=recall", "postgresql://u@db/recall?sslmode=require&password=x"):
        _write_config(tmp_path, recall_command=sys.executable, dsn=dsn)
        assert "password" in load_settings(tmp_path).problem(), dsn


def test_mirror_builtin_false_from_the_setup_wizard_turns_mirroring_off(tmp_path):
    """Invariant (ENV-001): `hermes memory setup` saves "false" as a string; it must read as False,
    and a value that is not a boolean is refused with a reason.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py ENV-001`): in
    re_call_hermes/config.py, `mirror_builtin = _as_bool(values["mirror_builtin"])` became `mirror_builtin =
    bool(values["mirror_builtin"])`; the test failed at `AssertionError: assert True is False`. Restored,
    green.
    """
    _write_config(tmp_path, recall_command=sys.executable, mirror_builtin="false")
    assert load_settings(tmp_path).mirror_builtin is False
    _write_config(tmp_path, recall_command=sys.executable, mirror_builtin="maybe")
    assert "not valid" in load_settings(tmp_path).problem()


def test_relative_store_and_memo_paths_resolve_under_the_data_folder(tmp_path, monkeypatch):
    """Invariant (ENV-002): the provider and the child (cwd = data folder) agree on where a relative
    memo_folder or sqlite path is, because both are made absolute under the data folder.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py ENV-002`): in
    re_call_hermes/config.py, `return path if path.is_absolute() else base / path` became `return path`; the
    test failed at `AssertionError: assert WindowsPath('notes') == (WindowsPath('C:/Users/<user>/A...`
    (account name redacted). Restored, green.
    """
    monkeypatch.chdir(tmp_path)
    _write_config(tmp_path, recall_command=sys.executable, memo_folder="notes", dsn="sqlite:///store/m.db")
    settings = load_settings(tmp_path)
    data = tmp_path / "re-call"
    assert settings.memo_folder == data / "notes"
    assert settings.store_path == data / "store" / "m.db"
    env = settings.child_env({})
    assert Path(env["RECALL_INDEX_ROOT"]).is_absolute()


def test_the_child_gets_proxy_and_model_cache_variables_and_a_profile_embedding_cache(tmp_path):
    """Invariant (ENV-004, DOC-001): proxy, CA, TMPDIR and model cache variables reach the child, and
    its embedding cache (vectors of memo text) is under the profile's data folder.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py ENV-004`): in
    re_call_hermes/config.py, `"HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "...` became
    `(deleted)`; the test failed at `AssertionError: HTTPS_PROXY`. Restored, green.
    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py DOC-001`): in
    re_call_hermes/config.py, `RECALL_EMBED_CACHE=str(self.data_dir / EMBED_CACHE_FILE),` became
    `(deleted)`; the test failed at `AssertionError: assert None ==
    'C://Users//<user>//AppData//Local//Temp//pytes...` (account name redacted). Restored, green.
    """
    _write_config(tmp_path, recall_command=sys.executable)
    base = {"HTTPS_PROXY": "http://proxy:3128", "SSL_CERT_FILE": "/ca.pem", "TMPDIR": "/t", "RECALL_MODEL_CACHE": "/m"}
    env = load_settings(tmp_path).child_env(base)
    for key, value in base.items():
        assert env.get(key) == value, key
    assert env.get("RECALL_EMBED_CACHE") == str(tmp_path / "re-call" / "embeddings.sqlite")


def test_the_plugin_declares_nothing_hermes_would_install():
    """Invariant (DEP-001): Hermes's package manager treats a [build-system] table, optional
    dependencies or dependency groups as "this plugin installs packages"
    (pm/plugin_declarations.py `installs`); none may be present.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py DEP-001`): in
    pyproject.toml, `[project] name = "recall-hermes"` became `[build-system] requires = ["setuptools"]
    [proje...`; the test failed at `AssertionError: assert 'build-system' not in {'build-system':
    {'requires': ['...`. Restored, green.
    """
    document = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    project = document.get("project", {})
    assert "build-system" not in document
    assert not any(project.get("optional-dependencies", {}).values())
    assert not document.get("dependency-groups")
    assert project.get("dependencies") == []


def test_whitespace_around_an_entry_does_not_strand_its_memo(harness):
    """Invariant (DAT-002): Hermes stores entries stripped and names the stored text in
    previous_content, so an add with surrounding whitespace is still forgotten by its remove.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py DAT-002`): in
    re_call_hermes/provider.py, `content = content.strip() previous = previous.strip() if ...` became
    `(deleted)`; the test failed at `assert False`. Restored, green.
    """
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}], "recall_forget_memo": [{"deleted": True}]})
    provider = harness.start()
    provider.on_memory_write("add", "memory", "Prefers tea\n")
    provider.on_memory_write("remove", "memory", "", {"previous_content": "Prefers tea"})
    assert wait_for(lambda: any(c["tool"] == "recall_forget_memo" for c in harness.events("call")), timeout=8)
    calls = harness.events("call")
    assert calls[0]["arguments"]["text"] == "Prefers tea"
    assert calls[1]["arguments"] == {"memo_id": "m1.md"}


def test_two_instances_on_one_profile_keep_each_others_mirror_entries(harness, monkeypatch):
    """Invariant (DAT-003): two provider instances (two gateway sessions) mirroring into one
    profile at the same time both end up in mirror.json.

    Red proof: NOT ESTABLISHED. A regression guard only. Since the L5.5 rework the lock covers
    millisecond reads and writes of mirror.json and every settle re-reads the file, so removing
    the lock leaves a race this test cannot hit on demand. The serialisation itself is covered by
    the nonbehavioral control test_the_lock_serialises_writers_within_one_process, and the
    cross-instance settle by the forced interleaving in the next test, which is red-proved.
    """
    from re_call_hermes.provider import ReCallMemoryProvider

    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}]}, delay={"recall_remember": 1.0})
    first = harness.start()
    second = ReCallMemoryProvider()
    second.initialize("s2", hermes_home=str(harness.home), platform="cli", agent_context="primary")
    try:
        assert wait_for(second._ready.is_set)
        first.on_memory_write("add", "user", "Lives in Rome")
        second.on_memory_write("add", "user", "Has a cat")
        mirror = harness.data / "mirror.json"

        def both():
            state = _read_mirror(mirror)
            return len(state.get("memos", {})) == 2 and not state.get("pending")

        assert wait_for(both, timeout=15)
    finally:
        second.shutdown()


def test_an_inflight_write_cannot_be_replayed_by_another_instance(tmp_path):
    """BUG-001 replaces concurrent replay with an exclusive durable submission claim.

    Red proof: removing the submitted check permits B's second call, so the blocked assertion fails.
    Once A settles, B may drain the queued removal normally.
    """
    _write_config(tmp_path, recall_command=sys.executable)
    settings = load_settings(tmp_path)
    path = settings.data_dir / provider_module.MIRROR_FILE
    path.write_text(json.dumps(_journal(settings, pending=[
        ["r1", "add", "user", "Has a cat", None], ["r2", "remove", "user", "", "Has a cat"]])), encoding="utf-8")
    a, b = provider_module.ReCallMemoryProvider(), provider_module.ReCallMemoryProvider()
    a._settings = b._settings = settings
    forgotten = []
    blocked = []

    def b_call(tool, arguments, timeout):
        if tool == "recall_forget_memo":
            forgotten.append(arguments["memo_id"])
            return {"deleted": True}
        return {"memo_id": "m1.md"}

    def a_call(tool, arguments, timeout):
        try:
            b._drain_pending()
        except McpError:
            blocked.append(True)
        return {"memo_id": "m1.md"}

    b._call = b_call
    a._call = a_call
    a._apply(path, "r1", "add", "user", "Has a cat", None)
    b._drain_pending()
    state = _read_mirror(path)
    assert blocked == [True]
    assert forgotten == ["m1.md"]
    assert state["memos"] == {} and state["pending"] == []


def test_a_late_settle_does_not_undo_another_instances_later_write(tmp_path):
    """A stale completion cannot overwrite a mapping already advanced by another writer.

    Red proof (Linux host, 2026-11-08): `python tests/mutations.py DAT-003-SETTLE` removed the
    stale-completion guard, overwrote the newer mapping and failed the state assertion. Restored, green.
    """
    _write_config(tmp_path, recall_command=sys.executable)
    provider = provider_module.ReCallMemoryProvider()
    provider._settings = settings = load_settings(tmp_path)
    path = settings.data_dir / provider_module.MIRROR_FILE
    state = _journal(settings, memos={"entry": "newer.md"})
    path.write_text(json.dumps(state), encoding="utf-8")
    provider._settle(path, "already-settled", new_key="entry", new_memo="older.md")
    assert _read_mirror(path) == state


def test_an_unsubmitted_journaled_write_is_completed_with_its_request_id(harness):
    """BUG-001 control: recovery remains valid before any server submission.

    DAT-004-REPLAY mutation still removes startup draining and fails the memo assertion.
    Unknown-outcome recovery is covered by test_pr1_regressions instead.
    """
    harness.scenario({"recall_remember": [{"memo_id": "m7.md"}]})
    _write_config(harness.home, **harness.config)
    state = _journal(load_settings(harness.home), pending=[["abc123", "add", "user", "Lives in Rome", None]])
    (harness.data / "mirror.json").write_text(json.dumps(state), encoding="utf-8")
    provider = harness.start()
    assert wait_for(lambda: "m7.md" in json.dumps(_read_mirror(harness.data / "mirror.json")), timeout=10)
    provider.shutdown()
    state = _read_mirror(harness.data / "mirror.json")
    assert list(state["memos"].values()) == ["m7.md"]
    assert state["pending"] == []
    assert harness.events("call")[0]["arguments"]["request_id"] == "hermes-abc123"


def test_a_refused_correction_still_stores_the_new_text(harness):
    """Invariant (DAT-005): when RE-call refuses the supersedes of a mirrored replace (the memo was
    forgotten or corrected through the model's own tools), the new text is stored on its own.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py DAT-005`): in
    re_call_hermes/provider.py, `if isinstance(answer, dict) and answer.get("conflict") in...` became `if
    False:`; the test failed at `assert False`. Restored, green.
    """
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}, {"refused": "no memo", "conflict": "target_missing"},
                                         {"memo_id": "m2.md"}]})
    provider = harness.start()
    provider.on_memory_write("add", "user", "Lives in Paris")
    provider.on_memory_write("replace", "user", "Lives in Rome", {"previous_content": "Lives in Paris"})
    mirror = harness.data / "mirror.json"
    assert wait_for(lambda: "m2.md" in json.dumps(_read_mirror(mirror)), timeout=10)
    calls = [c["arguments"] for c in harness.events("call")]
    assert calls[1]["supersedes"] == "m1.md" and "supersedes" not in calls[2]
    assert calls[2]["text"] == "Lives in Rome"



def test_the_lock_serialises_writers_within_one_process(tmp_path):
    """Nonbehavioral control for _locked itself (DAT-003): two threads cannot hold it at once. The
    behaviour test above carries the red proof."""
    order: list[str] = []
    path = tmp_path / "mirror.json"

    def hold(name):
        with provider_module._locked(path):
            order.append(name + "+")
            threading.Event().wait(0.2)
            order.append(name + "-")

    threads = [threading.Thread(target=hold, args=(n,)) for n in "ab"]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert order in (["a+", "a-", "b+", "b-"], ["b+", "b-", "a+", "a-"])


def test_an_unknown_write_blocks_a_later_replace(harness, monkeypatch):
    """BUG-001: a later replace cannot overtake a write with an unknown outcome.

    Red proof (Linux host, 2026-11-08): `python tests/mutations.py MIRROR-ORDER` removed the submitted
    guard and failed at two calls versus one. Restored, green. Normal replacement remains
    covered by test_a_refused_correction_still_stores_the_new_text and test_provider.
    """
    monkeypatch.setattr(provider_module, "TOOL_TIMEOUT_S", 0.2)
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}, {"memo_id": "m2.md"}]},
                     delay_first={"recall_remember": 0.6})
    provider = harness.start()
    provider.on_memory_write("add", "user", "Lives in Paris")
    provider.on_memory_write("replace", "user", "Lives in Rome", {"previous_content": "Lives in Paris"})
    provider.shutdown()
    state = _read_mirror(harness.data / "mirror.json")
    assert len(harness.events("call")) == 1
    assert len(state["pending"]) == 2 and not state["memos"]


def test_an_unbound_legacy_mirror_is_preserved_without_deletion(harness):
    """BUG-002 changes legacy migration: absent provenance cannot authorize a deletion.

    Red proof (Linux host, 2026-11-08): `python tests/mutations.py MIRROR-DESTINATION` bypassed the
    destination refusal and failed on an unwanted forget call. Restored, green. Same-destination
    bound recovery remains covered by the PR regression control.
    """
    from re_call_hermes.provider import _digest

    harness.scenario({"recall_forget_memo": [{"deleted": True}]})
    state = {_digest("user", "Has a cat"): {"memo_id": "m9.md", "entries": 1}}
    path = harness.data / "mirror.json"
    path.write_text(json.dumps(state), encoding="utf-8")
    provider = harness.start()
    provider.on_memory_write("remove", "user", "", {"previous_content": "Has a cat"})
    provider.shutdown()
    assert harness.events("call") == []
    assert _read_mirror(path) == state


def test_a_malformed_journal_entry_does_not_stop_the_writer(harness):
    """Nonbehavioral control (DAT-004 rework): a malformed pending entry is dropped, and the next
    write is mirrored. Shape validation only; the replay tests carry the red proofs."""
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}]})
    garbage = {"memos": {}, "pending": [["x"], "garbage"]}
    (harness.data / "mirror.json").write_text(json.dumps(garbage), encoding="utf-8")
    provider = harness.start()
    provider.on_memory_write("add", "user", "Marker")
    assert wait_for(lambda: [c["arguments"]["text"] for c in harness.events("call")] == ["Marker"], timeout=8)


def test_a_refusal_that_is_not_a_target_conflict_is_final(harness):
    """Invariant (DAT-005 rework): only a refused supersedes target (target_missing, stale_target,
    revision_mismatch) falls back to a plain remember; `memo_missing`, a memo the user forgot since,
    must not be re-created.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    DAT-005-FINAL`): in re_call_hermes/provider.py, `if isinstance(answer, dict) and answer.get("conflict")
    in...` became `if isinstance(answer, dict) and answer.get("ref...`; the test failed at `AssertionError:
    assert [None, 'm1.md', None] == [None, 'm1.md']`. Restored, green.
    """
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}, {"refused": "forgotten", "conflict": "memo_missing"}]})
    provider = harness.start()
    provider.on_memory_write("add", "user", "Lives in Paris")
    provider.on_memory_write("replace", "user", "Lives in Rome", {"previous_content": "Lives in Paris"})
    mirror = harness.data / "mirror.json"

    def settled():
        state = _read_mirror(mirror)
        return len(harness.events("call")) >= 2 and not state.get("pending")

    assert wait_for(settled, timeout=10)
    import time as _time

    _time.sleep(0.5)
    assert [c["arguments"].get("supersedes") for c in harness.events("call")] == [None, "m1.md"]


def test_an_unknown_write_is_retained_and_blocks_later_ones(harness, monkeypatch):
    """BUG-001 changes give-up behavior: unresolved writes remain explicit and block the journal.

    Red proof (Linux host, 2026-11-08): `python tests/mutations.py MIRROR-UNKNOWN` settled unknown
    outcomes and failed at four calls versus one. Restored, green.
    Definitive refusal remains covered by test_a_refusal_that_is_not_a_target_conflict_is_final.
    """
    monkeypatch.setattr(provider_module, "TOOL_TIMEOUT_S", 0.1)
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}]}, no_answer_text=["Poison"])
    provider = harness.start()
    for text in ("Poison", "Second", "Third", "Fourth"):
        provider.on_memory_write("add", "user", text)
    provider.shutdown()
    state = _read_mirror(harness.data / "mirror.json")
    assert len(harness.events("call")) == 1
    assert [op[3] for op in state["pending"]] == ["Poison", "Second", "Third", "Fourth"]
    assert state["memos"] == {}


def test_a_journaled_write_another_instance_applied_is_not_applied_again(harness):
    """Invariant (DAT-004 rework): _apply re-checks under the lock that its write is still pending,
    so a stale snapshot never replays a write another instance settled.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    DAT-004-RECHECK`): in re_call_hermes/provider.py, `if all(op[0] != request_id for op in
    state["pending"]): r...` became `(deleted)`; the test failed at `AssertionError: assert [{'event':
    'c...d': 'm6.md'}}] == []`. Restored, green.
    """
    from re_call_hermes.provider import _digest

    harness.scenario({"recall_forget_memo": [{"deleted": True}]})
    provider = harness.start()
    mirror = harness.data / "mirror.json"
    mirror.write_text(json.dumps(_journal(provider._settings, memos={_digest("user", "Has a cat"): "m6.md"})),
                      encoding="utf-8")
    provider._apply(mirror, "settled-elsewhere", "remove", "user", "", "Has a cat")
    assert harness.events("call") == []
