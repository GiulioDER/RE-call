"""Tests for the CCA run 20261112T120535Z fixes, one per CONFIRMED finding (ids in each docstring)."""

from __future__ import annotations

import json
import sys
import threading
import time
from pathlib import Path

from conftest import wait_for

from re_call_hermes import provider as provider_module
from re_call_hermes.config import load_settings
from re_call_hermes.provider import ReCallMemoryProvider

SEARCH_OK = {
    "abstained": False,
    "evidence": {"decision": "answer", "citable": ["c1"]},
    "hits": [{"memo_id": "job.md", "chunk_id": "c1", "verdict": "ok", "text": "Works at Acme"}],
}
# RE-call v0.15.0 recall_mcp/tool_surface.py:82, the `search` preset a non-Lite child runs.
RECALL_SEARCH_PRESET = {"recall_search", "recall_evidence"}


def _write_config(home: Path, **values) -> None:
    (home / "re-call").mkdir(parents=True, exist_ok=True)
    (home / "re-call" / "config.json").write_text(json.dumps(values), encoding="utf-8")


def _read(path: Path) -> dict:
    """mirror.json as the test sees it. On Windows a read can meet the provider's os.replace."""
    for _ in range(50):
        try:
            return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
        except (PermissionError, ValueError):
            threading.Event().wait(0.02)
    return {}


def _calls(harness, tool):
    return [c["arguments"] for c in harness.events("call") if c["tool"] == tool]


def test_a_write_the_dead_server_never_got_is_applied_once_it_is_back(harness, monkeypatch, caplog):
    """Invariant (BUG-001): a mirrored write refused before it was sent (the child had died) is not
    an unknown outcome: it stays pending, unclaimed, and is applied in order once the server is back.

    Red proof (2026-11-12, CCA run 20261112T120535Z, local Windows, `cca_checks tautology verify` against
    the pre-fix snapshot, and the same test run in a checkout of 86fdc52): with re_call_hermes/provider.py
    `_apply` and `_call` as they were (no NotSent, the write left in `submitted`), the test failed at
    `assert False` where `False = wait_for(...)`: the replay was refused and no recall_remember arrived.
    Restored, green.
    """
    harness.scenario({"recall_search": [SEARCH_OK], "recall_remember": [{"memo_id": "m1.md"}, {"memo_id": "m2.md"}]},
                     exit_on="recall_evidence")
    provider = harness.start()
    provider.handle_tool_call("recall_evidence", {"query": "x"})  # the scripted server exits on this call
    assert wait_for(lambda: not provider._client.alive)
    assert provider._ready.is_set()  # nothing has noticed the dead child yet: the case this is about
    provider.on_memory_write("add", "user", "Lives in Rome")
    assert wait_for(lambda: "could not mirror a built-in memory add" in caplog.text)

    monkeypatch.setattr(provider_module, "RESTART_INTERVAL_S", 0.0)
    provider.prefetch("where do I work?")  # notices the dead child and starts a new one
    assert wait_for(lambda: provider._ready.is_set() and provider._client.alive)
    provider.on_memory_write("add", "user", "Has a cat")
    assert wait_for(lambda: len(_calls(harness, "recall_remember")) == 2, timeout=10)
    assert [a["text"] for a in _calls(harness, "recall_remember")] == ["Lives in Rome", "Has a cat"]
    state = _read(harness.data / "mirror.json")
    assert sorted(state["memos"].values()) == ["m1.md", "m2.md"] and state["submitted"] == []


def test_a_write_taken_while_the_server_restarts_is_journaled_and_replayed(harness, monkeypatch):
    """Invariant (BUG-002): a built-in memory write the writer takes while the server is not ready
    reaches mirror.json before any wait, so shutdown cannot drop it and the next start applies it.

    Red proof (2026-11-12, CCA run 20261112T120535Z, local Windows, `cca_checks tautology verify` against
    the pre-fix snapshot, and the same test run in a checkout of 86fdc52): with re_call_hermes/provider.py
    `_write_loop` dropping the write before journaling it, the test failed at `AssertionError: assert [] ==
    ['remove']`. Restored, green.
    """
    tools = {"recall_remember": [{"memo_id": "m1.md"}], "recall_forget_memo": [{"deleted": 1}]}
    harness.scenario(tools, exit_on="recall_evidence")
    provider = harness.start()
    mirror = harness.data / "mirror.json"
    provider.on_memory_write("add", "user", "Lives in Paris")
    assert wait_for(lambda: "m1.md" in _read(mirror).get("memos", {}).values())
    provider.handle_tool_call("recall_evidence", {"query": "x"})  # the scripted server exits on this call
    assert wait_for(lambda: not provider._client.alive)

    harness.scenario(tools, exit_on="recall_evidence", init_delay=5)  # the restart takes a while
    monkeypatch.setattr(provider_module, "RESTART_INTERVAL_S", 0.0)
    provider._maybe_restart()
    assert provider._starting and not provider._ready.is_set()
    provider.on_memory_write("remove", "user", "", {"previous_content": "Lives in Paris"})
    provider.shutdown()
    assert [op[1] for op in _read(mirror)["pending"]] == ["remove"]

    harness.scenario(tools)
    harness.start()  # the next session's start replays the journal
    assert wait_for(lambda: _calls(harness, "recall_forget_memo") == [{"memo_id": "m1.md"}], timeout=10)


def test_a_start_that_cannot_make_its_folders_says_why(harness):
    """Invariant (BUG-003): when the start cannot create the memo folders, the provider records why,
    offers no tools and claims no memory, instead of a dead start thread and a silent 'starting'.

    Red proof (2026-11-12, CCA run 20261112T120535Z, local Windows, `cca_checks tautology verify` against
    the pre-fix snapshot, and the same test run in a checkout of 86fdc52): with re_call_hermes/provider.py
    `_start_client_once` creating the folders outside any handler, the test failed at `assert False` where
    `False = wait_for(...)`: `_start_error` stayed empty. Restored, green.
    """
    (harness.data / "memos").mkdir()
    (harness.data / "memos" / "hermes").write_text("a file where the assistant folder goes", encoding="utf-8")
    harness.scenario({"recall_search": [SEARCH_OK]})
    provider = harness.start(ready=False)
    assert wait_for(lambda: provider._start_error, timeout=10)
    assert "did not start" in provider._start_error
    assert provider.get_tool_schemas() == []
    assert provider.system_prompt_block() == ""


def test_a_restart_that_fails_is_not_waited_out(harness, monkeypatch):
    """Invariant (PERF-002): a prefetch that launches a restart which then fails returns when it
    fails, not after the whole cold-start wait of Hermes's 8 s turn budget.

    Red proof (2026-11-12, CCA run 20261112T120535Z, local Windows, `cca_checks tautology verify` against
    the pre-fix snapshot, and the same test run in a checkout of 86fdc52): with re_call_hermes/provider.py
    `prefetch` waiting on `_ready` alone, the test failed at `assert (8430.11... - 8424.10...) < 4.0`.
    Restored, green.
    """
    monkeypatch.setattr(provider_module, "RESTART_INTERVAL_S", 0.0)
    harness.scenario({"recall_search": [SEARCH_OK]})
    harness.config["recall_command"] = '"' + sys.executable + '" -c "import sys; sys.exit(1)"'
    provider = harness.start(ready=False)
    assert wait_for(lambda: provider._start_error and not provider._starting)
    started = time.monotonic()
    assert provider.prefetch("where do I work?") == ""
    assert time.monotonic() - started < 4.0


def test_a_postgres_store_is_offered_only_the_tools_its_server_registers(tmp_path, monkeypatch):
    """Invariant (DOC-002): on a non-Lite store the model is offered exactly the tools the child is
    asked to serve, and those are tools RE-call's `search` surface registers.

    Red proof (2026-11-12, CCA run 20261112T120535Z, local Windows, `cca_checks tautology verify` against
    the pre-fix snapshot, and the same test run in a checkout of 86fdc52): with re_call_hermes/provider.py
    `get_tool_schemas` offering READ_TOOLS to a non-Lite store, the test failed at `AssertionError: assert
    ({'recall_current_facts', 'recall_evidence', 'recall_search'} and ...) <= ...`. Restored, green.
    """
    _write_config(tmp_path, recall_command=sys.executable, dsn="postgresql://reader@db.example/recall")
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    offered = {s["name"] for s in ReCallMemoryProvider().get_tool_schemas()}
    assert offered and offered <= RECALL_SEARCH_PRESET
    requested = load_settings(tmp_path).child_env({})["RECALL_MCP_TOOLS"].replace(",", " ").split()
    assert set(requested) == offered


def test_setup_offers_the_saved_values_so_enter_keeps_them(tmp_path, monkeypatch):
    """Invariant (ENV-002): `hermes memory setup` prompts with each field's default and saves what
    Enter keeps, so the defaults offered are the profile's saved values.

    Red proof (2026-11-12, CCA run 20261112T120535Z, local Windows, `cca_checks tautology verify` against
    the pre-fix snapshot, and the same test run in a checkout of 86fdc52): with re_call_hermes/provider.py
    `get_config_schema` offering the static defaults, the test failed at `AssertionError: assert
    {'recall_comm..., 'k': 5, ...} == {'recall_comm...k': '12', ...}`. Restored, green.
    """
    saved = {"recall_command": "/opt/recall/bin/recall-mcp", "dsn": "", "memo_folder": "notes", "k": "12",
             "mirror_builtin": "false"}
    _write_config(tmp_path, **saved)
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    assert {f["key"]: f["default"] for f in ReCallMemoryProvider().get_config_schema()} == saved


def test_the_source_policy_reaches_the_child(tmp_path):
    """Invariant (ENV-003): the source policy `recall setup` honoured is the server's too, so its
    refresh and search keep the files it denies out of memory.

    Red proof (2026-11-12, CCA run 20261112T120535Z, local Windows, `cca_checks tautology verify` against
    the pre-fix snapshot, and the same test run in a checkout of 86fdc52): with re_call_hermes/config.py
    `PASSTHROUGH_ENV` lacking the variable, the test failed at `AssertionError: assert None ==
    '/etc/recall/policy.toml'`. Restored, green.
    """
    _write_config(tmp_path, recall_command=sys.executable)
    env = load_settings(tmp_path).child_env({"RECALL_SOURCE_POLICY_FILE": "/etc/recall/policy.toml"})
    assert env.get("RECALL_SOURCE_POLICY_FILE") == "/etc/recall/policy.toml"


def test_a_profile_rename_keeps_mirroring_into_the_same_store(harness):
    """Invariant (ENV-004): after Hermes renames the profile directory, the journal is bound to the
    same store at its new path, so a later remove still forgets the memo its entry became.

    Red proof (2026-11-12, CCA run 20261112T120535Z, local Windows, `cca_checks tautology verify` against
    the pre-fix snapshot, and the same test run in a checkout of 86fdc52): with re_call_hermes/provider.py
    lacking `on_profile_renamed` (the ABC's no-op ran), the test failed at `assert False` where `False =
    wait_for(...)`: the remove was refused as a changed destination. Restored, green.
    """
    tools = {"recall_remember": [{"memo_id": "m1.md"}], "recall_forget_memo": [{"deleted": 1}]}
    harness.scenario(tools)
    provider = harness.start(dsn="")
    provider.on_memory_write("add", "user", "Lives in Paris")
    assert wait_for(lambda: "m1.md" in _read(harness.data / "mirror.json").get("memos", {}).values())
    provider.shutdown()

    renamed = harness.home.with_name("renamed-home")
    harness.home.rename(renamed)
    ReCallMemoryProvider().on_profile_renamed(harness.home.name, renamed.name, str(renamed))
    harness.home, harness.data = renamed, renamed / "re-call"
    provider = harness.start(dsn="")
    provider.on_memory_write("remove", "user", "", {"previous_content": "Lives in Paris"})
    assert wait_for(lambda: _calls(harness, "recall_forget_memo") == [{"memo_id": "m1.md"}], timeout=10)
