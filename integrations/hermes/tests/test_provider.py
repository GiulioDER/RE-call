"""The provider against a scripted RE-call server running as a real child process."""

from __future__ import annotations

import json
import os
import sys
import time

import pytest
from conftest import wait_for

from re_call_hermes import provider as provider_module

SEARCH_OK = {
    "abstained": False,
    "evidence": {"decision": "answer", "citable": ["c1"]},
    "hits": [{"memo_id": "job.md", "chunk_id": "c1", "verdict": "ok", "text": "Works at Acme"}],
}
SEARCH_ABSTAIN = {"abstained": True, "evidence": {"citable": []}, "hits": []}

SIX_TOOLS = ["recall_search", "recall_evidence", "recall_current_facts",
             "recall_apply_fact", "recall_remember", "recall_forget_memo"]


def test_prefetch_injects_citable_memory_and_reports_the_count(harness):
    """Invariant: prefetch returns the rendered block and recall_status counts citable memories.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    PREFETCH-STATUS`): in re_call_hermes/provider.py, `self._status = None if self._settings is None or
    is_trivi...` became `if self._settings is None or is_trivial_prompt`; the test failed at
    `AssertionError: assert RecallStatus(provider_label='RE-call', count=1, glyph=...`. Restored, green.
    """
    harness.scenario({"recall_search": [SEARCH_OK, SEARCH_ABSTAIN]})
    provider = harness.start()
    text = provider.prefetch("where do I work?")
    assert "[ok] memo job.md: Works at Acme" in text
    assert provider.recall_status().count == 1
    provider.prefetch("what is my cat called?")
    assert provider.recall_status() is None  # reflects only the LAST prefetch
    calls = harness.events("call")
    assert calls[0]["arguments"] == {"query": "where do I work?", "k": 5}


def test_trivial_prompts_never_reach_the_server(harness):
    """Invariant: a bare acknowledgement costs no search.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py TRIVIAL`): in
    re_call_hermes/provider.py, `if self._settings is None or is_trivial_prompt(query):` became `if
    self._settings is None:`; the test failed at `AssertionError: assert '## RE-call m...Works at Acme' ==
    ''`. Restored, green.
    """
    harness.scenario({"recall_search": [SEARCH_OK]})
    provider = harness.start()
    assert provider.prefetch("thanks!") == ""
    assert provider.prefetch("/reset") == ""
    assert harness.events("call") == []


def test_prefetch_gives_up_before_hermes_does(harness, monkeypatch):
    """Invariant: a slow search returns "" at the provider's own deadline, not the tool timeout.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py DEADLINE`):
    in re_call_hermes/provider.py, `self._settings.k}, PREFETCH_DEADLINE_S)` became `self._settings.k},
    TOOL_TIMEOUT_S)`; the test failed at `AssertionError: assert '## RE-call m...Works at Acme' == ''`.
    Restored, green.
    """
    monkeypatch.setattr(provider_module, "PREFETCH_DEADLINE_S", 0.5)
    monkeypatch.setattr(provider_module, "TOOL_TIMEOUT_S", 30.0)
    harness.scenario({"recall_search": [SEARCH_OK]}, delay={"recall_search": 3})
    provider = harness.start()
    started = time.monotonic()
    assert provider.prefetch("where do I work?") == ""
    assert time.monotonic() - started < 2.5


def test_a_refused_search_is_reported_not_hidden(harness):
    """Invariant: a tool error (an uncalibrated store, say) becomes one visible line.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py TOOL-ERROR`):
    in re_call_hermes/provider.py, `except ToolError as exc: return render_unavailable(str(ex...` became
    `(deleted)`; the test failed at `AssertionError: assert 'could not answer this turn: calibration
    required' in ''`. Restored, green.
    """
    harness.scenario({"recall_search": [{"isError": True, "text": "calibration required"}]})
    provider = harness.start()
    text = provider.prefetch("where do I work?")
    assert "could not answer this turn: calibration required" in text
    assert provider.recall_status() is None


def test_the_child_gets_a_minimal_environment_and_the_data_directory(harness, monkeypatch):
    """Invariant: Hermes's secrets never reach the child; trust stays strict; cwd holds no .env.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py ENV`): in
    re_call_hermes/config.py, `env = {k: base[k] for k in PASSTHROUGH_ENV if k in base}` became `env =
    dict(base)`; the test failed at `AssertionError: assert 'OPENROUTER_API_KEY' not in {...`
    (environment redacted). Restored, green.
    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py ENV-CWD`): in
    re_call_hermes/provider.py, `cwd=settings.data_dir, # no project .env` became `cwd=Path.cwd(), # no
    project .env`; the test failed at `AssertionError: assert 'c:/users/<user>...recall-hermes' ==
    'c:/users/<user>...-hom...` (account name redacted). Restored, green.
    """
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-not-a-real-key")
    harness.scenario({"recall_search": [SEARCH_OK]})
    harness.start()
    start = harness.events("start")[0]
    env = start["env"]
    assert "OPENROUTER_API_KEY" not in env
    assert env["RECALL_TRUST_MODE"] == "strict"
    assert env["RECALL_EMBEDDER"] == "fastembed"
    assert env["RECALL_MCP_TOOLS"] == "assistant"
    assert env["RECALL_ASSISTANT_MEMOS"].startswith(env["RECALL_INDEX_ROOT"])
    assert start["cwd"].replace("\\", "/").lower() == str(harness.data).replace("\\", "/").lower()


def test_write_tools_are_offered_only_to_the_primary_agent_on_lite(harness):
    """Invariant: a subagent or cron run, or a PostgreSQL store, gets search only.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py TOOLS`): in
    re_call_hermes/provider.py, `if settings.lite and self._primary:` became `if settings.lite:`; the test
    failed at `AssertionError: assert ['re_call_sea..._call_forget'] == ['re_call_search']`. Restored,
    green.
    """
    harness.scenario({"recall_search": [SEARCH_OK]})
    primary = harness.start()
    assert [s["name"] for s in primary.get_tool_schemas()] == SIX_TOOLS
    primary.shutdown()
    sub = harness.start(agent_context="subagent")
    assert [s["name"] for s in sub.get_tool_schemas()] == ["recall_search", "recall_evidence", "recall_current_facts"]
    refused = json.loads(sub.handle_tool_call("recall_remember", {"text": "x"}))
    assert "does not offer" in refused["error"]
    assert not [c for c in harness.events("call") if c["tool"] == "recall_remember"]


def test_tool_calls_forward_to_the_server_tool_with_their_arguments(harness):
    """Invariant: each provider tool reaches the matching RE-call tool; blank arguments are dropped.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py FORWARD`): in
    re_call_hermes/provider.py, `if v not in (None, "")}` became `}`; the test failed at `AssertionError:
    assert [('recall_rem...d': 'm1.md'})] == [('recall_rem...d': ...`. Restored, green.
    """
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}], "recall_forget_memo": [{"deleted": True}]})
    provider = harness.start()
    assert json.loads(provider.handle_tool_call("recall_remember", {"text": "Lives in Rome", "supersedes": ""})) == {
        "memo_id": "m1.md"
    }
    provider.handle_tool_call("recall_forget_memo", {"memo_id": "m1.md"})
    calls = [(c["tool"], c["arguments"]) for c in harness.events("call")]
    assert calls == [("recall_remember", {"text": "Lives in Rome"}), ("recall_forget_memo", {"memo_id": "m1.md"})]


def test_builtin_memory_writes_are_mirrored_with_supersession(harness):
    """Invariant: add remembers, replace supersedes the memo the old entry became, remove forgets it.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py MIRROR`): in
    re_call_hermes/provider.py, `arguments["supersedes"] = supersedes` became `pass`; the test failed at
    `AssertionError: assert [('recall_rem...d': 'm2.md'})] == [('recall_rem...d': ...`. Restored, green.
    """
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}, {"memo_id": "m2.md"}],
                      "recall_forget_memo": [{"deleted": True}]})
    provider = harness.start()
    provider.on_memory_write("add", "user", "Lives in Paris")
    provider.on_memory_write("replace", "user", "Lives in Rome", {"previous_content": "Lives in Paris"})
    provider.on_memory_write("remove", "user", "", {"previous_content": "Lives in Rome"})
    assert wait_for(lambda: len(harness.events("call")) == 3)
    calls = [(c["tool"], c["arguments"]) for c in harness.events("call")]
    request_ids = [args.pop("request_id", None) for tool, args in calls if tool == "recall_remember"]
    assert all(r and r.startswith("hermes-") for r in request_ids) and len(set(request_ids)) == 2
    assert calls == [
        ("recall_remember", {"text": "Lives in Paris"}),
        ("recall_remember", {"text": "Lives in Rome", "supersedes": "m1.md"}),
        ("recall_forget_memo", {"memo_id": "m2.md"}),
    ]


def test_a_destructive_mirror_without_the_entry_identity_is_skipped(harness):
    """Invariant: replace or remove with no previous_content (older Hermes) writes nothing.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    MIRROR-GUARD`): in re_call_hermes/provider.py, `if action in ("replace", "remove") and previous is
    None:` became `if False:`; the test failed at `AssertionError: assert ['New text', 'Marker'] ==
    ['Marker']`. Restored, green.
    """
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}], "recall_forget_memo": [{"deleted": True}]})
    provider = harness.start()
    provider.on_memory_write("replace", "memory", "New text", {})
    provider.on_memory_write("remove", "memory", "", {})
    provider.on_memory_write("add", "memory", "Marker")
    assert wait_for(lambda: len(harness.events("call")) >= 1)
    time.sleep(0.3)
    assert [c["arguments"]["text"] for c in harness.events("call")] == ["Marker"]


def test_a_crashed_server_is_restarted(harness, monkeypatch):
    """Invariant: after the child exits, a later turn starts it again and recall resumes.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py RESTART`): in
    re_call_hermes/provider.py, `self._maybe_restart() raise McpError(self._start_error` became `raise
    McpError(self._start_error`; the test failed at `assert False`. Restored, green.
    """
    monkeypatch.setattr(provider_module, "RESTART_INTERVAL_S", 0.0)
    harness.scenario({"recall_search": [SEARCH_OK]}, exit_on="recall_remember")
    provider = harness.start()
    provider.handle_tool_call("recall_remember", {"text": "boom"})
    assert wait_for(lambda: not provider._client.alive)
    provider.prefetch("where do I work?")  # notices the dead child and starts a new one
    assert wait_for(lambda: provider._ready.is_set() and provider._client.alive)
    assert "Works at Acme" in provider.prefetch("where do I work?")
    assert len(harness.events("start")) == 2


def test_a_missing_store_makes_the_provider_unavailable_with_the_setup_command(harness, monkeypatch):
    """Invariant: no store means inactive, and the reason tells the user how to create it.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py STORE`): in
    re_call_hermes/config.py, `if store is not None and not store.is_file():` became `if False:`; the test
    failed at `AssertionError: assert 'setup --lite' in ''`. Restored, green.
    """
    from re_call_hermes.config import load_settings

    harness.scenario({})
    (harness.data / "memory.db").unlink()
    (harness.data / "config.json").write_text(json.dumps(harness.config), encoding="utf-8")
    problem = load_settings(harness.home).problem()
    assert "setup --lite" in problem
    provider = harness.start(ready=False)
    assert provider.get_tool_schemas() == []
    assert harness.events("start") == []


def test_a_dsn_with_a_password_is_refused(tmp_path):
    """Invariant: a credential in the config file is refused (catalog rule 11).

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py PASSWORD`):
    in re_call_hermes/config.py, `if not self.lite and (urlsplit(self.dsn).password or _DSN...` became `if
    False:`; the test failed at `AssertionError: assert 'password' in ''`. Restored, green.
    """
    from re_call_hermes.config import load_settings

    (tmp_path / "re-call").mkdir()
    (tmp_path / "re-call" / "config.json").write_text(
        json.dumps({"recall_command": sys.executable, "dsn": "postgresql://me:hunter2@localhost/recall"}),
        encoding="utf-8",
    )
    assert "password" in load_settings(tmp_path).problem()


def test_register_hands_out_a_new_instance_each_time():
    """Invariant: the multiplexing contract, one instance per load.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py REGISTER`):
    in re_call_hermes/__init__.py, `ctx.register_memory_provider(ReCallMemoryProvider())` became
    `ctx.register_memory_provider(_SHARED) _SHARED =...`; the test failed at `assert
    <re_call_hermes.provider.ReCallMemoryProvider object at 0x000001E01CC4...`. Restored, green.
    """
    import re_call_hermes

    got = []

    class Ctx:
        def register_memory_provider(self, p):
            got.append(p)

    re_call_hermes.register(Ctx())
    re_call_hermes.register(Ctx())
    assert got[0] is not got[1]


@pytest.mark.parametrize("command", ["recall-mcp-that-does-not-exist"])
def test_an_unknown_command_is_reported(tmp_path, command):
    """Nonbehavioral control for the message text; the store check above carries the red proof."""
    from re_call_hermes.config import load_settings

    (tmp_path / "re-call").mkdir()
    (tmp_path / "re-call" / "config.json").write_text(json.dumps({"recall_command": command}), encoding="utf-8")
    assert "was not found" in load_settings(tmp_path).problem()


def test_an_identical_entry_is_mirrored_once(harness):
    """Invariant: the same built-in entry added twice maps to one memo, never an orphan.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    MIRROR-ONCE`): in re_call_hermes/provider.py, `if action == "remove" or present:` became `if action ==
    "remove":`; the test failed at `AssertionError: assert ['Has a cat',...at', 'Marker'] == ['Has a cat',
    'Marker']`. Restored, green.
    """
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}, {"memo_id": "m2.md"}]})
    provider = harness.start()
    provider.on_memory_write("add", "user", "Has a cat")
    provider.on_memory_write("add", "user", "Has a cat")
    provider.on_memory_write("add", "user", "Marker")
    assert wait_for(lambda: any(c["arguments"]["text"] == "Marker" for c in harness.events("call")))
    assert [c["arguments"]["text"] for c in harness.events("call")] == ["Has a cat", "Marker"]


def test_a_lost_remember_answer_is_preserved_without_retry(harness, monkeypatch):
    """BUG-001 changes the old retry contract: an unknown result needs reconciliation.

    Red proof (Linux host, 2026-11-08): `python tests/mutations.py MIRROR-NO-RETRY` reintroduced
    transport retry and failed at two calls versus one. Restored, green; normal-write controls remain.
    """
    monkeypatch.setattr(provider_module, "TOOL_TIMEOUT_S", 0.2)
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}]}, delay_first={"recall_remember": 0.5})
    provider = harness.start()
    provider.on_memory_write("add", "user", "Lives in Rome")
    provider.shutdown()
    state = json.loads((harness.data / "mirror.json").read_text(encoding="utf-8"))
    assert len(harness.events("call")) == 1
    assert len(state["pending"]) == 1 and not state["memos"]


def test_a_corrupt_mirror_file_does_not_end_mirroring(harness):
    """Invariant: one failed mirror write is logged and the next write still goes through.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    WRITER-SURVIVES`): in re_call_hermes/provider.py, `except Exception: # one bad write` became `except
    McpError: # one bad write`; the test failed at `assert False`. Restored, green.
    """
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}]})
    provider = harness.start()
    mirror = harness.data / "mirror.json"
    mirror.write_text("{not json", encoding="utf-8")
    provider.on_memory_write("add", "user", "First")
    time.sleep(0.5)
    mirror.write_text("{}", encoding="utf-8")
    provider.on_memory_write("add", "user", "Second")
    assert wait_for(lambda: [c["arguments"]["text"] for c in harness.events("call")] == ["Second"], timeout=10)


def test_shutdown_while_the_server_starts_leaves_no_server(harness):
    """Invariant: a child still starting when Hermes shuts the provider down is closed, not adopted.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    CLOSED-DURING-START`): in re_call_hermes/provider.py, `if self._closed: # shutdown() ran while this
    child was st...` became `(deleted)`; the test failed at `assert False`. Restored, green.
    """
    harness.scenario({"recall_search": [SEARCH_OK]}, init_delay=1.5)
    # No writer thread: shutdown() would otherwise wait on it, and so on the start, hiding the race.
    provider = harness.start(ready=False, mirror_builtin=False)
    assert wait_for(lambda: harness.events("start"))
    provider.shutdown()
    assert wait_for(lambda: harness.events("exit"), timeout=8)
    assert provider._client is None


def test_two_launches_start_one_server(harness):
    """Invariant: a second start request while one is in flight starts nothing.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py ONE-LAUNCH`):
    in re_call_hermes/provider.py, `if self._closed or self._starting: return` became `if self._closed:
    return`; the test failed at `AssertionError: assert 2 == 1`. Restored, green.
    """
    harness.scenario({"recall_search": [SEARCH_OK]}, init_delay=1.0)
    provider = harness.start(ready=False)
    provider._launch("re-call-second")
    assert wait_for(provider._ready.is_set)
    time.sleep(1.5)
    assert len(harness.events("start")) == 1


def test_an_abandoned_prefetch_cannot_set_a_later_turns_status(harness, monkeypatch):
    """Invariant: a prefetch that finishes after the next turn began leaves that turn's status alone.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    STATUS-TOKEN`): in re_call_hermes/provider.py, `if rendered.citable and self._prefetch_token is token:`
    became `if rendered.citable:`; the test failed at `AssertionError: assert
    RecallStatus(provider_label='RE-call', count=1, glyph=...`. Restored, green.
    """
    import threading

    monkeypatch.setattr(provider_module, "PREFETCH_DEADLINE_S", 5.0)
    harness.scenario({"recall_search": [SEARCH_OK]}, delay={"recall_search": 1.5})
    provider = harness.start()
    late = threading.Thread(target=provider.prefetch, args=("where do I work?",))
    late.start()
    assert wait_for(lambda: harness.events("call"))
    assert provider.prefetch("thanks") == ""  # the next turn
    late.join(10)
    assert provider.recall_status() is None


def test_a_corrupt_config_is_reported_not_raised(tmp_path):
    """Invariant: an unreadable config.json makes the provider unavailable with a reason.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    CONFIG-ERROR`): in re_call_hermes/config.py, `except (ValueError, TypeError) as exc: # reported by
    prob...` became `except TypeError as exc: # reported by problem()`; the test failed at `assert 'is not
    valid' in "JSONDecodeError('Expecting property name enclosed i...`. Restored, green.
    """
    from re_call_hermes.config import load_settings

    (tmp_path / "re-call").mkdir()
    (tmp_path / "re-call" / "config.json").write_text("{not json", encoding="utf-8")
    try:
        problem = load_settings(tmp_path).problem()
    except Exception as exc:  # the defect: activation raises instead of reporting
        problem = repr(exc)
    assert "is not valid" in problem


def test_a_duplicate_add_does_not_keep_a_removed_entry_alive(harness):
    """Invariant (DAT-001): Hermes holds a text at most once and reports a duplicate add as a
    successful no-op, so after add, add again, remove, the memo is forgotten.

    This replaces the refcount test of 01eec97, which pinned the defect DAT-001 confirmed: it
    assumed two identical built-in entries, which Hermes's store never holds
    (tools/memory_tool_store.py: an existing entry answers "Entry already exists").

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    DAT-001-DUP-ADD`): in re_call_hermes/provider.py, `if action == "remove" or present:` became `if action
    == "remove":`; the test failed at `AssertionError: assert ['recall_reme..._forget_memo'] ==
    ['recall_reme..._for...`. Restored, green.
    """
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}], "recall_forget_memo": [{"deleted": True}]})
    provider = harness.start()
    provider.on_memory_write("add", "user", "Has a cat")
    provider.on_memory_write("add", "user", "Has a cat")
    provider.on_memory_write("remove", "user", "", {"previous_content": "Has a cat"})
    assert wait_for(lambda: any(c["tool"] == "recall_forget_memo" for c in harness.events("call")), timeout=8)
    assert [c["tool"] for c in harness.events("call")] == ["recall_remember", "recall_forget_memo"]
    assert harness.events("call")[1]["arguments"] == {"memo_id": "m1.md"}


def test_tools_are_declared_before_initialize(harness, monkeypatch):
    """Invariant: Hermes collects tool schemas in add_provider(), before initialize(), so an
    uninitialized instance must already declare them for a configured profile.

    Red proof (2026-11-08, CCA run 20261108T171507Z, local Windows, `python tests/mutations.py
    TOOLS-BEFORE-INIT`): in re_call_hermes/provider.py, `settings = self._settings or
    load_settings(self._home_for...` became `settings = self._settings if settings is None: ...`; the test
    failed at `AssertionError: assert [] == ['re_call_sea..._call_forget']`. Restored, green.
    """
    from re_call_hermes.provider import ReCallMemoryProvider

    (harness.data / "config.json").write_text(json.dumps(harness.config), encoding="utf-8")
    monkeypatch.setenv("HERMES_HOME", str(harness.home))
    names = [s["name"] for s in ReCallMemoryProvider().get_tool_schemas()]
    assert names == SIX_TOOLS


def test_the_first_prefetch_waits_for_a_starting_server(harness):
    """Invariant: a new session's first turn, which arrives while the server is still starting,
    gets memory once the server is ready, inside the prefetch budget (found by the local Hermes
    chat test of 2026-11-08: every one-shot chat got no memory).

    Red proof (2026-11-08, local Windows, `python tests/mutations.py COLD-START-WAIT`): in
    re_call_hermes/provider.py, `if not (self._starting and self._ready.wait(PREFETCH_BUDGET_S - MIN_SE`
    became `if True:`; the test failed at `AssertionError: assert 'Works at Acme' in ''`. Restored, green.
    Also red before the fix: 'Works at Acme' not in ''.
    """
    harness.scenario({"recall_search": [SEARCH_OK]}, init_delay=1.5)
    provider = harness.start(ready=False)
    started = time.monotonic()
    text = provider.prefetch("where do I work?")
    assert "Works at Acme" in text
    assert time.monotonic() - started < provider_module.PREFETCH_BUDGET_S


def test_a_failed_start_is_not_waited_on(harness):
    """Invariant: when the server cannot start, prefetch returns at once instead of spending the
    cold-start wait on a start that will never finish.

    Red proof (2026-11-08, local Windows, `python tests/mutations.py FAILED-START-NO-WAIT`): in
    re_call_hermes/provider.py, `if not (self._starting and self._ready.wait(PREFETCH_BUDGET_S - MIN_SE`
    became `if not self._ready.wait(PREFETCH_BUDGET_S - MIN_SEARCH_S):`; the test failed at `assert
    (41107.0830611 - 41101.082254) < 1.0`. Restored, green.
    """
    harness.scenario({"recall_search": [SEARCH_OK]})
    harness.config["recall_command"] = '"' + sys.executable + '" -c "import sys; sys.exit(1)"'
    provider = harness.start(ready=False)
    assert wait_for(lambda: not provider._starting and provider._start_error)
    started = time.monotonic()
    assert provider.prefetch("where do I work?") == ""
    assert time.monotonic() - started < 1.0


def test_initialize_returns_only_once_the_server_is_ready(harness):
    """Invariant: Hermes runs the first turn right after initialize(), so initialize() waits for
    the server (bounded by START_TIMEOUT_S) and the first prefetch finds it ready (found by the
    local Hermes chat test of 2026-11-09: a cold start missed the prefetch budget).

    Red proof (2026-11-09, local Windows, `python tests/mutations.py INIT-WAITS`): in
    re_call_hermes/provider.py, `starter.join()` became `pass`; the test failed at `assert False`.
    Restored, green.
    """
    from re_call_hermes.provider import ReCallMemoryProvider

    harness.scenario({"recall_search": [SEARCH_OK]}, init_delay=1.5)
    (harness.data / "config.json").write_text(json.dumps(harness.config), encoding="utf-8")
    os.environ["FAKE_SCENARIO"] = str(harness.scenario_path)
    os.environ["FAKE_LOG"] = str(harness.log_path)
    provider = ReCallMemoryProvider()
    harness.provider = provider
    provider.initialize("s1", hermes_home=str(harness.home), platform="cli", agent_context="primary")
    assert provider._ready.is_set()
