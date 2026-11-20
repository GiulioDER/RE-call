"""The model-facing write tools honour Hermes's memory.write_approval (owner request, 2026-11-20).

The unit tests drive the fake gate in tests/fakes/tools/write_approval.py; they skip with a real
Hermes on the path, where tests/test_contract.py checks the real gate instead."""

from __future__ import annotations

import json

import pytest
from conftest import HERMES_SRC

pytestmark = pytest.mark.skipif(bool(HERMES_SRC), reason="drives the fake gate; the contract test covers the real one")


@pytest.fixture
def gate(monkeypatch):
    from tools import write_approval

    monkeypatch.setattr(write_approval, "CALLS", [])
    monkeypatch.setattr(write_approval, "FAIL", None)
    monkeypatch.setattr(write_approval, "ENABLED", False)
    monkeypatch.setattr(write_approval, "DECISION", write_approval.GateDecision(allow=True))
    return write_approval


def _calls(harness, tool):
    return [c for c in harness.events("call") if c["tool"] == tool]


def test_a_write_hermes_would_stage_is_refused_and_not_sent(harness, gate):
    """Invariant: with memory.write_approval on and nobody to approve (Hermes would stage the write),
    recall_remember is refused with the reason and never reaches RE-call, and the gate is asked about
    the text being saved.

    Red proof (2026-11-20, `python tests/mutations.py APPROVAL-GATED`): in re_call_hermes/provider.py,
    `if tool_name in WRITE_TOOLS:` became `if False:`; the test failed at `AssertionError: assert
    'error' in {'memo_id': 'm1.md'}`. Restored, green.
    """
    gate.DECISION = gate.GateDecision(stage=True, message="Staged for approval")
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}]})
    provider = harness.start()
    answer = json.loads(provider.handle_tool_call("recall_remember", {"text": "Lives in Rome"}))
    assert "error" in answer
    assert "memory.write_approval is on" in answer["error"]
    assert _calls(harness, "recall_remember") == []
    assert gate.CALLS[0]["subsystem"] == "memory" and '"text": "Lives in Rome"' in gate.CALLS[0]["detail"]


def test_a_write_the_user_denies_is_refused_with_hermes_message(harness, gate):
    """Invariant: a write the user denies at the inline prompt is refused with Hermes's own message,
    for each of the three write tools.

    Red proof (2026-11-20, `python tests/mutations.py APPROVAL-BLOCKED`): in
    re_call_hermes/provider.py, `if decision.blocked:` became `if False:`; the test failed at
    `AssertionError: assert 'memory.write_approval is on and nobody...' == 'Memory write denied by
    user.'`. Restored, green.
    """
    gate.DECISION = gate.GateDecision(blocked=True, message="Memory write denied by user.")
    harness.scenario({})
    provider = harness.start()
    fact = {"namespace": "user", "subject": "user", "predicate": "lives_in", "object": "Rome",
            "evidence_card_ids": ["c1"], "request_id": "r1"}
    for tool, args in (("recall_remember", {"text": "Lives in Rome"}), ("recall_forget_memo", {"memo_id": "m1"}),
                       ("recall_apply_fact", fact)):
        assert json.loads(provider.handle_tool_call(tool, args))["error"] == "Memory write denied by user."
    assert harness.events("call") == []


def test_a_failed_check_refuses_only_while_approval_may_be_on(harness, gate):
    """Invariant: when the gate check itself raises, the write is refused if approval is on or its
    state cannot be read, and goes ahead if approval is off.

    Red proof (2026-11-20, `python tests/mutations.py APPROVAL-FAIL-CLOSED`): in
    re_call_hermes/provider.py, `            gated = True` became `            gated = False`; the test
    failed at `AssertionError: assert 'error' in {'memo_id': 'm2.md'}`. Restored, green.
    """
    gate.FAIL = RuntimeError("config unreadable")
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}, {"memo_id": "m2.md"}]})
    provider = harness.start()
    gate.ENABLED = False
    assert json.loads(provider.handle_tool_call("recall_remember", {"text": "a"})) == {"memo_id": "m1.md"}
    gate.ENABLED = True
    assert "check failed" in json.loads(provider.handle_tool_call("recall_remember", {"text": "b"}))["error"]
    gate.ENABLED = RuntimeError("still unreadable")
    assert "error" in json.loads(provider.handle_tool_call("recall_remember", {"text": "c"}))
    assert len(_calls(harness, "recall_remember")) == 1


def test_reads_are_never_gated(harness, gate):
    """Invariant: the read tools never consult the gate, so search works with approval on.

    Red proof (2026-11-20, `python tests/mutations.py APPROVAL-READS-FREE`): in
    re_call_hermes/provider.py, `if tool_name in WRITE_TOOLS:` became `if True:`; the test failed at
    `AssertionError: assert 'error' not in {'error': 'memory.write_approval is on and nobody...'}`.
    Restored, green.
    """
    from test_provider import SEARCH_OK

    gate.DECISION = gate.GateDecision(stage=True)
    harness.scenario({"recall_search": [SEARCH_OK]})
    provider = harness.start()
    answer = json.loads(provider.handle_tool_call("recall_search", {"query": "where do I live?"}))
    assert "error" not in answer
    assert gate.CALLS == []


def test_the_prompt_shows_what_a_write_replaces_and_forgets(harness, gate):
    """Invariant (review BUG-001, BUG-004): the approval prompt names the memo a correction replaces
    and shows the text of the memo a forget deletes, so the user never approves a hidden effect.

    Red proof (2026-11-20, `python tests/mutations.py APPROVAL-SHOWS-SUPERSEDES`): in
    re_call_hermes/provider.py, the `line += f", replacing memo ...` line became `pass`; the test
    failed at `AssertionError: assert "replacing memo 'm5.md'" in "RE-call: remember 'Lives in Milan'"`.
    Restored, green.
    Red proof (2026-11-20, `python tests/mutations.py APPROVAL-SHOWS-FORGOTTEN`): in
    re_call_hermes/provider.py, `text = _memo_text(folder, memo_id) if folder is not None else None`
    became `text = None`; the test failed at `AssertionError: assert 'Has a cat' in '{...'`. Restored,
    green.
    """
    harness.scenario({"recall_remember": [{"memo_id": "m6.md"}], "recall_forget_memo": [{"deleted": True}]})
    provider = harness.start()
    provider.handle_tool_call("recall_remember", {"text": "Lives in Milan", "supersedes": "m5.md"})
    assert "replacing memo 'm5.md'" in gate.CALLS[-1]["summary"]
    assert '"supersedes": "m5.md"' in gate.CALLS[-1]["detail"]
    folder = provider._settings.assistant_folder
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "note-0003.md").write_text("---\nvalid_from: 2026-11-01\n---\nHas a cat\n", encoding="utf-8")
    provider.handle_tool_call("recall_forget_memo", {"memo_id": "note-0003"})
    assert "Has a cat" in gate.CALLS[-1]["detail"]


def test_a_gate_that_fails_to_import_refuses(harness, gate, monkeypatch, tmp_path):
    """Invariant (review BUG-003): only a Hermes without the gate module lets a write through
    unasked; a gate module that fails while importing refuses, since that is a check that failed.

    Red proof (2026-11-20, `python tests/mutations.py APPROVAL-IMPORT-CLOSED`): in
    re_call_hermes/provider.py, `if isinstance(exc, ModuleNotFoundError) and exc.name in (...)` became
    `if True:`; the test failed at `AssertionError: assert 'error' in {'memo_id': 'm1.md'}`.
    Restored, green.
    """
    import sys
    import types

    broken = tmp_path / "broken_tools"
    broken.mkdir()
    (broken / "write_approval.py").write_text('raise ImportError("a dependency of the gate is missing")\n',
                                              encoding="utf-8")
    package = types.ModuleType("tools")
    package.__path__ = [str(broken)]
    monkeypatch.setitem(sys.modules, "tools", package)
    monkeypatch.delitem(sys.modules, "tools.write_approval", raising=False)
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}]})
    provider = harness.start()
    answer = json.loads(provider.handle_tool_call("recall_remember", {"text": "Lives in Rome"}))
    assert "error" in answer
    assert harness.events("call") == []


def test_a_hermes_without_the_gate_module_still_honours_the_setting(harness, gate, monkeypatch, tmp_path):
    """Invariant: on a Hermes whose `tools` package has no `write_approval` (moved or removed in a
    release), the write tools read memory.write_approval through the public `hermes_cli.config`:
    refused while it is on, written while it is off. Never written unasked, never all refused.

    Red proof (2026-11-20, `python tests/mutations.py APPROVAL-NO-GATE-CONFIG`): in
    re_call_hermes/provider.py, `return APPROVAL_NO_GATE if _approval_configured() else None` became
    `return None`; the test failed at `AssertionError: assert {'memo_id': 'm1.md'} == {'error': ...}`.
    Restored, green.
    Red proof (2026-11-20, `python tests/mutations.py APPROVAL-NO-GATE-REMOVED`): in
    re_call_hermes/provider.py, the condition became the merged #1474 one, `if isinstance(exc,
    ModuleNotFoundError) and exc.name in (...)`; the test failed at `AssertionError: assert {'error':
    'th...d to RE-call'} == {'error': 'me...d to RE-call'}` (the check-failed refusal, the one that
    also refused every write with approval off). Restored, green.
    """
    import sys
    import types

    from re_call_hermes.provider import APPROVAL_NO_GATE

    empty = tmp_path / "tools_without_gate"
    empty.mkdir()
    package = types.ModuleType("tools")
    package.__path__ = [str(empty)]
    monkeypatch.setitem(sys.modules, "tools", package)
    monkeypatch.delitem(sys.modules, "tools.write_approval", raising=False)
    setting = {"memory": {"write_approval": "on"}}
    config = types.ModuleType("hermes_cli.config")
    config.load_config_readonly = lambda: setting
    config.cfg_get = lambda cfg, *keys, default=None: cfg.get(keys[0], {}).get(keys[1], default)
    monkeypatch.setitem(sys.modules, "hermes_cli", types.ModuleType("hermes_cli"))
    monkeypatch.setitem(sys.modules, "hermes_cli.config", config)
    harness.scenario({"recall_remember": [{"memo_id": "m1.md"}]})
    provider = harness.start()
    assert json.loads(provider.handle_tool_call("recall_remember", {"text": "a"})) == {"error": APPROVAL_NO_GATE}
    assert harness.events("call") == []
    setting["memory"]["write_approval"] = False
    assert json.loads(provider.handle_tool_call("recall_remember", {"text": "a"})) == {"memo_id": "m1.md"}
