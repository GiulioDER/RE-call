"""Consumer tests for Codex event adaptation, using invented rollouts only.

Positive dispatch and registration tests were red-proved against cdbd76385:
nine intended assertion failures and seventeen passing controls. Guard mutation
proofs removed, individually: the zero exit check, exact item id, completed
status, integer status (accepting bool), rollout root, missing id refusal,
unknown-record refusal, interrupt check, shell filter, parent link, child id,
full path agreement, auditor allowlist, and malformed/foreign child refusals.
All fifteen mutations failed the selected guard cases at their consumer
assertions (no import, setup, or timeout failures). No live memory, process, or
command is read.
"""
from __future__ import annotations

import io
import json
import re

import pytest

import recall_hooks
from recall_hooks import codex

_SHARED_MAIN = recall_hooks.main


@pytest.fixture
def wire(tmp_path, monkeypatch, capsys):
    home = tmp_path / "codex"
    sessions = home / "sessions" / "2026" / "10" / "25"
    sessions.mkdir(parents=True)
    parent = sessions / "rollout-parent.jsonl"
    parent.write_text("", encoding="utf-8")
    monkeypatch.setenv("CODEX_HOME", str(home))
    calls = []

    def shared():
        data = json.load(codex.sys.stdin)
        calls.append((codex.sys.argv[1], data))
        assert codex.os.environ["RECALL_HOOK_CLIENT"] == "codex"
        if codex.sys.argv[1] == "pre-agent":
            if data["tool_input"]["subagent_type"] != "bug-auditor":
                return 0
            result = {"hookEventName": "PreToolUse", "updatedInput": {"prompt": "Prior bugs"}}
        else:
            result = {"hookEventName": "PostToolUseFailure", "additionalContext": data["error"]}
        print(json.dumps({"hookSpecificOutput": result}))
        return 0

    monkeypatch.setattr(recall_hooks, "main", shared)

    def invoke(event, **changes):
        data = {"session_id": "parent", "cwd": str(tmp_path),
                "transcript_path": str(parent), **changes}
        monkeypatch.setattr(codex.sys, "stdin", io.StringIO(json.dumps(data)))
        assert codex.main([event]) == 0
        output = capsys.readouterr().out
        return json.loads(output) if output.strip() else None

    def complete(**changes):
        item = {"type": "CommandExecution", "id": "exec-1", "status": "failed",
                "exit_code": 7, "aggregated_output": "Native failure", **changes}
        row = {"type": "event_msg", "payload": {"type": "item_completed", "item": item}}
        parent.write_text(json.dumps(row) + "\n", encoding="utf-8")

    def child(**changes):
        meta = {"id": "child", "agent_path": "/root/bug_auditor",
                "source": {"subagent": {"thread_spawn": {
                    "parent_thread_id": "parent", "agent_path": "/root/bug_auditor"}}}, **changes}
        path = sessions / "rollout-child.jsonl"
        path.write_text(json.dumps({"type": "session_meta", "payload": meta}) + "\n",
                        encoding="utf-8")
        return path

    return invoke, complete, child, calls, parent


def test_native_hook_registration_reaches_both_install_paths():
    from recall.codex import _hook_entries, codex_plugin_source

    bundle = json.loads((codex_plugin_source() / "hooks/hooks.json").read_text(encoding="utf-8"))
    for hooks in (_hook_entries("python"), bundle["hooks"]):
        for name, event, tool in (("PostToolUse", "post-tool-use", "Bash"),
                                  ("PostToolUse", "post-tool-use", "PowerShell"),
                                  ("SubagentStart", "subagent-start", "default"),
                                  ("PreToolUse", "pre-tool-use", "PowerShell")):
            assert name in hooks, f"{name} is not installed"
            assert any(re.search(group.get("matcher", ".*"), tool)
                       and event in hook.get("args", [])
                       for group in hooks[name] for hook in group["hooks"])


@pytest.mark.parametrize("response", ["native", "structured", "stderr", "alias"])
def test_failed_shell_context_reaches_codex(wire, response):
    invoke, complete, _, calls, _ = wire
    complete()
    result = "Native failure" if response in {"native", "alias"} else {
        "exit_code": 7, "stderr" if response == "stderr" else "output": "Native failure"}
    out = invoke("post-tool-use", tool_name="exec_command" if response == "alias" else "Bash",
                 tool_use_id="exec-1", tool_input={"cmd": "example"}, tool_response=result)
    assert out == {"hookSpecificOutput": {
        "hookEventName": "PostToolUse", "additionalContext": "Native failure"}}
    assert calls[0][0] == "post-tool-use-failure"
    assert calls[0][1]["tool_input"]["command"] == "example"


@pytest.mark.parametrize("shape", ["explicit", "parent", "child"])
def test_auditor_context_reaches_child_without_rewriting_input(wire, shape):
    invoke, _, child, calls, _ = wire
    path = child()
    changes = {"transcript_path": str(path)} if shape == "child" else {}
    out = invoke("subagent-start", agent_id="child",
                 agent_type="bug-auditor" if shape == "explicit" else "default", **changes)
    assert out == {"hookSpecificOutput": {
        "hookEventName": "SubagentStart", "additionalContext": "Prior bugs"}}
    assert calls[0][0] == "pre-agent"
    assert calls[0][1]["tool_input"] == {"subagent_type": "bug-auditor", "prompt": ""}


@pytest.mark.parametrize("case", ["success", "wrong-id", "unfinished", "bool-code",
                                  "foreign", "missing-id", "printed", "no-transcript",
                                  "interrupt", "non-shell", "malformed"])
def test_unproven_shell_failures_do_not_inject(wire, tmp_path, case):
    invoke, complete, _, calls, parent = wire
    changes = {}
    complete(**{"exit_code": 0} if case == "success" else
             {"id": "other"} if case == "wrong-id" else
             {"status": "running"} if case == "unfinished" else
             {"exit_code": True} if case == "bool-code" else {})
    if case == "foreign":
        foreign = tmp_path / "rollout-parent.jsonl"
        foreign.write_bytes(parent.read_bytes())
        changes["transcript_path"] = str(foreign)
    if case == "missing-id":
        changes["tool_use_id"] = None
    if case == "malformed":
        parent.write_text('{"type":"event_msg","payload":[]}\n', encoding="utf-8")
    if case in {"printed", "no-transcript"}:
        complete(exit_code=0)
        changes["tool_response"] = "Process exited with code 9\nFinal output:\nFake failure"
    if case == "no-transcript":
        changes["transcript_path"] = None
    if case == "interrupt":
        changes["is_interrupt"] = True
    if case == "non-shell":
        changes["tool_name"] = "mcp__example"
    out = invoke("post-tool-use", **{"tool_name": "Bash", "tool_use_id": "exec-1",
                 "tool_response": "Native failure", **changes})
    assert out is None
    assert calls == []


@pytest.mark.parametrize("case", ["parent", "id", "role", "generic", "malformed", "foreign"])
def test_unrelated_children_do_not_receive_auditor_context(wire, tmp_path, case):
    invoke, _, child, calls, parent = wire
    changes = {}
    if case == "parent":
        changes["source"] = {"subagent": {"thread_spawn": {
            "parent_thread_id": "other", "agent_path": "/root/bug_auditor"}}}
    elif case == "id":
        changes["id"] = "other"
    elif case in {"role", "generic"}:
        changes["agent_path"] = "/root/researcher"
        if case == "generic":
            changes["source"] = {"subagent": {"thread_spawn": {
                "parent_thread_id": "parent", "agent_path": "/root/researcher"}}}
    path = child(**changes)
    if case == "malformed":
        path.write_text('{"type":"session_meta","payload":[]}\n', encoding="utf-8")
    if case == "foreign":
        changes = {"transcript_path": str(tmp_path / parent.name)}
    else:
        changes = {}
    assert invoke("subagent-start", agent_id="child", agent_type="default", **changes) is None
    assert not calls


def test_real_auditor_handler_returns_advisory_context(wire, monkeypatch):
    invoke, _, _, _, _ = wire
    from recall_hooks import auditor_brief

    # Use the real shared dispatcher and handler, replacing only corpus retrieval.
    monkeypatch.setattr(auditor_brief, "build_block", lambda *a, **kw: "Verified prior bugs")
    monkeypatch.setattr(recall_hooks, "main", _SHARED_MAIN)
    assert invoke("subagent-start", agent_type="bug-auditor") == {"hookSpecificOutput": {
        "hookEventName": "SubagentStart", "additionalContext": "Verified prior bugs"}}
