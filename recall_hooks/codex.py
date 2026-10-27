"""Codex payload adapter for the shared RE-call hook implementation.

Lifecycle, prompt, and write events use the shared handlers directly. PostToolUse
adapts proven shell failures to the shared failure handler, and SubagentStart
delivers the shared auditor brief as advisory child context.

It marks every delegated run with `RECALL_HOOK_CLIENT=codex`, which is how the general knowledge
event log tells the two clients apart. The knowledge home, the log and the on/off switch are NOT
redirected with `RECALL_HOOK_CONFIG_HOME`: they are resolved from the project config home, so Claude
and Codex share one corpus, one log and one switch.
"""
from __future__ import annotations

import io
from contextlib import redirect_stdout
import json
import os
from pathlib import Path
import re
import sys
from typing import Any

_MAX_RECORD_BYTES = 1_048_576
_IDENTIFIER = re.compile(r"[A-Za-z0-9_-]{1,180}")


def _session_path(data: dict[str, Any], session: object = None) -> Path | None:
    """Accept only the identified rollout below this client's configured sessions root."""
    raw = data.get("transcript_path")
    session = data.get("session_id") if session is None else session
    if not isinstance(raw, str) or not isinstance(session, str) or not _IDENTIFIER.fullmatch(session):
        return None
    if raw.startswith("\\\\?\\"):
        raw = raw[4:]
    from recall.codex import codex_home

    root = (codex_home() / "sessions").resolve()
    path = Path(raw).resolve()
    if path.is_relative_to(root) and path.name.endswith(f"-{session}.jsonl"):
        return path
    return None


def _native_completion(data: dict[str, Any]) -> dict[str, Any] | None:
    """Version-checked compatibility path for raw stdout without an exit status.

    Native rollout records are not a stable public API. Read a bounded tail and
    require the exact completed command id; unknown formats remain unclassified.
    Never interpret a status header printed by a command as completion metadata.
    """
    path, ident = _session_path(data), data.get("tool_use_id")
    if path is None or not isinstance(ident, str) or not _IDENTIFIER.fullmatch(ident):
        return None
    with path.open("rb") as stream:
        stream.seek(0, 2)
        stream.seek(max(0, stream.tell() - _MAX_RECORD_BYTES))
        lines = stream.read(_MAX_RECORD_BYTES).decode("utf-8", "replace").splitlines()
    for line in reversed(lines):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict) or row.get("type") != "event_msg":
            continue
        payload = row.get("payload")
        if not isinstance(payload, dict) or payload.get("type") != "item_completed":
            continue
        item = payload.get("item")
        if isinstance(item, dict) and item.get("type") == "CommandExecution" and item.get("id") == ident:
            if item.get("status") in {"completed", "failed"} and type(item.get("exit_code")) is int:
                return item
            return None
    return None


def _failure(data: dict[str, Any]) -> dict[str, Any] | None:
    if data.get("tool_name") not in {"Bash", "PowerShell"} or data.get("is_interrupt") is True:
        return None
    response = data.get("tool_response")
    if isinstance(response, dict):
        code = response.get("exit_code")
        error = response.get("output", response.get("stderr", ""))
    elif isinstance(response, str):
        native = _native_completion(data)
        if native is None:
            return None
        code = native["exit_code"]
        error = native.get("aggregated_output", response)
    else:
        return None
    if type(code) is not int or code == 0 or not isinstance(error, str):
        return None
    return {**data, "error": error}


def _auditor_type(data: dict[str, Any]) -> str:
    """Recover named default children only from metadata linked to this parent."""
    from recall_hooks.auditor_brief import auditor_types

    role = data.get("agent_type")
    if role == "default":
        child_id = data.get("agent_id")
        if not isinstance(child_id, str) or not _IDENTIFIER.fullmatch(child_id):
            return ""
        parent = _session_path(data) or _session_path(data, child_id)
        if parent is None:
            return ""
        role = ""
        # The startup payload may name the child transcript but keeps the parent
        # session_id. Both wire shapes require the child's positive parent link.
        for index, child in enumerate(parent.parent.glob(f"*-{child_id}.jsonl")):
            if index >= 256:
                break
            if _session_path({"transcript_path": str(child)}, child_id) is None:
                continue
            with child.open("rb") as stream:
                line = stream.readline(_MAX_RECORD_BYTES + 1)
            if len(line) > _MAX_RECORD_BYTES:
                continue
            row = json.loads(line)
            if not isinstance(row, dict) or row.get("type") != "session_meta":
                continue
            meta = row.get("payload")
            if not isinstance(meta, dict) or meta.get("id") != child_id:
                continue
            source = meta.get("source")
            subagent = source.get("subagent") if isinstance(source, dict) else None
            spawn = subagent.get("thread_spawn") if isinstance(subagent, dict) else None
            if not isinstance(spawn, dict) or spawn.get("parent_thread_id") != data.get("session_id"):
                continue
            name = spawn.get("agent_path")
            if isinstance(name, str) and name == meta.get("agent_path"):
                role = name.rsplit("/", 1)[-1].replace("_", "-")
                break
    if isinstance(role, str) and role.rsplit(":", 1)[-1].casefold() in auditor_types():
        return role
    return ""


def _emit_context(event: str, raw: str) -> None:
    """Forward advisory text only, never a permission decision or input rewrite."""
    if not raw.strip():
        return
    result = json.loads(raw)["hookSpecificOutput"]
    context = (result.get("updatedInput", {}).get("prompt") if event == "subagent-start"
               else result.get("additionalContext"))
    if isinstance(context, str) and context.strip():
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "SubagentStart" if event == "subagent-start" else "PostToolUse",
            "additionalContext": context.strip(),
        }}))


def _normalise(payload: object, event: str) -> dict[str, object]:
    data = dict(payload) if isinstance(payload, dict) else {}
    if event == "user-prompt-submit" and not data.get("prompt"):
        for key in ("user_prompt", "text", "message", "input"):
            value = data.get(key)
            if isinstance(value, str) and value.strip():
                data["prompt"] = value
                break
    if not data.get("cwd"):
        data["cwd"] = os.getcwd()
    if data.get("tool_name") in {"exec_command", "functions.exec_command"}:
        data["tool_name"] = "Bash"
    tool_input = data.get("tool_input")
    if isinstance(tool_input, dict) and "command" not in tool_input and "cmd" in tool_input:
        data["tool_input"] = {**tool_input, "command": tool_input["cmd"]}
    return data


def main(argv: list[str] | None = None) -> int:
    """Run one Codex hook while preserving the shared fail-open contract."""
    args = list(sys.argv[1:] if argv is None else argv)
    event = args[0] if args else ""
    if event not in {
        "session-start",
        "pre-compact",
        "pre-tool-use",
        "user-prompt-submit",
        "session-end",
        "post-tool-use",
        "subagent-start",
    }:
        return 0
    try:
        # Codex frames one JSON payload per hook invocation. Read the complete frame: truncating a
        # large but valid prompt/tool payload would turn it into invalid JSON and silently lose the
        # memory check. The hook remains fail-open if the client sends malformed or unreasonable
        # input.
        buffer = getattr(sys.stdin, "buffer", None)
        raw = buffer.read().decode("utf-8-sig") if buffer is not None else sys.stdin.read()
        payload = _normalise(json.loads(raw.lstrip("\ufeff")) if raw.strip() else {}, event)
        shared_event = event
        if event == "post-tool-use":
            failed = _failure(payload)
            if failed is None:
                return 0
            payload, shared_event = failed, "post-tool-use-failure"
        elif event == "subagent-start":
            role = _auditor_type(payload)
            if not role:
                return 0
            payload = {**payload, "tool_name": "Agent",
                       "tool_input": {"subagent_type": role, "prompt": ""}}
            shared_event = "pre-agent"
        if event == "session-end":
            # Codex does not expose CLAUDE_PID. Its hook process and its local MCP child are both
            # descended from the Codex client process, so that concrete parent PID is the session
            # identity used by the shared cleanup path. A marker is included for transports whose
            # command line carries the client token instead of a usable parent chain.
            parent_pid = str(os.getppid())
            payload["_client_pid"] = parent_pid
            payload["_client_mark"] = os.environ.get("RECALL_MCP_CLIENT", f"codex-{parent_pid}")
            payload["_client_session_id"] = (
                os.environ.get("RECALL_MCP_SESSION_ID") or payload.get("session_id") or ""
            )
        from recall.codex import codex_integration_dir

        previous_config_home = os.environ.get("RECALL_HOOK_CONFIG_HOME")
        previous_project_home = os.environ.get("RECALL_PROJECT_CONFIG_HOME")
        previous_client = os.environ.get("RECALL_HOOK_CLIENT")
        os.environ["RECALL_HOOK_CONFIG_HOME"] = str(codex_integration_dir())
        from recall_hooks import main as recall_main

        previous_argv = sys.argv
        previous_stdin = sys.stdin
        try:
            sys.argv = ["recall-codex-hooks", shared_event]
            os.environ["RECALL_HOOK_CLIENT"] = "codex"
            sys.stdin = io.StringIO(json.dumps(payload))
            if event in {"post-tool-use", "subagent-start"}:
                output = io.StringIO()
                with redirect_stdout(output):
                    recall_main()
                _emit_context(event, output.getvalue())
                return 0
            return int(recall_main())
        finally:
            sys.argv = previous_argv
            sys.stdin = previous_stdin
            if previous_config_home is None:
                os.environ.pop("RECALL_HOOK_CONFIG_HOME", None)
            else:
                os.environ["RECALL_HOOK_CONFIG_HOME"] = previous_config_home
            if previous_project_home is None:
                os.environ.pop("RECALL_PROJECT_CONFIG_HOME", None)
            else:
                os.environ["RECALL_PROJECT_CONFIG_HOME"] = previous_project_home
            if previous_client is None:
                os.environ.pop("RECALL_HOOK_CLIENT", None)
            else:
                os.environ["RECALL_HOOK_CLIENT"] = previous_client
    except (KeyboardInterrupt, SystemExit):
        return 0
    except BaseException:
        # Memory hints never deny a Codex action or prevent a prompt.
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
