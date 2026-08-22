from __future__ import annotations

import json

import pytest

from benchmarks.agent_ab import (
    CodexExecConfig,
    parse_codex_jsonl,
)
from benchmarks.agent_ab.codex_exec import CodexTranscriptError, _transcript_fields


def test_codex_config_builds_non_shell_command() -> None:
    config = CodexExecConfig(
        model="gpt-5.3-codex",
        sandbox="workspace-write",
        ignore_user_config=True,
        extra_args=("-c", "model_reasoning_effort=high"),
    )

    assert config.command("fix the bug") == [
        "codex",
        "exec",
        "--json",
        "--model",
        "gpt-5.3-codex",
        "--sandbox",
        "workspace-write",
        "--ephemeral",
        "--ignore-user-config",
        "-c",
        "model_reasoning_effort=high",
        "fix the bug",
    ]


def test_parse_codex_jsonl_extracts_usage_tools_and_recall_calls() -> None:
    lines = [
        {"type": "thread.started", "thread_id": "thread-1"},
        {"type": "turn.started"},
        {
            "type": "item.completed",
            "item": {
                "id": "mcp-1",
                "type": "mcp_tool_call",
                "name": "recall_search",
                "server": "recall",
                "status": "completed",
            },
        },
        {
            "type": "item.completed",
            "item": {
                "id": "cmd-1",
                "type": "command_execution",
                "command": "pytest -q",
                "aggregated_output": "5 passed",
            },
        },
        {
            "type": "item.completed",
            "item": {"id": "msg-1", "type": "agent_message", "text": "Done"},
        },
        {
            "type": "turn.completed",
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cached_input_tokens": 80,
                "reasoning_output_tokens": 4,
            },
        },
    ]

    events = parse_codex_jsonl("\n".join(json.dumps(line) for line in lines))
    fields = _transcript_fields(events, recall_markers=("recall",))

    assert fields["response"] == "Done"
    assert fields["input_tokens"] == 100
    assert fields["output_tokens"] == 20
    assert fields["cached_input_tokens"] == 80
    assert fields["reasoning_output_tokens"] == 4
    assert fields["model_turns"] == 1
    assert fields["recall_call_count"] == 1
    assert len(fields["tool_calls"]) == 2


def test_parse_codex_jsonl_rejects_non_object_lines() -> None:
    with pytest.raises(CodexTranscriptError, match="must be an object"):
        parse_codex_jsonl(json.dumps(["not an event"]))
