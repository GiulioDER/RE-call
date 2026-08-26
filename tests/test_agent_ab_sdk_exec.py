"""The SDK driver, exercised entirely without `claude-agent-sdk` installed.

Canned dict-shaped message entries go through the normalizer into `claude_exec`'s own parsing
core, so every assertion here is also an assertion that the two drivers share one field mapping.
The one gated live test at the bottom needs the SDK, the CLI, and an explicit env opt-in.
"""
from __future__ import annotations

import asyncio
import gzip
import json
import os
import shutil
import sys

import pytest

from benchmarks.agent_ab.arms import ArmSpec
from benchmarks.agent_ab.gate import check_session
from benchmarks.agent_ab.schema import RECALL_OFF, RECALL_ON
from benchmarks.agent_ab.sdk_exec import (
    SDKExecConfig,
    build_sdk_configs,
    build_sdk_record,
    make_sdk_runner,
    sdk_mcp_servers,
)

pytestmark = pytest.mark.benchharness

ROW = {"task_id": "t1", "user_input": "do the thing"}


def _entry(message_type: str, data: dict, stamp: str = "2026-08-26T12:00:00+00:00") -> dict:
    return {"message_type": message_type, "data": data, "received_at": stamp}


def _init_entry(tools: list[str], servers: list[dict] | None = None) -> dict:
    return _entry(
        "SystemMessage",
        {
            "subtype": "init",
            "data": {
                "tools": tools,
                "mcp_servers": servers or [],
                "claude_code_version": "2.1.238",
                "model": "anthropic/claude-haiku-4.5",
            },
        },
    )


def _result_entry(**overrides) -> dict:
    data = {
        "subtype": "success",
        "duration_ms": 1200,
        "duration_api_ms": 900,
        "is_error": False,
        "num_turns": 2,
        "session_id": "s-1",
        "total_cost_usd": 0.01,
        "usage": {"input_tokens": 34},
        "model_usage": {
            "anthropic/claude-haiku-4.5": {
                "inputTokens": 40,
                "cacheReadInputTokens": 1000,
                "cacheCreationInputTokens": 200,
                "outputTokens": 55,
            }
        },
        "result": "done",
        "permission_denials": [],
        "stop_reason": "end_turn",
    }
    data.update(overrides)
    return _entry("ResultMessage", data)


def test_importing_sdk_exec_never_imports_the_sdk() -> None:
    assert "benchmarks.agent_ab.sdk_exec" in sys.modules
    SDKExecConfig(model="m")
    assert "claude_agent_sdk" not in sys.modules


def test_the_config_has_no_system_prompt_field_so_replacement_is_unrepresentable() -> None:
    # The SDK's plain-string system_prompt REPLACES the base prompt; the baselines APPENDED.
    # The config makes the wrong form unrepresentable rather than merely discouraged.
    with pytest.raises(TypeError):
        SDKExecConfig(system_prompt="you are a helpful assistant")  # type: ignore[call-arg]
    config = SDKExecConfig(append_system_prompt_file="prompt.txt")
    assert config.append_system_prompt_file == "prompt.txt"


def test_stdio_spec_env_and_command_are_carried_into_the_sdk_server_config(tmp_path) -> None:
    from benchmarks.agent_ab.recall_server import StdioRecallSpec

    spec = StdioRecallSpec(dsn="postgresql://x/y", cwd=tmp_path, tenant="ten")
    servers = sdk_mcp_servers(spec)
    entry = servers[spec.server_name]
    assert entry["type"] == "stdio"
    assert entry["command"] == spec.python
    assert entry["args"] == ["-m", "recall_mcp.server"]
    assert entry["env"]["RECALL_ENV"] == "production"
    assert entry["env"]["PYTHONSAFEPATH"] == "1"
    assert entry["env"]["RECALL_TENANT"] == "ten"


def test_messages_normalize_to_events_the_existing_parser_accepts() -> None:
    entries = [
        _init_entry(["Read", "mcp__recall__recall_search"]),
        _entry(
            "AssistantMessage",
            {
                "content": [
                    {"__type__": "TextBlock", "text": "searching"},
                    {
                        "__type__": "ToolUseBlock",
                        "id": "call-1",
                        "name": "mcp__recall__recall_search",
                        "input": {"query": "q"},
                    },
                ],
                "parent_tool_use_id": None,
            },
        ),
        _entry(
            "UserMessage",
            {
                "content": [
                    {
                        "__type__": "ToolResultBlock",
                        "tool_use_id": "call-1",
                        "content": [{"__type__": "TextBlock", "text": "the memo"}],
                        "is_error": False,
                    }
                ]
            },
        ),
        _result_entry(),
    ]
    record = build_sdk_record(
        ROW, RECALL_ON, entries=entries, wall_time_ms=100.0, config=SDKExecConfig()
    )
    assert record.recall_call_count == 1
    assert record.retrieved_contexts == ("the memo",)
    assert record.metadata["init_present"] is True
    assert record.metadata["driver"] == "sdk"
    assert record.metadata["session_tools"] == ["Read", "mcp__recall__recall_search"]


def test_tool_results_pair_by_id_when_delivered_out_of_order() -> None:
    entries = [
        _init_entry(["mcp__recall__recall_search"]),
        _entry(
            "AssistantMessage",
            {
                "content": [
                    {"__type__": "ToolUseBlock", "id": "a", "name": "mcp__recall__recall_search", "input": {}},
                    {"__type__": "ToolUseBlock", "id": "b", "name": "Read", "input": {}},
                ]
            },
        ),
        _entry(
            "UserMessage",
            {"content": [{"__type__": "ToolResultBlock", "tool_use_id": "b", "content": "file text", "is_error": False}]},
        ),
        _entry(
            "UserMessage",
            {"content": [{"__type__": "ToolResultBlock", "tool_use_id": "a", "content": "memo text", "is_error": False}]},
        ),
        _result_entry(),
    ]
    record = build_sdk_record(
        ROW, RECALL_ON, entries=entries, wall_time_ms=1.0, config=SDKExecConfig()
    )
    by_name = {call["name"]: call for call in record.tool_calls}
    assert by_name["mcp__recall__recall_search"]["output"] == "memo text"
    assert by_name["Read"]["output"] == "file text"


def test_an_error_tool_result_with_string_content_does_not_crash() -> None:
    entries = [
        _init_entry(["Bash"]),
        _entry(
            "AssistantMessage",
            {"content": [{"__type__": "ToolUseBlock", "id": "x", "name": "Bash", "input": {"command": "boom"}}]},
        ),
        _entry(
            "UserMessage",
            {"content": [{"__type__": "ToolResultBlock", "tool_use_id": "x", "content": "command not found", "is_error": True}]},
        ),
        _result_entry(),
    ]
    record = build_sdk_record(
        ROW, RECALL_OFF, entries=entries, wall_time_ms=1.0, config=SDKExecConfig()
    )
    assert record.metadata["failed_tool_calls"] == 1


def test_input_tokens_sum_cache_components_from_model_usage() -> None:
    entries = [_init_entry(["Read"]), _result_entry()]
    record = build_sdk_record(
        ROW, RECALL_OFF, entries=entries, wall_time_ms=1.0, config=SDKExecConfig()
    )
    # 40 fresh + 1000 cache-read + 200 cache-creation, NOT usage.input_tokens's 34.
    assert record.input_tokens == 1240
    assert record.output_tokens == 55
    assert record.metadata["fresh_input_tokens"] == 40
    assert record.metadata["cache_read_input_tokens"] == 1000
    assert record.metadata["cache_creation_input_tokens"] == 200


def test_usage_is_the_fallback_when_model_usage_is_absent() -> None:
    entries = [
        _init_entry(["Read"]),
        _result_entry(model_usage=None, usage={"input_tokens": 10, "cache_read_input_tokens": 5, "output_tokens": 3}),
    ]
    record = build_sdk_record(
        ROW, RECALL_OFF, entries=entries, wall_time_ms=1.0, config=SDKExecConfig()
    )
    assert record.input_tokens == 15
    assert record.output_tokens == 3


def test_result_cost_is_recorded_as_untrusted_never_as_spend() -> None:
    entries = [_init_entry(["Read"]), _result_entry(total_cost_usd=1.23)]
    record = build_sdk_record(
        ROW, RECALL_OFF, entries=entries, wall_time_ms=1.0, config=SDKExecConfig()
    )
    assert record.metadata["reported_cost_usd_untrusted"] == 1.23
    assert record.system_cost_usd is None


def test_init_fields_feed_the_gate_and_a_missing_init_voids_the_session() -> None:
    on_without_tools = build_sdk_record(
        ROW,
        RECALL_ON,
        entries=[_init_entry(["Read"], servers=[{"name": "recall", "status": "failed"}]), _result_entry()],
        wall_time_ms=1.0,
        config=SDKExecConfig(),
    )
    verdict = check_session(on_without_tools)
    assert not verdict.admitted
    assert any("never available" in r or "no tool matching" in r for r in verdict.reasons)

    no_init = build_sdk_record(
        ROW, RECALL_ON, entries=[_result_entry()], wall_time_ms=1.0, config=SDKExecConfig()
    )
    assert not check_session(no_init).admitted

    off_with_tools = build_sdk_record(
        ROW,
        RECALL_OFF,
        entries=[_init_entry(["mcp__recall__recall_search"]), _result_entry()],
        wall_time_ms=1.0,
        config=SDKExecConfig(),
    )
    assert not check_session(off_with_tools).admitted

    clean_on = build_sdk_record(
        ROW,
        RECALL_ON,
        entries=[
            _init_entry(
                ["mcp__recall__recall_search"], servers=[{"name": "recall", "status": "connected"}]
            ),
            _result_entry(),
        ],
        wall_time_ms=1.0,
        config=SDKExecConfig(),
    )
    assert check_session(clean_on).admitted


def test_the_raw_typed_stream_is_written_as_messages_arrive(tmp_path, monkeypatch) -> None:
    # A fake query() that yields two messages then dies: the gz must already hold both lines.
    import types

    class FakeMessage:
        pass

    async def fake_query(*, prompt, options):
        yield {"content": [{"__type__": "TextBlock", "text": "one"}]}
        yield {"content": [{"__type__": "TextBlock", "text": "two"}]}
        raise RuntimeError("mid-session death")

    fake = types.ModuleType("claude_agent_sdk")

    class ClaudeAgentOptions:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)
            self.system_prompt = None

    fake.ClaudeAgentOptions = ClaudeAgentOptions  # type: ignore[attr-defined]
    fake.query = fake_query  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "claude_agent_sdk", fake)
    monkeypatch.setattr(
        "benchmarks.agent_ab.sdk_exec.resolve_claude_executable", lambda name="claude": "claude"
    )

    from benchmarks.agent_ab.sdk_exec import run_sdk_case

    config = SDKExecConfig(stream_dir=tmp_path)
    with pytest.raises(RuntimeError, match="mid-session death"):
        asyncio.run(run_sdk_case(ROW, RECALL_OFF, config))
    written = list(tmp_path.glob("*.jsonl.gz"))
    assert len(written) == 1
    with gzip.open(written[0], "rt", encoding="utf-8") as handle:
        lines = [json.loads(line) for line in handle]
    assert len(lines) == 2
    assert lines[0]["message_type"] == "dict"


def test_make_sdk_runner_requires_a_config_per_variant() -> None:
    with pytest.raises(ValueError, match="missing"):
        make_sdk_runner({RECALL_ON: SDKExecConfig()})


def test_build_sdk_configs_refuses_an_on_arm_without_a_recall_spec(tmp_path) -> None:
    prompt = tmp_path / "static.txt"
    prompt.write_text("static", encoding="utf-8")
    specs = {
        RECALL_ON: ArmSpec(profile="claude_md_recall", append_system_prompt_file=prompt),
        RECALL_OFF: ArmSpec.claude_md(prompt),
    }
    with pytest.raises(ValueError, match="recall_spec"):
        build_sdk_configs(specs, recall_spec=None, model="m", cwd=tmp_path)


def test_build_sdk_configs_gives_the_server_only_to_the_on_arm(tmp_path) -> None:
    from benchmarks.agent_ab.recall_server import StdioRecallSpec

    prompt = tmp_path / "static.txt"
    prompt.write_text("static", encoding="utf-8")
    spec = StdioRecallSpec(dsn="postgresql://x/y", cwd=tmp_path)
    specs = {
        RECALL_ON: ArmSpec(
            profile="claude_md_recall",
            append_system_prompt_file=prompt,
            extra_allowed_tools=("mcp__recall__recall_search",),
        ),
        RECALL_OFF: ArmSpec.claude_md(prompt),
    }
    configs = build_sdk_configs(
        specs, recall_spec=spec, model="m", cwd=tmp_path, extra_allowed_tools=("Write",)
    )
    assert configs[RECALL_ON].mcp_servers is not None
    assert configs[RECALL_OFF].mcp_servers is None
    assert "Write" in configs[RECALL_ON].allowed_tools
    assert "Write" in configs[RECALL_OFF].allowed_tools
    assert "mcp__recall__recall_search" in configs[RECALL_ON].allowed_tools
    assert configs[RECALL_ON].bare and configs[RECALL_OFF].bare


@pytest.mark.skipif(
    os.environ.get("AGENT_AB_SDK_LIVE") != "1",
    reason="live SDK smoke; set AGENT_AB_SDK_LIVE=1 (needs the agent extra, the CLI, and auth)",
)
@pytest.mark.skipif(shutil.which("claude") is None, reason="needs the Claude Code CLI on PATH")
@pytest.mark.timeout(300)
def test_sdk_live_readiness() -> None:
    pytest.importorskip("claude_agent_sdk")
    from benchmarks.agent_ab.sdk_exec import run_sdk_case

    config = SDKExecConfig(timeout_s=180.0, allowed_tools=())
    record = asyncio.run(
        run_sdk_case(
            {"task_id": "live", "user_input": "Reply with the single word READY."},
            RECALL_OFF,
            config,
        )
    )
    assert record.metadata["init_present"] is True
    assert record.input_tokens is not None and record.input_tokens > 0
    assert "READY" in (record.response or "").upper()
