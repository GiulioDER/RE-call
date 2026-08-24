"""Tests for the Claude Code adapter and the admission gate.

Both fixtures under `tests/fixtures/agent_ab/` are **real captured streams**, not hand-written
ones, recorded on 2026-08-20 against Claude Code 2.1.220. That matters for two of the tests here:
the out-of-order tool results and the string-shaped `tool_use_result` are properties of the actual
stream that a plausible hand-written fixture would not have reproduced, so a parser written
against an invented fixture passes its tests and fails on the first real session.
"""

from __future__ import annotations

import gzip
import json
import shutil
import sys
from pathlib import Path

import pytest

from benchmarks.agent_ab.claude_exec import (
    ClaudeExecConfig,
    ClaudeTranscriptError,
    build_record,
    init_event,
    resolve_claude_executable,
    parse_claude_stream_json,
    transcript_fields,
)
from benchmarks.agent_ab.gate import admit_pairs, check_session
from benchmarks.agent_ab.schema import RECALL_OFF, RECALL_ON, SessionRecord

#: Any existing absolute path satisfies `resolve_claude_executable`, which short-circuits
#: before it consults PATH. Using the running interpreter keeps these tests about the
#: command that gets built rather than about whether Claude Code happens to be installed:
#: on CI it is not, and three of these failed for that reason alone.
EXECUTABLE = sys.executable

FIXTURES = Path(__file__).parent / "fixtures" / "agent_ab"
BARE_TOOLS = FIXTURES / "claude_bare_tools.jsonl"
MCP_PENDING = FIXTURES / "claude_mcp_pending.jsonl"


def _stream(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _config(**kwargs) -> ClaudeExecConfig:
    kwargs.setdefault("strict_mcp_config", False)
    # Every config a test builds names an executable, so nothing here depends on Claude Code being
    # installed. Setting it in the helper rather than at each call site is what makes that true for
    # tests added later too: the CI runner has no `claude` on PATH.
    kwargs.setdefault("executable", EXECUTABLE)
    return ClaudeExecConfig(**kwargs)


def _record(path: Path, variant: str = RECALL_ON, **kwargs) -> SessionRecord:
    config = kwargs.pop("config", _config())
    return build_record(
        {"task_id": "t1", "user_input": "irrelevant"},
        variant,
        stream=_stream(path),
        wall_time_ms=1234.0,
        config=config,
        command=["claude", "--bare"],
        exit_code=0,
        **kwargs,
    )


# --------------------------------------------------------------------------- command building


def test_command_puts_prompt_after_flags_and_never_uses_a_shell() -> None:
    config = ClaudeExecConfig(
        executable=EXECUTABLE, model="anthropic/claude-haiku-4.5",
        mcp_config='{"mcpServers":{}}',
        allowed_tools=("Bash", "mcp__recall-memory__recall_search"),
        disallowed_tools=("Bash(docker *)",),
        permission_mode="dontAsk",
    )
    command = config.command("do the thing")
    assert "--bare" in command
    assert command[command.index("-p") + 1] == "do the thing"
    assert command[command.index("--output-format") + 1] == "stream-json"
    assert "--verbose" in command
    assert "--strict-mcp-config" in command
    assert command[command.index("--allowed-tools") + 1] == (
        "Bash,mcp__recall-memory__recall_search"
    )
    assert command[command.index("--disallowed-tools") + 1] == "Bash(docker *)"


@pytest.mark.skipif(shutil.which("claude") is None, reason="Claude Code is not installed here")
def test_the_npm_shim_is_resolved_to_a_native_binary() -> None:
    """A `.cmd` wrapper cannot be started at all, so resolution must reach the real executable.

    `create_subprocess_exec` takes an argv list and starts no shell, which is the property that
    stops a task prompt becoming a command. That is also why the npm batch shim is unusable and
    must be resolved past. Skipped rather than failed where Claude Code is absent: on a runner
    without it this asserts something about the machine, not about the code.
    """

    resolved = Path(resolve_claude_executable())
    assert resolved.stem.lower() == "claude"
    assert resolved.suffix.lower() not in {".cmd", ".bat", ".ps1"}


def test_strict_mcp_config_without_a_config_must_be_stated_explicitly() -> None:
    # The combination is legitimate for the off arm and catastrophic for the on arm, and the two
    # look identical on the command line, so it may not be reached by defaulting.
    with pytest.raises(ValueError, match="no MCP servers"):
        ClaudeExecConfig(executable=EXECUTABLE, strict_mcp_config=True, mcp_config=None)
    assert ClaudeExecConfig(executable=EXECUTABLE, strict_mcp_config=False).command("x")


# --------------------------------------------------------------------------- stream parsing


def test_parse_rejects_a_non_json_line_rather_than_skipping_it() -> None:
    # Claude Code prints startup warnings to stderr; a caller that merges the two streams gets
    # them interleaved here. Skipping the line would silently drop whatever followed it.
    stream = _stream(BARE_TOOLS).replace(
        "\n", "\nWarning: no stdin data received in 3s, proceeding without it.\n", 1
    )
    with pytest.raises(ClaudeTranscriptError, match="capture stderr separately"):
        parse_claude_stream_json(stream)


def test_real_stream_yields_tool_calls_a_response_and_usage() -> None:
    record = _record(BARE_TOOLS)
    assert record.error is None and record.success
    names = [call["name"] for call in record.tool_calls]
    assert names == ["Bash", "Bash"]
    assert "hello-from-bash" in record.response
    assert record.input_tokens and record.input_tokens > 0
    assert record.output_tokens and record.output_tokens > 0
    assert record.model_turns is not None
    # Wall time is measured around the subprocess, not taken from the stream.
    assert record.wall_time_ms == 1234.0
    assert record.metadata["stream_duration_ms"] is not None
    assert record.metadata["claude_code_version"] == "2.1.220"


def test_a_failed_tool_call_is_counted_and_does_not_crash_the_parser() -> None:
    # In the capture, the failing call's `tool_use_result` is a bare STRING while the succeeding
    # call's is a dict. A parser that assumes a mapping raises on exactly the calls that matter.
    record = _record(BARE_TOOLS)
    failures = [call for call in record.tool_calls if call.get("is_error")]
    assert len(failures) == 1
    assert "definitely-missing-dir" in failures[0]["args"]["command"]
    assert record.metadata["failed_tool_calls"] == 1


def test_tool_latency_is_paired_by_id_not_by_position() -> None:
    """The captured stream delivers the SECOND call's result first.

    Positional pairing produces plausible, wrong numbers for both calls, which is the worst kind
    of bug in a benchmark: it never raises and it always answers.
    """
    events = parse_claude_stream_json(_stream(BARE_TOOLS))
    starts, finishes = {}, {}
    for event in events:
        for block in event.get("message", {}).get("content", []) or []:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_use":
                starts[block["id"]] = event["timestamp"]
            elif block.get("type") == "tool_result":
                finishes[block["tool_use_id"]] = event["timestamp"]
    call_order = list(starts)
    result_order = list(finishes)
    assert call_order != result_order, "fixture no longer exercises out-of-order results"

    fields = transcript_fields(events)
    by_command = {
        call["args"].get("command"): call["latency_ms"] for call in fields.tool_calls
    }
    # 09.305 -> 12.045 and 09.505 -> 11.971. Positional pairing would give 2666 and 2540.
    assert by_command["ls definitely-missing-dir"] == pytest.approx(2740, abs=5)
    assert by_command["echo hello-from-bash"] == pytest.approx(2466, abs=5)


def test_the_raw_stream_is_written_before_it_is_parsed(tmp_path: Path, monkeypatch) -> None:
    """The transcript is the evidence, so it must survive a stream the parser rejects.

    Writing it after `build_record` would lose exactly the sessions worth looking at: the ones
    whose output was malformed enough to raise.
    """
    import asyncio

    from benchmarks.agent_ab import claude_exec

    stream_dir = tmp_path / "streams"
    config = _config(stream_dir=stream_dir)
    row = {"task_id": "task/with:awkward chars", "user_input": "go"}

    class _FakeProcess:
        returncode = 0

        async def communicate(self) -> tuple[bytes, bytes]:
            return b"this is not json\n", b""

    async def _fake_launch(*_args, **_kwargs):
        return _FakeProcess()

    monkeypatch.setattr(claude_exec.asyncio, "create_subprocess_exec", _fake_launch)
    with pytest.raises(ClaudeTranscriptError):
        asyncio.run(claude_exec.run_claude_case(row, RECALL_ON, config))

    written = list(stream_dir.glob("*.gz"))
    assert len(written) == 1, "the unparseable stream must still be on disk"
    assert "/" not in written[0].name and ":" not in written[0].name
    with gzip.open(written[0], "rt", encoding="utf-8") as handle:
        assert handle.read() == "this is not json\n"


def test_build_record_does_not_claim_a_stream_file_it_did_not_write() -> None:
    # Capture belongs to the run path. A record built from an already-saved stream must not
    # advertise a path, or the artifact index would point at files that are not there.
    record = _record(BARE_TOOLS, config=_config(stream_dir="/nowhere"))
    assert "stream_path" not in record.metadata


def test_conversation_opens_with_the_human_turn_the_stream_never_echoes() -> None:
    # Ragas MultiTurnSample is a conversation. Without the opening request, every agent metric
    # that reads the human turn scores the assistant against an empty ask.
    record = _record(BARE_TOOLS)
    assert record.conversation[0] == {"role": "user", "content": "irrelevant"}
    assert [turn["role"] for turn in record.conversation[1:3]] == ["assistant", "assistant"]
    assert any(turn["role"] == "tool" for turn in record.conversation)


def test_input_tokens_counts_cached_tokens_because_they_are_most_of_the_input() -> None:
    """Real numbers from a paired smoke session on 2026-08-20.

    `usage.input_tokens` was **34** while the session actually put **26,359** input tokens through
    the model, because cache reads and cache creation are not in that field. Reported naively the
    arms read 26 against 14; honestly they read about 26,359 against 18,189, a 45% difference. A
    summary built on the wrong field is not noisy, it is unrelated to the quantity it names, and
    nothing downstream could have caught it.
    """
    result = {
        "type": "result",
        "is_error": False,
        "num_turns": 4,
        "usage": {
            "input_tokens": 34,
            "cache_creation_input_tokens": 7320,
            "cache_read_input_tokens": 18459,
            "output_tokens": 941,
        },
        "modelUsage": {
            "anthropic/claude-haiku-4.5": {
                "inputTokens": 580,
                "outputTokens": 957,
                "cacheReadInputTokens": 18459,
                "cacheCreationInputTokens": 7320,
            }
        },
    }
    stream = json.dumps({"type": "system", "subtype": "init", "tools": []}) + "\n"
    stream += json.dumps(result) + "\n"
    record = build_record(
        {"task_id": "t", "user_input": "x"},
        RECALL_ON,
        stream=stream,
        wall_time_ms=1.0,
        config=_config(),
        command=["claude"],
    )
    assert record.input_tokens == 580 + 18459 + 7320 == 26359
    # modelUsage is the session aggregate; usage looks like the final request only, and the two
    # disagreed 580 to 34 on this very run. The aggregate wins.
    assert record.metadata["fresh_input_tokens"] == 580
    assert record.output_tokens == 957
    # The components survive, because a cache read and a fresh token are not priced alike.
    assert record.metadata["cache_read_input_tokens"] == 18459
    assert record.metadata["cache_creation_input_tokens"] == 7320


def test_usage_is_the_fallback_when_a_stream_has_no_model_aggregate() -> None:
    stream = json.dumps({"type": "system", "subtype": "init", "tools": []}) + "\n"
    stream += json.dumps(
        {
            "type": "result",
            "is_error": False,
            "num_turns": 1,
            "usage": {
                "input_tokens": 100,
                "cache_read_input_tokens": 900,
                "output_tokens": 7,
            },
        }
    ) + "\n"
    record = build_record(
        {"task_id": "t", "user_input": "x"},
        RECALL_ON,
        stream=stream,
        wall_time_ms=1.0,
        config=_config(),
        command=["claude"],
    )
    assert record.input_tokens == 1000
    assert record.output_tokens == 7


def test_untrusted_cost_is_recorded_under_a_name_that_says_so() -> None:
    record = _record(BARE_TOOLS)
    assert "reported_cost_usd_untrusted" in record.metadata
    assert record.metadata["reported_cost_usd_untrusted"] is not None
    assert record.system_cost_usd is None, "the gateway's estimate must not become the cost field"


# --------------------------------------------------------------------------- the admission gate


def test_the_pending_server_capture_is_the_failure_the_gate_exists_for() -> None:
    """This is the 2026-08-20 capture, verbatim: a successful run with no treatment applied."""
    events = parse_claude_stream_json(_stream(MCP_PENDING))
    init = init_event(events)
    assert init is not None
    assert init["mcp_servers"] == [{"name": "recall-memory", "status": "pending"}]
    assert not [tool for tool in init["tools"] if tool.startswith("mcp__recall")]
    result = [event for event in events if event["type"] == "result"][-1]
    # Nothing in the terminal state distinguishes this from a real on-arm session.
    assert result["is_error"] is False
    assert result["subtype"] == "success"
    assert result["permission_denials"] == []


def test_gate_rejects_an_on_arm_whose_server_never_connected() -> None:
    verdict = check_session(_record(MCP_PENDING, RECALL_ON))
    assert not verdict.admitted
    assert any("never available" in reason for reason in verdict.reasons)
    assert any("pending" in reason for reason in verdict.reasons)


def test_gate_admits_the_same_session_as_an_off_arm() -> None:
    # The identical transcript is perfectly good evidence for the arm that is supposed to have no
    # memory layer. What it can never be is evidence for the arm that is supposed to have one.
    assert check_session(_record(MCP_PENDING, RECALL_OFF)).admitted


def test_gate_rejects_an_off_arm_that_could_see_recall() -> None:
    record = _record(BARE_TOOLS, RECALL_OFF)
    leaked = SessionRecord.from_mapping(
        {
            **record.to_dict(),
            "metadata": {
                **record.metadata,
                "recall_tools_available": ["mcp__recall-memory__recall_search"],
            },
        }
    )
    verdict = check_session(leaked)
    assert not verdict.admitted
    assert any("differ by less" in reason for reason in verdict.reasons)


def test_available_but_never_called_is_a_note_and_stays_in_the_data() -> None:
    """Availability is the gate; usage is a finding. Conflating them manufactures a null."""
    record = _record(BARE_TOOLS, RECALL_ON)
    equipped = SessionRecord.from_mapping(
        {
            **record.to_dict(),
            "recall_call_count": 0,
            "metadata": {
                **record.metadata,
                "recall_tools_available": ["mcp__recall-memory__recall_search"],
                "mcp_servers": [{"name": "recall-memory", "status": "connected"}],
            },
        }
    )
    verdict = check_session(equipped)
    assert verdict.admitted
    assert any("never called" in note for note in verdict.notes)


def test_a_pair_is_discarded_whole_and_reported_by_task_id() -> None:
    on = _record(MCP_PENDING, RECALL_ON)
    off = _record(BARE_TOOLS, RECALL_OFF)
    report = admit_pairs([on, off])
    assert report.admitted == ()
    assert report.discarded_task_ids == ("t1",)
    summary = report.summary()
    assert summary["admitted_pairs"] == 0 and summary["discarded_pairs"] == 1
    # The account of what was dropped, and why, has to survive into the artifact.
    assert json.dumps(summary)
    assert any(verdict["reasons"] for verdict in summary["verdicts"])


def test_a_missing_arm_discards_the_pair_rather_than_unpairing_the_design() -> None:
    report = admit_pairs([_record(BARE_TOOLS, RECALL_OFF)])
    assert report.discarded_task_ids == ("t1",)
    assert any(
        "no recall_on record" in reason
        for verdict in report.verdicts
        for reason in verdict.reasons
    )


def test_a_session_with_no_init_event_cannot_be_verified_either_way() -> None:
    stream = "\n".join(
        line
        for line in _stream(BARE_TOOLS).splitlines()
        if json.loads(line).get("subtype") != "init"
    )
    record = build_record(
        {"task_id": "t1", "user_input": "x"},
        RECALL_ON,
        stream=stream,
        wall_time_ms=1.0,
        config=_config(),
        command=["claude"],
    )
    assert record.metadata["init_present"] is False
    verdict = check_session(record)
    assert not verdict.admitted
    assert any("never observed" in reason for reason in verdict.reasons)
