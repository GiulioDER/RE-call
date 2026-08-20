"""Claude Code adapter for the paired benchmark runner.

The adapter consumes the JSONL stream emitted by ``claude -p --output-format stream-json
--verbose``. Like `codex_exec`, it does not decide how RE-call is configured: the caller supplies
one `ClaudeExecConfig` per variant, so the only intended difference between the arms stays
explicit and reviewable in the command line each arm actually ran.

Three things about the stream were established by capture rather than by reading documentation,
and each of them breaks a parser written the obvious way:

- **Tool results arrive out of order.** In the recorded fixture the result for the *second*
  `tool_use` is delivered before the result for the first. Anything that pairs a call with a
  result positionally reports the wrong latency for both. Pairing is by `tool_use_id`.
- **`tool_use_result` is a dict on success and a bare string on failure.** A parser that reaches
  for `.get(...)` on it raises `AttributeError` on exactly the tool calls this benchmark exists to
  count.
- **`system/init` carries no timestamp**, so wall time is measured around the subprocess here and
  the stream's own `duration_ms` is kept beside it as metadata rather than used as the answer.

Every subprocess here is started with `create_subprocess_exec`, which takes an argument vector and
never involves a shell, so a task prompt cannot become a command.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from .runner import Runner
from .schema import SessionRecord, VARIANTS

#: Every Claude Code MCP tool is named ``mcp__<server>__<tool>``, so one prefix identifies a
#: server's whole tool surface without enumerating it.
DEFAULT_RECALL_TOOL_PREFIX = "mcp__recall"


class ClaudeTranscriptError(ValueError):
    """Raised when a Claude Code JSONL stream cannot be interpreted."""


def resolve_claude_executable(name: str = "claude") -> str:
    """Resolve `claude` to something `CreateProcess` can actually start.

    On Windows, npm installs `claude` as a `.cmd` batch shim. `create_subprocess_exec` takes an
    argument vector and never goes through a shell, which is the property that keeps a task prompt
    from becoming a command, and it means a batch file cannot be started: the attempt fails with
    ``FileNotFoundError: [WinError 2]`` naming nothing in particular. The shim's last line calls a
    real `claude.exe` alongside it, so prefer that.

    Running the shim through `cmd /c` instead would work and is rejected on purpose: it puts a
    shell back between this harness and a prompt written by whoever authored the task file.
    """

    candidate = Path(name)
    if candidate.exists() and candidate.is_file():
        return str(candidate)

    found = shutil.which(name)
    if found is None:
        raise FileNotFoundError(
            f"{name!r} is not on PATH. Install Claude Code, or pass an absolute path as "
            f"ClaudeExecConfig(executable=...)."
        )
    resolved = Path(found)
    if resolved.suffix.lower() in {".cmd", ".bat", ".ps1"}:
        native = (
            resolved.parent
            / "node_modules"
            / "@anthropic-ai"
            / "claude-code"
            / "bin"
            / "claude.exe"
        )
        if native.is_file():
            return str(native)
        raise FileNotFoundError(
            f"{found} is a shell wrapper that cannot be started without a shell, and no "
            f"native executable was found at {native}. Pass an absolute path to the real "
            f"binary as ClaudeExecConfig(executable=...)."
        )
    return str(resolved)


@dataclass(frozen=True)
class ClaudeExecConfig:
    """Configuration for one isolated ``claude -p`` arm.

    `bare` defaults to True and is what makes the arms comparable: ``--bare`` skips hooks, plugins,
    auto memory, MCP auto-discovery and `CLAUDE.md`, so context reaches the session only through
    the flags recorded here. Without it, whatever happens to sit in the working directory or in
    ``~/.claude`` joins the experiment as an unrecorded variable.
    """

    executable: str = "claude"
    model: str | None = None
    cwd: str | Path | None = None
    timeout_s: float = 1800.0
    env: Mapping[str, str] = field(default_factory=dict)
    extra_args: tuple[str, ...] = ()
    bare: bool = True
    mcp_config: str | None = None
    strict_mcp_config: bool = True
    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    append_system_prompt_file: str | Path | None = None
    permission_mode: str | None = None
    add_dirs: tuple[str, ...] = ()
    recall_tool_prefix: str = DEFAULT_RECALL_TOOL_PREFIX

    def __post_init__(self) -> None:
        if not self.executable.strip():
            raise ValueError("executable must not be empty")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not self.recall_tool_prefix.strip():
            raise ValueError("recall_tool_prefix must not be empty")
        if self.strict_mcp_config and self.mcp_config is None:
            # --strict-mcp-config only means "ignore discovered servers"; with no --mcp-config it
            # silently produces a session with no MCP at all. For the off arm that is the intent,
            # but it has to be stated, because the same command is how an on arm loses its server.
            raise ValueError(
                "strict_mcp_config with mcp_config=None yields a session with no MCP servers; "
                "pass strict_mcp_config=False, or set mcp_config, to say which you meant"
            )

    def command(self, prompt: str) -> list[str]:
        """Build an argument list without invoking a shell."""

        command = [resolve_claude_executable(self.executable)]
        if self.bare:
            command.append("--bare")
        command.extend(("-p", prompt))
        command.extend(("--output-format", "stream-json", "--verbose"))
        if self.model:
            command.extend(("--model", self.model))
        if self.mcp_config is not None:
            command.extend(("--mcp-config", self.mcp_config))
        if self.strict_mcp_config:
            command.append("--strict-mcp-config")
        if self.allowed_tools:
            command.extend(("--allowed-tools", ",".join(self.allowed_tools)))
        if self.disallowed_tools:
            command.extend(("--disallowed-tools", ",".join(self.disallowed_tools)))
        if self.append_system_prompt_file is not None:
            command.extend(
                ("--append-system-prompt-file", str(self.append_system_prompt_file))
            )
        if self.permission_mode:
            command.extend(("--permission-mode", self.permission_mode))
        for directory in self.add_dirs:
            command.extend(("--add-dir", directory))
        command.extend(self.extra_args)
        return command


def parse_claude_stream_json(stream: str) -> list[dict[str, Any]]:
    """Parse and validate a Claude Code JSONL stream into plain mappings.

    Raises on a malformed line rather than skipping it. A skipped line is a dropped tool call or a
    dropped result, and both of those are silent under-counts in the direction that flatters
    whichever arm produced them.
    """

    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(stream.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise ClaudeTranscriptError(
                f"invalid JSON on line {line_number}; capture stderr separately, because a "
                f"merged stderr puts Claude Code's startup warnings into this stream"
            ) from error
        if not isinstance(event, dict):
            raise ClaudeTranscriptError(f"stream-json line {line_number} must be an object")
        events.append(event)
    return events


def init_event(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return the ``system``/``init`` event, which reports the session's real tool surface."""

    for event in events:
        if event.get("type") == "system" and event.get("subtype") == "init":
            return event
    return None


def result_event(events: Sequence[Mapping[str, Any]]) -> Mapping[str, Any] | None:
    """Return the terminal ``result`` event."""

    for event in reversed(events):
        if event.get("type") == "result":
            return event
    return None


def _content_blocks(event: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    message = event.get("message")
    if not isinstance(message, Mapping):
        return []
    content = message.get("content")
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return []
    return [block for block in content if isinstance(block, Mapping)]


def _timestamp_ms(event: Mapping[str, Any]) -> float | None:
    raw = event.get("timestamp")
    if not isinstance(raw, str) or not raw:
        return None
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp() * 1000.0
    except ValueError:
        return None


def _text_of(value: Any) -> str:
    """Flatten a content value that may be a string, a block, or a list of blocks."""

    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, Mapping):
        return str(value.get("text", "")) if "text" in value else json.dumps(value)
    if isinstance(value, Sequence) and not isinstance(value, bytes):
        return "".join(_text_of(item) for item in value)
    return str(value)


def _is_subagent(event: Mapping[str, Any]) -> bool:
    return event.get("parent_tool_use_id") is not None


@dataclass(frozen=True)
class TranscriptFields:
    """Everything the record needs that has to be reconstructed from the event stream."""

    conversation: tuple[dict[str, Any], ...]
    tool_calls: tuple[dict[str, Any], ...]
    response: str
    recall_call_count: int
    recall_latency_ms: float | None
    retrieved_contexts: tuple[str, ...]
    failed_tool_calls: int
    subagent_tool_calls: int


def transcript_fields(
    events: Sequence[Mapping[str, Any]],
    *,
    recall_tool_prefix: str = DEFAULT_RECALL_TOOL_PREFIX,
) -> TranscriptFields:
    """Reconstruct conversation, tool calls, RE-call usage and failures from the stream."""

    # tool_use_id -> the call, so a result can find its call however far away it lands.
    calls: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    conversation: list[dict[str, Any]] = []
    response = ""

    for event in events:
        event_type = event.get("type")
        if event_type == "assistant":
            blocks = _content_blocks(event)
            text = "".join(
                str(block.get("text", "")) for block in blocks if block.get("type") == "text"
            )
            turn_calls: list[dict[str, Any]] = []
            for block in blocks:
                if block.get("type") != "tool_use":
                    continue
                call_id = str(block.get("id") or f"call-{len(order)}")
                arguments = block.get("input")
                call = {
                    "id": call_id,
                    "name": str(block.get("name", "")),
                    "args": dict(arguments) if isinstance(arguments, Mapping) else {},
                    "started_ms": _timestamp_ms(event),
                    "subagent": _is_subagent(event),
                }
                calls[call_id] = call
                order.append(call_id)
                turn_calls.append({"name": call["name"], "args": call["args"]})
            if text or turn_calls:
                turn: dict[str, Any] = {"role": "assistant", "content": text}
                if turn_calls:
                    turn["tool_calls"] = turn_calls
                conversation.append(turn)
            if text and not _is_subagent(event):
                response = text
        elif event_type == "user":
            for block in _content_blocks(event):
                if block.get("type") != "tool_result":
                    continue
                call_id = str(block.get("tool_use_id", ""))
                call = calls.get(call_id)
                output = _text_of(block.get("content"))
                if call is None:
                    # A result with no matching call: record it rather than drop it, so the count
                    # of results never silently disagrees with the count of calls.
                    call = {
                        "id": call_id,
                        "name": "",
                        "args": {},
                        "started_ms": None,
                        "subagent": _is_subagent(event),
                    }
                    calls[call_id] = call
                    order.append(call_id)
                call["output"] = output
                call["is_error"] = bool(block.get("is_error"))
                finished = _timestamp_ms(event)
                started = call.get("started_ms")
                call["latency_ms"] = (
                    finished - started
                    if finished is not None and started is not None and finished >= started
                    else None
                )
                conversation.append({"role": "tool", "content": output})
        elif event_type == "result":
            final = event.get("result")
            if isinstance(final, str) and final:
                response = final

    ordered = [calls[call_id] for call_id in order]
    recall_calls = [call for call in ordered if call["name"].startswith(recall_tool_prefix)]
    latencies = [
        call["latency_ms"] for call in recall_calls if call.get("latency_ms") is not None
    ]
    contexts: list[str] = []
    for call in recall_calls:
        output = call.get("output")
        if isinstance(output, str) and output:
            contexts.append(output)

    tool_calls = tuple(
        {
            key: value
            for key, value in call.items()
            if key in {"name", "args", "output", "is_error", "latency_ms", "subagent"}
        }
        for call in ordered
    )
    return TranscriptFields(
        conversation=tuple(conversation),
        tool_calls=tool_calls,
        response=response,
        recall_call_count=len(recall_calls),
        # Total time the session spent inside RE-call, which is the overhead the memory layer
        # adds to the wall clock. None when nothing was measurable, never 0.
        recall_latency_ms=sum(latencies) if latencies else None,
        retrieved_contexts=tuple(contexts),
        failed_tool_calls=sum(1 for call in ordered if call.get("is_error")),
        subagent_tool_calls=sum(1 for call in ordered if call.get("subagent")),
    )


def _usage_fields(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {"input_tokens": None, "output_tokens": None, "model_turns": None}
    usage = result.get("usage")
    usage = usage if isinstance(usage, Mapping) else {}

    def integer(value: Any) -> int | None:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
        return int(value)

    return {
        "input_tokens": integer(usage.get("input_tokens")),
        "output_tokens": integer(usage.get("output_tokens")),
        "model_turns": integer(result.get("num_turns")),
        "cache_read_input_tokens": integer(usage.get("cache_read_input_tokens")),
        "cache_creation_input_tokens": integer(usage.get("cache_creation_input_tokens")),
    }


def _error_text(result: Mapping[str, Any] | None) -> str | None:
    if result is None:
        return "no result event in stream"
    if not result.get("is_error"):
        return None
    status = result.get("api_error_status")
    reason = str(result.get("terminal_reason") or result.get("subtype") or "error")
    detail = _text_of(result.get("result"))[:400]
    parts = [reason]
    if status:
        parts.append(f"(HTTP {status})")
    if detail:
        parts.append(f": {detail}")
    return " ".join(parts)


def build_record(
    row: Mapping[str, Any],
    variant: str,
    *,
    stream: str,
    wall_time_ms: float,
    config: ClaudeExecConfig,
    command: Sequence[str],
    exit_code: int | None = None,
    stderr: str = "",
) -> SessionRecord:
    """Turn one finished session's stream into a canonical `SessionRecord`."""

    events = parse_claude_stream_json(stream)
    init = init_event(events)
    result = result_event(events)
    fields = transcript_fields(events, recall_tool_prefix=config.recall_tool_prefix)
    usage = _usage_fields(result)

    session_tools = [str(name) for name in (init.get("tools") or [])] if init else []
    recall_tools = [
        name for name in session_tools if name.startswith(config.recall_tool_prefix)
    ]
    denials = list(result.get("permission_denials") or []) if result else []
    retries = [
        event
        for event in events
        if event.get("type") == "system" and event.get("subtype") == "api_retry"
    ]

    metadata: dict[str, Any] = {
        "command": list(command),
        "exit_code": exit_code,
        # Explicit, because every other gate signal below is read off `init`, and an absent init
        # event makes all of them look like a clean "no MCP servers" rather than "we never saw
        # what this session had".
        "init_present": init is not None,
        "claude_code_version": init.get("claude_code_version") if init else None,
        "model": (init.get("model") if init else None) or config.model,
        "session_id": result.get("session_id") if result else None,
        # The admission gate reads these three. They are the difference between "the arm had the
        # memory layer" and "the arm reported success without it".
        "mcp_servers": list(init.get("mcp_servers") or []) if init else [],
        "mcp_server_errors": list(init.get("mcp_server_errors") or []) if init else [],
        "recall_tools_available": recall_tools,
        "session_tools": session_tools,
        "stream_duration_ms": result.get("duration_ms") if result else None,
        "stream_duration_api_ms": result.get("duration_api_ms") if result else None,
        "ttft_ms": result.get("ttft_ms") if result else None,
        # Reported by the CLI, and measured wrong through a gateway: for 1,920 in / 45 out this
        # field read about 6x the list price of those tokens. Kept for completeness, never
        # published. Real spend is reconciled from the gateway's own accounting.
        "reported_cost_usd_untrusted": result.get("total_cost_usd") if result else None,
        "permission_denials": denials,
        "permission_denial_count": len(denials),
        "api_retries": len(retries),
        "failed_tool_calls": fields.failed_tool_calls,
        "subagent_tool_calls": fields.subagent_tool_calls,
        "cache_read_input_tokens": usage.get("cache_read_input_tokens"),
        "cache_creation_input_tokens": usage.get("cache_creation_input_tokens"),
        "stop_reason": result.get("stop_reason") if result else None,
        "stderr": stderr[-2000:] if stderr else "",
    }

    error = _error_text(result)
    user_input = str(row.get("user_input", ""))
    # The stream never echoes the prompt, but `MultiTurnSample` is a conversation and a
    # conversation that opens with the assistant is not one. Without this, every agent metric that
    # reads the human turn scores against an empty request.
    conversation = fields.conversation
    if user_input:
        conversation = ({"role": "user", "content": user_input},) + conversation
    return SessionRecord(
        task_id=str(row["task_id"]),
        variant=variant,
        # Whether the agent got the task right is decided by the graders, not here. This flag says
        # only that the session ran to completion, which is the precondition for grading it.
        success=error is None,
        user_input=user_input,
        response=fields.response,
        reference=row.get("reference"),
        retrieved_contexts=fields.retrieved_contexts,
        reference_contexts=tuple(row.get("reference_contexts") or ()),
        conversation=conversation,
        reference_tool_calls=tuple(row.get("reference_tool_calls") or ()),
        tool_calls=fields.tool_calls,
        recall_call_count=fields.recall_call_count,
        recall_latency_ms=fields.recall_latency_ms,
        input_tokens=usage["input_tokens"],
        output_tokens=usage["output_tokens"],
        model_turns=usage["model_turns"],
        wall_time_ms=wall_time_ms,
        error=error,
        metadata=metadata,
    )


async def run_claude_case(
    row: Mapping[str, Any], variant: str, config: ClaudeExecConfig
) -> SessionRecord:
    """Run one task in one arm and return its record."""

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    prompt = str(row.get("user_input", "")).strip()
    if not prompt:
        raise ValueError(f"task {row.get('task_id')!r} has an empty user_input")

    command = config.command(prompt)
    environment = dict(os.environ)
    environment.update({str(key): str(value) for key, value in config.env.items()})

    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *command,
        # Claude Code waits 3s for stdin before giving up and warning. Closing it returns that
        # 3 seconds on every session in the run.
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(config.cwd) if config.cwd else None,
        env=environment,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=config.timeout_s)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
        raise ClaudeTranscriptError(
            f"claude exceeded timeout_s={config.timeout_s} for task {row.get('task_id')!r}"
        ) from None
    wall_time_ms = (time.perf_counter() - started) * 1000.0

    return build_record(
        row,
        variant,
        stream=stdout.decode("utf-8", errors="replace"),
        wall_time_ms=wall_time_ms,
        config=config,
        command=command,
        exit_code=process.returncode,
        stderr=stderr.decode("utf-8", errors="replace"),
    )


def make_claude_runner(configs: Mapping[str, ClaudeExecConfig]) -> Runner:
    """Build a `Runner` that dispatches each variant to its own configuration."""

    missing = [variant for variant in VARIANTS if variant not in configs]
    if missing:
        raise ValueError(f"a configuration is required for every variant; missing {missing}")

    async def run_case(row: Mapping[str, Any], variant: str) -> SessionRecord:
        return await run_claude_case(row, variant, configs[variant])

    return run_case
