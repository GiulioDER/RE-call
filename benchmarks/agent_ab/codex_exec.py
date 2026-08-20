"""Codex CLI adapter for the paired benchmark runner.

The adapter consumes the JSONL stream emitted by ``codex exec --json``. It does not decide how
RE-call is configured. Callers provide separate environments or Codex configuration directories
for the two variants, so the benchmark can keep the only intended difference explicit.
"""

from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .runner import Runner
from .schema import SessionRecord, VARIANTS


class CodexTranscriptError(ValueError):
    """Raised when a Codex JSONL stream cannot be interpreted."""


@dataclass(frozen=True)
class CodexExecConfig:
    """Configuration for one isolated ``codex exec`` arm."""

    executable: str = "codex"
    model: str | None = None
    sandbox: str = "read-only"
    cwd: str | Path | None = None
    timeout_s: float = 1800.0
    env: Mapping[str, str] = field(default_factory=dict)
    extra_args: tuple[str, ...] = ()
    ephemeral: bool = True
    skip_git_repo_check: bool = False
    ignore_user_config: bool = False
    ignore_rules: bool = False
    recall_markers: tuple[str, ...] = ("recall",)

    def __post_init__(self) -> None:
        if not self.executable.strip():
            raise ValueError("executable must not be empty")
        if self.sandbox not in {"read-only", "workspace-write", "danger-full-access"}:
            raise ValueError("sandbox must be a supported Codex sandbox mode")
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if any(not marker.strip() for marker in self.recall_markers):
            raise ValueError("recall_markers must not contain empty values")

    def command(self, prompt: str) -> list[str]:
        """Build an argument list without invoking a shell."""

        command = [self.executable, "exec", "--json"]
        if self.model:
            command.extend(("--model", self.model))
        command.extend(("--sandbox", self.sandbox))
        if self.ephemeral:
            command.append("--ephemeral")
        if self.skip_git_repo_check:
            command.append("--skip-git-repo-check")
        if self.ignore_user_config:
            command.append("--ignore-user-config")
        if self.ignore_rules:
            command.append("--ignore-rules")
        command.extend(self.extra_args)
        command.append(prompt)
        return command


def _event_type(event: Mapping[str, Any]) -> str:
    event_type = event.get("type")
    if isinstance(event_type, str):
        return event_type
    message = event.get("msg")
    if isinstance(message, Mapping) and isinstance(message.get("type"), str):
        return str(message["type"])
    return ""


def _event_payload(event: Mapping[str, Any]) -> Mapping[str, Any]:
    message = event.get("msg")
    if not isinstance(event.get("type"), str) and isinstance(message, Mapping):
        return message
    return event


def _item(event: Mapping[str, Any]) -> Mapping[str, Any] | None:
    payload = _event_payload(event)
    value = payload.get("item")
    return value if isinstance(value, Mapping) else None


def _text(value: Any) -> str:
    if isinstance(value, str):
        return value
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return "".join(_text(part) for part in value)
    return "" if value is None else str(value)


def parse_codex_jsonl(stream: str) -> list[dict[str, Any]]:
    """Parse and validate a Codex JSONL stream into plain mappings."""

    events: list[dict[str, Any]] = []
    for line_number, line in enumerate(stream.splitlines(), 1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as error:
            raise CodexTranscriptError(f"invalid JSON on line {line_number}") from error
        if not isinstance(event, dict):
            raise CodexTranscriptError(f"Codex JSONL line {line_number} must be an object")
        events.append(event)
    return events


def _usage_totals(events: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cached_input_tokens": 0,
        "reasoning_output_tokens": 0,
    }
    observed = False
    for event in events:
        if _event_type(event) not in {"turn.completed", "turn_complete"}:
            continue
        usage = _event_payload(event).get("usage")
        if not isinstance(usage, Mapping):
            continue
        observed = True
        for key in totals:
            value = usage.get(key)
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                totals[key] += int(value)
    return totals if observed else {}


def _item_type(item: Mapping[str, Any]) -> str:
    value = item.get("type")
    return value if isinstance(value, str) else ""


def _is_tool_item(item: Mapping[str, Any]) -> bool:
    return _item_type(item) in {
        "apply_patch",
        "command_execution",
        "computer_call",
        "file_change",
        "mcp_tool_call",
        "mcp_call",
        "web_search",
    }


def _tool_mapping(item: Mapping[str, Any]) -> dict[str, Any]:
    item_type = _item_type(item)
    name = item.get("name") or item.get("tool_name") or item_type
    arguments = item.get("arguments") or item.get("input") or {}
    result = item.get("aggregated_output") or item.get("output") or item.get("result")
    mapping: dict[str, Any] = {
        "name": str(name),
        "type": item_type,
        "args": dict(arguments) if isinstance(arguments, Mapping) else {"raw": str(arguments)},
    }
    if result is not None:
        mapping["output"] = _text(result)
    if item.get("status") is not None:
        mapping["status"] = str(item["status"])
    return mapping


def _is_recall_item(item: Mapping[str, Any], markers: Sequence[str]) -> bool:
    if _item_type(item) not in {"mcp_tool_call", "mcp_call"}:
        return False
    identity = " ".join(
        str(item.get(key, ""))
        for key in ("name", "tool_name", "server", "server_name", "mcp_server")
    ).lower()
    return any(marker.lower() in identity for marker in markers)


def _transcript_fields(
    events: Sequence[Mapping[str, Any]],
    *,
    recall_markers: Sequence[str],
) -> dict[str, Any]:
    items: dict[str, Mapping[str, Any]] = {}
    conversations: dict[str, dict[str, Any]] = {}
    tool_calls: dict[str, dict[str, Any]] = {}
    for event in events:
        item = _item(event)
        if item is None:
            continue
        item_id = str(item.get("id") or f"item-{len(items)}")
        items[item_id] = item
        item_type = _item_type(item)
        if item_type == "agent_message":
            content = _text(item.get("text") or item.get("content"))
            conversations[item_id] = {"role": "assistant", "content": content}
        if _is_tool_item(item):
            tool_calls[item_id] = _tool_mapping(item)

    turn_completed = sum(
        _event_type(event) in {"turn.completed", "turn_complete"} for event in events
    )
    turn_started = sum(_event_type(event) in {"turn.started", "turn_start"} for event in events)
    usage = _usage_totals(events)
    thread_id = next(
        (
            str(event.get("thread_id"))
            for event in events
            if isinstance(event.get("thread_id"), str)
        ),
        None,
    )
    failed = any(
        _event_type(event) in {"turn.failed", "error", "turn_failed"} for event in events
    )
    errors = [
        _text(_event_payload(event).get("message") or _event_payload(event).get("error"))
        for event in events
        if _event_type(event) in {"turn.failed", "error", "turn_failed"}
    ]
    return {
        "response": next(
            (
                value["content"]
                for value in reversed(list(conversations.values()))
                if value["content"]
            ),
            "",
        ),
        "conversation": list(conversations.values()),
        "tool_calls": list(tool_calls.values()),
        "recall_call_count": sum(
            _is_recall_item(item, recall_markers) for item in items.values()
        ),
        "model_turns": turn_completed or turn_started or None,
        "failed": failed,
        "error_messages": [message for message in errors if message],
        "thread_id": thread_id,
        **usage,
    }


def _row_sequence(row: Mapping[str, Any], key: str) -> list[Any]:
    value = row.get(key, [])
    if value is None:
        return []
    if isinstance(value, (str, bytes)) or not isinstance(value, Sequence):
        raise TypeError(f"row[{key!r}] must be a sequence")
    return list(value)


async def run_codex_case(
    row: Mapping[str, Any],
    variant: str,
    config: CodexExecConfig,
) -> SessionRecord:
    """Run one Codex CLI case and convert its JSONL stream into a session record."""

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}")
    prompt = row.get("prompt") or row.get("user_input")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ValueError("row must contain a nonempty prompt or user_input")
    cwd = row.get("cwd") or config.cwd
    environment = os.environ.copy()
    environment.update(config.env)
    started = time.perf_counter()
    process = await asyncio.create_subprocess_exec(
        *config.command(prompt),
        cwd=str(cwd) if cwd is not None else None,
        env=environment,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    timed_out = False
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), config.timeout_s)
    except TimeoutError:
        timed_out = True
        process.kill()
        stdout, stderr = await process.communicate()
    wall_time_ms = (time.perf_counter() - started) * 1000
    events = parse_codex_jsonl(stdout.decode("utf-8", errors="replace"))
    fields = _transcript_fields(events, recall_markers=config.recall_markers)
    error_messages = list(fields.pop("error_messages"))
    returncode = process.returncode
    if timed_out:
        error_messages.insert(0, f"Codex timed out after {config.timeout_s:g} seconds")
    if returncode:
        error_messages.insert(0, f"Codex exited with status {returncode}")
    if stderr and returncode and not error_messages:
        error_messages.append("Codex wrote diagnostics to stderr")
    error = "; ".join(error_messages) or None
    metadata = dict(row.get("metadata", {})) if isinstance(row.get("metadata"), Mapping) else {}
    metadata.update(
        {
            "codex_thread_id": fields.pop("thread_id"),
            "codex_returncode": returncode,
            "codex_event_types": sorted({_event_type(event) for event in events if _event_type(event)}),
            "cached_input_tokens": fields.pop("cached_input_tokens", None),
            "reasoning_output_tokens": fields.pop("reasoning_output_tokens", None),
            "stderr_present": bool(stderr),
        }
    )
    return SessionRecord(
        task_id=str(row["task_id"]),
        variant=variant,
        success=returncode == 0 and not fields.pop("failed"),
        user_input=prompt,
        response=str(fields.pop("response")),
        reference=str(row["reference"]) if row.get("reference") is not None else None,
        retrieved_contexts=tuple(str(item) for item in _row_sequence(row, "retrieved_contexts")),
        reference_contexts=tuple(str(item) for item in _row_sequence(row, "reference_contexts")),
        conversation=tuple(fields.pop("conversation")),
        reference_tool_calls=tuple(_row_sequence(row, "reference_tool_calls")),
        tool_calls=tuple(fields.pop("tool_calls")),
        recall_call_count=int(fields.pop("recall_call_count")),
        input_tokens=fields.pop("input_tokens", None),
        output_tokens=fields.pop("output_tokens", None),
        model_turns=fields.pop("model_turns"),
        wall_time_ms=wall_time_ms,
        abstained=bool(row.get("abstained", False)),
        trust_verdicts=tuple(str(item) for item in _row_sequence(row, "trust_verdicts")),
        error=error,
        metadata=metadata,
    )


def make_codex_runner(
    configs: Mapping[str, CodexExecConfig],
) -> Runner:
    """Create a paired runner from separate Codex configurations."""

    missing = set(VARIANTS) - set(configs)
    if missing:
        raise ValueError(f"configs missing variants: {sorted(missing)}")

    async def runner(row: Mapping[str, Any], variant: str) -> SessionRecord:
        return await run_codex_case(row, variant, configs[variant])

    return runner
