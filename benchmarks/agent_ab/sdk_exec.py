"""Claude Agent SDK adapter for the paired benchmark runner.

Prior work: `benchmarks/agent_ab/claude_exec.py` is the driver this supersedes, and its parsing
core is REUSED here rather than reimplemented, so its three recorded stream findings keep one
owner. `benchmarks/agent_ab/NEXT-BENCHMARK-MULTI-PRODUCT.md` (approved 2026-08-22) is the plan
this is shaped for: the `MemoryAdapter` seam and per-session `CLAUDE_CONFIG_DIR` hermeticity are
its decisions, not new ones. `recall_interop/memory_benchmarks.py` is the existing example of
adapting RE-call to a foreign client contract and supplied the sync-in-thread pattern. No prior
SDK-driven harness existed in this repository: `claude_agent_sdk` appeared nowhere before this
module, which is why the driver is new and the endpoint, gate and schema around it are not.

Supersedes `claude_exec`'s subprocess-and-parse driver without touching it: `claude_exec.py`
produced every archived baseline and stays byte-identical so those runs remain reproducible from
the code that ran them, and so "only the driver differs" is checkable as a diff. This module
receives TYPED messages from the SDK, normalizes them back into the stream-json event shapes, and
hands them to `claude_exec`'s own parsing core — `transcript_fields`, `init_event`,
`result_event`, `_usage_fields` — so the three hard-won stream facts (results pair by
`tool_use_id`, an error result is a bare string, input tokens are fresh + cache-read +
cache-creation from the per-model aggregate) keep living in exactly one place.

Two equivalence decisions are load-bearing for comparability with the CLI-driven baselines:

- **The system prompt is never replaced.** The SDK's plain-string `system_prompt` REPLACES the
  default Claude Code prompt, while the baselines ran `--append-system-prompt-file`, which
  APPENDS. `SDKExecConfig` therefore has no system-prompt field at all: the append file travels
  as the literal CLI flag through `extra_args`, so both arms keep the same base prompt the
  archived runs had. The preset+append form is the successor's business, where it gets its own
  verification.
- **Hermeticity is `--bare`, passed literally.** `setting_sources` is left unset (the SDK then
  loads no user or project settings) AND the literal `--bare` flag is passed through
  `extra_args`, because `--bare` is what the baselines ran under and a literal flag has no
  semantic mapping to get wrong. The multi-product successor replaces this with a per-session
  `CLAUDE_CONFIG_DIR`, because competitor integrations ARE hooks and plugins that `--bare` would
  strip; nothing here blocks that — it is one `env` entry and dropping the flag.

The environment is always the FULL merged `os.environ | config.env`, mirroring
`run_claude_case`: whether `options.env` replaces or extends the child environment is exactly the
ambiguity that the MCP `env`-block lesson says not to gamble on.
"""

from __future__ import annotations

import asyncio
import dataclasses
import gzip
import json
import os
import re
import time
from collections.abc import Mapping, Sequence
from contextlib import aclosing
from dataclasses import dataclass, field
from datetime import datetime, timezone
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .arms import BASE_TOOLS, DENIED_TOOLS, OFF_ARM_PROFILES, ON_ARM_PROFILES, ArmSpec
from .claude_exec import (
    DEFAULT_RECALL_TOOL_PREFIX,
    ClaudeExecConfig,
    build_record,
    resolve_claude_executable,
)
from .runner import Runner
from .schema import RECALL_OFF, RECALL_ON, SessionRecord, VARIANTS

if TYPE_CHECKING:  # pragma: no cover
    from .recall_server import StdioRecallSpec


class SDKTranscriptError(ValueError):
    """Raised when an SDK session cannot be interpreted or does not finish."""


def sdk_version() -> str | None:
    try:
        return importlib_metadata.version("claude-agent-sdk")
    except importlib_metadata.PackageNotFoundError:
        return None


def _jsonable(value: Any) -> Any:
    """Serialize a typed message losslessly enough to be evidence. Must never raise.

    Dataclasses keep their class name under `__type__`, because `dataclasses.asdict` erases the
    one thing that distinguishes a `TextBlock` from a `ToolUseBlock`. A serializer crash would
    cost exactly the transcript it exists to keep, so the fallback is `repr`.
    """

    try:
        if dataclasses.is_dataclass(value) and not isinstance(value, type):
            out: dict[str, Any] = {"__type__": type(value).__name__}
            for item in dataclasses.fields(value):
                out[item.name] = _jsonable(getattr(value, item.name))
            return out
        if isinstance(value, Mapping):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value
        if isinstance(value, (list, tuple, set)):
            return [_jsonable(item) for item in value]
        if isinstance(value, (datetime, Path)):
            return str(value)
        return repr(value)
    except Exception:  # noqa: BLE001 - evidence capture must survive anything
        return repr(value)


@dataclass(frozen=True)
class SDKExecConfig:
    """Configuration for one isolated Agent SDK arm.

    Deliberately has NO system-prompt field: see the module docstring. `mcp_servers` takes
    SDK-native server configs — stdio command dicts today, in-process SDK server objects when the
    `recall_agent` arm lands — which is the seam the multi-product successor plugs into.
    """

    model: str | None = None
    cwd: str | Path | None = None
    timeout_s: float = 1800.0
    env: Mapping[str, str] = field(default_factory=dict)
    bare: bool = True
    mcp_servers: Mapping[str, Any] | None = None
    allowed_tools: tuple[str, ...] = ()
    disallowed_tools: tuple[str, ...] = ()
    append_system_prompt_file: str | Path | None = None
    permission_mode: str | None = "acceptEdits"
    recall_tool_prefix: str = DEFAULT_RECALL_TOOL_PREFIX
    #: Pinned so the SDK cannot resolve a different CLI than the one `environment.json` versions.
    cli_path: str | None = None
    stream_dir: str | Path | None = None

    def __post_init__(self) -> None:
        if self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive")
        if not self.recall_tool_prefix.strip():
            raise ValueError("recall_tool_prefix must not be empty")

    def options(self) -> Any:
        """Build `ClaudeAgentOptions`. Imports the SDK here, not at module load."""

        from claude_agent_sdk import ClaudeAgentOptions

        extra_args: dict[str, str | None] = {}
        if self.bare:
            extra_args["bare"] = None
        if self.append_system_prompt_file is not None:
            extra_args["append-system-prompt-file"] = str(self.append_system_prompt_file)
        if self.mcp_servers:
            # The same guarantee build_configs states for the CLI arms: the only server present
            # is the one named. With setting_sources unset there is nothing to discover, and the
            # literal flag makes that a recorded fact rather than an inference.
            extra_args["strict-mcp-config"] = None

        environment = dict(os.environ)
        environment.update({str(key): str(value) for key, value in self.env.items()})

        options = ClaudeAgentOptions(
            model=self.model,
            cwd=str(self.cwd) if self.cwd else None,
            env=environment,
            mcp_servers=dict(self.mcp_servers) if self.mcp_servers else {},
            allowed_tools=list(self.allowed_tools),
            disallowed_tools=list(self.disallowed_tools),
            permission_mode=self.permission_mode,
            extra_args=extra_args,
            cli_path=self.cli_path or resolve_claude_executable(),
        )
        if getattr(options, "system_prompt", None) is not None:  # pragma: no cover - see docstring
            raise SDKTranscriptError(
                "a plain-string system_prompt REPLACES the base prompt and breaks comparability; "
                "the append file must travel as extra_args['append-system-prompt-file']"
            )
        return options

    def summary(self) -> dict[str, Any]:
        """A redacted account of the resolved options for the record. Never env values."""

        return {
            "model": self.model,
            "bare": self.bare,
            "mcp_server_names": sorted(self.mcp_servers) if self.mcp_servers else [],
            "allowed_tools": list(self.allowed_tools),
            "disallowed_tools": list(self.disallowed_tools),
            "append_system_prompt_file": (
                str(self.append_system_prompt_file)
                if self.append_system_prompt_file is not None
                else None
            ),
            "permission_mode": self.permission_mode,
            "timeout_s": self.timeout_s,
            "env_keys": sorted(str(key) for key in self.env),
        }


def sdk_mcp_servers(spec: "StdioRecallSpec") -> dict[str, Any]:
    """The SDK-native form of the stdio server config `StdioRecallSpec.write_mcp_config` writes.

    Same interpreter, same `spec.env()` block, so `RECALL_ENV=production` and the Windows import
    fixes (`PYTHONPATH`, `PYTHONSAFEPATH=1`, `APPDATA`/`SystemRoot` passthrough) are carried
    unchanged. No `cwd` key: `spec.env()` was built precisely to make the server's import
    independent of whatever directory the session runs from.
    """

    return {
        spec.server_name: {
            "type": "stdio",
            "command": spec.python,
            "args": ["-m", "recall_mcp.server"],
            "env": spec.env(),
        }
    }


_BLOCK_KIND = {
    "TextBlock": "text",
    "ToolUseBlock": "tool_use",
    "ToolResultBlock": "tool_result",
}


def _block_event(block: Mapping[str, Any]) -> dict[str, Any] | None:
    """One serialized content block back into its stream-json shape. Unknown kinds are dropped
    deliberately (thinking blocks and the like carry nothing the record reads)."""

    kind = block.get("__type__")
    if kind is None:
        raw = block.get("type")
        if raw in {"text", "tool_use", "tool_result"}:
            return dict(block)
        if "tool_use_id" in block:
            kind = "ToolResultBlock"
        elif "name" in block and "id" in block:
            kind = "ToolUseBlock"
        elif "text" in block:
            kind = "TextBlock"
    mapped = _BLOCK_KIND.get(str(kind))
    if mapped == "text":
        return {"type": "text", "text": str(block.get("text", ""))}
    if mapped == "tool_use":
        return {
            "type": "tool_use",
            "id": block.get("id"),
            "name": block.get("name"),
            "input": block.get("input"),
        }
    if mapped == "tool_result":
        return {
            "type": "tool_result",
            "tool_use_id": block.get("tool_use_id"),
            "content": block.get("content"),
            "is_error": block.get("is_error"),
        }
    return None


def _message_blocks(content: Any) -> list[dict[str, Any]]:
    if not isinstance(content, Sequence) or isinstance(content, (str, bytes)):
        return []
    blocks = []
    for item in content:
        if isinstance(item, Mapping):
            mapped = _block_event(item)
            if mapped is not None:
                blocks.append(mapped)
    return blocks


def events_from_messages(entries: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Normalize captured typed messages into the event dicts `claude_exec`'s parser consumes.

    Timestamps are the ARRIVAL clock stamped at receipt, because typed messages carry none of
    their own. That is a different measurement basis than the CLI stream's own timestamps, which
    is why `recall_latency_ms` is demoted to recorded-not-falsifying in the replication's
    preregistration.
    """

    events: list[dict[str, Any]] = []
    for entry in entries:
        message_type = str(entry.get("message_type", ""))
        data = entry.get("data")
        data = data if isinstance(data, Mapping) else {}
        stamp = entry.get("received_at")
        if message_type in {"AssistantMessage", "UserMessage"}:
            content = data.get("content")
            if message_type == "AssistantMessage" and isinstance(content, str):
                blocks: list[dict[str, Any]] = [{"type": "text", "text": content}]
            else:
                blocks = _message_blocks(content)
            if not blocks:
                continue
            events.append(
                {
                    "type": "assistant" if message_type == "AssistantMessage" else "user",
                    "timestamp": stamp,
                    "parent_tool_use_id": data.get("parent_tool_use_id"),
                    "message": {"content": blocks},
                }
            )
        elif message_type == "SystemMessage":
            payload = data.get("data")
            payload = payload if isinstance(payload, Mapping) else {}
            events.append(
                {
                    "type": "system",
                    "subtype": data.get("subtype"),
                    "timestamp": stamp,
                    **{
                        key: value
                        for key, value in payload.items()
                        if key not in {"type", "subtype", "timestamp"}
                    },
                }
            )
        elif message_type == "ResultMessage":
            events.append(
                {
                    "type": "result",
                    "timestamp": stamp,
                    "subtype": data.get("subtype"),
                    "duration_ms": data.get("duration_ms"),
                    "duration_api_ms": data.get("duration_api_ms"),
                    "is_error": data.get("is_error"),
                    "num_turns": data.get("num_turns"),
                    "session_id": data.get("session_id"),
                    "total_cost_usd": data.get("total_cost_usd"),
                    "usage": data.get("usage"),
                    # The rename that makes `_usage_fields`' token rule apply verbatim: input is
                    # fresh + cache-read + cache-creation summed across the per-model aggregate.
                    "modelUsage": data.get("model_usage"),
                    "result": data.get("result"),
                    "permission_denials": data.get("permission_denials"),
                    "stop_reason": data.get("stop_reason"),
                    "api_error_status": data.get("api_error_status"),
                    "terminal_reason": data.get("terminal_reason"),
                }
            )
        # Anything else (stream events, unknown future types) carries nothing the record reads.
    return events


def build_sdk_record(
    row: Mapping[str, Any],
    variant: str,
    *,
    entries: Sequence[Mapping[str, Any]],
    wall_time_ms: float,
    config: SDKExecConfig,
    stderr: str = "",
) -> SessionRecord:
    """Normalize, then reuse `claude_exec.build_record` so the field mapping has one owner."""

    events = events_from_messages(entries)
    stream = "\n".join(json.dumps(event, default=str) for event in events)
    shim = ClaudeExecConfig(
        model=config.model,
        recall_tool_prefix=config.recall_tool_prefix,
        strict_mcp_config=False,
    )
    record = build_record(
        row,
        variant,
        stream=stream,
        wall_time_ms=wall_time_ms,
        config=shim,
        command=("claude-agent-sdk",),
        exit_code=None,
        stderr=stderr,
    )
    return record.__class__.from_mapping(
        {
            **record.to_dict(),
            "metadata": {
                **record.metadata,
                "driver": "sdk",
                "sdk_version": sdk_version(),
                "options": config.summary(),
                # `metadata["stderr"]` is "" on every SDK record, and `claude_exec.build_record`
                # (frozen) collapses even an explicit None to "", so an unobserved stderr is
                # byte-identical to a clean one. This flag is the distinction, in this driver's
                # own metadata layer: the SDK gives no stderr channel here, so nothing was seen
                # rather than nothing was written. Null-not-zero, in the only place it can be said.
                #
                # Derived from the argument rather than hardcoded False, so that wiring a stderr
                # channel later cannot leave the flag asserting the opposite of the field.
                "stderr_observed": bool(stderr),
            },
        }
    )


async def run_sdk_case(
    row: Mapping[str, Any], variant: str, config: SDKExecConfig
) -> SessionRecord:
    """Run one task in one arm through the SDK and return its record."""

    if variant not in VARIANTS:
        raise ValueError(f"variant must be one of {VARIANTS}, got {variant!r}")
    prompt = str(row.get("user_input", "")).strip()
    if not prompt:
        raise ValueError(f"task {row.get('task_id')!r} has an empty user_input")

    from claude_agent_sdk import query

    options = config.options()

    handle = None
    stream_path: Path | None = None
    if config.stream_dir is not None:
        directory = Path(config.stream_dir)
        directory.mkdir(parents=True, exist_ok=True)
        safe_task = re.sub(r"[^A-Za-z0-9._#-]", "_", str(row["task_id"]))
        stream_path = directory / f"{safe_task}.{variant}.jsonl.gz"
        handle = gzip.open(stream_path, "wt", encoding="utf-8")

    entries: list[dict[str, Any]] = []
    started = time.perf_counter()
    try:
        try:
            async with asyncio.timeout(config.timeout_s):
                # `aclosing`, because `async for` does NOT close its iterator when the loop body
                # raises (PEP 533 was deferred), and the SDK's `query()` is a bare
                # async-for-yield with no try/finally of its own. Without this, a failure in the
                # body below (a gzip write on a full disk, say) leaves the generator suspended
                # and the spawned Claude Code process alive, outliving the case that started it
                # and competing with the next one for tokens and for the stdio recall server.
                # The timeout path was already safe (cancellation is thrown INTO the generator,
                # so its finally runs); this makes both paths deterministic.
                async with aclosing(query(prompt=prompt, options=options)) as stream:
                    async for message in stream:
                        entry = {
                            "message_type": type(message).__name__,
                            "data": _jsonable(message),
                            "received_at": datetime.now(timezone.utc).isoformat(),
                        }
                        entries.append(entry)
                        if handle is not None:
                            # Written as messages arrive, so a session the normalizer later
                            # rejects is still on disk to look at.
                            handle.write(json.dumps(entry) + "\n")
                            handle.flush()
        except TimeoutError:
            raise SDKTranscriptError(
                f"claude-agent-sdk exceeded timeout_s={config.timeout_s} for task "
                f"{row.get('task_id')!r}"
            ) from None
    finally:
        if handle is not None:
            handle.close()
    wall_time_ms = (time.perf_counter() - started) * 1000.0

    record = build_sdk_record(
        row,
        variant,
        entries=entries,
        wall_time_ms=wall_time_ms,
        config=config,
        stderr="",
    )
    if stream_path is None:
        return record
    return record.__class__.from_mapping(
        {
            **record.to_dict(),
            "metadata": {**record.metadata, "stream_path": stream_path.name},
        }
    )


def build_sdk_configs(
    specs: Mapping[str, ArmSpec],
    *,
    recall_spec: "StdioRecallSpec | None" = None,
    in_process_servers: Mapping[str, Any] | None = None,
    model: str,
    cwd: str | Path,
    timeout_s: float = 1800.0,
    env: Mapping[str, str] | None = None,
    permission_mode: str = "acceptEdits",
    extra_allowed_tools: tuple[str, ...] = (),
    cli_path: str | None = None,
) -> dict[str, SDKExecConfig]:
    """Build one `SDKExecConfig` per variant, with `cli_path` resolved ONCE by the caller.

    `cli_path` is a parameter rather than something resolved here, because resolving it here
    would make merely BUILDING a config require the Claude Code CLI on PATH, which no test and no
    CI job has. The run script resolves it once and passes it, so a run still pins one executable
    for all its sessions (`SDKExecConfig.options` falls back to a lazy resolve otherwise).

    Mirror of `arms.build_configs`: everything except
    the memory configuration is passed identically to both arms from this one call."""

    missing = [variant for variant in VARIANTS if variant not in specs]
    if missing:
        raise ValueError(f"an ArmSpec is required for every variant; missing {missing}")
    if specs[RECALL_ON].profile not in ON_ARM_PROFILES:
        raise ValueError(
            f"the {RECALL_ON} arm must use one of {ON_ARM_PROFILES}, got "
            f"{specs[RECALL_ON].profile!r}"
        )
    if specs[RECALL_OFF].profile not in OFF_ARM_PROFILES:
        raise ValueError(
            f"the {RECALL_OFF} arm must use one of {OFF_ARM_PROFILES}, got "
            f"{specs[RECALL_OFF].profile!r}"
        )
    # Exactly ONE memory source for the on arm. Both are accepted because the point of the
    # in-process arm is to differ from the stdio arm in the transport and in nothing else, and
    # "nothing else" is only checkable if the two are built by the same function from the same
    # specs. Passing both would silently pick one and quietly answer a different question.
    if (recall_spec is None) == (in_process_servers is None):
        raise ValueError(
            f"the {RECALL_ON} arm profile {specs[RECALL_ON].profile!r} needs exactly one memory "
            f"source: recall_spec (an external stdio server) OR in_process_servers (SDK tools "
            f"served in this process), not both and not neither"
        )
    # Written as an if/else on the same variable the check above narrowed, rather than a ternary
    # on the other one: the exactly-one rule is obvious to a reader and invisible to a type
    # checker, which cannot see that `recall_spec` is non-None in this branch.
    if in_process_servers is not None:
        on_arm_servers = dict(in_process_servers)
    else:
        assert recall_spec is not None  # guaranteed by the exactly-one check above
        on_arm_servers = sdk_mcp_servers(recall_spec)

    configs: dict[str, SDKExecConfig] = {}
    for variant, spec in specs.items():
        allowed = BASE_TOOLS + extra_allowed_tools + spec.extra_allowed_tools
        configs[variant] = SDKExecConfig(
            model=model,
            cwd=cwd,
            timeout_s=timeout_s,
            env=dict(env or {}),
            bare=True,
            cli_path=cli_path,
            mcp_servers=on_arm_servers if variant == RECALL_ON else None,
            allowed_tools=allowed,
            disallowed_tools=DENIED_TOOLS,
            append_system_prompt_file=spec.append_system_prompt_file,
            permission_mode=permission_mode,
            recall_tool_prefix=spec.recall_tool_prefix,
        )
    return configs


def make_sdk_runner(configs: Mapping[str, SDKExecConfig]) -> Runner:
    """Build a `Runner` that dispatches each variant to its own configuration."""

    missing = [variant for variant in VARIANTS if variant not in configs]
    if missing:
        raise ValueError(f"a configuration is required for every variant; missing {missing}")

    async def run_case(row: Mapping[str, Any], variant: str) -> SessionRecord:
        return await run_sdk_case(row, variant, configs[variant])

    return run_case
