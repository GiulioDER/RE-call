"""Run the preregistered query construction arms against the remote MCP server.

The input is a frozen JSON list containing at least ``task_id``, ``original_prompt`` and
``query``. Any gold fields are copied to the artifact for scoring but are never sent to the
original model or to RE call.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import shlex
import tempfile
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib import error, request

import anyio
import mcp_types as types
from mcp import ClientSession
from mcp.shared.message import SessionMessage


MODEL = "deepseek/deepseek-v4-pro"
BASE_URL = "https://openrouter.ai/api/v1"
REASONING_EFFORT = "medium"
MAX_TOKENS = 1024
TEMPERATURE = 0


def _message_text(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "".join(
            item["text"]
            for item in value
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        ).strip()
    return ""


def _post_json(*, url: str, api_key: str, payload: dict[str, object], timeout: float) -> dict[str, Any]:
    body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/openai/recall",
        "X-Title": "RE-call query construction benchmark",
    }
    req = request.Request(url, data=body, headers=headers, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        value = json.loads(response.read().decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("provider returned a non-object response")
    return value


def _ask_original_model(
    prompt: str,
    *,
    endpoint: str,
    api_key: str,
    model: str,
    reasoning_effort: str,
    max_tokens: int,
    timeout: float,
    retries: int,
) -> tuple[dict[str, object], dict[str, object], str]:
    payload: dict[str, object] = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are the original calling agent. Answer the retrieval challenge only. "
                    "Return the requested JSON object and do not answer the user."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": TEMPERATURE,
        "max_tokens": max_tokens,
        "response_format": {"type": "json_object"},
        "reasoning": {"effort": reasoning_effort},
    }
    last_error = "provider request failed"
    for attempt in range(retries):
        started = time.perf_counter()
        try:
            response = _post_json(
                url=endpoint,
                api_key=api_key,
                payload=payload,
                timeout=timeout,
            )
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
                raise ValueError("provider returned no choices")
            message = choices[0].get("message")
            content = _message_text(message.get("content") if isinstance(message, dict) else None)
            usage = response.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            details = usage.get("completion_tokens_details")
            details = details if isinstance(details, dict) else {}
            finish_reason = str(choices[0].get("finish_reason") or "unreported")
            if not content:
                raise ValueError(
                    "provider returned empty content "
                    f"(finish_reason={finish_reason}, "
                    f"prompt_tokens={int(usage.get('prompt_tokens', 0) or 0)}, "
                    f"completion_tokens={int(usage.get('completion_tokens', 0) or 0)}, "
                    f"reasoning_tokens={int(details.get('reasoning_tokens', 0) or 0)})"
                )
            frame = json.loads(content)
            if not isinstance(frame, dict):
                raise ValueError("provider frame must be a JSON object")
            metadata = {
                "provider_id": "openrouter",
                "model_id": model,
                "model_revision": str(response.get("model") or "unreported"),
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
                "reasoning_tokens": int(details.get("reasoning_tokens", 0) or 0),
                "finish_reason": finish_reason,
                "latency_ms": max(0, int((time.perf_counter() - started) * 1000)),
                "prompt_sha256": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            }
            return frame, metadata, content
        except (error.HTTPError, error.URLError, TimeoutError, ValueError, json.JSONDecodeError) as exc:
            if isinstance(exc, error.HTTPError):
                detail = exc.read(512).decode("utf-8", errors="replace").replace("\n", " ")
                last_error = f"HTTP {exc.code}: {detail[:240]}"
            else:
                last_error = f"{type(exc).__name__}: {exc}"[:280]
            if attempt + 1 < retries:
                time.sleep(min(30.0, 2.0**attempt))
    raise RuntimeError(f"original model failed after {retries} attempts: {last_error}")


def _server_command(
    tenant: str,
    embedder: str,
    index_root: str,
    profile: str,
    pinned_generation_id: str | None,
) -> tuple[str, list[str]]:
    ssh = os.environ.get(
        "RECALL_SSH_EXECUTABLE",
        str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "OpenSSH" / "ssh.exe"),
    )
    ssh_config = str(Path.home() / ".ssh" / "config").replace("\\", "/")
    pin = (
        f"RECALL_PINNED_GENERATION_ID={shlex.quote(pinned_generation_id)} "
        if pinned_generation_id
        else ""
    )
    remote = (
        "cd ~/recall-repos && set -a && . ./.env && set +a && "
        "RECALL_ENV=production RECALL_TRUST_MODE=production "
        "RECALL_BENCHMARK_PIN=1 "
        f"RECALL_TENANT={shlex.quote(tenant)} RECALL_EMBEDDER={shlex.quote(embedder)} "
        f"RECALL_INDEX_ROOT={shlex.quote(index_root)} "
        f"RECALL_RETRIEVAL_PROFILE={shlex.quote(profile)} "
        f"{pin}"
        "exec .venv/bin/python -m recall_mcp.server"
    )
    return ssh, ["-T", "-o", "BatchMode=yes", "-F", ssh_config, "vps2", remote]


@asynccontextmanager
async def _ssh_stdio_client(
    command: str,
    args: list[str],
    *,
    errlog: Any,
) -> Any:
    """Run SSH with asyncio pipes; AnyIO's Windows pipe bridge drops frames."""

    process = await asyncio.create_subprocess_exec(
        command,
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    read_send, read_receive = anyio.create_memory_object_stream[SessionMessage | Exception](0)
    write_send, write_receive = anyio.create_memory_object_stream[SessionMessage](0)

    async def read_stdout() -> None:
        try:
            while line := await process.stdout.readline():
                try:
                    message = types.jsonrpc_message_adapter.validate_json(line, by_name=False)
                except ValueError as exc:
                    await read_send.send(exc)
                else:
                    await read_send.send(SessionMessage(message))
        finally:
            await read_send.aclose()

    async def read_stderr() -> None:
        while chunk := await process.stderr.readline():
            errlog.write(chunk.decode("utf-8", errors="replace"))
            errlog.flush()

    async def write_stdin() -> None:
        try:
            async for message in write_receive:
                process.stdin.write(message.message.model_dump_json(by_alias=True, exclude_unset=True).encode() + b"\n")
                await process.stdin.drain()
        finally:
            process.stdin.close()
            await process.stdin.wait_closed()

    tasks = [
        asyncio.create_task(read_stdout()),
        asyncio.create_task(read_stderr()),
        asyncio.create_task(write_stdin()),
    ]
    try:
        yield read_receive, write_send
    finally:
        await write_send.aclose()
        await write_receive.aclose()
        try:
            await asyncio.wait_for(process.wait(), timeout=3)
        except TimeoutError:
            process.kill()
            await process.wait()
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        await read_receive.aclose()


async def _tool_payload(result: object) -> dict[str, object]:
    text = "".join(
        block.text
        for block in getattr(result, "content", [])
        if getattr(block, "type", None) == "text"
    )
    if not text:
        raise RuntimeError("MCP returned an empty response")
    payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("MCP returned a non-object response")
    return payload


def _assert_generation(payload: dict[str, object], expected_generation_id: str | None) -> None:
    """Fail the apparatus if a pinned benchmark response is not from that snapshot."""

    if expected_generation_id is None:
        return
    generation = payload.get("generation")
    if isinstance(generation, dict):
        actual = generation.get("generation_id")
    else:
        actual = payload.get("generation_id")
        if actual is None and isinstance(payload.get("retrieval"), dict):
            actual = payload["retrieval"].get("generation_id")
    if actual != expected_generation_id:
        raise RuntimeError(
            "benchmark generation mismatch: "
            f"expected {expected_generation_id!r}, got {actual!r}"
        )


async def _run_arm(
    session: ClientSession,
    item: dict[str, object],
    *,
    arm: str,
    graph_expansion: str,
    model: str,
    endpoint: str,
    api_key: str,
    reasoning_effort: str,
    max_tokens: int,
    provider_timeout: float,
    retries: int,
    expected_generation_id: str | None,
    challenge_marker: str | None,
) -> dict[str, object]:
    original_prompt = str(item["original_prompt"])
    query = str(item["query"])
    tool_calls: list[dict[str, object]] = []
    model_calls: list[dict[str, object]] = []
    challenge = await session.call_tool(
        "recall_query_construction_challenge",
        {
            "original_prompt": original_prompt,
            "query": query,
            "arm": arm,
            "round_index": 0,
            "graph_expansion": graph_expansion,
            **({"expected_generation_id": expected_generation_id} if expected_generation_id else {}),
        },
    )
    payload = await _tool_payload(challenge)
    _assert_generation(payload, expected_generation_id)
    tool_calls.append(payload)
    initial_round = payload.get("round_index")
    if type(initial_round) is not int or initial_round != 0:
        raise ValueError("MCP returned an invalid initial construction round")
    while payload.get("status") == "challenge":
        prompt = str(payload["challenge_prompt"] if "challenge_prompt" in payload else payload["next_challenge_prompt"])
        if challenge_marker and challenge_marker not in prompt:
            raise RuntimeError(
                f"challenge prompt marker missing; expected {challenge_marker!r}"
            )
        try:
            frame, provider, raw_frame = await asyncio.to_thread(
                _ask_original_model,
                prompt,
                endpoint=endpoint,
                api_key=api_key,
                model=model,
                reasoning_effort=reasoning_effort,
                max_tokens=max_tokens,
                timeout=provider_timeout,
                retries=retries,
            )
        except RuntimeError as exc:
            fallback = await session.call_tool(
                "recall_reasoning_query",
                {
                    "query": query,
                    "k": 5,
                    "mode": "retrieval_only",
                    "graph_expansion": graph_expansion,
                },
            )
            fallback_payload = await _tool_payload(fallback)
            _assert_generation(fallback_payload, expected_generation_id)
            tool_calls.append(fallback_payload)
            return {
                "task_id": item.get("task_id"),
                "arm": arm,
                "original_prompt": original_prompt,
                "query": query,
                "gold": {key: value for key, value in item.items() if key.startswith("gold") or key.endswith("_ids")},
                "final": fallback_payload,
                "tool_calls": tool_calls,
                "model_calls": model_calls,
                "fallback": {"phase": "original_model", "reason": str(exc)},
                "apparatus": {
                    "expected_generation_id": expected_generation_id,
                    "challenge_marker": challenge_marker,
                },
            }
        model_calls.append({"frame": frame, "raw": raw_frame, "provider": provider})
        if "next_round_index" in payload:
            next_round = payload["next_round_index"]
        elif "challenge_prompt" in payload:
            # The first challenge is answered at the round it declares. Follow-up challenges must
            # carry an explicit next_round_index so a stale continuation cannot repeat round zero.
            next_round = initial_round
        else:
            raise ValueError("MCP omitted the next construction round")
        if type(next_round) is not int or not 0 <= next_round < 2:
            raise ValueError("MCP returned an invalid construction round")
        continuation = await session.call_tool(
            "recall_query_construction_challenge",
            {
                "original_prompt": original_prompt,
                "query": query,
                "arm": arm,
                "round_index": next_round,
                "frame": frame,
                "expected_generation_id": expected_generation_id
                or (
                    payload.get("generation", {}).get("generation_id")
                    if isinstance(payload.get("generation"), dict)
                    else None
                ),
                "graph_expansion": graph_expansion,
            },
        )
        payload = await _tool_payload(continuation)
        _assert_generation(payload, expected_generation_id)
        tool_calls.append(payload)
        if len(model_calls) >= 2:
            break
    return {
        "task_id": item.get("task_id"),
        "arm": arm,
        "original_prompt": original_prompt,
        "query": query,
        "gold": {key: value for key, value in item.items() if key.startswith("gold") or key.endswith("_ids")},
        "final": payload,
        "tool_calls": tool_calls,
        "model_calls": model_calls,
        "apparatus": {
            "expected_generation_id": expected_generation_id,
            "challenge_marker": challenge_marker,
        },
    }


def _item_key(arm: str, index: int) -> str:
    return f"{arm}:{index}"


def _item_digest(item: dict[str, object]) -> str:
    encoded = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _checkpoint_settings(args: argparse.Namespace, input_sha256: str) -> dict[str, object]:
    return {
        "input_sha256": input_sha256,
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "temperature": TEMPERATURE,
        "max_tokens": args.max_tokens,
        "timeout": args.timeout,
        "retries": args.retries,
        "tenant": args.tenant,
        "embedder": args.embedder,
        "index_root": args.index_root,
        "profile": args.profile,
        "graph_expansion": args.graph_expansion,
        "gold_class": args.gold_class,
        "pinned_generation_id": args.pinned_generation_id,
        "challenge_marker": args.challenge_marker,
    }


def _load_checkpoint(
    path: Path,
    *,
    expected_settings: dict[str, object],
    resume: bool,
) -> dict[str, dict[str, object]]:
    if not path.exists():
        return {}
    if not resume:
        raise SystemExit(f"checkpoint exists; pass --resume to continue it: {path}")
    completed: dict[str, dict[str, object]] = {}
    meta_seen = False
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"invalid checkpoint JSON at line {line_number}: {path}") from exc
            if not isinstance(record, dict):
                raise SystemExit(f"invalid checkpoint record at line {line_number}: {path}")
            if record.get("type") == "meta":
                meta_seen = True
                if record.get("settings") != expected_settings:
                    raise SystemExit("checkpoint settings do not match this benchmark invocation")
            elif record.get("type") == "row":
                key = record.get("key")
                digest = record.get("item_sha256")
                row = record.get("row")
                if isinstance(key, str) and isinstance(digest, str) and isinstance(row, dict):
                    completed[key] = {"item_sha256": digest, "row": row}
            else:
                raise SystemExit(f"unknown checkpoint record at line {line_number}: {path}")
    if not meta_seen:
        raise SystemExit(f"checkpoint has no metadata record: {path}")
    return completed


async def main_async(args: argparse.Namespace) -> None:
    items = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise SystemExit("input must contain a JSON list")
    items = [dict(item) for item in items[: args.limit or None]]
    if args.gold_class:
        items = [item for item in items if item.get("gold_class") == args.gold_class]
    if not items:
        raise SystemExit("population filter selected no input items")
    input_sha256 = hashlib.sha256(args.input.read_bytes()).hexdigest()
    checkpoint_path = args.checkpoint or args.output.with_name(args.output.name + ".checkpoint.jsonl")
    checkpoint_exists = checkpoint_path.exists()
    settings = _checkpoint_settings(args, input_sha256)
    completed = _load_checkpoint(path=checkpoint_path, expected_settings=settings, resume=args.resume)
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is required")
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    ssh, command = _server_command(
        args.tenant,
        args.embedder,
        args.index_root,
        args.profile,
        args.pinned_generation_id,
    )
    diagnostics_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    diagnostics_path = Path(diagnostics_file.name)
    diagnostics_file.close()
    rows_by_key: dict[str, dict[str, object]] = {}
    item_digests = {
        _item_key(arm, index): _item_digest(item)
        for arm in ("baseline", "original_loop", "pyramid")
        for index, item in enumerate(items, start=1)
    }
    for key, saved in completed.items():
        if key in item_digests and saved["item_sha256"] == item_digests[key]:
            rows_by_key[key] = saved["row"]
    checkpoint_handle = checkpoint_path.open("a", encoding="utf-8")
    if not checkpoint_exists:
        checkpoint_handle.write(json.dumps({"type": "meta", "settings": settings}, ensure_ascii=False) + "\n")
        checkpoint_handle.flush()
        os.fsync(checkpoint_handle.fileno())
    checkpoint_lock = asyncio.Lock()

    async def save_row(key: str, row: dict[str, object]) -> None:
        async with checkpoint_lock:
            checkpoint_handle.write(
                json.dumps(
                    {"type": "row", "key": key, "item_sha256": item_digests[key], "row": row},
                    ensure_ascii=False,
                )
                + "\n"
            )
            checkpoint_handle.flush()
            os.fsync(checkpoint_handle.fileno())

    async def run_one(arm: str, index: int, item: dict[str, object], semaphore: asyncio.Semaphore) -> tuple[str, dict[str, object]]:
        key = _item_key(arm, index)
        async with semaphore:
            print(f"{arm} {index}/{len(items)} {item.get('task_id', index)}", flush=True)
            if arm == "baseline":
                result = await session.call_tool(
                    "recall_reasoning_query",
                    {
                        "query": str(item["query"]),
                        "k": 5,
                        "mode": "retrieval_only",
                        "graph_expansion": args.graph_expansion,
                    },
                )
                payload = await _tool_payload(result)
                _assert_generation(payload, args.pinned_generation_id)
                row = {
                    "task_id": item.get("task_id"),
                    "arm": arm,
                    "original_prompt": item.get("original_prompt"),
                    "query": item.get("query"),
                    "gold": {key: value for key, value in item.items() if key.startswith("gold") or key.endswith("_ids")},
                    "final": payload,
                    "tool_calls": [payload],
                    "model_calls": [],
                    "apparatus": {
                        "expected_generation_id": args.pinned_generation_id,
                        "challenge_marker": args.challenge_marker,
                    },
                }
            else:
                row = await _run_arm(
                    session,
                    item,
                    arm=arm,
                    graph_expansion=args.graph_expansion,
                    model=args.model,
                    endpoint=endpoint,
                    api_key=api_key,
                    reasoning_effort=args.reasoning_effort,
                    max_tokens=args.max_tokens,
                    provider_timeout=args.timeout,
                    retries=args.retries,
                    expected_generation_id=args.pinned_generation_id,
                    challenge_marker=args.challenge_marker,
                )
            return key, row

    try:
        diagnostics_handle = diagnostics_path.open("w", encoding="utf-8")
        try:
            async with _ssh_stdio_client(ssh, command, errlog=diagnostics_handle) as (read, write):
                async with ClientSession(read, write) as session:
                    await asyncio.wait_for(session.initialize(), timeout=60)
                    listed = await asyncio.wait_for(session.list_tools(), timeout=30)
                    names = {tool.name for tool in listed.tools}
                    if "recall_query_construction_challenge" not in names:
                        raise RuntimeError("VPS2 MCP does not expose recall_query_construction_challenge")
                    semaphore = asyncio.Semaphore(args.workers)
                    for arm in ("baseline", "original_loop", "pyramid"):
                        tasks = [
                            asyncio.create_task(run_one(arm, index, item, semaphore))
                            for index, item in enumerate(items, start=1)
                            if _item_key(arm, index) not in rows_by_key
                        ]
                        for task in asyncio.as_completed(tasks):
                            key, row = await task
                            rows_by_key[key] = row
                            await save_row(key, row)
        finally:
            diagnostics_handle.close()
    finally:
        diagnostics = diagnostics_path.read_text(encoding="utf-8")
        diagnostics_path.unlink(missing_ok=True)
        if diagnostics.strip():
            print(f"[server diagnostics] {diagnostics.strip()}", flush=True)
        checkpoint_handle.close()
    ordered_keys = [
        _item_key(arm, index)
        for arm in ("baseline", "original_loop", "pyramid")
        for index in range(1, len(items) + 1)
    ]
    if len(rows_by_key) != len(ordered_keys):
        raise SystemExit("benchmark ended before all rows completed; resume from the checkpoint")
    rows = [rows_by_key[key] for key in ordered_keys]
    artifact = {
        "artifact": "RE-call original model query construction benchmark",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": str(args.input),
        "input_sha256": input_sha256,
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "temperature": TEMPERATURE,
        "max_tokens": args.max_tokens,
        "graph_expansion": args.graph_expansion,
        "workers": args.workers,
        "gold_class": args.gold_class,
        "pinned_generation_id": args.pinned_generation_id,
        "challenge_marker": args.challenge_marker,
        "checkpoint": str(checkpoint_path),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"rows": len(rows), "output": str(args.output)}))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--base-url", default=BASE_URL)
    parser.add_argument("--reasoning-effort", choices=("none", "minimal", "low", "medium", "high"), default=REASONING_EFFORT)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--api-key-env", default="OPENROUTER_API_KEY")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--gold-class", choices=("miss", "control"), help="run only one frozen population class")
    parser.add_argument("--pinned-generation-id", help="read one immutable VPS2 generation snapshot")
    parser.add_argument("--challenge-marker", help="require this literal marker in every challenge prompt")
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage:voyage-4")
    parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--graph-expansion", choices=("off", "one_hop"), default="one_hop")
    parser.add_argument("--workers", type=int, default=1, help="maximum concurrent benchmark cases")
    parser.add_argument("--resume", action="store_true", help="resume completed rows from the checkpoint sidecar")
    parser.add_argument("--checkpoint", type=Path, help="checkpoint JSONL path (defaults beside output)")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.retries < 1 or args.timeout <= 0 or args.limit < 0 or args.workers < 1:
        raise SystemExit("max tokens, retries, timeout, limit, and workers must be valid")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
