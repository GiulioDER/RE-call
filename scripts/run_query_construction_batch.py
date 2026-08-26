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
            if not content:
                raise ValueError("provider returned empty content")
            frame = json.loads(content)
            if not isinstance(frame, dict):
                raise ValueError("provider frame must be a JSON object")
            usage = response.get("usage")
            usage = usage if isinstance(usage, dict) else {}
            metadata = {
                "provider_id": "openrouter",
                "model_id": model,
                "model_revision": str(response.get("model") or "unreported"),
                "prompt_tokens": int(usage.get("prompt_tokens", 0) or 0),
                "completion_tokens": int(usage.get("completion_tokens", 0) or 0),
                "total_tokens": int(usage.get("total_tokens", 0) or 0),
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


def _server_command(tenant: str, embedder: str, index_root: str, profile: str) -> tuple[str, list[str]]:
    ssh = os.environ.get(
        "RECALL_SSH_EXECUTABLE",
        str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "OpenSSH" / "ssh.exe"),
    )
    ssh_config = str(Path.home() / ".ssh" / "config").replace("\\", "/")
    remote = (
        "cd ~/recall-repos && set -a && . ./.env && set +a && "
        "RECALL_ENV=production RECALL_TRUST_MODE=production "
        f"RECALL_TENANT={shlex.quote(tenant)} RECALL_EMBEDDER={shlex.quote(embedder)} "
        f"RECALL_INDEX_ROOT={shlex.quote(index_root)} "
        f"RECALL_RETRIEVAL_PROFILE={shlex.quote(profile)} "
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
        },
    )
    payload = await _tool_payload(challenge)
    tool_calls.append(payload)
    initial_round = payload.get("round_index")
    if type(initial_round) is not int or initial_round != 0:
        raise ValueError("MCP returned an invalid initial construction round")
    while payload.get("status") == "challenge":
        prompt = str(payload["challenge_prompt"] if "challenge_prompt" in payload else payload["next_challenge_prompt"])
        try:
            frame, provider, raw_frame = _ask_original_model(
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
                "expected_generation_id": payload.get("generation", {}).get("generation_id")
                if isinstance(payload.get("generation"), dict)
                else None,
                "graph_expansion": graph_expansion,
            },
        )
        payload = await _tool_payload(continuation)
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
    }


async def main_async(args: argparse.Namespace) -> None:
    items = json.loads(args.input.read_text(encoding="utf-8"))
    if not isinstance(items, list):
        raise SystemExit("input must contain a JSON list")
    items = [dict(item) for item in items[: args.limit or None]]
    api_key = os.environ.get(args.api_key_env, "").strip()
    if not api_key:
        raise SystemExit(f"{args.api_key_env} is required")
    endpoint = args.base_url.rstrip("/") + "/chat/completions"
    ssh, command = _server_command(args.tenant, args.embedder, args.index_root, args.profile)
    diagnostics_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    diagnostics_path = Path(diagnostics_file.name)
    diagnostics_file.close()
    rows: list[dict[str, object]] = []
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
                    for arm in ("baseline", "original_loop", "pyramid"):
                        for index, item in enumerate(items, start=1):
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
                                rows.append({
                                    "task_id": item.get("task_id"),
                                    "arm": arm,
                                    "original_prompt": item.get("original_prompt"),
                                    "query": item.get("query"),
                                    "gold": {key: value for key, value in item.items() if key.startswith("gold") or key.endswith("_ids")},
                                    "final": payload,
                                    "tool_calls": [payload],
                                    "model_calls": [],
                                })
                            else:
                                rows.append(
                                    await _run_arm(
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
                                    )
                                )
        finally:
            diagnostics_handle.close()
    finally:
        diagnostics = diagnostics_path.read_text(encoding="utf-8")
        diagnostics_path.unlink(missing_ok=True)
        if diagnostics.strip():
            print(f"[server diagnostics] {diagnostics.strip()}", flush=True)
    artifact = {
        "artifact": "RE-call original model query construction benchmark",
        "measured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "input": str(args.input),
        "input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
        "model": args.model,
        "base_url": args.base_url,
        "reasoning_effort": args.reasoning_effort,
        "temperature": TEMPERATURE,
        "max_tokens": args.max_tokens,
        "graph_expansion": args.graph_expansion,
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
    parser.add_argument("--tenant", default="memory")
    parser.add_argument("--embedder", default="voyage:voyage-4")
    parser.add_argument("--index-root", default="/home/sentiment/recall-repos/memory")
    parser.add_argument("--profile", default="fast")
    parser.add_argument("--graph-expansion", choices=("off", "one_hop"), default="one_hop")
    args = parser.parse_args()
    if args.max_tokens < 1 or args.retries < 1 or args.timeout <= 0 or args.limit < 0:
        raise SystemExit("max tokens, retries, timeout, and limit must be valid")
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
