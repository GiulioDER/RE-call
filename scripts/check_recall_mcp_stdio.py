"""Read only handshake check for the RE-call MCP stdio server."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
from pathlib import Path

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


def _command() -> tuple[str, list[str]]:
    ssh = os.environ.get(
        "RECALL_SSH_EXECUTABLE",
        str(Path(os.environ.get("WINDIR", r"C:\Windows")) / "System32" / "OpenSSH" / "ssh.exe"),
    )
    ssh_config = str(Path.home() / ".ssh" / "config").replace("\\", "/")
    remote = (
        "cd ~/recall-repos && set -a && . ./.env && set +a && "
        f"RECALL_TENANT={os.environ.get('RECALL_TEST_TENANT', 'memory')} "
        f"RECALL_EMBEDDER={os.environ.get('RECALL_TEST_EMBEDDER', 'voyage:voyage-4')} "
        f"RECALL_INDEX_ROOT={os.environ.get('RECALL_TEST_INDEX_ROOT', '/home/sentiment/recall-repos/memory')} "
        "exec .venv/bin/python -m recall_mcp.server"
    )
    return ssh, ["-T", "-o", "BatchMode=yes", "-F", ssh_config, "vps2", remote]


async def _check() -> list[str]:
    ssh, args = _command()
    params = StdioServerParameters(command=ssh, args=args)
    diagnostics_file = tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", delete=False)
    diagnostics_path = Path(diagnostics_file.name)
    diagnostics_file.close()
    try:
        with diagnostics_path.open("w", encoding="utf-8") as diagnostics:
            async with stdio_client(params, errlog=diagnostics) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.list_tools()
                    return sorted(tool.name for tool in result.tools)
    finally:
        diagnostics = diagnostics_path.read_text(encoding="utf-8")
        diagnostics_path.unlink(missing_ok=True)
        if diagnostics:
            print("[server diagnostics]", file=os.sys.stderr)
            print(diagnostics, file=os.sys.stderr)


async def _raw_check() -> None:
    ssh, args = _command()
    print(json.dumps({"ssh": ssh, "args": args}), flush=True)
    process = await asyncio.create_subprocess_exec(
        ssh,
        *args,
        stdin=asyncio.subprocess.PIPE,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    request = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "initialize",
        "params": {
            "protocolVersion": "2025-06-18",
            "capabilities": {},
            "clientInfo": {"name": "recall-codex-debug", "version": "1.0"},
        },
    }
    assert process.stdin is not None
    assert process.stdout is not None
    assert process.stderr is not None
    process.stdin.write((json.dumps(request) + "\n").encode("utf-8"))
    await process.stdin.drain()
    print("request-sent", flush=True)
    try:
        response = await asyncio.wait_for(process.stdout.readline(), timeout=8)
    except asyncio.TimeoutError:
        response = b"<timeout>"
    print(json.dumps({"response": response.decode("utf-8", errors="replace")}), flush=True)
    process.terminate()
    try:
        await asyncio.wait_for(process.wait(), timeout=3)
    except asyncio.TimeoutError:
        process.kill()
        await process.wait()
    stderr = await process.stderr.read()
    returncode = process.returncode
    print(json.dumps({
        "response": response.decode("utf-8", errors="replace"),
        "stderr": stderr.decode("utf-8", errors="replace"),
        "returncode": returncode,
    }, indent=2))


if __name__ == "__main__":
    if os.environ.get("RECALL_MCP_RAW_CHECK") == "1":
        asyncio.run(_raw_check())
    else:
        print("\n".join(asyncio.run(_check())))
