"""A minimal MCP client over stdio for one child `recall-mcp` process, standard library only.

MCP's stdio transport is newline-delimited JSON-RPC 2.0. This client sends `initialize`, the
`notifications/initialized` notification, then `tools/call` requests, and matches answers to
requests by id on a reader thread, so a request that timed out cannot be answered by the next
one's response.
"""

from __future__ import annotations

import itertools
import json
import logging
import queue
import subprocess
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "re-call-hermes", "version": "0.1.0"}

ThreadFactory = Callable[..., threading.Thread]


class McpError(RuntimeError):
    """The server could not be reached, did not answer in time, or answered with an error."""


class ToolError(McpError):
    """The tool ran and reported an error (`isError`), with the server's text."""


class NotSent(McpError):
    """The request never left this process because the server was not running, so its outcome is known."""


def _close_quietly(stream: Any) -> None:
    try:
        stream.close()
    except (OSError, ValueError):  # the child is gone, or a writer thread closed it first
        pass


def _default_thread_factory(target: Callable[[], None], *, name: str) -> threading.Thread:
    return threading.Thread(target=target, name=name, daemon=True)


class StdioMcpClient:
    """One child process, started by `start()`, answering `call_tool()` until `close()`."""

    def __init__(
        self,
        argv: Sequence[str],
        *,
        env: Mapping[str, str],
        cwd: str | Path,
        stderr_path: str | Path,
        thread_factory: ThreadFactory = _default_thread_factory,
    ) -> None:
        self._argv = list(argv)
        self._env = dict(env)
        self._cwd = str(cwd)
        self._stderr_path = Path(stderr_path)
        self._thread_factory = thread_factory
        self._ids = itertools.count(1)
        # Lines reach the child through one writer thread, so a child that stops reading its input
        # blocks that thread and never a caller: callers only wait, and only until their deadline.
        self._outbox: queue.Queue[bytes | None] = queue.Queue()
        self._pending: dict[int, dict[str, Any]] = {}
        # Ids whose caller gave up: their late answers are dropped rather than kept forever.
        self._abandoned: dict[int, None] = {}
        self._cond = threading.Condition()
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_file: Any = None

    @property
    def alive(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def start(self, timeout: float) -> None:
        """Launch the child and complete the MCP handshake within `timeout` seconds."""
        self._stderr_path.parent.mkdir(parents=True, exist_ok=True)
        self._stderr_file = open(self._stderr_path, "ab")  # noqa: SIM115 - closed in close()
        self._proc = subprocess.Popen(
            self._argv,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=self._stderr_file,
            cwd=self._cwd,
            env=self._env,
        )
        self._thread_factory(self._read_loop, name="re-call-mcp-reader").start()
        self._thread_factory(self._write_loop, name="re-call-mcp-writer").start()
        self._request(
            "initialize",
            {"protocolVersion": PROTOCOL_VERSION, "capabilities": {}, "clientInfo": CLIENT_INFO},
            timeout,
        )
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def call_tool(self, name: str, arguments: Mapping[str, Any], timeout: float) -> Any:
        """The tool's answer, parsed from its JSON text. Raises ToolError when `isError` is set."""
        result = self._request("tools/call", {"name": name, "arguments": dict(arguments)}, timeout)
        content = result.get("content")
        parts = content if isinstance(content, list) else []
        text = "".join(
            str(part.get("text", "")) for part in parts if isinstance(part, dict) and part.get("type") == "text"
        )
        if result.get("isError"):  # a refusal stays a refusal, content or not, so it is never retried
            raise ToolError(text or f"{name} failed")
        try:  # no content list leaves "" here, which is not JSON either
            return json.loads(text)
        except ValueError as exc:
            raise McpError(f"{name} answered with text that is not JSON: {text[:200]!r}") from exc

    def close(self, timeout: float = 5.0) -> None:
        proc, self._proc = self._proc, None
        self._outbox.put(None)
        if proc is not None:
            stdin = proc.stdin
            if stdin is not None:
                # Closing flushes input the child has not read; a wedged child never reads it, so the
                # close runs off this thread, and the kill below is what finally unblocks it.
                self._thread_factory(lambda: _close_quietly(stdin), name="re-call-mcp-close").start()
            try:
                proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill()
                try:
                    proc.wait(timeout=timeout)
                except subprocess.TimeoutExpired:
                    logger.warning("re-call: the server did not exit after kill (pid %s)", proc.pid)
            # stdout is left to the reader thread, which ends at EOF: closing it here would wait on
            # the lock that thread's blocked read holds.
        if self._stderr_file is not None:
            self._stderr_file.close()
            self._stderr_file = None
        with self._cond:
            self._cond.notify_all()

    # -- internals -----------------------------------------------------------------------------

    def _send(self, message: Mapping[str, Any]) -> None:
        proc = self._proc
        if proc is None or proc.stdin is None or proc.poll() is not None:
            raise NotSent("the RE-call server is not running")
        self._outbox.put((json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8"))

    def _write_loop(self) -> None:
        proc = self._proc
        while proc is not None and proc.stdin is not None:
            line = self._outbox.get()
            if line is None:
                return
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
            except (OSError, ValueError):  # the child closed its input, or close() closed ours
                return

    def _request(self, method: str, params: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        request_id = next(self._ids)
        deadline = time.monotonic() + timeout
        self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)})
        with self._cond:
            while request_id not in self._pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    self._abandon(request_id)
                    raise McpError(f"{method} got no answer within {timeout:.1f}s")
                if not self.alive:
                    self._abandon(request_id)
                    raise McpError("the RE-call server exited")
                self._cond.wait(min(remaining, 0.5))
            message = self._pending.pop(request_id)
        if "error" in message:
            error = message["error"]
            detail = error.get("message", error) if isinstance(error, dict) else error
            raise McpError(f"{method}: {detail}")
        result = message.get("result")
        if not isinstance(result, dict):
            raise McpError(f"{method} answered with a result that is not an object")
        return result

    def _abandon(self, request_id: int) -> None:
        # Called with self._cond held. One int per answer that never comes; a capped set would let an
        # evicted id's late answer sit in _pending forever instead.
        self._abandoned[request_id] = None

    def _read_loop(self) -> None:
        proc = self._proc
        if proc is None or proc.stdout is None:
            return
        for raw in proc.stdout:
            try:
                message = json.loads(raw)
            except ValueError:
                logger.debug("re-call: ignoring a line that is not JSON-RPC")
                continue
            # Notifications and server requests carry no answer to us; only answers have our ids.
            if isinstance(message, dict) and isinstance(message.get("id"), int) and "method" not in message:
                with self._cond:
                    if message["id"] in self._abandoned:
                        del self._abandoned[message["id"]]
                        continue
                    self._pending[message["id"]] = message
                    self._cond.notify_all()
        with self._cond:
            self._cond.notify_all()
