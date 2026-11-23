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
import subprocess
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

PROTOCOL_VERSION = "2025-06-18"
CLIENT_INFO = {"name": "re-call-hermes", "version": "0.1.0"}
MAX_PENDING = 16
MAX_REQUEST_BYTES = 8 * 1024 * 1024
MAX_QUEUED_REQUEST_BYTES = 2 * MAX_REQUEST_BYTES
TIMEOUTS_BEFORE_RESTART = 2

ThreadFactory = Callable[..., threading.Thread]


class McpError(RuntimeError):
    """The server could not be reached, did not answer in time, or answered with an error."""


class ToolError(McpError):
    """The call was explicitly refused by the server or failed local input validation."""


class LiteWriterBusy(McpError):
    """The forget tool could not acquire its write lane and made no mutation."""


class NotSent(McpError):
    """The request never left this process and can be attempted again."""


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
        self._outbox: deque[tuple[int | None, bytes]] = deque()
        self._pending: dict[int, dict[str, Any] | None] = {}
        self._dispatched: set[int] = set()
        self._timed_out: dict[int, None] = {}  # bounded ids of dispatched requests that expired
        self._cond = threading.Condition()
        self._handshaken = False
        self._broken = False
        self._consecutive_timeouts = 0
        self._proc: subprocess.Popen[bytes] | None = None
        self._stderr_file: Any = None

    @property
    def alive(self) -> bool:
        proc = self._proc
        return proc is not None and proc.poll() is None

    @property
    def stalled(self) -> bool:
        with self._cond:
            return self._consecutive_timeouts >= TIMEOUTS_BEFORE_RESTART

    @property
    def healthy(self) -> bool:
        with self._cond:
            return self.alive and not self._broken

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
        with self._cond:
            self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})
            self._handshaken = True

    def call_tool(self, name: str, arguments: Mapping[str, Any], timeout: float) -> Any:
        """The tool's answer, parsed from its JSON text. Raises ToolError when `isError` is set."""
        result = self._request("tools/call", {"name": name, "arguments": dict(arguments)}, timeout)
        content = result.get("content")
        parts = content if isinstance(content, list) else []
        text = "".join(
            str(part.get("text", "")) for part in parts if isinstance(part, dict) and part.get("type") == "text"
        )
        if result.get("isError"):  # a refusal stays a refusal, content or not, unless explicitly pre-write
            if name == "recall_forget_memo" and result["isError"] is True:
                # The MCP SDK prefixes a tool's error text with its name.
                try:
                    refusal = json.loads(text.removeprefix(f"Error executing tool {name}: "))
                except ValueError:
                    refusal = None
                if (isinstance(refusal, dict) and refusal.keys() == {"error", "message"}
                        and refusal["error"] == "lite_writer_busy" and isinstance(refusal["message"], str)):
                    raise LiteWriterBusy(refusal["message"])
            raise ToolError(text or f"{name} failed")
        try:  # no content list leaves "" here, which is not JSON either
            return json.loads(text)
        except ValueError as exc:
            raise McpError(f"{name} answered with text that is not JSON: {text[:200]!r}") from exc

    def close(self, timeout: float = 5.0) -> None:
        with self._cond:
            proc, self._proc = self._proc, None
            stderr_file, self._stderr_file = self._stderr_file, None
            self._break_transport()
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
        if stderr_file is not None:
            stderr_file.close()

    # -- internals -----------------------------------------------------------------------------

    def _break_transport(self) -> None:
        # Call with the condition held so no further frame can enter a broken transport.
        self._broken = True
        self._outbox.clear()
        self._timed_out.clear()
        self._cond.notify_all()

    def _send(self, message: Mapping[str, Any], request_id: int | None = None) -> None:
        # The caller holds the condition across registration and queueing.
        proc = self._proc
        if self._broken or proc is None or proc.stdin is None or proc.poll() is not None:
            raise NotSent("the RE-call server is not running")
        if len(self._outbox) >= MAX_PENDING:
            raise NotSent("too many outstanding RE-call requests")
        line = (json.dumps(message, separators=(",", ":")) + "\n").encode("utf-8")
        if len(line) > MAX_REQUEST_BYTES:
            raise ToolError("the RE-call request is too large")
        if len(line) + sum(len(frame) for _, frame in self._outbox) > MAX_QUEUED_REQUEST_BYTES:
            raise NotSent("the RE-call outgoing queue is full")
        self._outbox.append((request_id, line))
        self._cond.notify_all()

    def _write_loop(self) -> None:
        proc = self._proc
        while proc is not None and proc.stdin is not None:
            with self._cond:
                self._cond.wait_for(lambda: self._broken or self._outbox)
                if self._broken:
                    return
                request_id, line = self._outbox.popleft()
                if request_id is not None:
                    if request_id not in self._pending:
                        continue  # a caller timed out while its frame was still queued
                    self._dispatched.add(request_id)
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
            except (OSError, ValueError):  # the child closed its input, or close() closed ours
                with self._cond:
                    self._break_transport()
                return

    def _request(self, method: str, params: Mapping[str, Any], timeout: float) -> dict[str, Any]:
        request_id = next(self._ids)
        deadline = time.monotonic() + timeout
        with self._cond:
            if len(self._pending) >= MAX_PENDING:
                raise NotSent("too many outstanding RE-call requests")
            self._pending[request_id] = None
            try:
                self._send({"jsonrpc": "2.0", "id": request_id, "method": method, "params": dict(params)},
                           request_id)
                while (message := self._pending[request_id]) is None:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        if self._handshaken and not self._broken:
                            if request_id in self._dispatched:
                                if len(self._timed_out) >= MAX_PENDING:
                                    del self._timed_out[next(iter(self._timed_out))]
                                self._timed_out[request_id] = None
                            self._consecutive_timeouts += 1
                            if self._consecutive_timeouts >= TIMEOUTS_BEFORE_RESTART:
                                self._break_transport()
                        error = McpError if request_id in self._dispatched else NotSent
                        raise error(f"{method} got no answer within {timeout:.1f}s")
                    if self._broken or not self.alive:
                        error = McpError if request_id in self._dispatched else NotSent
                        raise error("the RE-call server disconnected")
                    self._cond.wait(min(remaining, 0.5))
            finally:
                del self._pending[request_id]
                self._dispatched.discard(request_id)
        if "error" in message:
            error = message["error"]
            detail = error.get("message", error) if isinstance(error, dict) else error
            raise McpError(f"{method}: {detail}")
        result = message.get("result")
        if not isinstance(result, dict):
            raise McpError(f"{method} answered with a result that is not an object")
        return result

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
            if isinstance(message, dict) and type(message.get("id")) is int and "method" not in message:
                with self._cond:
                    request_id = message["id"]
                    if self._broken or request_id not in self._pending or self._pending[request_id] is not None:
                        if not self._broken and request_id in self._timed_out:
                            error = message.get("error")
                            if (message.get("jsonrpc") == "2.0"
                                    and ("result" in message) != ("error" in message)
                                    and (isinstance(message.get("result"), dict) or (
                                        isinstance(error, dict) and type(error.get("code")) is int
                                        and isinstance(error.get("message"), str)
                                    ))):
                                del self._timed_out[request_id]
                                self._consecutive_timeouts = 0  # health only, no waiter to answer
                        continue
                    self._pending[request_id] = message
                    self._consecutive_timeouts = 0
                    self._cond.notify_all()
        _close_quietly(proc.stdout)
        with self._cond:
            self._break_transport()
