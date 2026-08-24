"""A pre-warmed RE-call MCP server for the on arm, and the readiness check that proves it.

Why HTTP rather than the stdio server the MCP config would otherwise name: starting
`recall_mcp.server` over stdio takes **13.3 s** on this corpus, almost all of it loading the
embedder, and that cost is paid again for every session. Two consequences, and the second is the
one that matters:

1. Across a few hundred sessions it is an hour of pure startup.
2. It lands *inside the on arm's measured wall time*, so the memory layer is charged for a model
   load that a warm deployment never pays. The benchmark would be reporting embedder startup and
   calling it retrieval overhead.

A single long-lived server, started once and shared by every on-arm session, removes both. It also
sidesteps the failure this whole harness is built around: a stdio server that is slow to start is
a server that is *absent* on Claude Code 2.1.220, which does not wait for a pending MCP server.

`recall_mcp.build_auth` refuses to open an HTTP listener with no authentication configured, which
is correct and non-negotiable, so this writes a short-lived static token file into a private
temporary directory and removes it on exit. The token never enters the repository, and the MCP
config is written as a **file** rather than passed inline so the bearer token never appears in a
command line that gets recorded into a session artifact.
"""

from __future__ import annotations

import json
import os
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: Scope name the server expects. `read` alone is rejected: valid scopes are namespaced.
READ_SCOPE = "recall:read"
DEFAULT_SERVER_NAME = "recall-memory"


class WarmServerError(RuntimeError):
    """Raised when the warm server cannot be started or cannot be proven ready."""


@dataclass
class WarmRecallServer:
    """Start one authenticated streamable-http RE-call server and prove it answers.

    Use as a context manager. The server outlives every session in the run, which is the point:
    the embedder is loaded once.
    """

    dsn: str
    cwd: str | Path
    tenant: str = "memory"
    host: str = "127.0.0.1"
    port: int = 5480
    embedder: str = "fastembed"
    trust_mode: str = "development"
    python: str = field(default_factory=lambda: sys.executable)
    startup_timeout_s: float = 240.0
    server_name: str = DEFAULT_SERVER_NAME
    extra_env: dict[str, str] = field(default_factory=dict)

    _process: subprocess.Popen[bytes] | None = field(default=None, init=False, repr=False)
    _state_dir: Path | None = field(default=None, init=False, repr=False)
    _token: str = field(default="", init=False, repr=False)
    _log_path: Path | None = field(default=None, init=False, repr=False)
    handshake_ms: float | None = field(default=None, init=False)

    # ------------------------------------------------------------------ addresses and config

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}/mcp"

    @property
    def token(self) -> str:
        if not self._token:
            raise WarmServerError("the server has not been started, so no token exists")
        return self._token

    @property
    def mcp_config_path(self) -> Path:
        """Path to the `--mcp-config` file for the on arm.

        A path, not inline JSON, so the bearer token stays out of every recorded command line.
        """

        if self._state_dir is None:
            raise WarmServerError("the server has not been started")
        return self._state_dir / "mcp-config.json"

    def tool_prefix(self) -> str:
        return f"mcp__{self.server_name}__"

    # ------------------------------------------------------------------ lifecycle

    def __enter__(self) -> "WarmRecallServer":
        self.start()
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()

    def start(self) -> "WarmRecallServer":
        if self._process is not None:
            raise WarmServerError("already started")
        if _port_is_open(self.host, self.port):
            raise WarmServerError(
                f"{self.host}:{self.port} is already listening. Refusing to attach to a server "
                f"this run did not start: its tenant, corpus and trust mode are unknown, and "
                f"they are the experiment. Choose another port."
            )

        state = Path(tempfile.mkdtemp(prefix="recall-ab-"))
        self._state_dir = state
        self._token = secrets.token_urlsafe(32)
        tokens_path = state / "tokens.json"
        _write_private(
            tokens_path,
            json.dumps(
                {
                    "principals": [
                        {
                            "name": "agent-ab-harness",
                            "tenant": self.tenant,
                            "token": self._token,
                            "scopes": [READ_SCOPE],
                        }
                    ]
                }
            ),
        )
        _write_private(
            self.mcp_config_path,
            json.dumps(
                {
                    "mcpServers": {
                        self.server_name: {
                            "type": "http",
                            "url": self.url,
                            "headers": {"Authorization": f"Bearer {self._token}"},
                        }
                    }
                }
            ),
        )

        environment = dict(os.environ)
        environment.update(
            {
                "RECALL_TRANSPORT": "streamable-http",
                "RECALL_HOST": self.host,
                "RECALL_PORT": str(self.port),
                # Both are required for an HTTP transport: they are published in the server's
                # protected-resource metadata. With hand-provisioned tokens they are its own URL.
                "RECALL_AUTH_ISSUER_URL": f"http://{self.host}:{self.port}",
                "RECALL_AUTH_RESOURCE_URL": self.url,
                "RECALL_AUTH_TOKENS_FILE": str(tokens_path),
                "RECALL_DSN": self.dsn,
                "RECALL_EMBEDDER": self.embedder,
                "RECALL_TRUST_MODE": self.trust_mode,
                "RECALL_TENANT": self.tenant,
            }
        )
        environment.update(self.extra_env)
        # A static token file is refused outright under RECALL_ENV=production, and that refusal is
        # correct. Make sure an inherited value cannot turn this into a confusing startup crash.
        environment.pop("RECALL_ENV", None)

        self._log_path = state / "server.log"
        log = self._log_path.open("wb")
        self._process = subprocess.Popen(  # noqa: S603 - argv list, no shell
            [self.python, "-m", "recall_mcp.server"],
            cwd=str(self.cwd),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            self._await_listening()
        except Exception:
            self.close()
            raise
        return self

    def _await_listening(self) -> None:
        deadline = time.monotonic() + self.startup_timeout_s
        while time.monotonic() < deadline:
            if self._process is not None and self._process.poll() is not None:
                raise WarmServerError(
                    f"the RE-call server exited with code {self._process.returncode} during "
                    f"startup. Server log:\n{self.log_tail()}"
                )
            if _port_is_open(self.host, self.port):
                return
            time.sleep(0.5)
        raise WarmServerError(
            f"the RE-call server did not listen on {self.host}:{self.port} within "
            f"{self.startup_timeout_s}s. Server log:\n{self.log_tail()}"
        )

    def log_tail(self, limit: int = 3000) -> str:
        if self._log_path is None or not self._log_path.exists():
            return "(no server log)"
        return self._log_path.read_text(encoding="utf-8", errors="replace")[-limit:]

    def close(self) -> None:
        if self._process is not None:
            self._process.terminate()
            try:
                self._process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=20)
            self._process = None
        if self._state_dir is not None:
            # The token dies with the run. Nothing here is meant to survive it.
            shutil.rmtree(self._state_dir, ignore_errors=True)
            self._state_dir = None
        self._token = ""

    # ------------------------------------------------------------------ readiness proof

    async def check(self, query: str = "why must each session start its own database container") -> dict[str, Any]:
        """Complete a real MCP handshake, list tools, and run one controlled search.

        This is the standalone form of the four conditions the run gate needs: initialize
        completes, the tool list contains RE-call's tools, a search succeeds, and its latency is
        recorded. Failing here before a run is cheap; discovering it afterwards costs the run.
        """

        from mcp import ClientSession
        from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

        started = time.perf_counter()
        # This transport takes credentials on a pre-built HTTP client, not as a `headers` kwarg,
        # and yields a two-tuple. Both differ from the older `streamablehttp_client` API.
        http_client = create_mcp_http_client(
            headers={"Authorization": f"Bearer {self.token}"}
        )
        async with http_client:
            async with streamable_http_client(self.url, http_client=http_client) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    handshake_ms = (time.perf_counter() - started) * 1000.0
                    self.handshake_ms = handshake_ms
                    listed = await session.list_tools()
                    tools = sorted(tool.name for tool in listed.tools)
                    search_started = time.perf_counter()
                    result = await session.call_tool("recall_search", {"query": query})
                    search_ms = (time.perf_counter() - search_started) * 1000.0

        payload: dict[str, Any] = {}
        for block in result.content:
            text = getattr(block, "text", None)
            if isinstance(text, str):
                try:
                    payload = json.loads(text)
                except json.JSONDecodeError:
                    payload = {"raw": text[:400]}
                break

        if "recall_search" not in tools:
            raise WarmServerError(f"recall_search is missing from the tool list: {tools}")
        return {
            "url": self.url,
            "tenant": self.tenant,
            "handshake_ms": round(handshake_ms, 1),
            "search_ms": round(search_ms, 1),
            "tool_count": len(tools),
            "tools": tools,
            "abstained": payload.get("abstained"),
            "trust_state": payload.get("trust_state"),
            "failure_code": payload.get("failure_code"),
            "calibrated": payload.get("calibrated"),
            "hit_count": len(payload.get("hits") or payload.get("results") or []),
        }


#: Environment a Claude Code stdio MCP server needs before it can even import.
#:
#: An MCP config's `env` block **replaces** the environment rather than extending it, and the
#: omission fails twice over, misleadingly, on Windows:
#:
#: - without `APPDATA`, Python cannot find the user site-packages directory and the server dies on
#:   `ModuleNotFoundError: No module named 'anyio'`, which reads as a broken install;
#: - with `APPDATA` but without `SystemRoot`, the import succeeds and Winsock fails instead, with
#:   `OSError: [WinError 10106] The requested service provider could not be loaded`.
#:
#: Neither message mentions the environment. Both were reproduced on 2026-08-21.
STDIO_PASSTHROUGH_ENV: tuple[str, ...] = (
    "APPDATA",
    "LOCALAPPDATA",
    "SystemRoot",
    "PATH",
    "PATHEXT",
    "TEMP",
    "TMP",
    "USERPROFILE",
    "HOMEDRIVE",
    "HOMEPATH",
    "SYSTEMDRIVE",
    "COMSPEC",
    "NUMBER_OF_PROCESSORS",
    "PROCESSOR_ARCHITECTURE",
    "LANG",
    "LC_ALL",
)


@dataclass
class StdioRecallSpec:
    """A per-session stdio RE-call server, which is the only way to serve a CALIBRATED corpus.

    RE-call reads generations only under `RECALL_ENV=production` (`recall_mcp/server.py`), and
    `build_auth` refuses a static bearer token under production because such a token has no
    expiry, issuer or revocation path. Those two rules together mean the warm HTTP server above
    **cannot** carry a calibrated generation without standing up OIDC. stdio needs no
    authentication, because there is no remote caller to authenticate, so it can.

    The cost is a cold start per session, measured at about 11 s here, almost all of it loading the
    embedder. That is affordable only because Claude Code **2.1.221 and later wait** for a pending
    MCP server before the first turn. Below that version the session simply runs without the tool
    and reports success, which is the failure `gate.py` exists to catch.
    """

    dsn: str
    cwd: str | Path
    tenant: str = "default"
    embedder: str = "fastembed"
    python: str = field(default_factory=lambda: sys.executable)
    server_name: str = DEFAULT_SERVER_NAME
    #: Left unset so the strict trust policy applies. A calibrated corpus should satisfy the
    #: policy rather than need it relaxed; setting this would hide a calibration that did not take.
    trust_mode: str | None = None
    _state_dir: Path | None = field(default=None, init=False, repr=False)

    def env(self) -> dict[str, str]:
        environment = {
            key: os.environ[key] for key in STDIO_PASSTHROUGH_ENV if key in os.environ
        }
        environment.update(
            {
                "RECALL_ENV": "production",
                "RECALL_TRANSPORT": "stdio",
                "RECALL_DSN": self.dsn,
                "RECALL_EMBEDDER": self.embedder,
                "RECALL_TENANT": self.tenant,
                "PYTHONUTF8": "1",
                # The config's `cwd` is not what decides whether `python -m recall_mcp.server`
                # can import anything: measured 2026-08-21, the same config connects when the
                # SESSION runs from the repository root and fails when it runs from an artifact
                # directory, so the server inherits the session's working directory rather than
                # the one it was given. `PYTHONPATH` makes the import independent of both, which
                # is what a benchmark that runs sessions in a scratch directory needs.
                "PYTHONPATH": str(Path(self.cwd).resolve()),
                # ⛔ And PYTHONPATH alone is not enough once sessions run somewhere real.
                #
                # The server inherits the SESSION's working directory, and `python -m` puts that
                # directory FIRST on sys.path, ahead of PYTHONPATH. The task-success benchmark runs
                # each session in a sandbox restored from a fixture, and several of those fixtures
                # contain their own `recall/` package, because the tasks are about this repository.
                # So the fixture shadowed the real package and the server died on
                # `ModuleNotFoundError: No module named 'recall.calibration'`.
                #
                # What that looks like from outside is the failure this whole harness is built to
                # catch: `mcp_servers: [{"name": "recall-memory", "status": "failed"}]`, no recall
                # tool in the session, and a perfectly ordinary-looking transcript. Measured
                # 2026-08-21, cwd=sandbox fails and cwd=sandbox plus this variable gives 16 tools.
                # PYTHONSAFEPATH drops the working directory from sys.path entirely, which is what
                # a server that must not import from wherever its caller happens to stand wants.
                "PYTHONSAFEPATH": "1",
            }
        )
        if self.trust_mode:
            environment["RECALL_TRUST_MODE"] = self.trust_mode
        return environment

    def tool_prefix(self) -> str:
        return f"mcp__{self.server_name}__"

    def write_mcp_config(self, destination: str | Path) -> Path:
        target = Path(destination)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(
                {
                    "mcpServers": {
                        self.server_name: {
                            "command": self.python,
                            "args": ["-m", "recall_mcp.server"],
                            "cwd": str(self.cwd),
                            "env": self.env(),
                        }
                    }
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        self._state_dir = target.parent
        return target

    async def check(self, query: str = "how do I limit the number of CPU threads the embedder uses") -> dict[str, Any]:
        """Drive one stdio server directly, and report what the strict trust policy decided."""

        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        params = StdioServerParameters(
            command=self.python,
            args=["-m", "recall_mcp.server"],
            env=self.env(),
            cwd=str(self.cwd),
        )
        started = time.perf_counter()
        async with stdio_client(params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                handshake_ms = (time.perf_counter() - started) * 1000.0
                listed = await session.list_tools()
                tools = sorted(tool.name for tool in listed.tools)
                search_started = time.perf_counter()
                result = await session.call_tool("recall_search", {"query": query})
                search_ms = (time.perf_counter() - search_started) * 1000.0

        text = next((getattr(b, "text", "") for b in result.content if getattr(b, "text", None)), "")
        try:
            payload = json.loads(text)
        except json.JSONDecodeError:
            # A refusal comes back as prose, not JSON, and it is the most informative failure
            # this check can produce: it names the tenant, generation and calibration status.
            raise WarmServerError(f"recall_search was refused: {text[:300]}") from None
        if "recall_search" not in tools:
            raise WarmServerError(f"recall_search is missing from the tool list: {tools}")
        return {
            "transport": "stdio",
            "tenant": self.tenant,
            "handshake_ms": round(handshake_ms, 1),
            "search_ms": round(search_ms, 1),
            "tool_count": len(tools),
            "abstained": payload.get("abstained"),
            "trust_state": payload.get("trust_state"),
            "failure_code": payload.get("failure_code"),
            "calibrated": payload.get("calibrated"),
            "generation_id": payload.get("generation_id"),
            "calibration_id": payload.get("calibration_id"),
        }


def _port_is_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.settimeout(timeout)
        return probe.connect_ex((host, port)) == 0


def _write_private(path: Path, text: str) -> None:
    """Write a secret to a file only this user can read."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    try:
        path.chmod(0o600)
    except OSError:  # pragma: no cover - best effort on platforms without POSIX modes
        pass
