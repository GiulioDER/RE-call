"""A small, per-session loopback relay for the write-time hook.

The hook process is intentionally short-lived.  This module keeps the expensive PostgreSQL
connection in a helper owned by the Claude session instead.  The transport is loopback TCP rather
than a Unix-only socket so the same implementation works on Windows.  Every request carries a
random token stored in a user-private state file; a random local port alone is not authentication.

This is best-effort infrastructure.  A relay failure is reported to the caller as ``None`` and the
write-time hook falls back to its existing cooldown behavior.  It can never deny a client tool call.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import secrets
import socket
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

from . import claude_config_home, load_config

STATE_DIR_NAME = "recall-hook-relays"
STATE_PREFIX = "relay-"
REQUEST_TIMEOUT_SECONDS = 5.5
START_TIMEOUT_SECONDS = 8.0
IDLE_TIMEOUT_SECONDS = 15 * 60.0
RETRY_BACKOFF_SECONDS = 5.0


class RelayUnavailable(RuntimeError):
    """The helper could not serve a request."""


def _session_key(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8", "surrogatepass")).hexdigest()[:32]


def state_path(session_id: str) -> Path:
    """Return the state path without putting an untrusted session id in a filename."""

    return claude_config_home() / STATE_DIR_NAME / f"{STATE_PREFIX}{_session_key(session_id)}.json"


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=str(path.parent), prefix=f".{path.name}.", suffix=".tmp")
    temporary = Path(name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(document, sort_keys=True) + "\n")
        try:
            temporary.chmod(0o600)
        except OSError:
            pass
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _read(path: Path) -> dict[str, Any] | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return document if isinstance(document, dict) else None


def _fingerprint(config: dict[str, Any], options: dict[str, Any]) -> str:
    stable = {
        "dsn": str(config.get("dsn", "")),
        "tenant": str(config.get("tenant", "default")),
        "k": int(options["k"]),
        "connect_timeout": float(options["connect_timeout"]),
    }
    return hashlib.sha256(json.dumps(stable, sort_keys=True).encode()).hexdigest()


def _pid_alive(pid: Any) -> bool:
    try:
        numeric = int(pid)
        if numeric <= 0:
            return False
        os.kill(numeric, 0)
        return True
    except (OSError, TypeError, ValueError):
        return False


def _endpoint_request(state: dict[str, Any], message: dict[str, Any]) -> dict[str, Any]:
    host = str(state["host"])
    port = int(state["port"])
    with socket.create_connection((host, port), timeout=REQUEST_TIMEOUT_SECONDS) as connection:
        connection.settimeout(REQUEST_TIMEOUT_SECONDS)
        connection.sendall((json.dumps(message, separators=(",", ":")) + "\n").encode())
        buffer = b""
        while b"\n" not in buffer:
            chunk = connection.recv(65536)
            if not chunk:
                raise RelayUnavailable("relay closed the connection")
            buffer += chunk
        response = json.loads(buffer.split(b"\n", 1)[0].decode("utf-8"))
    if not isinstance(response, dict):
        raise RelayUnavailable("relay returned a malformed response")
    return response


def _stop_state(path: Path, state: dict[str, Any] | None = None) -> None:
    state = state or _read(path)
    if state and state.get("port"):
        try:
            _endpoint_request(state, {"op": "shutdown", "token": str(state.get("token", ""))})
        except Exception:
            pass
    try:
        path.unlink()
    except OSError:
        pass


def _spawn(path: Path, token: str, fingerprint: str) -> dict[str, Any]:
    _atomic_write(path, {
        "version": 1,
        "host": "127.0.0.1",
        "port": 0,
        "pid": 0,
        "token": token,
        "fingerprint": fingerprint,
        "created_at": time.time(),
    })
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
    else:
        # A session hook must not retain a pipe or process-group dependency on the short-lived
        # client invocation that started it.
        creationflags = 0
    subprocess.Popen(
        [sys.executable, "-m", "recall_hooks", "write-time-relay", "--state", str(path)],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        close_fds=os.name != "nt",
        start_new_session=os.name != "nt",
        creationflags=creationflags,
    )
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        state = _read(path)
        if state and int(state.get("port", 0) or 0) > 0:
            try:
                response = _endpoint_request(
                    state, {"op": "ping", "token": str(state.get("token", ""))}
                )
                if response.get("status") == "ok":
                    return state
            except Exception:
                pass
        time.sleep(0.05)
    raise RelayUnavailable("relay did not become ready")


def _ensure(session_id: str, config: dict[str, Any], options: dict[str, Any]) -> tuple[Path, dict[str, Any]]:
    path = state_path(session_id)
    fingerprint = _fingerprint(config, options)
    state = _read(path)
    if state and state.get("fingerprint") == fingerprint and int(state.get("port", 0) or 0) > 0:
        try:
            response = _endpoint_request(
                state, {"op": "ping", "token": str(state.get("token", ""))}
            )
            if response.get("status") == "ok":
                return path, state
        except Exception:
            pass
        _stop_state(path, state)
    elif state:
        _stop_state(path, state)

    token = secrets.token_urlsafe(32)
    return path, _spawn(path, token, fingerprint)


def search(
    session_id: str,
    query: str,
    config: dict[str, Any],
    options: dict[str, Any],
) -> list[tuple[str, str, float]]:
    """Search through the session relay, restarting one crashed helper at most once."""

    if not session_id:
        raise RelayUnavailable("client supplied no session id")
    last_error: Exception | None = None
    for _ in range(2):
        path, state = _ensure(session_id, config, options)
        try:
            response = _endpoint_request(
                state,
                {"op": "query", "token": str(state.get("token", "")), "query": query},
            )
            status = response.get("status")
            if status == "ok":
                rows = response.get("hits", [])
                return [(str(row[0]), str(row[1]), float(row[2])) for row in rows]
            raise RelayUnavailable(str(response.get("error", "relay query failed")))
        except (OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            last_error = exc
            _stop_state(path, state)
    raise RelayUnavailable(str(last_error or "relay query failed"))


def stop(session_id: str) -> None:
    """Ask one session's helper to exit and remove its state, best effort."""

    if not session_id:
        return
    path = state_path(session_id)
    _stop_state(path)
    # The child writes its final heartbeat before replying to shutdown.  On Windows the replace
    # and unlink can cross briefly, so make the postcondition of SessionEnd deterministic without
    # waiting for the process itself or killing a possibly-reused PID.
    for _ in range(10):
        if not path.exists():
            break
        try:
            path.unlink()
        except OSError:
            time.sleep(0.02)


def stop_all() -> None:
    """Used by uninstall so no relay survives removal of the hook configuration."""

    directory = claude_config_home() / STATE_DIR_NAME
    try:
        paths = list(directory.glob(f"{STATE_PREFIX}*.json"))
    except OSError:
        return
    for path in paths:
        _stop_state(path)


def _serve(state_path_value: Path) -> int:
    """Run the child process.  It owns the socket and one PostgreSQL connection."""

    state = _read(state_path_value)
    if not state or not state.get("token"):
        return 0
    config = load_config()
    if not config.get("dsn"):
        return 0
    # Local import keeps SessionStart and the ordinary hook path free of psycopg import cost.
    from .write_time import _search_connection, settings

    options = settings(config)
    token = str(state["token"])
    connection: Any = None
    next_connect = 0.0
    stop_requested = False

    def connect() -> Any:
        import psycopg

        return psycopg.connect(
            str(config["dsn"]),
            connect_timeout=options["connect_timeout"],
            options="-c statement_timeout=5s",
        )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.settimeout(1.0)
    state = {**state, "host": "127.0.0.1", "port": listener.getsockname()[1], "pid": os.getpid()}
    _atomic_write(state_path_value, state)

    try:
        while not stop_requested:
            if time.time() - float(state.get("last_used", state.get("created_at", time.time()))) > IDLE_TIMEOUT_SECONDS:
                break
            try:
                client, _ = listener.accept()
            except socket.timeout:
                continue
            with client:
                try:
                    client.settimeout(REQUEST_TIMEOUT_SECONDS)
                    data = b""
                    while b"\n" not in data and len(data) <= 256 * 1024:
                        chunk = client.recv(65536)
                        if not chunk:
                            break
                        data += chunk
                    request = json.loads(data.split(b"\n", 1)[0].decode("utf-8"))
                    valid = hmac.compare_digest(str(request.get("token", "")), token)
                    if not valid:
                        response = {"status": "error", "error": "unauthorized"}
                    elif request.get("op") == "ping":
                        response = {"status": "ok"}
                    elif request.get("op") == "shutdown":
                        response = {"status": "ok"}
                        stop_requested = True
                    elif request.get("op") == "query":
                        now = time.monotonic()
                        if connection is None and now >= next_connect:
                            try:
                                connection = connect()
                            except Exception as exc:  # noqa: BLE001 - relay is fail-open
                                next_connect = now + RETRY_BACKOFF_SECONDS
                                response = {"status": "unavailable", "error": type(exc).__name__}
                            else:
                                response = None
                        else:
                            response = None
                        if response is None:
                            try:
                                hits = _search_connection(connection, str(request.get("query", "")), config, options)
                                response = {"status": "ok", "hits": hits}
                            except Exception as exc:  # noqa: BLE001 - relay is fail-open
                                try:
                                    connection.close()
                                except Exception:
                                    pass
                                connection = None
                                next_connect = time.monotonic() + RETRY_BACKOFF_SECONDS
                                response = {"status": "unavailable", "error": type(exc).__name__}
                    else:
                        response = {"status": "error", "error": "unknown operation"}
                except Exception as exc:  # noqa: BLE001 - malformed local input must not crash child
                    response = {"status": "error", "error": type(exc).__name__}
                state["last_used"] = time.time()
                try:
                    _atomic_write(state_path_value, state)
                except OSError:
                    pass
                try:
                    client.sendall((json.dumps(response, separators=(",", ":")) + "\n").encode())
                except OSError:
                    pass
    finally:
        if connection is not None:
            try:
                connection.close()
            except Exception:
                pass
        listener.close()
        current = _read(state_path_value)
        if current and int(current.get("pid", 0) or 0) == os.getpid():
            try:
                state_path_value.unlink()
            except OSError:
                pass
    return 0


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if "--state" not in args:
        return 0
    try:
        path = Path(args[args.index("--state") + 1])
        return _serve(path)
    except Exception:
        return 0


__all__ = ["RelayUnavailable", "main", "search", "state_path", "stop", "stop_all"]
