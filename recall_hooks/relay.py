"""A small, per-session loopback relay for the write-time hook.

The hook process is intentionally short-lived.  This module keeps the expensive PostgreSQL
connection in a helper owned by the Claude session instead.  The transport is loopback TCP rather
than a Unix-only socket so the same implementation works on Windows.  Every request carries a
random token stored in a user-private state file; a random local port alone is not authentication.

This is best-effort infrastructure.  A relay failure raises ``RelayUnavailable`` to the hook
boundary, where the write-time hook records cooldown and fails open. It can never deny a client
tool call.
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
from contextlib import contextmanager
from pathlib import Path
from collections.abc import Iterator
from typing import Any, Callable, Protocol, TypedDict, cast

from . import claude_config_home, load_config

STATE_DIR_NAME = "recall-hook-relays"
STATE_PREFIX = "relay-"
# The client hook is configured with a five-second timeout.  Keep both relay waits below that
# ceiling so a failed helper cannot outlive the hook invocation that owns it.
REQUEST_TIMEOUT_SECONDS = 4.0
START_TIMEOUT_SECONDS = 4.0
STOP_WAIT_SECONDS = 6.0
SEARCH_BUDGET_SECONDS = 4.5
IDLE_TIMEOUT_SECONDS = 15 * 60.0
RETRY_BACKOFF_SECONDS = 5.0
LOCK_TIMEOUT_SECONDS = START_TIMEOUT_SECONDS
LOCK_STALE_SECONDS = START_TIMEOUT_SECONDS * 3.0


class RelayState(TypedDict, total=False):
    """Persisted fields used to find and authenticate one session's helper."""

    version: int
    host: str
    port: int
    pid: int
    token: str
    fingerprint: str
    created_at: float
    last_used: float


class RelayResponse(TypedDict, total=False):
    """Response envelope exchanged over the local relay protocol."""

    status: str
    error: str
    hits: list[Any]


class ConnectionLike(Protocol):
    """Small structural surface needed from psycopg without importing it in the hook."""

    def execute(self, *args: Any, **kwargs: Any) -> Any: ...

    def close(self) -> Any: ...


class RelayUnavailable(RuntimeError):
    """The helper could not serve a request."""


class RelayServiceUnavailable(RelayUnavailable):
    """The helper answered, but its database is currently unavailable."""


def _session_key(session_id: str) -> str:
    return hashlib.sha256(session_id.encode("utf-8", "surrogatepass")).hexdigest()[:32]


def state_path(session_id: str) -> Path:
    """Return the state path without putting an untrusted session id in a filename."""

    return claude_config_home() / STATE_DIR_NAME / f"{STATE_PREFIX}{_session_key(session_id)}.json"


def _lock_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.lock")


def _stop_marker_path(path: Path) -> Path:
    return path.with_name(f"{path.name}.stop")


@contextmanager
def _state_lock(path: Path, deadline: float | None = None) -> Iterator[None]:
    """Serialize state inspection and helper startup across concurrent hook processes."""

    lock = _lock_path(path)
    lock.parent.mkdir(parents=True, exist_ok=True)
    wait_deadline = time.monotonic() + LOCK_TIMEOUT_SECONDS
    if deadline is not None:
        wait_deadline = min(wait_deadline, deadline)
    acquired = False
    while time.monotonic() < wait_deadline:
        try:
            fd = os.open(str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        except FileExistsError:
            try:
                if time.time() - lock.stat().st_mtime > LOCK_STALE_SECONDS:
                    lock.unlink()
                    continue
            except OSError:
                pass
            time.sleep(0.02)
            continue
        else:
            try:
                os.write(fd, str(os.getpid()).encode("ascii"))
            finally:
                os.close(fd)
            acquired = True
            break
    if not acquired:
        raise RelayUnavailable("relay state lock timeout")
    try:
        yield
    finally:
        try:
            lock.unlink()
        except OSError:
            pass


def _atomic_write(path: Path, document: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        path.parent.chmod(0o700)
    except OSError:
        pass
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


def _read(path: Path) -> RelayState | None:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return cast(RelayState, document) if isinstance(document, dict) else None


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


def _endpoint_request(
    state: RelayState, message: dict[str, Any], timeout: float = REQUEST_TIMEOUT_SECONDS
) -> RelayResponse:
    host = str(state["host"])
    port = int(state["port"])
    if host != "127.0.0.1" or not 1 <= port <= 65535:
        raise RelayUnavailable("relay endpoint is not a valid loopback address")
    timeout = max(0.01, min(float(timeout), REQUEST_TIMEOUT_SECONDS))
    with socket.create_connection((host, port), timeout=timeout) as connection:
        connection.settimeout(timeout)
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
    return cast(RelayResponse, response)


def _wait_for_exit(pid: Any) -> bool:
    """Wait briefly for a helper to exit without ever killing a possibly-reused PID."""

    if not _pid_alive(pid):
        return True
    deadline = time.monotonic() + STOP_WAIT_SECONDS
    while time.monotonic() < deadline:
        if not _pid_alive(pid):
            return True
        time.sleep(0.05)
    return not _pid_alive(pid)


def _retire_state_locked(path: Path, state: RelayState | None = None) -> None:
    """Signal a helper to stop; the caller must hold the per-state lock."""

    state = state or _read(path)
    if not state:
        path.unlink(missing_ok=True)
        _stop_marker_path(path).unlink(missing_ok=True)
        return
    token = str(state.get("token", ""))
    if token:
        try:
            _atomic_write(_stop_marker_path(path), {"token": token})
        except OSError:
            pass
    path.unlink(missing_ok=True)
    if not _pid_alive(state.get("pid")):
        _stop_marker_path(path).unlink(missing_ok=True)


def _retire_state(
    path: Path, state: RelayState | None = None, deadline: float | None = None
) -> None:
    """Retire one owned helper without racing a replacement state."""

    try:
        with _state_lock(path, deadline):
            current = _read(path)
            expected_token = str((state or {}).get("token", ""))
            current_token = str((current or {}).get("token", ""))
            if expected_token and current and current_token != expected_token:
                return
            _retire_state_locked(path, current or state)
    except RelayUnavailable:
        # Cleanup is best effort and must not consume the query's remaining budget.
        return


def _stop_state(path: Path, state: RelayState | None = None) -> bool:
    """Request shutdown and remove state only after the owned helper has exited."""

    marker = _stop_marker_path(path)
    with _state_lock(path):
        current = _read(path)
        if state and current and current.get("token") != state.get("token"):
            return True
        state = current or state
        if not state:
            path.unlink(missing_ok=True)
            marker.unlink(missing_ok=True)
            return True
        pid = state.get("pid")
        token = str(state.get("token", ""))
        if token:
            try:
                _atomic_write(marker, {"token": token})
            except OSError:
                pass
        # Prevent a concurrent query from adopting a helper that is already stopping. The child
        # still has the marker and its PID, so the caller can wait for real process termination.
        path.unlink(missing_ok=True)
    if state and state.get("port"):
        try:
            _endpoint_request(state, {"op": "shutdown", "token": str(state.get("token", ""))})
        except Exception:
            pass
    stopped = _wait_for_exit(pid)
    if stopped:
        try:
            with _state_lock(path):
                current_marker = _read(marker)
                if isinstance(current_marker, dict) and current_marker.get("token") == token:
                    marker.unlink(missing_ok=True)
        except RelayUnavailable:
            pass
    return stopped


def _initial_state(token: str, fingerprint: str) -> RelayState:
    return {
        "version": 1,
        "host": "127.0.0.1",
        "port": 0,
        "pid": 0,
        "token": token,
        "fingerprint": fingerprint,
        "created_at": time.time(),
    }


def _spawn(
    path: Path,
    token: str,
    fingerprint: str,
    deadline: float | None = None,
    initialize: bool = True,
) -> RelayState:
    if initialize:
        _atomic_write(path, _initial_state(token, fingerprint))
    creationflags = 0
    if os.name == "nt":
        creationflags = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0) | getattr(
            subprocess, "DETACHED_PROCESS", 0
        )
    else:
        # A session hook must not retain a pipe or process-group dependency on the short-lived
        # client invocation that started it.
        creationflags = 0
    try:
        subprocess.Popen(
            [sys.executable, "-m", "recall_hooks", "write-time-relay", "--state", str(path)],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            close_fds=os.name != "nt",
            start_new_session=os.name != "nt",
            creationflags=creationflags,
        )
    except Exception:
        try:
            with _state_lock(path):
                current = _read(path)
                if current and current.get("token") == token:
                    path.unlink(missing_ok=True)
                    current_marker = _read(_stop_marker_path(path))
                    if isinstance(current_marker, dict) and current_marker.get("token") == token:
                        _stop_marker_path(path).unlink(missing_ok=True)
        except RelayUnavailable:
            pass
        raise
    start_deadline = time.monotonic() + START_TIMEOUT_SECONDS
    if deadline is not None:
        start_deadline = min(start_deadline, deadline)
    while time.monotonic() < start_deadline:
        state = _read(path)
        if state and int(state.get("port", 0) or 0) > 0:
            try:
                response = _endpoint_request(
                    state,
                    {"op": "ping", "token": str(state.get("token", ""))},
                    timeout=max(0.01, start_deadline - time.monotonic()),
                )
                if response.get("status") == "ok":
                    return state
            except Exception:
                pass
        time.sleep(min(0.05, max(0.01, start_deadline - time.monotonic())))
    # The child is detached and may have read the initial state just before the deadline. Leave a
    # tokenized stop marker for that child, rather than only removing the state it can recreate.
    try:
        with _state_lock(path):
            current = _read(path)
            if current and current.get("token") == token:
                _atomic_write(_stop_marker_path(path), {"token": token})
                path.unlink(missing_ok=True)
    except (OSError, RelayUnavailable):
        pass
    raise RelayUnavailable("relay did not become ready")


def _ensure(
    session_id: str,
    config: dict[str, Any],
    options: dict[str, Any],
    deadline: float | None = None,
) -> tuple[Path, RelayState]:
    path = state_path(session_id)
    fingerprint = _fingerprint(config, options)
    start = False
    with _state_lock(path, deadline):
        state = _read(path)
        if state and state.get("fingerprint") == fingerprint and int(state.get("port", 0) or 0) > 0:
            # The query itself is the liveness check.  A separate ping adds one loopback
            # round trip to every tool call and still races the query it precedes.  If the helper
            # is stale, search retires it after the query transport fails and retries once.
            return path, state
        if state and state.get("fingerprint") == fingerprint and not int(state.get("port", 0) or 0):
            token = str(state.get("token", ""))
        else:
            if state:
                _retire_state_locked(path, state)
            token = secrets.token_urlsafe(32)
            _atomic_write(path, _initial_state(token, fingerprint))
            start = True

    if start:
        return path, _spawn(path, token, fingerprint, deadline, False)

    # Another hook process owns startup. Wait for its claimed token instead of launching a second
    # detached child, and retry the state claim only after that owner has disappeared.
    wait_deadline = deadline if deadline is not None else time.monotonic() + START_TIMEOUT_SECONDS
    while time.monotonic() < wait_deadline:
        state = _read(path)
        if state and state.get("token") == token and int(state.get("port", 0) or 0) > 0:
            return path, state
        if not state or state.get("token") != token:
            return _ensure(session_id, config, options, deadline)
        time.sleep(0.02)
    raise RelayUnavailable("relay startup budget exhausted")


def search(
    session_id: str,
    query: str,
    config: dict[str, Any],
    options: dict[str, Any],
) -> list[tuple[str, str, float]]:
    """Search through the session relay, restarting one crashed helper at most once.

    Raises:
        RelayUnavailable: the local helper could not be reached or returned malformed data.
        RelayServiceUnavailable: the helper is alive but its database is unavailable. Both are
            caught by the outer write-time hook, which fails open and starts its cooldown.
    """

    if not session_id:
        raise RelayUnavailable("client supplied no session id")
    last_error: Exception | None = None
    deadline = time.monotonic() + SEARCH_BUDGET_SECONDS
    for _ in range(2):
        if time.monotonic() >= deadline:
            break
        path, state = _ensure(session_id, config, options, deadline)
        try:
            response = _endpoint_request(
                state,
                {
                    "op": "query",
                    "token": str(state.get("token", "")),
                    "query": query[: _max_query_chars()],
                },
                timeout=max(0.01, deadline - time.monotonic()),
            )
            status = response.get("status")
            if status == "ok":
                rows = response.get("hits", [])
                return [(str(row[0]), str(row[1]), float(row[2])) for row in rows]
            raise RelayServiceUnavailable(str(response.get("error", "relay query failed")))
        except RelayServiceUnavailable:
            raise
        except (RelayUnavailable, OSError, ValueError, KeyError, IndexError, TypeError, json.JSONDecodeError) as exc:
            last_error = exc
            _retire_state(path, state, deadline=deadline)
    raise RelayUnavailable(str(last_error or "relay query failed"))


def _max_query_chars() -> int:
    """Read the protocol limit without importing the hook module at relay import time."""

    from .write_time import MAX_QUERY_CHARS

    return MAX_QUERY_CHARS


def stop(session_id: str) -> None:
    """Ask one session's helper to exit and remove its state, best effort."""

    if not session_id:
        return
    path = state_path(session_id)
    try:
        _stop_state(path)
    except Exception:
        # SessionEnd is best effort.  If another hook owns the lock, its own lifecycle call will
        # reconcile the state; never race it by deleting a newly-created relay state.
        return


def stop_all() -> None:
    """Used by uninstall so no relay survives removal of the hook configuration."""

    directory = claude_config_home() / STATE_DIR_NAME
    try:
        paths = list(directory.glob(f"{STATE_PREFIX}*.json"))
    except OSError:
        return
    for path in paths:
        try:
            _stop_state(path)
        except Exception:
            # Do not delete a state file while another hook is starting or querying that session.
            continue


def _read_request(client: socket.socket) -> dict[str, Any]:
    """Read one bounded newline framed request from a relay client."""

    client.settimeout(REQUEST_TIMEOUT_SECONDS)
    data = b""
    while b"\n" not in data and len(data) <= 256 * 1024:
        chunk = client.recv(65536)
        if not chunk:
            break
        data += chunk
    request = json.loads(data.split(b"\n", 1)[0].decode("utf-8"))
    if not isinstance(request, dict):
        raise ValueError("relay request is not an object")
    return request


def _handle_request(
    request: dict[str, Any],
    token: str,
    connection: ConnectionLike | None,
    next_connect: float,
    config: dict[str, Any],
    options: dict[str, Any],
    connect: Callable[[], ConnectionLike],
) -> tuple[RelayResponse, ConnectionLike | None, float, bool, bool]:
    """Authenticate and dispatch one request, returning updated connection state."""

    authenticated = hmac.compare_digest(str(request.get("token", "")), token)
    if not authenticated:
        return {"status": "error", "error": "unauthorized"}, connection, next_connect, False, False
    operation = request.get("op")
    if operation == "ping":
        return {"status": "ok"}, connection, next_connect, False, True
    if operation == "shutdown":
        return {"status": "ok"}, connection, next_connect, True, True
    if operation != "query":
        return {"status": "error", "error": "unknown operation"}, connection, next_connect, False, True

    now = time.monotonic()
    if connection is None and now >= next_connect:
        try:
            connection = connect()
        except Exception as exc:  # noqa: BLE001, fail-open relay boundary
            return (
                {"status": "unavailable", "error": type(exc).__name__},
                None,
                now + RETRY_BACKOFF_SECONDS,
                False,
                True,
            )
    if connection is None:
        return (
            {"status": "unavailable", "error": "connection backoff"},
            None,
            next_connect,
            False,
            True,
        )
    try:
        # The client bounds this too. Keeping the child bound makes direct protocol callers safe.
        from .write_time import MAX_QUERY_CHARS, _search_connection

        hits = _search_connection(connection, str(request.get("query", ""))[:MAX_QUERY_CHARS], config, options)
        return {"status": "ok", "hits": hits}, connection, next_connect, False, True
    except Exception as exc:  # noqa: BLE001, fail-open relay boundary
        try:
            connection.close()
        except Exception:
            pass
        return (
            {"status": "unavailable", "error": type(exc).__name__},
            None,
            time.monotonic() + RETRY_BACKOFF_SECONDS,
            False,
            True,
        )


def _write_state_if_owned(path: Path, state: RelayState, token: str) -> None:
    """Persist heartbeat state only while this helper still owns the tokenized state path."""

    try:
        with _state_lock(path):
            current = _read(path)
            if (
                current
                and current.get("token") == token
                and int(current.get("pid", 0) or 0) == os.getpid()
            ):
                _atomic_write(path, state)
    except RelayUnavailable:
        pass


def _cleanup_owned_state(path: Path, token: str) -> None:
    """Remove only this helper's state and stop marker, never a replacement's files."""

    try:
        with _state_lock(path):
            current = _read(path)
            if (
                current
                and current.get("token") == token
                and int(current.get("pid", 0) or 0) == os.getpid()
            ):
                path.unlink(missing_ok=True)
            marker = _read(_stop_marker_path(path))
            if isinstance(marker, dict) and marker.get("token") == token:
                _stop_marker_path(path).unlink(missing_ok=True)
    except RelayUnavailable:
        pass


def _publish_startup_state(path: Path, state: RelayState, token: str) -> bool:
    """Publish the listening endpoint only while the claimed token still owns the path."""

    try:
        with _state_lock(path):
            current = _read(path)
            stopping = _read(_stop_marker_path(path))
            if (
                not current
                or current.get("token") != token
                or (isinstance(stopping, dict) and stopping.get("token") == token)
            ):
                return False
            _atomic_write(path, state)
            return True
    except RelayUnavailable:
        return False


def _serve(state_path_value: Path) -> int:
    """Run the child process.  It owns the socket and one PostgreSQL connection."""

    state = _read(state_path_value)
    if not state or not state.get("token"):
        return 0
    config = load_config()
    if not config.get("dsn"):
        return 0
    from .write_time import settings

    options = settings(config)
    token = str(state["token"])
    connection: ConnectionLike | None = None
    next_connect = 0.0

    def connect() -> ConnectionLike:
        import psycopg

        return psycopg.connect(
            str(config["dsn"]),
            autocommit=True,
            connect_timeout=options["connect_timeout"],
            options="-c statement_timeout=5s",
        )

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(8)
    listener.settimeout(1.0)
    state = {**state, "host": "127.0.0.1", "port": listener.getsockname()[1], "pid": os.getpid()}
    if not _publish_startup_state(state_path_value, state, token):
        listener.close()
        return 0
    stop_requested = False

    try:
        while not stop_requested:
            if not state_path_value.exists():
                break
            stopping = _read(_stop_marker_path(state_path_value))
            if isinstance(stopping, dict) and stopping.get("token") == token:
                break
            if time.time() - float(state.get("last_used", state.get("created_at", time.time()))) > IDLE_TIMEOUT_SECONDS:
                break
            try:
                client, _ = listener.accept()
            except socket.timeout:
                continue
            with client:
                authenticated = False
                request: dict[str, Any] = {}
                try:
                    request = _read_request(client)
                    response, connection, next_connect, stop_requested, authenticated = _handle_request(
                        request, token, connection, next_connect, config, options, connect
                    )
                except Exception as exc:  # noqa: BLE001, malformed local input must not crash child
                    response = {"status": "error", "error": type(exc).__name__}
                stopping = _read(_stop_marker_path(state_path_value))
                if isinstance(stopping, dict) and stopping.get("token") == token:
                    stop_requested = True
                if authenticated and not stop_requested:
                    state["last_used"] = time.time()
                if not stop_requested and request.get("op") == "query":
                    _write_state_if_owned(state_path_value, state, token)
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
        _cleanup_owned_state(state_path_value, token)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run the detached relay child for the state file supplied by ``--state``.

    The process is intentionally silent and returns zero even when startup fails, because the
    caller is a best-effort hook and must never turn relay infrastructure into a client failure.
    """

    args = list(sys.argv[1:] if argv is None else argv)
    if "--state" not in args:
        return 0
    try:
        path = Path(args[args.index("--state") + 1])
        return _serve(path)
    except Exception:
        return 0


__all__ = [
    "RelayServiceUnavailable",
    "RelayUnavailable",
    "main",
    "search",
    "state_path",
    "stop",
    "stop_all",
]
