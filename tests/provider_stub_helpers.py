"""A local HTTP endpoint that counts requests, for testing retry policy against a real SDK.

Every SDK ships its own retry layer inside its transport. A faked client object cannot see that
layer, so a test built on one is blind to exactly the thing under test: whether the SDK retries
underneath our own retry loop and multiplies with it. Counting real POSTs against a real socket
is the only evidence that distinguishes "three attempts" from "three times three".

Used by ``test_embeddings_retry_policy`` and ``test_mtrag_generation_retry_policy``.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

#: A minimal but complete ``/v1/embeddings`` success body. The SDK parses it into a real
#: ``CreateEmbeddingResponse``, so an embedder's dim probe sees a genuine 3-wide vector.
EMBEDDINGS_OK: dict[str, Any] = {
    "object": "list",
    "model": "stub",
    "data": [{"object": "embedding", "index": 0, "embedding": [0.0, 0.0, 1.0]}],
    "usage": {"prompt_tokens": 1, "total_tokens": 1},
}

#: The same for ``/v1/chat/completions``, parsed into a real ``ChatCompletion`` so the success
#: path exercises the same object the caller reads. ``finish_reason`` is "stop" rather than
#: "length" so the happy path does not trip any truncation guard.
CHAT_OK: dict[str, Any] = {
    "id": "chatcmpl-stub",
    "object": "chat.completion",
    "created": 0,
    "model": "stub",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "ok"},
            "finish_reason": "stop",
        }
    ],
    "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
}

#: Stripped for the duration of every stub. ``httpx`` reads these with ``trust_env=True`` and
#: applies them to 127.0.0.1 as readily as to anything else, so a developer or CI runner
#: exporting ``HTTP_PROXY`` would send every request in these files to their proxy instead of to
#: the stub. That does not fail cleanly: the SDK's default read timeout is 600s, and this suite's
#: ``timeout_method = "thread"`` takes the whole session down rather than one test.
_PROXY_VARS = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY")


class ProviderStub:
    """Counts POSTs and answers with whatever status the test has armed."""

    def __init__(self, ok_body: dict[str, Any], port: int) -> None:
        self.ok_body = ok_body
        self.status = 200
        self.body: dict[str, Any] = {}
        self.count = 0
        self.base_url = f"http://127.0.0.1:{port}/v1"
        #: Extra response headers, for tests about what a provider ASKS for rather than what it
        #: returns. `Retry-After` is the reason this exists.
        self.headers: dict[str, str] = {}
        #: Clients registered with `track`, closed before the listener stops. See `provider_stub`.
        self.clients: list[Any] = []
        #: Recorded so teardown can prove these threads actually ended. Keep-alive means one
        #: thread serves every attempt of a test, and it outlives the request.
        self.handler_threads: set[threading.Thread] = set()

    def arm(self, status: int, message: str) -> None:
        """Fail every later request with ``status``, and reset the counter.

        ``message`` must avoid every marker in ``_is_transient``'s text fallback, so that a test
        measures the numeric status branch rather than an accident of provider wording.
        """
        self.status = status
        self.body = {"error": {"message": message}}
        self.count = 0

    def track(self, client: Any) -> Any:
        """Register an SDK client so teardown closes its connection. Returns it unchanged.

        Not optional bookkeeping. ``ThreadingHTTPServer`` sets ``daemon_threads``, which makes
        ``server_close``'s join a no-op, and HTTP/1.1 keep-alive leaves each handler thread
        blocked reading a socket nobody closes. Shutting the listener without closing the client
        end leaks a live thread per test.
        """
        self.clients.append(client)
        return client


def _handler_for(stub: ProviderStub) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
            stub.handler_threads.add(threading.current_thread())
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            stub.count += 1
            raw = json.dumps(stub.ok_body if stub.status == 200 else stub.body).encode()
            self.send_response(stub.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for name, value in stub.headers.items():
                self.send_header(name, value)
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args: object) -> None:
            """Silence the default stderr access log so the suite output stays pristine."""

    return _Handler


def _stop(
    server: ThreadingHTTPServer, thread: threading.Thread, stub: ProviderStub, *, join: float
) -> list[str]:
    """Close the clients, stop the listener, and report any teardown problem: a close that
    raised, or a thread that outlived the block.

    Returns the leaks rather than asserting them, so the caller decides whether they are worth
    raising. Each step is guarded: a client whose ``close`` raises must not be able to skip the
    server shutdown behind it, or one bad teardown leaks the listening socket into the rest of
    the session.

    That guarantee is structural, a ``finally`` around the whole close phase, rather than a bet on
    the breadth of an ``except``. Catching ``Exception`` leaves ``KeyboardInterrupt`` and
    ``SystemExit`` free to skip the shutdown and strand the listener plus its accept loop for the
    rest of the session, which is the exact outcome the paragraph above promises cannot happen.
    Neither is exotic here: a ``KeyboardInterrupt`` is delivered at an arbitrary bytecode boundary
    in the main thread, so Ctrl-C during a slow teardown lands inside the close loop as readily as
    anywhere else. One residual is deliberate and measured: a ``BaseException`` raised while
    closing still skips the REMAINING closes, so a handler thread can outlive that teardown. The
    listener cannot, and on a ``KeyboardInterrupt`` the process is ending regardless.

    A close that raises is reported rather than swallowed. Passing silently left the operator
    reading "a stub handler thread outlived the test" about a connection nobody could close, five
    seconds of joins away from the actual cause and pointing at the wrong subsystem.
    """
    problems: list[str] = []
    try:
        try:
            for index, client in enumerate(stub.clients):
                close = getattr(client, "close", None)
                if callable(close):
                    try:
                        close()
                    except Exception as exc:  # noqa: BLE001 - reported below, never fatal
                        problems.append(
                            f"closing tracked client {index} "
                            f"({type(client).__name__}) raised: {exc!r}"
                        )
        finally:
            try:
                server.shutdown()
            finally:
                server.server_close()
    except BaseException as escaping:
        # `problems` is read below the whole block, so anything escaping the shutdown phase, or a
        # BaseException out of a close, would otherwise discard every failure recorded up to that
        # point and restore exactly the silence this reporting was added to remove.
        for note in problems:
            escaping.add_note(note)
        raise

    leaks: list[str] = list(problems)
    thread.join(timeout=join)
    if thread.is_alive():
        leaks.append("the stub's accept loop outlived the test")
    for handler in stub.handler_threads:
        handler.join(timeout=join)
        if handler.is_alive():
            leaks.append(f"a stub handler thread outlived the test ({handler.name})")
    return leaks


@contextmanager
def provider_stub(ok_body: dict[str, Any]) -> Iterator[ProviderStub]:
    """Serve ``ok_body`` on an ephemeral loopback port until the block exits.

    Proxy variables are cleared here rather than in a fixture so that both this module's callers
    and any future one get the protection by construction; see ``_PROXY_VARS``. ``NO_PROXY`` is
    then set to the loopback host, which covers the case of a proxy configured somewhere this
    process cannot see. The restore is armed BEFORE the listener is built, so that a bind failure
    on a locked-down runner cannot leak a stripped environment into every test that follows.

    A teardown problem, a close that raised or a leaked thread, is reported only when the body
    itself succeeded. Raising it from a ``finally``
    unconditionally would replace the assertion the test actually failed on, demoting the real
    message to ``__context__`` and leaving "a stub handler thread outlived the test" as the
    headline for a failure that has nothing to do with threads. That is reachable in normal use:
    a client registered with `track` only after its constructor returns is untracked if the
    constructor is what raised, and the constructor makes a request of its own.
    """
    saved = {var: os.environ.get(var) for var in _PROXY_VARS}
    saved.update({var.lower(): os.environ.get(var.lower()) for var in _PROXY_VARS})
    try:
        for var in saved:
            os.environ.pop(var, None)
        os.environ["NO_PROXY"] = "127.0.0.1,localhost"

        server = ThreadingHTTPServer(("127.0.0.1", 0), BaseHTTPRequestHandler)
        stub = ProviderStub(ok_body, server.server_port)
        server.RequestHandlerClass = _handler_for(stub)
        # 0.05 rather than the 0.5 default: `shutdown` waits out one poll interval, which was
        # costing half a second of teardown per test for nothing.
        thread = threading.Thread(
            target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True
        )
        thread.start()
        failed = False
        try:
            yield stub
        except BaseException:
            failed = True
            raise
        finally:
            # A failing test waits 0.5s rather than 5s per thread: the leak is not the finding
            # there, and the joins are pure latency on a red run.
            leaks = _stop(server, thread, stub, join=0.5 if failed else 5.0)
            if leaks and not failed:
                raise AssertionError("; ".join(leaks))
    finally:
        for var, value in saved.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
