"""One retry policy per request, and exactly one.

`OpenRouterLLM.complete` wraps every call in `retry_with_backoff`, so that policy owns retries.
The `openai` SDK ships its own retry layer underneath, and unless it is switched off the two
multiply: a policy that says "four attempts" spends twelve requests, and the outer full-jitter
backoff (which exists so a fleet does not remarch onto the provider in lockstep) ends up
wrapping an inner loop that smears its own doubling schedule by only ``1 - 0.25 * random()`` — a
25% jitter rather than a draw across the interval, so it separates a fleet far less than the
layer wrapping it.

That arithmetic is worst exactly where it hurts most. A benchmark run makes thousands of calls
back to back, which is the traffic shape a provider rate-limits; answering a 429 with triple the
requests aims the multiplication at a provider that has just said it is overloaded, and triples
the bill for it.

These tests count real HTTP requests against a local stub rather than asserting on constructor
keywords, because the keyword is the mechanism and the request count is the behaviour. A fake
`chat.completions.create` would be blind to the layer under test: the SDK's retries live below
that method, inside its transport.
"""
from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from benchmarks.llm import OpenRouterLLM

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness

openai = pytest.importorskip("openai")

#: A minimal but complete `/v1/chat/completions` success body, parsed by the SDK into a real
#: `ChatCompletion` — so the success path exercises the same object `_complete_once` reads.
_OK_BODY = {
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


def _handler_for(state: dict) -> type[BaseHTTPRequestHandler]:
    class _Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def do_POST(self) -> None:  # noqa: N802 - name fixed by BaseHTTPRequestHandler
            # Recorded so teardown can prove this thread actually ended. Keep-alive means one
            # thread serves every attempt of a test, and it outlives the request.
            state["handler_threads"].add(threading.current_thread())
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            state["count"] += 1
            status = state["status"]
            body = _OK_BODY if status == 200 else state["body"]
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def log_message(self, *args: object) -> None:
            """Silence the default stderr access log so the suite output stays pristine."""

    return _Handler


@pytest.fixture
def stub_provider(monkeypatch):
    """A real HTTP endpoint that counts requests and replies with whatever the test arms.

    Proxy variables are stripped first. `httpx` reads them with `trust_env=True` and applies
    them to 127.0.0.1 as readily as to anything else, so a developer or CI runner exporting
    `HTTP_PROXY` would send every request in this file to their proxy instead of the stub.
    That does not fail cleanly: the SDK's default read timeout is 600s, and this suite's
    `timeout_method = "thread"` takes the whole session down rather than one test.

    Teardown closes the client end before stopping the server. `ThreadingHTTPServer` sets
    `daemon_threads`, which makes `server_close`'s join a no-op, and HTTP/1.1 keep-alive leaves
    each handler thread blocked reading a socket nobody closes. Shutting the listener without
    closing the connection leaks a live thread per test.
    """
    for var in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
        monkeypatch.delenv(var.lower(), raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")

    state: dict = {
        "status": 200,
        "body": {},
        "count": 0,
        "handler_threads": set(),
        "built": [],
    }
    server = ThreadingHTTPServer(("127.0.0.1", 0), _handler_for(state))
    # 0.05 rather than the 0.5 default: `shutdown` waits out one poll interval, which was
    # costing half a second of teardown per test for nothing.
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05})
    thread.daemon = True
    thread.start()
    state["url"] = f"http://127.0.0.1:{server.server_port}/v1"
    try:
        yield state
    finally:
        for llm in state["built"]:
            close = getattr(llm._client, "close", None)
            if callable(close):
                close()
        server.shutdown()
        server.server_close()
        thread.join(timeout=5)
        assert not thread.is_alive(), "the stub's accept loop outlived the test"
        for handler in state["handler_threads"]:
            handler.join(timeout=5)
            assert not handler.is_alive(), "a stub handler thread outlived the test"


def _llm(stub_provider, **kw) -> OpenRouterLLM:
    """An LLM pointed at the stub, registered so teardown can close its connection.

    `sleep` is swallowed rather than monkeypatched: `OpenRouterLLM` already injects it for
    exactly this, so the retry path runs at test speed without reaching into another module.
    """
    llm = OpenRouterLLM(
        model="stub-model",
        api_key="test",
        base_url=stub_provider["url"],
        sleep=lambda _seconds: None,
        **kw,
    )
    stub_provider["built"].append(llm)
    return llm


def _armed(stub_provider, status: int, message: str, **kw) -> OpenRouterLLM:
    """As `_llm`, with the stub armed to fail.

    `message` must avoid every marker in `_is_transient`'s text fallback, so each test measures
    the numeric status branch rather than an accident of provider wording.
    """
    stub_provider["status"] = status
    stub_provider["body"] = {"error": {"message": message}}
    return _llm(stub_provider, **kw)


def test_a_rate_limit_costs_the_retry_policys_attempts_and_no_more(stub_provider):
    """One 429 must cost 3 requests, which is what `max_attempts=3` means.

    With the SDK's own 2 retries left on, the layers multiply to 9.
    """
    llm = _armed(stub_provider, 429, "please slow down", max_attempts=3)

    with pytest.raises(openai.RateLimitError):
        llm.complete("sys", "user")

    assert stub_provider["count"] == 3


def test_an_upstream_server_error_costs_the_retry_policys_attempts_and_no_more(stub_provider):
    """Same arithmetic for 5xx. 507 is used deliberately: it is not one of the literal markers
    in the text fallback, so only the numeric branch can classify it transient."""
    llm = _armed(stub_provider, 507, "insufficient storage upstream", max_attempts=3)

    with pytest.raises(openai.InternalServerError):
        llm.complete("sys", "user")

    assert stub_provider["count"] == 3


def test_the_shipped_default_spends_four_requests_on_a_rate_limit_not_twelve(stub_provider):
    """The default `max_attempts=4` is what a real BEAM run uses, so pin the default itself.

    Asserting only against an explicit `max_attempts=3` would leave the shipped configuration
    untested, and it is the one that bills.
    """
    llm = _armed(stub_provider, 429, "please slow down")
    assert llm.max_attempts == 4, "this test exists to pin the default; it moved"

    with pytest.raises(openai.RateLimitError):
        llm.complete("sys", "user")

    assert stub_provider["count"] == 4


def test_a_bad_request_is_not_retried(stub_provider):
    """A permanent 4xx must still fail fast, at one request.

    The SDK does not retry 400 either, so this arm would pass with the defect present. It is
    here to prove the fix removed a multiplication rather than the retry itself.
    """
    llm = _armed(stub_provider, 400, "input is not valid for this model", max_attempts=3)

    with pytest.raises(openai.BadRequestError):
        llm.complete("sys", "user")

    assert stub_provider["count"] == 1


def test_a_success_costs_exactly_one_request(stub_provider):
    """The happy path must not be perturbed: one call, one request, the content returned."""
    llm = _llm(stub_provider)

    assert llm.complete("sys", "user") == "ok"

    assert stub_provider["count"] == 1
    assert llm.usage()["calls"] == 1
