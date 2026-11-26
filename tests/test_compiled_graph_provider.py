"""The compile client: one request per source, and a failure table that never spends past a stop.

Every test drives a fake transport; nothing reaches a network. Red proof, recorded 2026-11-26
(production symbols in ``recall.compiled_graph.provider``):

- ``test_credit_statuses_stop_after_one_call``: 402 dropped from ``_CREDIT_STATUSES`` fails (it is
  then rejected as an ordinary client error, not a credit stop).
- ``test_credit_body_stops``: the credit-body check removed fails with ``SourceRejected``.
- ``test_rate_limit_honours_retry_after_clamped``: ``RETRY_AFTER_CAP_SECONDS = 60`` fails on the
  sleeps.
- ``test_server_error_retried_then_rejected``: ``ATTEMPTS = 1`` fails on the call count.
- ``test_other_client_error_rejected_without_retry``: 404 added to the retried statuses fails on
  the call count.
- ``test_size_refusal_is_payload_too_large``: the size-message check removed fails with
  ``SourceRejected``.
- ``test_unparseable_reply_not_resent``: a parse failure retried fails on the call count (3).
- ``test_truncated_output_rejected_without_retry``: a cut-off reply parsed anyway fails.
- ``test_usage_read_from_reply``: usage ignored fails on the token counts.
- ``test_missing_required_variable_named``: a default model fails on ``pytest.raises``.
- ``test_request_shape``: the ``response_format`` line removed fails on the request body.
- ``test_api_key_never_in_errors``: the response body echoed into the message fails the check.
- ``test_unparseable_reply_not_resent``: a rejected 200 retried (``continue``) fails on the count.
- ``test_fenced_json_reply_accepted``: fence stripping disabled fails with ``SourceRejected``.
- ``test_error_object_in_ok_reply_stops_for_credit``: the 402 code check removed fails.
- ``test_error_object_in_ok_reply_rejects_source``: every error object read as credit fails.
- ``test_key_limit_stops``: ``key limit`` removed from ``_CREDIT_BODY`` fails.
- ``test_rate_limit_body_is_retried_not_stopped``: the ``status != 429`` exclusion removed fails
  with ``CreditStop`` (the body mentions credits).
- ``test_each_attempt_reserves_the_whole_request``: reserving without ``TEMPLATE_SLACK_TOKENS``
  fails on the reserved sizes.
- ``test_budget_refusal_mid_retry_stops``: a refused reservation ending the loop (``break``) fails
  with ``SourceRejected`` instead of ``BudgetExhausted``.
- ``test_network_errors_retried``: catching ``OSError`` only lets ``IncompleteRead`` escape.
- ``test_rejection_carries_status``: the status not passed fails on ``status == 404``.
- ``test_repr_hides_url_credentials``: ``repr`` printing ``base_url`` as given fails.
"""

from __future__ import annotations

import json
import re
import socket
import threading
from collections.abc import Callable, Mapping

import pytest

from recall.compiled_graph.budget import Budget
from recall.compiled_graph.provider import (
    ATTEMPTS,
    TEMPLATE_SLACK_TOKENS,
    BudgetExhausted,
    RETRY_AFTER_CAP_SECONDS,
    CompiledGraphConfigError,
    CompileReply,
    CreditStop,
    PayloadTooLarge,
    ProviderConfig,
    SourceRejected,
    compile_source,
    urllib_transport,
)

KEY = "sk-test-not-a-real-key-123456"
ENV = {"RECALL_COMPILED_GRAPH_API_KEY": KEY, "RECALL_COMPILED_GRAPH_MODEL": "openai/gpt-6-luna-pro"}
PAYLOAD = {"source": "notes/a.md", "anchors": [{"id": "a000_x", "excerpt": "text"}]}


def _ok(content: object, finish: str = "stop", usage: Mapping[str, int] | None = None) -> bytes:
    body: dict[str, object] = {
        "choices": [{"message": {"content": json.dumps(content)}, "finish_reason": finish}]
    }
    if usage is not None:
        body["usage"] = dict(usage)
    return json.dumps(body).encode()


class FakeTransport:
    def __init__(self, *responses: tuple[int, Mapping[str, str], bytes]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, object], Mapping[str, str]]] = []

    def __call__(
        self, url: str, body: bytes, headers: Mapping[str, str], timeout: float
    ) -> tuple[int, Mapping[str, str], bytes]:
        self.calls.append((url, json.loads(body), headers))
        return self.responses.pop(0) if len(self.responses) > 1 else self.responses[0]


class RecordingBudget(Budget):
    def __init__(self, max_usd: float = 100.0) -> None:
        super().__init__(max_usd=max_usd, cost_in_per_1k=0.002, cost_out_per_1k=0.008)
        self.reserved: list[int] = []

    def reserve(self, request_bytes: int) -> bool:
        self.reserved.append(request_bytes)
        return super().reserve(request_bytes)


def _run(
    transport: FakeTransport, sleeps: list[float] | None = None, budget: Budget | None = None
) -> CompileReply:
    recorded = sleeps if sleeps is not None else []
    return compile_source(
        ProviderConfig.from_env(ENV), transport, PAYLOAD, budget or RecordingBudget(),
        sleep=recorded.append,
    )


@pytest.mark.parametrize("status", [401, 402])
def test_credit_statuses_stop_after_one_call(status: int) -> None:
    transport = FakeTransport((status, {}, b'{"error":{"message":"no"}}'))
    with pytest.raises(CreditStop):
        _run(transport)
    assert len(transport.calls) == 1


@pytest.mark.parametrize("message", ["insufficient_quota", "Insufficient credits on account"])
def test_credit_body_stops(message: str) -> None:
    transport = FakeTransport((403, {}, json.dumps({"error": {"message": message}}).encode()))
    with pytest.raises(CreditStop):
        _run(transport)
    assert len(transport.calls) == 1


def test_rate_limit_honours_retry_after_clamped() -> None:
    transport = FakeTransport(
        (429, {"Retry-After": "30"}, b"{}"),
        (429, {"Retry-After": "2"}, b"{}"),
        (200, {}, _ok({"records": []})),
    )
    sleeps: list[float] = []
    _run(transport, sleeps)
    assert RETRY_AFTER_CAP_SECONDS == 10.0
    assert sleeps == [10.0, 2.0]
    assert len(transport.calls) == 3


def test_server_error_retried_then_rejected() -> None:
    transport = FakeTransport((503, {}, b"{}"))
    with pytest.raises(SourceRejected):
        _run(transport)
    assert ATTEMPTS == 3
    assert len(transport.calls) == 3


def test_other_client_error_rejected_without_retry() -> None:
    transport = FakeTransport((404, {}, b'{"error":{"message":"no such model"}}'))
    with pytest.raises(SourceRejected):
        _run(transport)
    assert len(transport.calls) == 1


@pytest.mark.parametrize("status", [400, 413])
def test_size_refusal_is_payload_too_large(status: int) -> None:
    body = json.dumps({"error": {"message": "This model's maximum context length is 128000 tokens"}})
    transport = FakeTransport((status, {}, body.encode()))
    with pytest.raises(PayloadTooLarge):
        _run(transport)
    assert len(transport.calls) == 1


def test_unparseable_reply_not_resent() -> None:
    bad = json.dumps({"choices": [{"message": {"content": "not json"}, "finish_reason": "stop"}]})
    transport = FakeTransport((200, {}, bad.encode()))
    with pytest.raises(SourceRejected):
        _run(transport)
    assert len(transport.calls) == 1


def test_fenced_json_reply_accepted() -> None:
    fenced = '```json\n{"records": []}\n```'
    body = json.dumps({"choices": [{"message": {"content": fenced}, "finish_reason": "stop"}]})
    assert _run(FakeTransport((200, {}, body.encode()))).content == {"records": []}


def test_error_object_in_ok_reply_stops_for_credit() -> None:
    body = json.dumps({"error": {"code": 402, "message": "Provider returned error"}})
    transport = FakeTransport((200, {}, body.encode()))
    with pytest.raises(CreditStop):
        _run(transport)
    assert len(transport.calls) == 1


def test_error_object_in_ok_reply_rejects_source() -> None:
    body = json.dumps({"error": {"code": 400, "message": "Provider returned error"}})
    transport = FakeTransport((200, {}, body.encode()))
    with pytest.raises(SourceRejected):
        _run(transport)
    assert len(transport.calls) == 1


def test_key_limit_stops() -> None:
    body = json.dumps({"error": {"message": "Key limit exceeded (total limit)"}})
    with pytest.raises(CreditStop):
        _run(FakeTransport((403, {}, body.encode())))


def test_rate_limit_body_is_retried_not_stopped() -> None:
    limited = json.dumps(
        {"error": {"message": "Rate limit exceeded: free-models-per-day. Add credits to unlock more"}}
    )
    transport = FakeTransport((429, {}, limited.encode()), (200, {}, _ok({"records": []})))
    _run(transport)
    assert len(transport.calls) == 2


def test_each_attempt_reserves_the_whole_request() -> None:
    budget = RecordingBudget()
    transport = FakeTransport(
        (503, {}, b"{}"),
        (200, {}, _ok({"records": []}, usage={"prompt_tokens": 1_000, "completion_tokens": 10})),
    )
    _run(transport, budget=budget)
    sent = len(json.dumps(transport.calls[0][1]).encode())
    assert len(budget.reserved) == 2
    assert all(reserved >= sent + TEMPLATE_SLACK_TOKENS for reserved in budget.reserved)
    first = (budget.reserved[0] / 1000) * 0.002 + 2.4 * 0.008
    assert abs(budget.spent_usd - (first + 0.002 + 0.01 * 0.008)) < 1e-9


def test_budget_refusal_mid_retry_stops() -> None:
    budget = RecordingBudget(max_usd=0.03)
    transport = FakeTransport((503, {}, b"{}"))
    with pytest.raises(BudgetExhausted):
        _run(transport, budget=budget)
    assert len(transport.calls) == 1


def test_network_errors_retried() -> None:
    import http.client

    failures: list[BaseException] = [TimeoutError("slow"), http.client.IncompleteRead(b"")]

    def flaky(url: str, body: bytes, headers: Mapping[str, str], timeout: float):  # type: ignore[no-untyped-def]  # mirrors the Transport alias
        if failures:
            raise failures.pop(0)
        return 200, {}, _ok({"records": []})

    assert _run(flaky).content == {"records": []}  # type: ignore[arg-type]  # a function, not FakeTransport


def test_rejection_carries_status() -> None:
    with pytest.raises(SourceRejected) as caught:
        _run(FakeTransport((404, {}, b"{}")))
    assert caught.value.status == 404


def test_repr_hides_url_credentials() -> None:
    config = ProviderConfig(api_key=KEY, model="m", base_url="https://user:secret@example.test/v1")
    assert "secret" not in repr(config)
    assert KEY not in repr(config)


def test_truncated_output_rejected_without_retry() -> None:
    transport = FakeTransport((200, {}, _ok({"records": []}, finish="length")))
    with pytest.raises(SourceRejected):
        _run(transport)
    assert len(transport.calls) == 1


def test_usage_read_from_reply() -> None:
    transport = FakeTransport(
        (200, {}, _ok({"records": []}, usage={"prompt_tokens": 1_200, "completion_tokens": 80}))
    )
    reply = _run(transport)
    assert reply.content == {"records": []}
    assert (reply.input_tokens, reply.output_tokens) == (1_200, 80)
    no_usage = _run(FakeTransport((200, {}, _ok({"records": []}))))
    assert (no_usage.input_tokens, no_usage.output_tokens) == (None, None)


def test_request_shape() -> None:
    transport = FakeTransport((200, {}, _ok({"records": []})))
    _run(transport)
    url, body, headers = transport.calls[0]
    assert url == "https://openrouter.ai/api/v1/chat/completions"
    assert body["model"] == "openai/gpt-6-luna-pro"
    assert body["temperature"] == 0
    assert body["response_format"] == {"type": "json_object"}
    messages = body["messages"]
    assert isinstance(messages, list) and messages[1]["content"].startswith("<stored_data>")
    assert headers["Authorization"] == f"Bearer {KEY}"


@pytest.mark.parametrize("missing", ["RECALL_COMPILED_GRAPH_API_KEY", "RECALL_COMPILED_GRAPH_MODEL"])
def test_missing_required_variable_named(missing: str) -> None:
    env = {name: value for name, value in ENV.items() if name != missing}
    with pytest.raises(CompiledGraphConfigError, match=missing):
        ProviderConfig.from_env(env)


def test_api_key_never_in_errors() -> None:
    echoed = json.dumps({"error": {"message": f"bad key {KEY}"}}).encode()
    for status in (401, 404, 503):
        with pytest.raises((CreditStop, SourceRejected)) as caught:
            _run(FakeTransport((status, {}, echoed)))
        assert KEY not in str(caught.value)
        assert KEY not in repr(caught.value)


class _LoopbackServer:
    """A one-thread HTTP/1.1 server on 127.0.0.1 that answers each request with ``reply``.

    It reads the whole request before answering, so a client error cannot be a reset before the
    status line; ``requests`` holds each request's head, lower-cased.
    """

    def __init__(self, reply: Callable[[int], bytes]) -> None:
        self.requests: list[bytes] = []
        self._reply = reply
        self._sock = socket.socket()
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(8)
        self._sock.settimeout(5)
        self.port = self._sock.getsockname()[1]
        threading.Thread(target=self._serve, daemon=True).start()

    def _serve(self) -> None:
        while True:
            try:
                conn, _ = self._sock.accept()
            except OSError:
                return
            with conn:
                data = b""
                while b"\r\n\r\n" not in data:
                    data += conn.recv(65536)
                head, _, rest = data.partition(b"\r\n\r\n")
                length = re.search(rb"(?i)content-length: *(\d+)", head)
                while length and len(rest) < int(length.group(1)):
                    rest += conn.recv(65536)
                self.requests.append(head.lower())
                conn.sendall(self._reply(self.port))
                conn.shutdown(socket.SHUT_WR)

    def close(self) -> None:
        self._sock.close()


def test_lost_body_after_200_is_not_sent_again() -> None:
    """A 200 whose body breaks mid-read was generated and billed, so it is never resent.

    Red proof, 2026-11-26: against ``urllib_transport`` and ``compile_source`` before the fix
    (the read failure surfaced as ``IncompleteRead`` and was retried as a network error) this
    fails on ``len(server.requests) == 1`` with 3.
    """
    server = _LoopbackServer(
        lambda _port: b"HTTP/1.1 200 OK\r\nContent-Length: 500\r\n\r\n{\"choices\":"
    )
    config = ProviderConfig(api_key=KEY, model="m", base_url=f"http://127.0.0.1:{server.port}/v1")
    try:
        with pytest.raises(SourceRejected) as caught:
            compile_source(config, urllib_transport, PAYLOAD, RecordingBudget(), sleep=lambda _s: None)
    finally:
        server.close()
    assert len(server.requests) == 1
    assert caught.value.status == 200


def test_redirect_never_carries_the_key() -> None:
    """A provider's redirect is refused, so the bearer key never reaches the redirect's target.

    Red proof, 2026-11-26: with the default ``urlopen`` opener (before the fix) urllib follows
    the 302 as a GET carrying ``Authorization``, and this fails on ``target.requests == []``.
    """
    target = _LoopbackServer(lambda _port: b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\n{}")
    origin = _LoopbackServer(
        lambda _port: (
            f"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:{target.port}/steal\r\n"
            "Content-Length: 0\r\n\r\n"
        ).encode()
    )
    config = ProviderConfig(api_key=KEY, model="m", base_url=f"http://127.0.0.1:{origin.port}/v1")
    try:
        with pytest.raises(SourceRejected) as caught:
            compile_source(config, urllib_transport, PAYLOAD, RecordingBudget(), sleep=lambda _s: None)
    finally:
        origin.close()
        target.close()
    assert target.requests == []
    assert caught.value.status == 302
