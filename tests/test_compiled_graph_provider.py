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
- ``test_invalid_json_retried_then_rejected``: no retry on a parse failure fails on the count.
- ``test_truncated_output_rejected_without_retry``: a cut-off reply parsed anyway fails.
- ``test_usage_read_from_reply``: usage ignored fails on the token counts.
- ``test_missing_required_variable_named``: a default model fails on ``pytest.raises``.
- ``test_request_shape``: the ``response_format`` line removed fails on the request body.
- ``test_api_key_never_in_errors``: the response body echoed into the message fails the check.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import pytest

from recall.compiled_graph.provider import (
    ATTEMPTS,
    RETRY_AFTER_CAP_SECONDS,
    CompiledGraphConfigError,
    CompileReply,
    CreditStop,
    PayloadTooLarge,
    ProviderConfig,
    SourceRejected,
    compile_source,
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


def _run(transport: FakeTransport, sleeps: list[float] | None = None) -> CompileReply:
    recorded = sleeps if sleeps is not None else []
    return compile_source(ProviderConfig.from_env(ENV), transport, PAYLOAD, sleep=recorded.append)


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


def test_invalid_json_retried_then_rejected() -> None:
    bad = json.dumps({"choices": [{"message": {"content": "not json"}, "finish_reason": "stop"}]})
    transport = FakeTransport((200, {}, bad.encode()))
    with pytest.raises(SourceRejected):
        _run(transport)
    assert len(transport.calls) == 3


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
