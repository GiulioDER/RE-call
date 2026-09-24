"""C9 must not fail an AML Coding sample permanently on inputs it can store.

Each test below guards one path found by the 2026-09-24 pre-run audit where a legitimate Add got a
response that no AML retry could repair: a permanent 422, or a 503 that repeats identically on all
32 attempts. Every behaviour test records the mutation it was proved red against; the proofs were
run with this file unchanged and the named production line mutated back to its pre-fix form.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest
from starlette.testclient import TestClient

from recall_aml.app import create_app
from recall_aml.config import HostedSettings
from recall_aml.context_overflow import (
    ContextOverflowGuard,
    byte_bounded_parts,
    utf8_prefix,
)
from recall_aml.service import HostedService
from recall_aml.variants import variant


C9 = "C9_routed_specialists_grounded_graph_atomic"
HEADERS = {"x-api-key": "k"}


class RecordingRepository:
    """Records what an Add would persist, per namespace."""

    distributed_locks = False

    def __init__(self) -> None:
        self.persisted: dict[str, list[Any]] = {}
        self.receipts: list[str] = []

    def _write(self, kind: str, chunks: Any) -> int:
        chunks = list(chunks)
        self.persisted.setdefault(kind, []).extend(chunks)
        return len(chunks)

    def get_receipt(self, *args: Any) -> None:
        return None

    def record_receipt(self, tenant: str, request_id: str, fingerprint: str, result: str) -> None:
        self.receipts.append(request_id)

    def prior_records(self, *args: Any, **kwargs: Any) -> list[Any]:
        return []

    def persist(self, tenant: str, chunks: Any) -> int:
        return self._write("raw", chunks)

    def persist_graph(self, tenant: str, chunks: Any) -> int:
        return self._write("graph", chunks)

    def persist_specialist(self, tenant: str, profile: str, chunks: Any) -> int:
        return self._write("specialist", chunks)

    def persist_atomic_views(self, tenant: str, profile: Any, chunks: Any) -> int:
        return len(list(chunks))


class FailingCompiler:
    def __init__(self) -> None:
        self.calls = 0

    def compile_anchored_v3(self, *args: Any) -> list[Any]:
        self.calls += 1
        raise TimeoutError("provider timeout")


def c9_client(repository: RecordingRepository, compiler: Any) -> TestClient:
    behavior = variant(C9)
    service = HostedService(
        repository,
        compiler,
        object(),
        behavior=behavior,
        multimodal_embedder=object(),
        specialist_retrievers={behavior.context_embedding_profile: object()},
    )
    settings = HostedSettings(database_url="postgresql://unused", api_key="k", git_commit="c")
    return TestClient(create_app(settings, service))


def add(client: TestClient, messages: list[dict[str, Any]]) -> Any:
    return client.post(
        "/v1/add",
        headers=HEADERS,
        json={"request_id": "r1", "user_id": "u", "session_id": "s", "messages": messages},
    )


def test_c9_add_survives_a_compiler_failure_on_a_whitespace_led_message():
    """A compiler failure must not turn into a permanent 422 for records C9 discards anyway.

    Red proof: `recall_aml/service.py`, the compiler `except` branch in `_add_once`, mutated back
    to always `records = deterministic_extract(...)`. This test then failed on the status
    assertion with 503: `require_substance` raises because the first 300 characters are all
    whitespace. On master, where any `ValueError` mapped to 422, the same input was a 422.
    """
    repository = RecordingRepository()
    response = add(
        c9_client(repository, FailingCompiler()),
        [{"role": "tool", "content": "\n" * 300 + "Traceback: ValueError boom"}],
    )

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["compiler_fallback"] is True
    assert body["compiled_count"] == 0
    assert body["raw_count"] >= 1


def test_a_blank_tool_call_turn_is_dropped_instead_of_refusing_the_add():
    """A tool-call-only assistant turn has empty content; the rest of the Add must be stored.

    Red proof: `recall_aml/models.py`, `Message.validate_content`, with the blank check restored
    (`if not value.strip(): raise ValueError("content must not be blank")`). This test then failed
    on the status assertion with 422.
    """
    repository = RecordingRepository()
    response = add(
        c9_client(repository, FailingCompiler()),
        [
            {"role": "user", "content": "fix the failing parser test"},
            {"role": "assistant", "content": ""},
            {"role": "tool", "content": "tests/test_parser.py::test_utf8 PASSED"},
        ],
    )

    assert response.status_code == 200, response.text
    stored = " ".join(chunk.text for chunk in repository.persisted["raw"])
    assert "fix the failing parser test" in stored
    assert "test_utf8 PASSED" in stored


def test_tool_call_fields_on_a_message_are_ignored_not_refused():
    """Fields beside role and content (`name`, `tool_call_id`, ...) must not refuse the Add.

    Red proof: `recall_aml/models.py`, the `model_config` line of `Message` deleted, so it
    inherits `extra="forbid"` from `StrictModel` again. This test then failed on the status
    assertion with 422 (`extra_forbidden`).
    """
    repository = RecordingRepository()
    response = add(
        c9_client(repository, FailingCompiler()),
        [
            {
                "role": "tool",
                "content": "exit code 0",
                "name": "bash",
                "tool_call_id": "call_1",
            }
        ],
    )

    assert response.status_code == 200, response.text
    assert "exit code 0" in " ".join(chunk.text for chunk in repository.persisted["raw"])


def test_an_add_whose_messages_are_all_blank_is_durable_and_empty():
    """Nothing to store is a 200 with a receipt, and no provider is called for it.

    Red proof: `recall_aml/service.py`, the `if not request.messages:` short circuit in
    `_add_once` deleted. This test then failed on `assert compiler.calls == 0` (1 == 0): the
    Add sent a compiler request for a session with no messages at all.
    """
    repository = RecordingRepository()
    compiler = FailingCompiler()
    response = add(
        c9_client(repository, compiler),
        [{"role": "assistant", "content": ""}, {"role": "assistant", "content": "  \n"}],
    )

    assert response.status_code == 200, response.text
    assert response.json()["raw_count"] == 0
    assert repository.receipts == ["r1"]
    assert compiler.calls == 0
    assert repository.persisted == {}


def test_a_value_error_inside_the_service_is_a_logged_503_not_a_silent_422(caplog):
    """422 is permanent to AML, so it is reserved for requests that cannot be parsed.

    Red proof: `recall_aml/app.py`, `protected`, with `except InvalidRequest` widened back to
    `except (ValidationError, ValueError)`. This test then failed on the status assertion with
    422, and no `hosted_request_failed` line was logged.
    """

    class OrderingFault(RecordingRepository):
        def persist(self, tenant: str, chunks: Any) -> int:
            raise ValueError("stable Code4 ordering requires an integer segment")

    client = c9_client(OrderingFault(), FailingCompiler())
    with caplog.at_level(logging.ERROR, logger="recall_aml"):
        response = add(client, [{"role": "user", "content": "fix the parser"}])

    assert response.status_code == 503
    assert any(record.getMessage() == "hosted_request_failed" for record in caplog.records)


def test_an_unparseable_request_is_still_a_422():
    """The narrowed mapping must keep refusing what can never be accepted.

    Red proof: `recall_aml/app.py`, `_parse`, with its `except` re-raising the original error
    instead of `InvalidRequest`. This test then failed on the status assertion with 503.
    """
    client = c9_client(RecordingRepository(), FailingCompiler())
    response = client.post(
        "/v1/add",
        headers=HEADERS,
        json={"request_id": "r1", "session_id": "s", "messages": []},
    )
    assert response.status_code == 422


class Rejection(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(f"HTTP {status}")
        self.http_status = status


class TokenBoundContext:
    """A stand-in Context endpoint that refuses any request above ``limit`` UTF-8 bytes."""

    dim = 2
    name = "voyage-context:voyage-context-4"
    profile = None

    def __init__(self, limit: int, status: int = 400) -> None:
        self.limit = limit
        self.status = status
        self.requests: list[list[list[str]]] = []

    def embed_document_groups(self, groups: list[list[str]]) -> list[list[list[float]]]:
        self.requests.append(groups)
        size = sum(len(text.encode("utf-8")) for group in groups for text in group)
        if size > self.limit:
            raise Rejection(self.status)
        return [[[float(len(text)), 1.0] for text in group] for group in groups]

    def embed_query(self, text: str) -> list[float]:
        self.requests.append([[text]])
        if len(text.encode("utf-8")) > self.limit:
            raise Rejection(self.status)
        return [float(len(text)), 1.0]


def test_an_accepted_context_request_is_sent_exactly_as_before():
    """The guard is invisible to a request Voyage accepts: one call, the same groups.

    Red proof: `ContextOverflowGuard.embed_document_groups` mutated to skip the first call and
    always refit. This test then failed on the request-list equality (two parts sent, not one).
    """
    inner = TokenBoundContext(limit=10_000)
    texts = ["alpha " * 50, "beta " * 50]

    vectors = ContextOverflowGuard(inner, limit_bytes=300).embed_passages(texts)

    assert inner.requests == [[texts]]
    assert len(vectors) == 2


def test_a_refused_context_request_is_refitted_and_keeps_every_chunk_in_order():
    """A Voyage 400 on an oversize document must still yield one vector per chunk, in order.

    Red proof: `ContextOverflowGuard.embed_document_groups` mutated to re-raise every error.
    This test then failed on `assert vectors is not None`.
    """
    inner = TokenBoundContext(limit=1_000)
    texts = ["x" * 600, "y" * 600, "中" * 900, "z" * 10]
    guard = ContextOverflowGuard(inner, limit_bytes=1_000)

    try:
        vectors = guard.embed_passages(texts)
    except Rejection:
        vectors = None

    assert vectors is not None
    assert len(vectors) == len(texts)
    assert [vector[0] for vector in vectors][:2] == [600.0, 600.0]
    # The 2,700-byte CJK chunk is cut to its first 333 whole characters for the embedding only.
    assert vectors[2][0] == 333.0
    assert vectors[3][0] == 10.0
    for request in inner.requests[1:]:
        assert sum(len(t.encode("utf-8")) for group in request for t in group) <= 1_000


def test_a_transient_context_error_is_not_refitted():
    """Only a 400 is re-planned; a 429 or 5xx stays retryable and untouched.

    Red proof: `_is_request_rejection` mutated to return True for every error. This test then
    failed on `pytest.raises` with DID NOT RAISE: the 429 was refitted into a smaller request
    and swallowed instead of reaching the caller as retryable.
    """
    inner = TokenBoundContext(limit=10, status=429)
    guard = ContextOverflowGuard(inner, limit_bytes=5)

    with pytest.raises(Rejection):
        guard.embed_passages(["a much longer text than ten bytes"])
    assert len(inner.requests) == 1


def test_a_refused_context_query_is_cut_to_the_byte_budget():
    """The Context route embeds the Search question with no truncation of its own.

    Red proof: `ContextOverflowGuard.embed_query` mutated to re-raise every error. This test then
    failed on `assert vector is not None`.
    """
    inner = TokenBoundContext(limit=100)
    guard = ContextOverflowGuard(inner, limit_bytes=100)

    try:
        vector = guard.embed_query("q" * 500)
    except Rejection:
        vector = None

    assert vector is not None
    assert vector[0] == 100.0


def test_utf8_budgeting_never_splits_a_character():
    """Nonbehavioural helper check: the cut lands on a character boundary, under the budget."""
    assert utf8_prefix("中文", 4) == "中"
    assert utf8_prefix("short", 100) == "short"
    parts = byte_bounded_parts(["aaaa", "bbbb", "cc"], 8)
    assert parts == [["aaaa", "bbbb"], ["cc"]]


def test_the_context_specialist_is_served_through_the_guard(monkeypatch):
    """Production wiring: the Context 4 specialist embedder is wrapped, the primary one is not.

    Red proof: `recall_aml/__main__.py`, `_resolve_hosted_embedders`, with the
    `ContextOverflowGuard(...)` wrapper removed. This test then failed on the `isinstance`
    assertion.
    """
    import recall_aml.__main__ as entry

    monkeypatch.setattr(
        entry, "resolve_registered_embedder", lambda profile, env: TokenBoundContext(10_000)
    )
    settings = HostedSettings(
        database_url="postgresql://unused",
        api_key="k",
        git_commit="c",
        voyage_api_key="v",
    )
    behavior = variant(C9)

    primary, specialists = entry._resolve_hosted_embedders(settings, behavior)

    assert not isinstance(primary, ContextOverflowGuard)
    assert isinstance(specialists[behavior.context_embedding_profile], ContextOverflowGuard)
