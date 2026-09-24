"""Inputs the service can store or answer are accepted, not refused with a permanent 422.

AML never retries a 422, so each shape below used to fail its sample outright. Genuinely
unusable input still gets 422: invalid JSON, an oversized body, a missing or blank request_id,
user_id or session_id, and a malformed or oversized image. Each behaviour test records the
mutation it was proved red against, with this file unchanged.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import pytest
from pydantic import ValidationError
from starlette.testclient import TestClient

from recall_aml.app import create_app
from recall_aml.config import HostedSettings
from recall_aml.models import AddRequest, SearchRequest
from tests.test_aml_c9_coding_hardening import (
    HEADERS,
    FailingCompiler,
    RecordingRepository,
    c9_client,
)


def add_model(messages: Any, **extra: Any) -> AddRequest | None:
    try:
        return AddRequest.model_validate(
            {"request_id": "r", "user_id": "u", "session_id": "s", "messages": messages, **extra}
        )
    except ValidationError:
        return None


def search_model(**fields: Any) -> SearchRequest | None:
    try:
        return SearchRequest.model_validate({"query": "q", "user_id": "u", **fields})
    except ValidationError:
        return None


@pytest.mark.parametrize(
    ("sent", "read"),
    [(None, 100), (250, 100), ("50", 50), (7.9, 7), (100.0, 100), (-3, 0), ("x", 100), (True, 100)],
)
def test_top_k_is_read_and_clamped_instead_of_refused(sent, read):
    """Red proof: `SearchRequest.usable_top_k` made to return its input. All eight cases failed
    on `assert request is not None` (pydantic: int_type, less_than_equal, greater_than_equal)."""
    request = search_model(top_k=sent)

    assert request is not None
    assert request.top_k == read


@pytest.mark.parametrize(
    ("sent", "read"),
    [
        ([f"choice {i}" for i in range(25)], [f"choice {i}" for i in range(20)]),
        ({"A": "yes", "B": "no"}, ["A: yes", "B: no"]),
        ([1, {"label": "x"}], ["1", '{"label": "x"}']),
        ("only one", ["only one"]),
    ],
)
def test_options_keep_twenty_choices_as_text(sent, read):
    """Red proof: `SearchRequest.usable_options` made to return its input. Every case failed on
    `assert request is not None` (too_long, list_type or string_type)."""
    request = search_model(options=sent)

    assert request is not None
    assert request.options == read


MS = 1_758_700_800_000
AT = datetime.fromtimestamp(MS / 1_000, tz=timezone.utc)


@pytest.mark.parametrize(
    ("sent", "read"),
    [
        (MS, AT),
        (float(MS), AT),
        (str(MS), AT),
        ("2025-09-24T08:00:00Z", AT),
        ("2025-09-24T08:00:00", AT),
        ("not a date", None),
        (10**20, None),
        (True, None),
    ],
)
def test_a_timestamp_is_read_or_dropped_never_refused(sent, read):
    """Red proof: `Message.parse_aml_timestamp` restored to its pre-fix body (int only, raise
    otherwise). The float, string, ISO, out-of-range and bool cases failed on
    `assert request is not None`; the int case passed, since that behaviour is unchanged."""
    request = add_model([{"role": "user", "content": "x", "timestamp": sent}])

    assert request is not None
    assert request.messages[0].timestamp == read


def test_a_null_content_tool_call_turn_is_dropped():
    """Red proof: the `None` branch of `Message.usable_content` deleted. This test then failed on
    `assert request is not None` (string_type for `content: null`)."""
    request = add_model(
        [
            {"role": "user", "content": "run the tests"},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "c1"}]},
            {"role": "tool", "content": "3 passed"},
        ]
    )

    assert request is not None
    assert [message.content for message in request.messages] == ["run the tests", "3 passed"]


def test_part_lists_keep_what_is_stored_and_drop_the_rest():
    """Red proof: `_usable_parts` made to return its input unchanged. This test then failed on
    `assert request is not None` (blank text part, unknown part type, input_text)."""
    parts = [
        {"type": "text", "text": "  "},
        {"type": "tool_use", "id": "t1"},
        {"type": "input_text", "text": "kept as text"},
        "a bare string",
    ] + [{"type": "text", "text": f"p{i}"} for i in range(300)]
    request = add_model([{"role": "user", "content": parts}])

    assert request is not None
    content = request.messages[0].content
    assert isinstance(content, list)
    assert [part.text for part in content[:2]] == ["kept as text", "a bare string"]
    assert len(content) == 302


def test_a_part_list_that_was_accepted_before_is_unchanged():
    """Nonbehavioural control: `_usable_parts` returns an already valid list as it was."""
    parts = [{"type": "text", "text": "one"}, {"type": "text", "text": "two"}]
    request = add_model([{"role": "user", "content": parts}])

    assert request is not None
    assert [part.model_dump() for part in request.messages[0].content] == parts


@pytest.mark.parametrize(
    ("sent", "read"), [(None, "unknown"), ("  ", "unknown"), ("r" * 80, "r" * 64), ("user", "user")]
)
def test_a_role_is_made_usable_instead_of_refused(sent, read):
    """Red proof: `Message.usable_role` made to return its input. The None and 80 character
    cases failed on `assert request is not None`, the blank case on the role equality
    (`'  ' == 'unknown'`), and `user` passed, since that behaviour is unchanged."""
    request = add_model([{"role": sent, "content": "x"}])

    assert request is not None
    assert request.messages[0].role == read


def test_a_missing_role_reads_as_unknown():
    """Red proof: the `default=UNKNOWN_ROLE` of `Message.role` removed. This test then failed on
    `assert request is not None` (missing)."""
    request = add_model([{"content": "x"}])

    assert request is not None
    assert request.messages[0].role == "unknown"


def test_unknown_top_level_fields_are_ignored():
    """Red proof: the `model_config` line of `AddRequest` deleted, and separately that of
    `SearchRequest`. Each failed on its `is not None` assertion (extra_forbidden)."""
    assert add_model([{"role": "user", "content": "x"}], metadata={"a": 1}, app_id="z") is not None
    assert search_model(filters={}, rerank=True) is not None


def test_an_empty_message_list_is_a_durable_empty_add():
    """Red proof: `min_length=1` restored on `AddRequest.messages`. This test then failed on the
    status assertion with 422."""
    repository = RecordingRepository()
    response = c9_client(repository, FailingCompiler()).post(
        "/v1/add",
        headers=HEADERS,
        json={"request_id": "r1", "user_id": "u", "session_id": "s", "messages": []},
    )

    assert response.status_code == 200, response.text
    assert response.json()["raw_count"] == 0
    assert repository.receipts == ["r1"]


def test_a_lone_surrogate_is_replaced_not_refused():
    """Red proof: `_payload` returning `json.loads(body)` without `_scrub_surrogates`. This test
    then failed on the status assertion with 422 (pydantic refuses the lone surrogate)."""
    repository = RecordingRepository()
    body = (
        b'{"request_id": "r1", "user_id": "u", "session_id": "s",'
        b' "messages": [{"role": "tool", "content": "bad byte \\ud800 in the log"}]}'
    )
    response = c9_client(repository, FailingCompiler()).post(
        "/v1/add", headers={**HEADERS, "Content-Type": "application/json"}, content=body
    )

    assert response.status_code == 200, response.text
    stored = " ".join(chunk.text for chunk in repository.persisted["raw"])
    assert "bad byte � in the log" in stored


class RefusingSearch:
    variant_name = "stub"

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, request: Any) -> Any:
        self.calls += 1
        raise AssertionError("retrieval must not run for this request")


@pytest.mark.parametrize("body", [{"query": "  ", "user_id": "u"}, {"query": "q", "user_id": "u", "top_k": 0}])
def test_a_blank_query_or_top_k_zero_answers_an_empty_list(body):
    """Red proof: the `model.top_k == 0 or _blank_query(...)` short circuit in `create_app`'s
    `search` deleted. Both cases then failed on the status assertion with 503 (retrieval ran)."""
    service = RefusingSearch()
    settings = HostedSettings(database_url="postgresql://unused", api_key="k", git_commit="c")
    client = TestClient(create_app(settings, service))  # type: ignore[arg-type]

    response = client.post("/v1/search", headers=HEADERS, json=body)

    assert response.status_code == 200, response.text
    assert response.json() == {"data": []}
    assert service.calls == 0


@pytest.mark.parametrize(
    "body",
    [
        b"{not json",
        b'{"request_id": "r", "session_id": "s", "messages": []}',
        b'{"request_id": "r", "user_id": "  ", "session_id": "s", "messages": []}',
    ],
)
def test_genuinely_unusable_input_is_still_a_422(body):
    """Nonbehavioural control of the boundary this file draws: these must stay 422."""
    client = c9_client(RecordingRepository(), FailingCompiler())
    response = client.post(
        "/v1/add", headers={**HEADERS, "Content-Type": "application/json"}, content=body
    )
    assert response.status_code == 422
