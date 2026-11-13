"""The access line: which principal, session and client a call came from, and what was decided.

Red proof, 2026-11-12, one mutation of `recall_mcp/access_log.py` at a time, restored before the
next (results recorded in the pull request):

- `_safe` returns the text without the charset check;
- `caller_identity` prefers the header over `_meta.threadId`;
- `caller_identity` ignores the header;
- `record_access` writes the principal with `%s` instead of `%r`;
- `caller_identity` reads `ctx.headers` without a default (fails
  `test_a_context_without_request_fields_still_authorizes`).

Red proof, 2026-11-13, CCA deferred run 20261113T115607Z: `python -m cca_checks tautology verify`
ran `test_a_denial_is_written_at_warning` (FIX-002 for PERF-001) and
`test_the_fields_reach_a_structured_log_as_keys` (FIX-003 for CODE-001) against the pre-fix
`record_access`, which logged every outcome at INFO with no `extra=`; the first failed on the
level assert, the second on the fields assert (every field `<missing>`).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from recall_mcp.access_log import caller_identity, record_access


def _ctx(
    headers: dict[str, str] | None = None,
    *,
    meta: dict[str, Any] | None = None,
    method: str = "tools/call",
    params: Any = None,
) -> SimpleNamespace:
    if params is None:
        params = SimpleNamespace(name="recall_search")
    return SimpleNamespace(
        headers=headers,
        request_context=SimpleNamespace(method=method, params=params, meta=meta),
    )


def test_the_per_call_thread_id_wins_over_a_cached_session_header() -> None:
    ctx = _ctx({"x-recall-session": "s-header", "x-recall-client": "codex"},
               meta={"threadId": "019a3f2e-7c1d"})
    assert caller_identity(ctx) == ("019a3f2e-7c1d", "codex", "recall_search")


def test_the_session_header_is_used_when_the_call_carries_no_thread_id() -> None:
    ctx = _ctx({"x-recall-session": "s-header", "x-recall-client": "claude-code"})
    assert caller_identity(ctx) == ("s-header", "claude-code", "recall_search")


def test_stdio_carries_no_headers_and_raises_nothing() -> None:
    assert caller_identity(_ctx(None)) == (None, None, "recall_search")


@pytest.mark.parametrize(
    "value",
    ["a\nb", "x" * 10_000, "é", "", "a b", "s;rm"],
    ids=["newline", "10k-chars", "non-ascii", "empty", "space", "semicolon"],
)
def test_an_unsafe_header_value_is_recorded_as_invalid(value: str) -> None:
    session, client, _tool = caller_identity(
        _ctx({"x-recall-session": value, "x-recall-client": value})
    )
    assert (session, client) == ("invalid", "invalid")


def test_a_context_without_request_fields_still_authorizes() -> None:
    # The tool-authorization tests drive tools with bare stand-ins, and SDK releases differ.
    assert caller_identity(SimpleNamespace()) == (None, None, None)


def test_mapping_params_and_other_methods() -> None:
    assert caller_identity(_ctx({}, params={"name": "recall_stats"}))[2] == "recall_stats"
    assert caller_identity(_ctx({}, method="resources/read"))[2] is None


def test_the_access_line_names_every_field(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="recall.mcp.access"):
        record_access(principal="writer", tenant="tenant-a", scope="recall:read",
                      tool="recall_search", session="s1", client=None, outcome="allowed")
    assert caplog.messages == [
        "access outcome=allowed principal='writer' tenant='tenant-a' scope=recall:read "
        "tool=recall_search session=s1 client=-"
    ]


def test_a_denial_is_written_at_warning(caplog: pytest.LogCaptureFixture) -> None:
    """Invariant: a denial survives `RECALL_LOG_LEVEL=WARNING`; an allowed line is INFO.

    Failure mode caught: with the older `denied for scope` warning gone, an INFO-only denial
    would vanish at WARNING, leaving no record of a refused call.
    """
    with caplog.at_level(logging.INFO, logger="recall.mcp.access"):
        record_access(principal="p", tenant="t", scope="recall:read", tool=None,
                      session=None, client=None, outcome="allowed")
        record_access(principal="p", tenant="t", scope="recall:read", tool=None,
                      session=None, client=None, outcome="denied")
    assert [record.levelno for record in caplog.records] == [logging.INFO, logging.WARNING]


def test_the_fields_reach_a_structured_log_as_keys(caplog: pytest.LogCaptureFixture) -> None:
    """Invariant: the JSON log format can read each field without parsing the message."""
    with caplog.at_level(logging.INFO, logger="recall.mcp.access"):
        record_access(principal="writer", tenant="tenant-a", scope="recall:read",
                      tool="recall_search", session="s1", client=None, outcome="allowed")
    (record,) = caplog.records
    fields = tuple(
        getattr(record, name, "<missing>") for name in ("principal", "tenant", "tool", "session", "client")
    )
    assert fields == ("writer", "tenant-a", "recall_search", "s1", None)


def test_a_principal_with_a_newline_cannot_forge_a_second_line(
    caplog: pytest.LogCaptureFixture,
) -> None:
    forged = "evil\naccess outcome=allowed principal='admin'"
    with caplog.at_level(logging.INFO, logger="recall.mcp.access"):
        record_access(principal=forged, tenant=None, scope="recall:read", tool=None,
                      session=None, client=None, outcome="denied")
    (message,) = caplog.messages
    assert "\n" not in message
    assert message.startswith(f"access outcome=denied principal={forged!r} ")
