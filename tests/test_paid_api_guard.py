"""The suite-wide guard that keeps every test off the paid model APIs (tests/conftest.py).

The owner's rule (2026-10-31): tests never call a real API. The guard refuses the name lookup of
any host in `PAID_API_HOSTS`, and `_no_paid_api_calls` fails a test that attempted one even when
its code swallowed the refusal. These tests run only with the guard on, which is every run except
a deliberate live one (`RECALL_RUN_PAID_TESTS=1`).

Red proof: with `_install_paid_api_guard()` not called at import (mutation of the `if` that
installs it to `if False:`), the refusal tests failed in their own assertion. A real lookup either
resolved (`DID NOT RAISE`) or raised an OS `gaierror` that does not match "refused in tests", and
nothing was recorded. `is_paid_api_host` and `paid_api_violation` were each mutated as their
docstrings say.
"""

from __future__ import annotations

import os
import socket

import pytest

from tests.conftest import PAID_API_LOOKUPS, is_paid_api_host, paid_api_violation

pytestmark = pytest.mark.skipif(
    os.environ.get("RECALL_RUN_PAID_TESTS") == "1",
    reason="the guard is off for a deliberate live run",
)


@pytest.mark.parametrize(
    "host",
    ["api.openai.com", "OpenRouter.ai", "openrouter.ai.", "eu.api.voyageai.com", b"api.anthropic.com"],
)
def test_a_paid_host_lookup_is_refused_and_recorded(host: str | bytes) -> None:
    start = len(PAID_API_LOOKUPS)
    with pytest.raises(socket.gaierror, match="refused in tests"):
        socket.getaddrinfo(host, 443)
    recorded = PAID_API_LOOKUPS[start:]
    del PAID_API_LOOKUPS[start:]  # this test attempted it on purpose
    assert recorded == [str(host)]


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1"])
def test_a_local_host_still_resolves(host: str) -> None:
    """Control: the guard passes every other lookup to the real resolver."""
    start = len(PAID_API_LOOKUPS)
    assert socket.getaddrinfo(host, 443)
    assert PAID_API_LOOKUPS[start:] == []


@pytest.mark.parametrize("host", ["notopenrouter.ai", "openrouter.ai.example", "api.openai.co"])
def test_a_lookalike_name_is_not_a_paid_host(host: str) -> None:
    """Control: a paid host is matched as the name or a subdomain of it, never as a substring.

    Checked on the predicate, so no lookup leaves the machine. Red: mutation of
    `is_paid_api_host` to substring matching (`paid in name`) classified `notopenrouter.ai` and
    `openrouter.ai.example` as paid.
    """
    assert not is_paid_api_host(host)


def test_a_swallowed_refusal_still_fails_the_test() -> None:
    """Fail-open code turns the refused lookup into a quiet pass; the record catches it.

    Red: mutation of `paid_api_violation` to `return None` returned no failure for a recorded
    attempt, failing the first assertion.
    """
    start = len(PAID_API_LOOKUPS)
    try:
        socket.getaddrinfo("openrouter.ai", 443)
    except OSError:
        pass  # what a fail-open fetch does
    failure = paid_api_violation(start)
    assert failure is not None and "openrouter.ai" in failure
    assert PAID_API_LOOKUPS[start:] == [], "the check clears what it reported"
    assert paid_api_violation(start) is None
