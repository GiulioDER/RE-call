"""The Add-time compiler resends only a request that could succeed on a resend.

On the official Full of 2026-09-25 the production journal held 76,150 HTTP 402 (credit
exhausted) answers for 25,372 fallback Adds: every refused compile was sent three times, with
fixed 0.25 s and 0.5 s sleeps between, and no resend of a 402 can succeed. The errors here are the
OpenAI SDK's own classes, built from real ``httpx`` responses, so the attribute the compiler
reads (``status_code``) and the header it honours (``retry-after``) are the ones production sees.

Red proof, 2026-09-26, each by mutating the named line of ``recall_aml/compiler.py`` with this
file unchanged, watching the named assertion fail, then restoring it:

* ``test_a_permanent_client_error_is_sent_once_by_the_anchored_compile``: deleting the
  ``if status != 400 and not _resend_can_succeed(exc): raise`` guard in the v3 attempt loop of
  ``OpenAICompiler._compile_anchored`` made three calls for every status and failed
  ``assert len(calls) == 1`` (``assert 3 == 1``).
* ``test_a_permanent_client_error_is_sent_once_by_json``: deleting the
  ``if not _resend_can_succeed(exc): raise`` guard in ``OpenAICompiler._json`` failed
  ``assert len(calls) == 1`` (``assert 3 == 1``) for the v1 and v2 compiles.
* ``test_a_resendable_failure_is_still_retried``: returning ``status is None`` from
  ``_resend_can_succeed`` (so 408, 409, 429 and 5xx were no longer resent) failed
  ``assert len(calls) == 3`` (``assert 1 == 3``) for 408, 409, 429, 500 and 503.
* ``test_a_400_within_the_budget_is_not_resent``: deleting the
  ``if status == 400 and not refitted: raise`` clause failed ``assert len(calls) == 1``
  (``assert 3 == 1``).
* ``test_retry_after_is_honoured_on_a_429_and_bounded``: making ``_retry_delay`` return the fixed
  ``0.25 * (2**attempt)`` failed ``assert sleeps == [3.0, 3.0]`` (``[0.25, 0.5]``).

The refit of a 400 over the budget is proved by
``tests/test_aml_compiler_input_cap.py::test_an_over_window_payload_is_fitted_after_its_first_failure``,
unchanged and still green, since that path must behave exactly as before.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from recall_aml.compiler import COMPILER_MAX_RETRY_AFTER_SECONDS, OpenAICompiler
from recall_aml.models import Message

openai = pytest.importorskip("openai")
httpx = pytest.importorskip("httpx")


def _status_error(status: int, headers: dict[str, str] | None = None) -> Exception:
    """The exception the OpenAI SDK raises for ``status``, exactly as its client builds it."""
    response = httpx.Response(
        status,
        headers=headers or {},
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions"),
    )
    client = openai.OpenAI(api_key="test", base_url="https://openrouter.ai/api/v1", max_retries=0)
    error: Exception = client._make_status_error_from_response(response)
    return error


def _failing_compiler(
    error: Exception, sleeps: list[float] | None = None
) -> tuple[OpenAICompiler, list[dict[str, Any]]]:
    calls: list[dict[str, Any]] = []

    def create(**request: Any) -> Any:
        calls.append(request)
        raise error

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    sink = sleeps if sleeps is not None else []
    return OpenAICompiler(client, sleep=sink.append), calls


def _messages() -> list[Message]:
    return [Message(role="user", content="moved the queue to postgres and pytest passed")]


PERMANENT = (400, 401, 402, 403, 404, 422)
RESENDABLE = (408, 409, 429, 500, 503)


@pytest.mark.parametrize("status", PERMANENT)
def test_a_permanent_client_error_is_sent_once_by_the_anchored_compile(status: int) -> None:
    compiler, calls = _failing_compiler(_status_error(status))

    with pytest.raises(openai.APIStatusError):
        compiler.compile_anchored_v3(_messages(), "s", [])

    assert len(calls) == 1


@pytest.mark.parametrize("status", PERMANENT)
@pytest.mark.parametrize("method", ["compile", "compile_anchored"])
def test_a_permanent_client_error_is_sent_once_by_json(status: int, method: str) -> None:
    compiler, calls = _failing_compiler(_status_error(status))

    with pytest.raises(openai.APIStatusError):
        getattr(compiler, method)(_messages(), "s", [])

    assert len(calls) == 1


@pytest.mark.parametrize("status", RESENDABLE)
@pytest.mark.parametrize("method", ["compile", "compile_anchored", "compile_anchored_v3"])
def test_a_resendable_failure_is_still_retried(status: int, method: str) -> None:
    compiler, calls = _failing_compiler(_status_error(status))

    with pytest.raises(openai.APIStatusError):
        getattr(compiler, method)(_messages(), "s", [])

    assert len(calls) == 3


@pytest.mark.parametrize("method", ["compile", "compile_anchored", "compile_anchored_v3"])
def test_a_timeout_is_still_retried(method: str) -> None:
    """Nonbehavioural control: an error without a status is resent exactly as before."""
    timeout = openai.APITimeoutError(
        request=httpx.Request("POST", "https://openrouter.ai/api/v1/chat/completions")
    )
    sleeps: list[float] = []
    compiler, calls = _failing_compiler(timeout, sleeps)

    with pytest.raises(openai.APITimeoutError):
        getattr(compiler, method)(_messages(), "s", [])

    assert len(calls) == 3
    assert sleeps == [0.25, 0.5]


def test_a_400_within_the_budget_is_not_resent() -> None:
    compiler, calls = _failing_compiler(_status_error(400))

    with pytest.raises(openai.BadRequestError):
        compiler.compile_anchored_v3(_messages(), "s", [])

    assert len(calls) == 1


@pytest.mark.parametrize(
    ("header", "expected"),
    [("3", 3.0), ("45", COMPILER_MAX_RETRY_AFTER_SECONDS), ("0.1", None)],
)
def test_retry_after_is_honoured_on_a_429_and_bounded(header: str, expected: float | None) -> None:
    sleeps: list[float] = []
    compiler, calls = _failing_compiler(_status_error(429, {"retry-after": header}), sleeps)

    with pytest.raises(openai.RateLimitError):
        compiler.compile_anchored_v3(_messages(), "s", [])

    assert len(calls) == 3
    if expected is None:
        # Shorter than the fixed backoff: the backoff stands.
        assert sleeps == [0.25, 0.5]
    else:
        assert sleeps == [expected, expected]
