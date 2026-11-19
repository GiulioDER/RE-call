"""`RECALL_REASONING_ANSWER_MAX_CALLS_PER_MIN` bounds answer calls, per process (issue 1405).

Invariant: a resolved answer provider makes at most the configured number of calls per minute,
counted across every provider the process resolves and across both backends, and a refused call
never reaches the transport. `off` removes the ceiling; anything else that is not a finite
positive number is refused naming the variable.

Failure mode it catches: `.env.example` documents this variable as the only bound on paid answer
calls, but its reader (`_configure_answer_rate` and the gate in `recall/answer_provider.py`) was
deleted by 410cfd81b (#585), so every call went through however many were made.

Red proof receipt ``answer-spend-ceiling-01``, 2026-11-19, against master `e7b7f92b9`
(`OllamaAnswerProvider.__call__` with no gate, `resolve_answer_provider` not reading the variable):
the refusal tests and the malformed-value test failed with `DID NOT RAISE`. The `off` test passed
there, as it must, and its red proof is by mutation: with `off` parsed as the default ceiling it
failed on the 31st call. Production symbols targeted: `OllamaAnswerProvider.__call__` and
`resolve_answer_provider`.

The transport is faked at `urllib.request.urlopen` as the module calls it, so both clients run
their real code and no request leaves the process.
"""

from __future__ import annotations

import io
import json
from collections.abc import Iterator

import pytest

import recall.answer_provider as ap

_PAID = {
    "RECALL_REASONING_ANSWER_ENABLED": "1",
    "RECALL_REASONING_ANSWER_PROVIDER": "openrouter",
    "RECALL_REASONING_ANSWER_MODEL": "test/model",
    "RECALL_REASONING_ANSWER_REVISION": "test-revision",
    "RECALL_REASONING_ANSWER_API_KEY": "test-key-not-real",
}
_LOCAL = {
    "RECALL_REASONING_ANSWER_ENABLED": "1",
    "RECALL_REASONING_ANSWER_PROVIDER": "ollama",
    "RECALL_REASONING_ANSWER_MODEL": "test-model",
}
_CONTENT = json.dumps({"answer": None, "citations": [], "insufficient_evidence": True})


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str]]:
    """Record every request that would have left the process, and answer it locally."""
    sent: list[str] = []

    def _urlopen(req, timeout=None):  # noqa: ANN001, ARG001 - the urllib signature
        sent.append(req.full_url)
        if req.full_url.endswith("/api/chat"):
            raw = {"message": {"content": _CONTENT}, "prompt_eval_count": 1, "eval_count": 1}
        else:
            raw = {"choices": [{"message": {"content": _CONTENT}}], "usage": {"total_tokens": 2}}
        return io.BytesIO(json.dumps(raw).encode("utf-8"))

    monkeypatch.setattr(ap.request, "urlopen", _urlopen)
    yield sent
    # The ceiling is process-wide by design, so leave it unconfigured for the next test.
    ap.resolve_answer_provider({**_LOCAL, "RECALL_REASONING_ANSWER_MAX_CALLS_PER_MIN": "off"})


def test_a_paid_call_past_the_ceiling_is_refused_before_it_is_sent(transport: list[str]) -> None:
    provider = ap.resolve_answer_provider({**_PAID, "RECALL_REASONING_ANSWER_MAX_CALLS_PER_MIN": "2"})
    assert provider is not None

    provider("system", "user")
    provider("system", "user")
    with pytest.raises(RuntimeError, match="RECALL_REASONING_ANSWER_MAX_CALLS_PER_MIN") as refused:
        provider("system", "user")

    assert type(refused.value).__name__ == "AnswerRateLimited"
    assert len(transport) == 2, "a refused call must not reach the endpoint"


def test_the_ceiling_is_shared_by_every_provider_the_process_resolves(
    transport: list[str],
) -> None:
    """Resolving again must not refill it, and the two backends draw on one budget."""
    ceiling = {"RECALL_REASONING_ANSWER_MAX_CALLS_PER_MIN": "3"}
    first = ap.resolve_answer_provider({**_PAID, **ceiling})
    assert first is not None
    first("system", "user")
    first("system", "user")

    second = ap.resolve_answer_provider({**_LOCAL, **ceiling})
    assert second is not None
    second("system", "user")
    with pytest.raises(RuntimeError, match="calls/min"):
        second("system", "user")
    assert len(transport) == 3


def test_the_word_off_removes_the_ceiling(transport: list[str]) -> None:
    provider = ap.resolve_answer_provider({**_PAID, "RECALL_REASONING_ANSWER_MAX_CALLS_PER_MIN": "off"})
    assert provider is not None
    for _ in range(40):
        provider("system", "user")
    assert len(transport) == 40


def test_a_ceiling_below_one_call_a_minute_still_allows_a_call(transport: list[str]) -> None:
    """`0.5` means one call every two minutes, not none. Red proof: with the bucket's capacity
    equal to the configured rate (0.5 tokens, so never a whole one) the first call raised
    `AnswerRateLimited`."""
    provider = ap.resolve_answer_provider({**_PAID, "RECALL_REASONING_ANSWER_MAX_CALLS_PER_MIN": "0.5"})
    assert provider is not None
    provider("system", "user")
    with pytest.raises(RuntimeError, match="calls/min"):
        provider("system", "user")
    assert len(transport) == 1


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "lots"])
def test_a_ceiling_that_is_not_a_finite_positive_number_is_refused(
    transport: list[str], value: str
) -> None:
    with pytest.raises(ValueError, match="RECALL_REASONING_ANSWER_MAX_CALLS_PER_MIN"):
        ap.resolve_answer_provider({**_PAID, "RECALL_REASONING_ANSWER_MAX_CALLS_PER_MIN": value})
    assert transport == []
