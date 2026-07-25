"""Token metering — accumulator + the create() wrapper, offline (no real OpenAI call)."""
from __future__ import annotations

from typing import Any

import pytest

from benchmarks import usage


@pytest.fixture(autouse=True)
def _clean_counter() -> Any:
    usage.reset()
    yield
    usage.reset()


class _Usage:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens


class _Resp:
    def __init__(self, usage_obj: _Usage | None) -> None:
        self.usage = usage_obj


def test_record_and_snapshot_accumulate() -> None:
    usage.record(10, 5)
    usage.record(3, 2)
    assert usage.snapshot() == {"calls": 2, "prompt_tokens": 13, "completion_tokens": 7}


def test_reset_clears() -> None:
    usage.record(10, 5)
    usage.reset()
    assert usage.snapshot() == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def test_wrap_counts_usage_and_returns_response_unchanged() -> None:
    resp = _Resp(_Usage(100, 40))

    def original(_this: Any, *_a: Any, **_k: Any) -> _Resp:
        return resp

    wrapped = usage._wrap(original)
    out = wrapped(object(), model="x")
    assert out is resp  # transparent — the caller gets the real response
    assert usage.snapshot() == {"calls": 1, "prompt_tokens": 100, "completion_tokens": 40}


def test_wrap_ignores_a_response_without_usage() -> None:
    # a streaming response, or one the endpoint returns without a usage block, must not crash or count
    def original(_this: Any, *_a: Any, **_k: Any) -> _Resp:
        return _Resp(None)

    usage._wrap(original)(object())
    assert usage.snapshot() == {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0}


def test_wrap_is_marked_so_install_is_idempotent() -> None:
    wrapped = usage._wrap(lambda _this: None)
    assert getattr(wrapped, "__metered__", False) is True
