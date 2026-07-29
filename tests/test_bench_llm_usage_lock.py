"""The harness token counters must only be mutated under their lock.

One `OpenRouterLLM` is driven concurrently by `benchmarks.beam.run`'s worker pool (8 threads by
default) and `+=` on a dict value is a read-modify-write. The process-wide meter this figure gets
SUBTRACTED FROM (`benchmarks.usage`) is already lock-guarded, so lost updates here made
`harness < total` and published a spuriously positive `memory_layer` — the field whose entire job
is to show that RE-call's retrieval path spends no tokens.

Tested as an INVARIANT, not as a race. Spawning threads and asserting a total would be flaky by
construction, and a flaky test in a repo that publishes numbers is worse than no test: it teaches
people to re-run until green. Instead the usage dict is wrapped so that any mutation taken without
the lock held fails immediately and deterministically. Removing the `with self._usage_lock:` in
`benchmarks/llm.py` turns this red on the first call.

The `openai` SDK is FAKED into `sys.modules` rather than skipped around, following
`test_bench_llm_max_tokens.py`. `openai` lives in the `bench` extra and CI installs `.[dev]`, so
an `importorskip` here would make this guard silently absent in the one place it most needs to
run — `_complete_once` imports the SDK unconditionally, before it checks whether a client is
already set, so presetting `_client` is not enough on its own.
"""
from __future__ import annotations

import sys
import threading
import types
from typing import Any

import pytest

from benchmarks.llm import CompletionTruncated, OpenRouterLLM


class _LockAssertingDict(dict):
    """A dict that refuses to be written to unless `lock` is held."""

    def __init__(self, inner: dict, lock: threading.Lock) -> None:
        super().__init__(inner)
        self._lock = lock

    def __setitem__(self, key: str, value: object) -> None:
        assert self._lock.locked(), f"usage[{key!r}] mutated without holding _usage_lock"
        super().__setitem__(key, value)


def _install_fake_openai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    prompt_tokens: int = 11,
    completion_tokens: int = 7,
    finish_reason: str | None = "stop",
) -> None:
    class _FakeCompletions:
        def create(self, **_: Any) -> types.SimpleNamespace:
            choice = types.SimpleNamespace(
                message=types.SimpleNamespace(content="ok"), finish_reason=finish_reason
            )
            usage = types.SimpleNamespace(
                prompt_tokens=prompt_tokens, completion_tokens=completion_tokens
            )
            return types.SimpleNamespace(choices=[choice], usage=usage)

    class _FakeOpenAI:
        def __init__(self, **_: Any) -> None:
            self.chat = types.SimpleNamespace(completions=_FakeCompletions())

    module = types.ModuleType("openai")
    module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)


def _armed(monkeypatch: pytest.MonkeyPatch, **kwargs: Any) -> OpenRouterLLM:
    _install_fake_openai(monkeypatch, **kwargs)
    llm = OpenRouterLLM(model="test/model", api_key="k")
    llm._usage = _LockAssertingDict(llm._usage, llm._usage_lock)  # noqa: SLF001
    return llm


def test_usage_counters_are_only_mutated_under_the_lock(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _armed(monkeypatch)
    assert llm.complete("sys", "user") == "ok"
    assert llm.usage() == {"calls": 1, "prompt_tokens": 11, "completion_tokens": 7}


def test_usage_accumulates_across_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    llm = _armed(monkeypatch)
    for _ in range(3):
        llm.complete("sys", "user")
    assert llm.usage() == {"calls": 3, "prompt_tokens": 33, "completion_tokens": 21}


def test_usage_snapshot_is_a_copy_not_the_live_dict(monkeypatch: pytest.MonkeyPatch) -> None:
    # `run.py` subtracts `harness` from `total`; handing out the live dict would let a later call
    # mutate a number that had already been published.
    llm = _armed(monkeypatch)
    llm.complete("sys", "user")
    snapshot = llm.usage()
    llm.complete("sys", "user")
    assert snapshot["calls"] == 1


def test_a_truncated_completion_is_counted_and_still_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Usage is recorded before the truncation check, which is correct — the tokens WERE spent —
    # but the call must still raise rather than return a half-written answer to be scored.
    _install_fake_openai(monkeypatch, finish_reason="length")
    llm = OpenRouterLLM(model="test/model", api_key="k", max_attempts=1)
    with pytest.raises(CompletionTruncated):
        llm.complete("sys", "user")
    assert llm.usage()["calls"] == 1, "the spend happened and must still be reported"
