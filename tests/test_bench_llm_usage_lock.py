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
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from benchmarks.llm import OpenRouterLLM


class _LockAssertingDict(dict):
    """A dict that refuses to be written to unless `lock` is held."""

    def __init__(self, inner: dict, lock: threading.Lock) -> None:
        super().__init__(inner)
        self._lock = lock

    def __setitem__(self, key: str, value: object) -> None:
        assert self._lock.locked(), f"usage[{key!r}] mutated without holding _usage_lock"
        super().__setitem__(key, value)


def _stub_client(prompt: int = 11, completion: int = 7, finish: str = "stop") -> SimpleNamespace:
    response = SimpleNamespace(
        usage=SimpleNamespace(prompt_tokens=prompt, completion_tokens=completion),
        choices=[SimpleNamespace(finish_reason=finish, message=SimpleNamespace(content="ok"))],
    )
    return SimpleNamespace(
        chat=SimpleNamespace(completions=SimpleNamespace(create=lambda **_: response))
    )


def _armed(**kwargs: object) -> OpenRouterLLM:
    llm = OpenRouterLLM(model="test/model", api_key="k", **kwargs)  # type: ignore[arg-type]
    # Set the client so `_complete_once` never imports or contacts openai.
    llm._client = _stub_client()  # noqa: SLF001
    llm._usage = _LockAssertingDict(llm._usage, llm._usage_lock)  # noqa: SLF001
    return llm


def test_usage_counters_are_only_mutated_under_the_lock() -> None:
    llm = _armed()
    assert llm.complete("sys", "user") == "ok"
    assert llm.usage() == {"calls": 1, "prompt_tokens": 11, "completion_tokens": 7}


def test_usage_accumulates_across_calls() -> None:
    llm = _armed()
    for _ in range(3):
        llm.complete("sys", "user")
    assert llm.usage() == {"calls": 3, "prompt_tokens": 33, "completion_tokens": 21}


def test_usage_snapshot_is_a_copy_not_the_live_dict() -> None:
    # `run.py` subtracts `harness` from `total`; handing out the live dict would let a later call
    # mutate a number that had already been published.
    llm = _armed()
    llm.complete("sys", "user")
    snapshot = llm.usage()
    llm.complete("sys", "user")
    assert snapshot["calls"] == 1


def test_a_truncated_completion_is_not_counted_as_a_clean_one() -> None:
    # Usage is recorded before the truncation check, which is correct — the tokens WERE spent —
    # but the call must still raise rather than return a half-written answer to be scored.
    from benchmarks.llm import CompletionTruncated

    llm = OpenRouterLLM(model="test/model", api_key="k", max_attempts=1)
    llm._client = _stub_client(finish="length")  # noqa: SLF001
    with pytest.raises(CompletionTruncated):
        llm.complete("sys", "user")
    assert llm.usage()["calls"] == 1, "the spend happened and must still be reported"
