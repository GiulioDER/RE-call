"""One `OpenRouterLLM` builds exactly one `openai` client, however many threads drive it.

`benchmarks.beam.run` hands ONE instance to a `ThreadPoolExecutor(max_workers=8)`, and
`_complete_once` built its client behind an unguarded check-then-set. Eight threads released
together from a barrier onto one instance built eight `OpenAI` clients; seven were overwritten by
the next assignment and dropped, each still owning an `httpx.Client` whose connection pool nobody
closes. Requests stayed correct, because every client is built from the same arguments including
`max_retries=0`, so the damage is leaked sockets and repeated setup rather than a wrong answer.
That is why it survived every behavioural test in this directory.

The client is deliberately NOT built eagerly in `__init__`, which would delete the race outright.
`openai` lives in the `bench` extra, `dev` excludes it on purpose, and CI installs `.[dev]`, so the
SDK is absent from the environment that runs these tests. `OpenRouterLLM` is documented as
constructible without it and `test_bench_llm.py::test_openrouter_llm_builds_with_defaults`
constructs one with no `openai` in `sys.modules` at all. Building at construction would move the
ImportError from the first request onto every construction: green on a developer machine that
happens to have the extra installed, red in CI. Measured, not assumed; see the commit message.

Two tests, because the invariant and the race fail differently:

* `test_the_unset_client_is_only_ever_touched_under_the_lock` is deterministic, on ONE thread, for
  every regression a single thread can observe. It is the `_LockAssertingDict` trick from
  `test_bench_llm_usage_lock.py` aimed at the `_client` slot rather than at the usage counters.
* `test_threads_released_together_build_exactly_one_client` is the property as the worker pool
  actually meets it. It can never be FALSELY red, because the lock makes "exactly one" hold under
  every schedule; as a bug detector it is probabilistic, which is why the invariant above is the
  one carrying the weight.

What must hold is narrower than "never touch `_client` unlocked". Once the client is set it is
never reset, so reading it unlocked is safe, and `_complete_once` does exactly that when it issues
the request. The property is about the UNSET slot: while `_client` is still None, neither reading
it nor assigning it may happen outside the guard, because that read is half of the check-then-set.

The instrumentation lives in `tests.lazy_client_lock_helpers`, shared with the identical guard on
`benchmarks.voyage_rerank.VoyageReranker`. Read that module before changing anything here: it
records the five shapes this assertion went through, four of which were GREEN against the mutant
that replaced them, and why watching the slot and watching the build are both required.

The residue is honest rather than closed: a single-threaded test can prove the guard is present,
blocking, owned by its caller, always the same lock object, unbroken between the two halves, and
covering the construction itself. It cannot prove that mutual exclusion actually excludes. That is
what the barrier test measures, and it is why neither test is redundant.
"""
from __future__ import annotations

import sys
import threading
import time
import types
from collections.abc import Callable
from typing import Any

import pytest

from benchmarks.llm import OpenRouterLLM
from tests.lazy_client_lock_helpers import asserting_lazy_client, built_under_the_lock

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness

#: Threads released at once onto one instance. `benchmarks.beam.run` defaults to 8 workers, and
#: that pool is the deployment this guard exists for.
WORKERS = 8


def _install_counting_openai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_construct: Callable[[], None] | None = None,
    failures: int = 0,
) -> list[Any]:
    """Fake `openai` that records every client it builds; returns the list, mutated in place.

    `on_construct` runs at the END of the constructor, after the client is recorded, so a caller
    can either inspect the state the build is happening in or hold the window open without being
    able to suppress the count by raising.
    """
    built: list[Any] = []
    built_lock = threading.Lock()
    remaining_failures = [None] * failures

    class _FakeCompletions:
        def create(self, **_: Any) -> types.SimpleNamespace:
            try:
                # Popped rather than tested-then-popped. A check-then-act on a shared list is the
                # exact shape this file prosecutes, and the threaded test uses this same fake.
                remaining_failures.pop()
            except IndexError:
                pass
            else:
                raise RuntimeError("connection reset by peer")
            choice = types.SimpleNamespace(
                message=types.SimpleNamespace(content="ok"), finish_reason="stop"
            )
            return types.SimpleNamespace(choices=[choice], usage=None, model="stub/rev")

    class _FakeOpenAI:
        def __init__(self, **_: Any) -> None:
            # Recorded BEFORE the hook, so the count and the hook are two independent
            # signals. Appending afterwards makes production code that wraps its build in
            # `try: ... except Exception:` erase both at once: the hook's AssertionError is
            # swallowed and the client it built is never counted.
            with built_lock:
                built.append(self)
            self.chat = types.SimpleNamespace(completions=_FakeCompletions())
            if on_construct is not None:
                on_construct()

    module = types.ModuleType("openai")
    module.OpenAI = _FakeOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "openai", module)
    return built


#: The instrumented subclass. `benchmarks.voyage_rerank` has the same defect and the same
#: fix, so the machinery lives in one place; what each shape of it failed to catch, and why
#: the build hook and the slot assertions are both needed, is documented there.
_AssertingLLM = asserting_lazy_client(OpenRouterLLM)


def test_the_unset_client_is_only_ever_touched_under_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder: list[Any] = []
    built = _install_counting_openai(monkeypatch, on_construct=built_under_the_lock(holder))
    # `max_attempts=1`: an AssertionError from the property must surface as itself and at once,
    # not after `retry_with_backoff` has paid three real `time.sleep` backoffs for it.
    llm = _AssertingLLM(model="test/model", api_key="k", max_attempts=1)
    holder.append(llm)
    # Laziness, pinned directly rather than inferred. `built_under_the_lock` reports an eager
    # build by RAISING, which production code is free to swallow: wrap the build in
    # `try: ... except Exception:`, as code around an optional extra often is, and the
    # assertion disappears with it. A count taken before the first call cannot be swallowed.
    assert built == [], "the client was built at construction; it is documented as lazy"

    assert llm.complete("sys", "user") == "ok"
    assert len(built) == 1
    # A second call proves the post-set read is reached and allowed, so the assertion above is
    # narrow by demonstration rather than by claim.
    assert llm.complete("sys", "user") == "ok"
    assert len(built) == 1
    # Nothing above would notice if the property stopped intercepting: a passthrough returns the
    # same client and the same count. This asserts the instrument itself ran.
    llm.assert_guard_was_live()


def test_a_failed_request_does_not_rebuild_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Write-once, driven rather than assumed.

    The helper has asserted against a reset since it was written, and until this test nothing in
    either lock file had ever made an SDK call fail, so that branch had never executed. "The pool
    may be poisoned, drop it and rebuild next time" is the ordinary reflex, and here it costs one
    client per failed attempt: at the shipped `max_attempts=4`, four for a single rate-limited
    completion. It also reopens the hazard the assertion names, since `_complete_once` reads the
    slot outside the lock to issue the request.
    """
    holder: list[Any] = []
    built = _install_counting_openai(
        monkeypatch, on_construct=built_under_the_lock(holder), failures=1
    )
    llm = _AssertingLLM(model="test/model", api_key="k", max_attempts=1)
    holder.append(llm)

    with pytest.raises(RuntimeError, match="connection reset"):
        llm.complete("sys", "user")
    assert len(built) == 1

    # Repeated, not once. A rebuild does not have to be triggered by the failure: "recycle every
    # N calls" is a rebuild on a trigger no single call reaches, and one succeeding call cannot
    # see it. 200 is far above any plausible N and costs milliseconds against a fake.
    for _ in range(200):
        assert llm.complete("sys", "user") == "ok"
    assert len(built) == 1, "the client was rebuilt after a failed request"
    llm.assert_guard_was_live()


def test_threads_released_together_build_exactly_one_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    start = threading.Barrier(WORKERS)
    holder: list[Any] = []
    owned = built_under_the_lock(holder)

    def _hold_the_window_open() -> None:
        # Every thread is inside the check-then-set at once. Unguarded, that is what turns the
        # leak from a matter of luck into a reproduction; guarded, it costs one 50ms build.
        owned()
        time.sleep(0.05)

    built = _install_counting_openai(monkeypatch, on_construct=_hold_the_window_open)
    # `_AssertingLLM` here too, not a bare `OpenRouterLLM`. Ownership is the half of the invariant
    # that only means anything under contention: a lone thread is the rightful owner of a lock it
    # took non-blocking, while seven threads that fell through a failed acquire are not. Running
    # the same assertions here turns that case into a named failure rather than a bare count.
    llm = _AssertingLLM(model="test/model", api_key="k", max_attempts=1)
    holder.append(llm)

    answers: list[str] = []
    answers_lock = threading.Lock()
    failures: list[BaseException] = []
    unreleased: list[BaseException] = []

    def _worker() -> None:
        try:
            start.wait(timeout=10)
        except threading.BrokenBarrierError as exc:
            # Kept OUT of `failures`. A runner too loaded to start eight threads inside
            # 10s is a scheduling timeout, and reporting it beside guard violations would
            # let it read as a lock failure, which is the one false red this test has.
            unreleased.append(exc)
            return
        try:
            answer = llm.complete("sys", "user")
        except BaseException as exc:  # noqa: BLE001 - re-reported below, never swallowed
            failures.append(exc)
        else:
            with answers_lock:
                answers.append(answer)

    # `daemon=True` and ONE deadline shared across the joins, not 30s each. Eight sequential
    # 30s joins is a 240s worst case inside a suite whose per-test budget is 120s with
    # `timeout_method = "thread"`, and that method takes the whole session down rather than
    # failing this test, so the leaked-thread assertion below would be unreachable in exactly the
    # case it is written for. Daemon threads then keep a wedged worker from holding up
    # interpreter shutdown after the failure has been reported.
    threads = [
        threading.Thread(target=_worker, name=f"llm-worker-{i}", daemon=True)
        for i in range(WORKERS)
    ]
    for thread in threads:
        thread.start()
    deadline = time.monotonic() + 30.0
    for thread in threads:
        thread.join(timeout=max(0.0, deadline - time.monotonic()))

    assert [t.name for t in threads if t.is_alive()] == []
    assert unreleased == [], "the runner could not release all workers in 10s; not a lock failure"
    assert failures == []
    assert answers == ["ok"] * WORKERS
    assert len(built) == 1, f"{len(built)} openai clients built for one OpenRouterLLM"
    llm.assert_guard_was_live()
    assert llm._client is built[0]  # noqa: SLF001 - the survivor is the one everybody used
