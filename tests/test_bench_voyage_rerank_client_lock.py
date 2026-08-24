"""One `VoyageReranker` builds exactly one `voyageai` client, however many threads drive it.

The same defect as `benchmarks.llm.OpenRouterLLM` had, in the same benchmark, reached by the same
worker pool. `_voyage_client` built its client behind an unguarded check-then-set while N threads
share one instance, so N clients were built and N-1 dropped, each still owning the HTTP connection
pool nobody closes. Reordering stays correct, since every client is built from the same api_key,
so the damage is leaked sockets and repeated setup rather than a wrong ranking.

The sharing is not assumed, it is traced. `benchmarks.beam.run` builds ONE `BeamRecallSystem`
before the conversation loop; that system resolves ONE reranker in its `__init__`
(`benchmarks.beam.systems`, `resolve_reranker`), which for a ``voyage:`` spec is a
`VoyageReranker`; its `_score` closure calls `system.retrieve(...)` and `_run_pool` maps that
closure over a `ThreadPoolExecutor(max_workers=args.workers)`, 8 by default. `retrieve` passes
`self._reranker` into the search on every query, so the reranker is reached concurrently, and
`run.py` says so itself beside the closure: "Retrieval happens INSIDE the worker".

The client stays LAZY rather than being built in `__init__`, for the reason the `openai` one does:
`voyageai` is an optional extra (`voyage`), `dev` excludes it, and CI installs `.[dev]`, so the SDK
is absent wherever these tests run. `tests/test_bench_voyage_rerank.py` constructs rerankers with
an injected fake client and no `voyageai` in `sys.modules` at all.

That injection seam is the one difference from the LLM case worth stating: `VoyageReranker(client=…)`
publishes a client during `__init__` ON PURPOSE, which is race-free (single-threaded, no other
thread holds a reference yet). The instrumentation allows it and leaves an EAGER build of the real
SDK to `built_under_the_lock`, which watches the constructor rather than the slot.

Two tests, because the invariant and the race fail differently, and what each can and cannot prove
is documented in `tests.lazy_client_lock_helpers` rather than repeated here.
"""
from __future__ import annotations

import sys
import threading
import time
import types
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

import pytest

from benchmarks.voyage_rerank import VoyageReranker
from recall.types import Chunk, ScoredChunk
from tests.lazy_client_lock_helpers import asserting_lazy_client, built_under_the_lock

#: Benchmark-harness coverage, not product coverage; product CI can deselect with
#: `-m 'not benchharness'`.
pytestmark = pytest.mark.benchharness

#: Threads released at once onto one instance, matching `benchmarks.beam.run`'s default workers.
WORKERS = 8

#: `publishes_at_init`: `VoyageReranker(client=...)` publishes during `__init__` by design,
#: which `OpenRouterLLM` never does. The flag is per class rather than blanket, so the LLM
#: guard keeps reporting a construction-time publish as the defect it is there.
_AssertingReranker = asserting_lazy_client(VoyageReranker, publishes_at_init=True)


def _hit(chunk_id: str, text: str) -> ScoredChunk:
    # Local rather than imported from `test_bench_voyage_rerank.py`: shared test fixtures in this
    # repository live in helper modules (`tests.conftest`, `tests.provider_stub_helpers`), and one
    # test module reaching into another's privates couples two files that have no reason to move
    # together.
    return ScoredChunk(
        chunk=Chunk(id=chunk_id, source="mem", text=text, metadata={}),
        score=0.9,
        indexed_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def _install_counting_voyageai(
    monkeypatch: pytest.MonkeyPatch,
    *,
    on_construct: Callable[[], None] | None = None,
    failures: int = 0,
) -> list[Any]:
    """Fake `voyageai` that records every client it builds; returns the list, mutated in place.

    `on_construct` runs at the END of the constructor, after the client is recorded, so a caller
    can either inspect the state the build is happening in or hold the window open without being
    able to suppress the count by raising.
    """
    built: list[Any] = []
    built_lock = threading.Lock()
    remaining_failures = [None] * failures

    class _FakeClient:
        def __init__(self, **_: Any) -> None:
            # Recorded BEFORE the hook; see the note in `built_under_the_lock`. Appending
            # afterwards would let a `try: ... except Exception:` around the production build
            # swallow the assertion AND lose the count, which is one escape rather than none.
            with built_lock:
                built.append(self)
            if on_construct is not None:
                on_construct()

        def rerank(self, query: str, documents: list[str], model: str, top_k: int) -> Any:
            try:
                # Popped rather than tested-then-popped. A check-then-act on a shared list is the
                # exact shape this file prosecutes, and the threaded test uses this same fake.
                remaining_failures.pop()
            except IndexError:
                pass
            else:
                raise RuntimeError("connection reset by peer")
            # Identity order: this file is about how many clients exist, not about reordering,
            # which `test_bench_voyage_rerank.py` already pins.
            items = [types.SimpleNamespace(index=i, relevance_score=1.0) for i in range(top_k)]
            return types.SimpleNamespace(results=items)

    module = types.ModuleType("voyageai")
    module.Client = _FakeClient  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "voyageai", module)
    return built


def test_the_unset_client_is_only_ever_touched_under_the_lock(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    holder: list[Any] = []
    built = _install_counting_voyageai(monkeypatch, on_construct=built_under_the_lock(holder))
    reranker = _AssertingReranker(api_key="k")
    holder.append(reranker)
    # Laziness, pinned directly rather than inferred. `built_under_the_lock` reports an eager
    # build by RAISING, which production code is free to swallow: wrap the build in
    # `try: ... except Exception:`, as code around an optional extra often is, and the
    # assertion disappears with it. A count taken before the first call cannot be swallowed,
    # and here it is the only net, since the injection seam means a construction-time publish
    # is legitimate for this class.
    assert built == [], "the client was built at construction; it is documented as lazy"
    hits = [_hit("a", "alpha"), _hit("b", "bravo")]

    assert [h.chunk.id for h in reranker.rerank("q", hits)] == ["a", "b"]
    assert len(built) == 1
    # A second call proves the post-set read is reached and allowed, so the assertion above is
    # narrow by demonstration rather than by claim.
    assert [h.chunk.id for h in reranker.rerank("q", hits)] == ["a", "b"]
    assert len(built) == 1
    # Nothing above would notice if the property stopped intercepting: a passthrough returns the
    # same client and the same count. This asserts the instrument itself ran.
    reranker.assert_guard_was_live()


def test_an_injected_client_is_not_mistaken_for_an_eager_build(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # `VoyageReranker(client=...)` is the documented offline seam and publishes during `__init__`,
    # which is race-free. The guard must not fire on it, and must never build an SDK client behind
    # it: a fake `voyageai` is installed precisely so that a stray build would be counted.
    def _nothing_should_reach_the_sdk() -> None:
        # A hook of its own rather than `built_under_the_lock([])`. That helper reads an empty
        # holder as "the fake fired before the instance was registered", which everywhere else is
        # a true proxy for construction time. Here the holder is empty forever by construction, so
        # it would report a build that happened mid-call as an eager build in `__init__`, which is
        # false and sends the reader looking at the wrong line. The condition this test actually
        # has is simpler and does not care when the build happened.
        raise AssertionError(
            "a voyageai client was built although one was injected: nothing behind the `client=` "
            "seam should reach the SDK, whenever the build happens"
        )

    built = _install_counting_voyageai(monkeypatch, on_construct=_nothing_should_reach_the_sdk)
    calls: list[str] = []

    def _injected_rerank(query: str, documents: list[str], model: str, top_k: int) -> Any:
        calls.append(query)
        # Reverse order, so the result cannot be mistaken for a pass-through of the input.
        return types.SimpleNamespace(
            results=[types.SimpleNamespace(index=i, relevance_score=1.0) for i in (1, 0)]
        )

    reranker = _AssertingReranker(client=types.SimpleNamespace(rerank=_injected_rerank))
    hits = [_hit("a", "alpha"), _hit("b", "bravo")]

    # The REORDERING, not merely the absence of a crash, and the call itself. Asserting only that
    # the result looks plausible would stay green if `rerank` short-circuited and never consulted
    # the injected client, which is a test that cannot tell "the seam works under the guard" from
    # "nothing happened".
    assert [h.chunk.id for h in reranker.rerank("q", hits)] == ["b", "a"]
    assert calls == ["q"]
    assert built == []
    # `lazily_built=False`: no read ever found the slot unset and no publish went
    # through the guarded branch, so only the storage half of the canary applies.
    reranker.assert_guard_was_live(lazily_built=False)


def test_a_failed_request_does_not_rebuild_the_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Write-once, driven rather than assumed.

    The helper has asserted against a reset since it was written, and until this test nothing in
    either file had ever made an SDK call fail, so that branch had never executed. It is not a
    hypothetical shape: "the pool may be poisoned, drop it and rebuild next time" is the ordinary
    reflex, and here it costs one client per failed request. `VoyageReranker` has no retry wrapper
    at all, so a run that rate-limits pays that on every query.
    """
    holder: list[Any] = []
    built = _install_counting_voyageai(
        monkeypatch, on_construct=built_under_the_lock(holder), failures=1
    )
    reranker = _AssertingReranker(api_key="k")
    holder.append(reranker)
    hits = [_hit("a", "alpha")]

    with pytest.raises(RuntimeError, match="connection reset"):
        reranker.rerank("q", hits)
    assert len(built) == 1

    # Repeated, not once. A rebuild does not have to be triggered by the failure: "recycle every
    # N calls" is a rebuild on a trigger no single call reaches, and one succeeding call cannot
    # see it. 200 is far above any plausible N and costs milliseconds against a fake.
    for _ in range(200):
        assert [h.chunk.id for h in reranker.rerank("q", hits)] == ["a"]
    assert len(built) == 1, "the client was rebuilt after a failed request"
    reranker.assert_guard_was_live()


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

    built = _install_counting_voyageai(monkeypatch, on_construct=_hold_the_window_open)
    reranker = _AssertingReranker(api_key="k")
    holder.append(reranker)
    hits = [_hit("a", "alpha")]

    orders: list[list[str]] = []
    orders_lock = threading.Lock()
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
            order = [h.chunk.id for h in reranker.rerank("q", hits)]
        except BaseException as exc:  # noqa: BLE001 - re-reported below, never swallowed
            failures.append(exc)
        else:
            with orders_lock:
                orders.append(order)

    # `daemon=True` and ONE deadline shared across the joins, not 30s each; see the same
    # construction in `test_bench_llm_client_lock.py` for why eight 30s joins would outlast the
    # suite's own 120s per-test budget and make the assertion below unreachable.
    threads = [
        threading.Thread(target=_worker, name=f"voyage-worker-{i}", daemon=True)
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
    assert orders == [["a"]] * WORKERS
    assert len(built) == 1, f"{len(built)} voyageai clients built for one VoyageReranker"
    reranker.assert_guard_was_live()
    assert reranker._client is built[0]  # noqa: SLF001 - the survivor is the one everybody used
