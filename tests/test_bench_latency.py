"""`benchmarks.latency` teardown: an arm must be both released and reclaimed before the next one.

`main` runs the arms sequentially in ONE process, so whatever arm 1 leaves behind is still there
while arm 2 is being timed. This module's entire output is a latency comparison between the two,
which makes residue from the previous arm a measurement error rather than untidiness.
"""
from __future__ import annotations

import gc
import random
import weakref
from typing import Any

import pytest

from benchmarks import latency as latency_module


class _RecordingSystem:
    """A `MemorySystem` that logs the teardown-relevant calls in order."""

    name = "fake"

    def __init__(self, events: list[str], explode_on_ingest: bool = False) -> None:
        self._events = events
        self._explode = explode_on_ingest

    def ingest(self, conversation: dict[str, Any]) -> None:
        if self._explode:
            raise RuntimeError("ingest exploded")

    def retrieve(self, question: str) -> str:
        return "ctx"

    def close(self) -> None:
        self._events.append("close")


class _RecordingGC:
    """Stands in for the `gc` module so the real collector is never touched by a unit test."""

    def __init__(self, events: list[str]) -> None:
        self._events = events

    def collect(self, *args: object) -> int:
        # The generation argument is recorded, not swallowed. `gc.collect(0)` would look identical
        # to a full collection here otherwise, while being unable to free a model that has already
        # been promoted out of the young generations.
        self._events.append(f"gc{list(args)}")
        return 0


_CONVS: list[dict[str, Any]] = [
    {"sample_id": "c1", "qa": [{"question": "q1"}, {"question": "q2"}]},
]


def _run(events: list[str], monkeypatch: pytest.MonkeyPatch, *, explode: bool = False) -> Any:
    monkeypatch.setattr(
        latency_module, "_build", lambda *a, **k: _RecordingSystem(events, explode)
    )
    monkeypatch.setattr(latency_module, "gc", _RecordingGC(events))
    return latency_module._run_arm(
        "mem0", _CONVS, "fastembed", "openai/gpt-4o-mini", 5, "postgresql://x/y",
        "bench_latency_chunks", 2, 1, 0, random.Random(0), "stamp",
    )


def test_run_arm_releases_the_arm_and_then_reclaims_its_memory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Both steps, in that order.

    `close()` releases HANDLES; it does not reclaim MEMORY. Both arms hold an embedder inside a
    reference cycle, so dropping the last reference is not enough: measured against the real
    class, dropping mem0's `HuggingFaceEmbedding` leaves its `SentenceTransformer` (~133MB of
    weights) alive until a collection runs. Without the collect, arm 2 is timed on a machine still
    holding arm 1's model.

    The order matters: collecting before the close would run while the handles are still
    referenced, which is the one moment the objects are guaranteed NOT to be collectable.
    """
    events: list[str] = []
    ingest_secs, retrieve_ms = _run(events, monkeypatch)

    assert ingest_secs and retrieve_ms  # the arm actually ran, so teardown is not vacuous
    assert events == ["close", "gc[]"]


def test_run_arm_actually_reclaims_the_arm_rather_than_merely_calling_collect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The EFFECT, with the real collector. Every other test here pins only the call.

    That distinction is the whole finding: a `gc.collect()` placed in `_run_arm`'s own `finally`
    frees nothing, because `system` and the bound method held in `close` are both still live
    locals IN THAT FRAME, so the arm is still reachable. Measured on the staged version:
    `gc.collect()` returned 0 and the arm stayed alive. Only deleting BOTH names first frees it
    — deleting `system` alone is not enough, because the bound method pins it independently.

    The arm here holds a reference CYCLE, so refcounting can never free it and the collector is
    the only thing that can. `gc` is disabled for the duration, so a passing assertion cannot be
    the work of an incidental background collection.
    """
    class _Payload:
        pass

    class _CyclicSystem:
        name = "fake"

        def __init__(self) -> None:
            self.payload = _Payload()
            self.payload.back = self  # type: ignore[attr-defined]

        def ingest(self, conversation: dict[str, Any]) -> None:
            return None

        def retrieve(self, question: str) -> str:
            return "ctx"

        def close(self) -> None:
            return None

    # A list rather than a closure variable: the fake `_build` must hand the arm over and keep no
    # reference of its own, or the test itself would be what keeps the cycle alive.
    holder = [_CyclicSystem()]
    alive = weakref.ref(holder[0].payload)
    monkeypatch.setattr(latency_module, "_build", lambda *a, **k: holder.pop())

    gc.disable()
    try:
        latency_module._run_arm(
            "mem0", _CONVS, "fastembed", "openai/gpt-4o-mini", 5, "postgresql://x/y",
            "bench_latency_chunks", 2, 1, 0, random.Random(0), "stamp",
        )
        assert alive() is None, "the arm's cycle survived _run_arm: the collect freed nothing"
    finally:
        gc.enable()


def test_the_collect_still_runs_when_close_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reclamation must not be conditional on the release succeeding.

    They are sequential statements in one `finally`, so without an inner guard a raising `close()`
    would skip the collect entirely — and a wedged arm is exactly the case where its memory is
    most worth reclaiming. The original exception must still propagate.
    """
    events: list[str] = []

    class _UnclosableSystem:
        name = "fake"

        def ingest(self, conversation: dict[str, Any]) -> None:
            return None

        def retrieve(self, question: str) -> str:
            return "ctx"

        def close(self) -> None:
            events.append("close")
            raise RuntimeError("close exploded")

    monkeypatch.setattr(latency_module, "_build", lambda *a, **k: _UnclosableSystem())
    monkeypatch.setattr(latency_module, "gc", _RecordingGC(events))

    with pytest.raises(RuntimeError, match="close exploded"):
        latency_module._run_arm(
            "mem0", _CONVS, "fastembed", "openai/gpt-4o-mini", 5, "postgresql://x/y",
            "bench_latency_chunks", 2, 1, 0, random.Random(0), "stamp",
        )

    assert events == ["close", "gc[]"]  # the collect ran anyway


def test_an_arm_with_nothing_to_close_is_still_reclaimed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The `close()`-less arm is precisely the one this must not skip.

    `RecallSystem` has no `close` — it holds a DSN string, not a connection — but it DOES hold an
    embedder, in the same reference cycle. Making the collect conditional on the duck-typed close
    would therefore skip it for the only arm that has nothing else to release, and `main` runs
    that arm first.
    """
    events: list[str] = []

    class _NoCloseSystem:
        name = "recall"

        def ingest(self, conversation: dict[str, Any]) -> None:
            return None

        def retrieve(self, question: str) -> str:
            return "ctx"

    assert not hasattr(_NoCloseSystem, "close")  # the precondition this test exists for
    monkeypatch.setattr(latency_module, "_build", lambda *a, **k: _NoCloseSystem())
    monkeypatch.setattr(latency_module, "gc", _RecordingGC(events))
    latency_module._run_arm(
        "recall", _CONVS, "fastembed", "openai/gpt-4o-mini", 5, "postgresql://x/y",
        "bench_latency_chunks", 2, 1, 0, random.Random(0), "stamp",
    )

    assert events == ["gc[]"]  # nothing to close, but the memory is still reclaimed


def test_the_arm_is_released_and_reclaimed_even_when_the_run_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The failure path is the one that matters: a crashed arm is exactly when residue is left.

    `main` catches nothing, so an arm that dies mid-run would otherwise hand the next one both a
    held lock and a resident model.
    """
    events: list[str] = []
    with pytest.raises(RuntimeError, match="ingest exploded"):
        _run(events, monkeypatch, explode=True)

    assert events == ["close", "gc[]"]
