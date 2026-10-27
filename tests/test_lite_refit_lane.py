"""The background calibration refit takes the lite store's `refit` lane (`recall.lite.writer_lock`).

Two MCP servers writing one lite file each schedule a background refit, and `_REFIT_LOCK` only
dedupes within a process. The lane makes them take turns, and the re-check inside it makes the
second one find the first one's fit and stand down instead of fitting the same corpus again.

Invariants and the failure each one catches:
- R1 a background refit that waited behind another process's fit re-checks inside the lane, finds
  the corpus covered, and does not fit again.
- R2 a background refit stands down at once, fitting nothing, when another refit is already QUEUED
  for the lane (that one starts later and sees this write too).

Red proof, 2026-10-13, against the merge of #911 and #913 before the hook (`_run_refit` fitting
without the lane), on the test host, each failing in its intended assertion:
- R1: "a refit that waited behind another fit fitted the same corpus again: ['recall-lite-refit']".
- R2: "a refit fitted while another refit was already queued: ['recall-lite-refit']".
"""
from __future__ import annotations

import threading
from pathlib import Path

import pytest

from recall.lite import LiteStore
from recall.lite import calibration as cal
from recall.lite.writer_lock import lite_lane, lite_lane_status
from tests.test_lite_calibration import ContentWordEmbedder
from tests.test_lite_carry_forward import _add, _fits, _fitted


def _due(tmp_path: Path) -> LiteStore:
    """A store whose fit no longer covers the corpus: a background refit is wanted."""
    store = _fitted(tmp_path)
    _add(store, tmp_path, 20)
    assert cal._refit_wanted(store), "the fixture did not make a refit due"
    return store


def _count_background_fits(monkeypatch: pytest.MonkeyPatch) -> tuple[list[str], object]:
    """Count `auto_calibrate` calls made through the module (the background refit's path)."""
    real = cal.auto_calibrate
    calls: list[str] = []

    def counting(store: LiteStore, embedder: object, *args: object, **kwargs: object) -> object:
        calls.append(threading.current_thread().name)
        return real(store, embedder, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(cal, "auto_calibrate", counting)
    return calls, real


def _hold_refit(store: LiteStore, purpose: str, then: object = None) -> tuple[threading.Event, threading.Event, threading.Thread]:
    """Another 'process' holds the refit lane until `go` is set; runs `then()` before releasing."""
    held, go = threading.Event(), threading.Event()

    def run() -> None:
        with lite_lane(store.path, lane="refit", purpose=purpose, wait_s=30):
            held.set()
            go.wait(60)
            if callable(then):
                then()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    assert held.wait(10), "the holder never took the refit lane"
    return go, held, thread


def test_a_refit_that_waited_finds_the_fresh_fit_and_stands_down(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R1."""
    store = _due(tmp_path)
    calls, real = _count_background_fits(monkeypatch)
    before = _fits(store)
    go, _held, holder = _hold_refit(store, "calibrate", then=lambda: real(store, ContentWordEmbedder()))  # type: ignore[operator]
    assert cal.schedule_refit(store, ContentWordEmbedder())
    go.set()
    holder.join(60)
    cal.wait_for_refit(store, timeout=60)
    assert calls == [], f"a refit that waited behind another fit fitted the same corpus again: {calls}"
    assert _fits(store) == before + 1, "exactly the holder's fit should have landed"


def test_a_refit_stands_down_when_another_is_already_queued(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """R2."""
    store = _due(tmp_path)
    calls, _real = _count_background_fits(monkeypatch)
    go, _held, holder = _hold_refit(store, "calibrate")
    queued_done = threading.Event()

    def queued() -> None:
        with lite_lane(store.path, lane="refit", purpose="calibrate", wait_s=30):
            pass
        queued_done.set()

    waiter = threading.Thread(target=queued, daemon=True)
    waiter.start()
    deadline = threading.Event()
    for _ in range(200):
        if len(lite_lane_status(store.path, "refit").waiters) == 1:
            break
        deadline.wait(0.05)
    assert len(lite_lane_status(store.path, "refit").waiters) == 1, "the queued refit never queued"
    try:
        assert cal.schedule_refit(store, ContentWordEmbedder())
        cal.wait_for_refit(store, timeout=10)
        assert calls == [], f"a refit fitted while another refit was already queued: {calls}"
    finally:
        go.set()
        holder.join(30)
        waiter.join(30)
