"""Progress for `recall generation build`, written as plain lines for a log file.

Why this exists (observed 2026-09-24 on VPS2): a project refresh held `embed.lock` for 45 minutes
and its log printed nothing after the `embedder:` line. The process was at its CPU quota with no
syscalls and no database socket, so it was embedding in memory and would write only at the end,
and that was indistinguishable from a hang. A session had to sample `/proc/<pid>/stat` twice to
prove it was working. Nothing said how far along it was, or whether chunks were being reused or
re-embedded.

So the build reports three things, all to a stream the caller chooses (the CLI uses stderr, which
the VPS2 scripts redirect into the log beside stdout):

1. A PLAN line before any embedding starts, saying how many objects can at most be reused and how
   many must be embedded, and saying loudly when the pipeline fingerprint moved so that nothing can
   be reused. That is the expensive surprise, and it is knowable at minute zero.
2. A PROGRESS line every `every` objects, and at least every `interval` seconds.
3. A HEARTBEAT from a daemon thread when the build loop itself is silent for `interval` seconds,
   naming the stage and the object it is on. One source's embedding call on a local CPU model can
   outlast any per-object cadence, and that is exactly where the silence was.

Lines, never a carriage return: the output lands in a file. The library emits events only
(`BuildProgressSink`); formatting and the thread live here, so a library caller that passes no
sink pays nothing and prints nothing.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol, TextIO

PREFIX = "[build]"


@dataclass(frozen=True)
class BuildPlan:
    """What the build can know about its cost before it embeds anything.

    `reusable` is an UPPER bound: it applies every reuse condition `_reuse_source` applies except
    the markdown body rule, which needs each document's text. `None` means the estimate could not
    be made, and `estimate_error` says why; a failed estimate never fails the build.
    """

    total: int
    reusable: int | None
    pipeline_fingerprint: str
    generations_with_fingerprint: int
    active_generation_id: str | None
    active_pipeline_fingerprint: str | None
    cache_enabled: bool
    estimate_error: str | None = None

    @property
    def fingerprint_changed(self) -> bool:
        return (
            self.active_pipeline_fingerprint is not None
            and self.active_pipeline_fingerprint != self.pipeline_fingerprint
        )

    @property
    def full_re_embed(self) -> bool:
        """An active generation exists, differs, and nothing retained shares the new fingerprint."""
        return self.fingerprint_changed and self.generations_with_fingerprint == 0


@dataclass(frozen=True)
class BuildCounters:
    total: int
    done: int = 0
    reused: int = 0
    embedded: int = 0
    empty: int = 0
    tombstoned: int = 0
    chunks_written: int = 0
    cache_hits: int | None = None
    cache_misses: int | None = None


class BuildProgressSink(Protocol):
    def planned(self, plan: BuildPlan) -> None: ...

    def update(
        self, counters: BuildCounters, *, stage: str, current: str | None = None
    ) -> None: ...

    def finished(self, counters: BuildCounters) -> None: ...


def _short(fingerprint: str | None) -> str:
    return (fingerprint or "none")[:12]


def format_plan(plan: BuildPlan) -> str:
    cache = "embedding cache on" if plan.cache_enabled else "embedding cache OFF"
    if plan.reusable is None:
        return (
            f"{PREFIX} plan: {plan.total} objects; reuse estimate unavailable "
            f"({plan.estimate_error}); {cache}"
        )
    if plan.full_re_embed:
        return (
            f"{PREFIX} plan: {plan.total} objects; FULL RE-EMBED: pipeline fingerprint "
            f"{_short(plan.pipeline_fingerprint)} differs from active generation "
            f"{plan.active_generation_id} ({_short(plan.active_pipeline_fingerprint)}) and no "
            f"retained generation shares it, so no chunk can be reused and all {plan.total} "
            f"objects will be chunked and embedded; {cache}"
            + (" (unchanged texts are served from it)" if plan.cache_enabled else "")
        )
    to_embed = plan.total - plan.reusable
    head = f"{PREFIX} plan: {plan.total} objects; up to {plan.reusable} reusable"
    if plan.generations_with_fingerprint:
        head += (
            f" from {plan.generations_with_fingerprint} generation(s) with pipeline fingerprint "
            f"{_short(plan.pipeline_fingerprint)}"
        )
    else:
        head += f" (no generation has pipeline fingerprint {_short(plan.pipeline_fingerprint)} yet)"
    if plan.fingerprint_changed:
        head += (
            f", which differs from active generation {plan.active_generation_id} "
            f"({_short(plan.active_pipeline_fingerprint)})"
        )
    return f"{head}; at least {to_embed} to chunk and embed; {cache}"


def format_counters(counters: BuildCounters, elapsed: float) -> str:
    line = (
        f"object {counters.done}/{counters.total}, reused {counters.reused}, "
        f"embedded {counters.embedded}, empty {counters.empty}, "
        f"tombstoned {counters.tombstoned}, chunks written {counters.chunks_written}"
    )
    if counters.cache_hits is not None and counters.cache_misses is not None:
        line += f", cache hits {counters.cache_hits} misses {counters.cache_misses}"
    return f"{line}, elapsed {elapsed:.0f}s"


@dataclass
class BuildProgressReporter:
    """The CLI's sink: plain lines to `stream`, and a heartbeat when the build goes quiet.

    Use it as a context manager so the heartbeat thread stops however the build ends.
    """

    stream: TextIO
    every: int = 25
    interval: float = 30.0
    heartbeat: bool = True
    clock: Callable[[], float] = time.monotonic
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    _stop: threading.Event = field(default_factory=threading.Event, init=False, repr=False)
    _thread: threading.Thread | None = field(default=None, init=False, repr=False)
    _started: float = field(default=0.0, init=False)
    _last_emit: float = field(default=0.0, init=False)
    _last_done_emitted: int = field(default=-1, init=False)
    _counters: BuildCounters | None = field(default=None, init=False)
    _stage: str = field(default="starting", init=False)
    _current: str | None = field(default=None, init=False)
    _stage_since: float = field(default=0.0, init=False)

    def __enter__(self) -> BuildProgressReporter:
        now = self.clock()
        self._started = self._last_emit = self._stage_since = now
        if self.heartbeat and self.interval > 0:
            self._thread = threading.Thread(
                target=self._beat, name="recall-build-heartbeat", daemon=True
            )
            self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=5)

    def _write(self, line: str) -> None:
        # Called with the lock held. A progress line must never be the reason a build dies.
        try:
            self.stream.write(line + "\n")
            self.stream.flush()
        except (OSError, ValueError):
            pass
        self._last_emit = self.clock()

    def planned(self, plan: BuildPlan) -> None:
        with self._lock:
            self._write(format_plan(plan))

    def update(
        self, counters: BuildCounters, *, stage: str, current: str | None = None
    ) -> None:
        with self._lock:
            now = self.clock()
            if stage != self._stage or current != self._current:
                self._stage_since = now
            self._counters, self._stage, self._current = counters, stage, current
            due_by_count = (
                self.every > 0
                and counters.done > 0
                and counters.done % self.every == 0
                and counters.done != self._last_done_emitted
            )
            due_by_time = self.interval > 0 and now - self._last_emit >= self.interval
            if due_by_count or due_by_time:
                self._emit_progress(now)

    def _emit_progress(self, now: float) -> None:
        assert self._counters is not None
        suffix = f", now {self._stage}" + (f" {self._current}" if self._current else "")
        self._write(f"{PREFIX} {format_counters(self._counters, now - self._started)}{suffix}")
        self._last_done_emitted = self._counters.done

    def finished(self, counters: BuildCounters) -> None:
        with self._lock:
            elapsed = self.clock() - self._started
            self._counters, self._stage, self._current = counters, "finished", None
            self._write(f"{PREFIX} done: {format_counters(counters, elapsed)}")

    def _beat(self) -> None:
        while not self._stop.wait(self.interval):
            with self._lock:
                now = self.clock()
                if self._stage == "finished" or now - self._last_emit < self.interval:
                    continue
                where = f" {self._current}" if self._current else ""
                progress = (
                    format_counters(self._counters, now - self._started)
                    if self._counters is not None
                    else f"elapsed {now - self._started:.0f}s"
                )
                self._write(
                    f"{PREFIX} still working: {self._stage}{where} for "
                    f"{now - self._stage_since:.0f}s; {progress}"
                )
