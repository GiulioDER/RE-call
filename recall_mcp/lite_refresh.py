"""The MCP server keeps a lite store current with its memo folder: the automatic half of `recall lite refresh`.

Why the server and not an OS scheduler or a file watcher: it is the only process that reads the
store, it runs exactly when somebody may search, and every client (Claude Code, Codex, an agent's
memory provider) starts the same server. A scheduled task fires when nobody searches and needs an
installer per platform; a watcher needs a new dependency. Design: recall-lab
`research/designs/2026-10-12-lite-auto-refresh.md`.

Two triggers:

1. **start**: one catch-up refresh in a daemon thread, for whatever was written while no client ran;
2. **before each retrieval**, at most once per `RECALL_LITE_REFRESH_INTERVAL` seconds (30): a stat
   walk of the folder. When it differs from the last snapshot this server saw refreshed, a refresh
   starts (one thread per server) and the retrieval waits for it up to `RECALL_LITE_REFRESH_WAIT`
   seconds (3), then answers from the corpus as it is.

Several servers on one file take turns through the store's write lane, which coalesces: N servers
seeing one change cause one refresh. `RECALL_LITE_AUTO_REFRESH=0` turns both triggers off.
"""

from __future__ import annotations

import logging
import math
import threading
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import TYPE_CHECKING

from recall.index_budget import IndexBudget, budget_from_env
from recall.lite import LiteStore
from recall.lite.refresh import Outcome, Snapshot, read_state, refresh, snapshot

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.security_policy import AccessContext, SourceSecurityPolicy

__all__ = ["LiteAutoRefresher"]

_log = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 30.0
DEFAULT_WAIT_S = 3.0
#: The retry backoff after a failed refresh of an unchanged folder: from twice the interval (at
#: least this) doubling to `MAX_BACKOFF_S`. A changed folder is retried at once.
MIN_BACKOFF_S = 60.0
MAX_BACKOFF_S = 600.0
_OFF = frozenset({"0", "false", "off", "no"})
_ON = frozenset({"", "1", "true", "on", "yes"})


def _enabled(env: Mapping[str, str]) -> bool:
    """RECALL_LITE_AUTO_REFRESH: on unless switched off; an unrecognised value is off, said so.

    Off rather than on for a value it cannot read (CCA ENV-003): somebody who wrote "disable"
    meant to change the default, and the default is on.
    """
    raw = env.get("RECALL_LITE_AUTO_REFRESH", "").strip().lower()
    if raw in _ON:
        return True
    if raw not in _OFF:
        _log.warning("RECALL_LITE_AUTO_REFRESH=%r is not a boolean (1/0, on/off); the automatic refresh is off", raw)
    return False


def _seconds(env: Mapping[str, str], name: str, default: float) -> float:
    """A number of seconds from `name`; the default for anything that is not a finite number.

    Capped at `threading.TIMEOUT_MAX`: a longer wait makes `Thread.join` raise OverflowError, which
    the fail-open check would swallow, so the search would not wait at all (CCA NUM-001).
    """
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        _log.warning("%s=%r is not a finite number of seconds; using %s", name, raw, default)
        return default
    return min(max(0.0, value), threading.TIMEOUT_MAX)


class LiteAutoRefresher:
    """Refreshes one lite store from one memo folder, in the background, for one server."""

    def __init__(
        self,
        store: LiteStore,
        embedder: Embedder,
        roots: list[Path],
        *,
        interval_s: float = DEFAULT_INTERVAL_S,
        wait_s: float = DEFAULT_WAIT_S,
        security_policy: SourceSecurityPolicy | None = None,
        security_context: AccessContext | None = None,
        budget: IndexBudget | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store, self._embedder, self._roots = store, embedder, roots
        self._interval_s, self._wait_s, self._clock = interval_s, wait_s, clock
        self._policy, self._context = security_policy, security_context
        self._budget = budget
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_check = -math.inf
        self._refreshed: str | None = None  # the snapshot fingerprint this server last saw current
        # The snapshot whose refresh last FAILED, and when it may be tried again. A failure that
        # repeats (embedder down, prune guard tripped) is retried after a doubling backoff, not
        # on every interval, each time stalling a search (CCA PERF-002).
        self._failed: str | None = None
        self._retry_at = -math.inf
        self._backoff_s = 0.0
        self._failure: str | None = None  # why this server's last refresh failed; None once one succeeds
        self._pending: Snapshot | None = None  # the check's snapshot, handed to the refresh it starts
        self._stopped = False

    @classmethod
    def from_env(
        cls,
        store: object,
        embedder: Embedder,
        env: Mapping[str, str],
        *,
        security_policy: SourceSecurityPolicy | None = None,
        security_context: AccessContext | None = None,
    ) -> LiteAutoRefresher | None:
        """A refresher for a lite store with `RECALL_INDEX_ROOT` set, unless switched off; else None.

        `security_policy` and `security_context` are the server's source policy and the context its
        own `recall_index` uses; the refresh applies them as that tool does (CCA SEC-001).
        """
        if not isinstance(store, LiteStore):
            return None
        if not _enabled(env):
            return None
        root = env.get("RECALL_INDEX_ROOT", "").strip()
        if not root:
            return None
        try:
            budget = budget_from_env(env)
        except ValueError as exc:
            # Unattended: a bad limit falls back to the defaults rather than to no limit at all.
            _log.warning("%s; the automatic refresh uses the default index budget", exc)
            budget = IndexBudget()
        return cls(
            store,
            embedder,
            [Path(root)],
            interval_s=_seconds(env, "RECALL_LITE_REFRESH_INTERVAL", DEFAULT_INTERVAL_S),
            wait_s=_seconds(env, "RECALL_LITE_REFRESH_WAIT", DEFAULT_WAIT_S),
            security_policy=security_policy,
            security_context=security_context,
            budget=budget,
        )

    def start(self) -> None:
        """The catch-up refresh. Returns at once; the refresh runs in a daemon thread."""
        self._kick()

    def before_retrieval(self) -> None:
        """Refresh first if the folder changed, waiting a bounded time. Never raises; call off the event loop."""
        try:
            with self._lock:
                now = self._clock()
                if now - self._last_check < self._interval_s:
                    return
                self._last_check = now
            current = snapshot(self._roots)
            fingerprint = current.fingerprint
            if fingerprint == self._refreshed:
                with self._lock:
                    self._failure = None  # back to a snapshot that refreshed: nothing is missing
                return
            if fingerprint == self._failed and self._refreshed_elsewhere(fingerprint):
                return
            with self._lock:
                if fingerprint == self._failed and now < self._retry_at:
                    return
                self._pending = current  # the refresh this check starts reuses the walk (CCA PERF-004)
            thread, started = self._kick()
            if thread is not None and started:
                # Only a refresh this check started is waited for. One already running (queued
                # behind another writer, or a long first embed) would otherwise stall a search
                # every interval for as long as it runs (CCA PERF-001).
                thread.join(self._wait_s)
        except Exception as exc:  # BROAD-CATCH: fail-open  # a refresh problem must never fail a search
            _log.warning("lite auto refresh check failed: %s", exc)

    def _refreshed_elsewhere(self, fingerprint: str) -> bool:
        """Did another refresher (a person's CLI run) bring this failing snapshot current? Then adopt it."""
        last = read_state(self._store.path).get("last_success") or {}
        if last.get("fingerprint") != fingerprint or last.get("stated") is None:
            return False
        with self._lock:
            self._refreshed, self._failed, self._failure, self._backoff_s = fingerprint, None, None, 0.0
        return True

    def warning(self) -> str | None:
        """What a retrieval should tell the agent while refreshing fails, or None."""
        with self._lock:
            failure = self._failure
        if failure is None:
            return None
        return (
            "Memory refresh is failing, so memos changed since the last successful refresh are not "
            f"searchable yet: {failure} Run `recall lite refresh --check` for details."
        )

    def stop(self, timeout_s: float = 10.0) -> None:
        """Start no more refreshes, and let a running one finish before the store closes.

        Called by the server's lifespan before it closes the store (CCA BUG-006): a refresh cut
        off by the close would record a closed-database FAILED as the last refresh.
        """
        with self._lock:
            self._stopped = True
            thread = self._thread
        if thread is not None:
            thread.join(timeout_s)

    def _kick(self) -> tuple[threading.Thread | None, bool]:
        """The refresh thread, started if none is running; and whether this call started it."""
        with self._lock:
            if self._stopped:
                return None, False
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="recall-lite-refresh", daemon=True)
                self._thread.start()
                return self._thread, True
            return self._thread, False

    def _run(self) -> None:
        with self._lock:
            before, self._pending = self._pending, None
        try:
            # In the background: a server is long-lived, so a refit can outlast this refresh.
            report = refresh(
                self._store,
                self._roots,
                self._embedder,
                background_calibration=True,
                security_policy=self._policy,
                security_context=self._context,
                budget=self._budget,
                before=before,
                key_root=self._roots[0],  # RECALL_INDEX_ROOT: the server's single root names files
            )
        except Exception as exc:  # BROAD-CATCH: fail-open  # logged; the next check retries
            _log.warning("lite auto refresh failed: %s", exc)
            with self._lock:
                self._failure = f"{type(exc).__name__}: {exc}"
            return
        _log.info("lite auto refresh: %s (%s)", report.outcome.value, report.reason)
        with self._lock:
            if report.outcome in (Outcome.REFRESHED, Outcome.UNCHANGED) and report.facts_stated >= 0:
                self._refreshed, self._failed, self._backoff_s = report.fingerprint, None, 0.0
                self._failure = None
            elif report.outcome is Outcome.FAILED or report.facts_stated < 0:
                self._failure = report.stated if report.facts_stated < 0 else report.reason
                repeated = report.fingerprint == self._failed
                self._backoff_s = (
                    min(self._backoff_s * 2, MAX_BACKOFF_S) if repeated else max(2 * self._interval_s, MIN_BACKOFF_S)
                )
                self._failed = report.fingerprint
                self._retry_at = self._clock() + self._backoff_s
