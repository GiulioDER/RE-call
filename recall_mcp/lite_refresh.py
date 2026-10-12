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

from recall.lite.refresh import Outcome, refresh, snapshot

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.lite.store import LiteStore

__all__ = ["LiteAutoRefresher"]

_log = logging.getLogger(__name__)

DEFAULT_INTERVAL_S = 30.0
DEFAULT_WAIT_S = 3.0
_OFF = frozenset({"0", "false", "off", "no"})


def _seconds(env: Mapping[str, str], name: str, default: float) -> float:
    raw = env.get(name, "").strip()
    if not raw:
        return default
    try:
        value = float(raw)
    except ValueError:
        _log.warning("%s=%r is not a number of seconds; using %s", name, raw, default)
        return default
    return max(0.0, value)


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
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._store, self._embedder, self._roots = store, embedder, roots
        self._interval_s, self._wait_s, self._clock = interval_s, wait_s, clock
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._last_check = -math.inf
        self._refreshed: str | None = None  # the snapshot fingerprint this server last saw current

    @classmethod
    def from_env(cls, store: object, embedder: Embedder, env: Mapping[str, str]) -> LiteAutoRefresher | None:
        """A refresher for a lite store with `RECALL_INDEX_ROOT` set, unless switched off; else None."""
        from recall.lite import LiteStore

        if not isinstance(store, LiteStore):
            return None
        if env.get("RECALL_LITE_AUTO_REFRESH", "").strip().lower() in _OFF:
            return None
        root = env.get("RECALL_INDEX_ROOT", "").strip()
        if not root:
            return None
        return cls(
            store,
            embedder,
            [Path(root)],
            interval_s=_seconds(env, "RECALL_LITE_REFRESH_INTERVAL", DEFAULT_INTERVAL_S),
            wait_s=_seconds(env, "RECALL_LITE_REFRESH_WAIT", DEFAULT_WAIT_S),
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
            if snapshot(self._roots).fingerprint == self._refreshed:
                return
            self._kick().join(self._wait_s)
        except Exception as exc:  # BROAD-CATCH: fail-open  # a refresh problem must never fail a search
            _log.warning("lite auto refresh check failed: %s", exc)

    def _kick(self) -> threading.Thread:
        with self._lock:
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._run, name="recall-lite-refresh", daemon=True)
                self._thread.start()
            return self._thread

    def _run(self) -> None:
        try:
            # In the background: a server is long-lived, so a refit can outlast this refresh.
            report = refresh(self._store, self._roots, self._embedder, background_calibration=True)
        except Exception as exc:  # BROAD-CATCH: fail-open  # logged; the next check retries
            _log.warning("lite auto refresh failed: %s", exc)
            return
        _log.info("lite auto refresh: %s (%s)", report.outcome.value, report.reason)
        if report.outcome in (Outcome.REFRESHED, Outcome.UNCHANGED):
            self._refreshed = report.fingerprint
