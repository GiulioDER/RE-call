"""Running work off the UI thread, and the two ways that silently goes wrong.

⛔ **This module exists so the lesson below has ONE implementation.** It was learned inside
`recall/desktop/ui.py` and cost real debugging; the graphical installer needs the same thing with a
progress channel, and copying it would have made a second copy of a rule whose failure mode is a
race that works often enough to look correct.

Two properties are load-bearing, and neither is obvious from reading Qt's documentation:

**1. The runnable must be KEPT.** `QRunnable` defaults to `autoDelete`, so the instant `run()`
returns Qt destroys it — taking its `QObject` signal source with it and purging the queued
cross-thread call before it is delivered. Measured on PySide6 6.11 with five identical jobs: **1 of
5 arrived** with the reference dropped, **5 of 5** with it held. Another run of the same code
delivered **0 of 5**. A garbage-collection race, not a deterministic failure, which is why it
survived so long.

**2. The emit itself can raise, and it does so at shutdown.** A worker outlives a window that stopped
waiting for the pool: it finishes, Qt has already deleted the signal source, and `emit` raises
`RuntimeError: Signal source has been deleted`. Reporting THAT on the failure signal raises the same
way from inside the handler, and PySide prints `Error calling Python override of QRunnable::run()`
with a traceback — indistinguishable from a crash, at the exact moment the user is closing the app.
Observed live, not theorised.

The base class holds both. `Job` is the zero-argument form the main window uses; `StreamingJob`
hands its function a `progress` callable so long work can report as it goes, which is what an
installer needs and what a single `done` signal cannot express.
"""

from __future__ import annotations

import importlib
from typing import Any

__all__ = ["CLOSE_WAIT_MS", "Job", "JobSignals", "StreamingJob"]

#: How long a window waits for background work before closing anyway, in milliseconds.
#:
#: Two seconds is chosen against what it is racing: a quick worker finishing normally, not a
#: `docker compose up`. Anything long enough to be worth waiting for is long enough to look frozen,
#: and the window closing while work continues is both true and better than a window that will not
#: close.
CLOSE_WAIT_MS = 2000

_qt_core: Any = None
try:  # pragma: no cover - exercised on environments without the desktop extra
    _qt_core = importlib.import_module("PySide6.QtCore")
except ImportError:  # pragma: no cover
    pass

# ⚠️ `getattr` rather than a re-assignment in the `except`, matching `recall/desktop/ui.py`. CI
# type-checks WITHOUT the desktop extra, so there the fallback branch is live and a
# `# type: ignore` on it is UNUSED, which `warn_unused_ignores` rejects; locally PySide6 is present
# and the same ignore is required. No single ignore comment can be correct in both, so the fix is
# to write it so no ignore is needed at all.
QObject: Any = getattr(_qt_core, "QObject", None)
QRunnable: Any = getattr(_qt_core, "QRunnable", None)
Signal: Any = getattr(_qt_core, "Signal", None)


if QObject is not None:

    class JobSignals(QObject):
        """The three things a background job can report.

        `progress` carries one line of human-readable status. It is a separate signal rather than
        repeated `done` emissions because `done` means finished, and a consumer that cannot tell
        "still working" from "finished" will re-enable its controls part way through.
        """

        done = Signal(object)
        failed = Signal(str)
        progress = Signal(str)

    class _BaseJob(QRunnable):
        """Carries the two properties in this module's docstring. Do not bypass it."""

        def __init__(self) -> None:
            super().__init__()
            self.signals = JobSignals()
            # ⚠️ Property 1. Set here rather than at every call site, because a call site that
            # forgets produces a race that passes its tests.
            self.setAutoDelete(False)

        def _call(self) -> Any:  # pragma: no cover - overridden
            raise NotImplementedError

        def run(self) -> None:
            """Run the job and report it, without ever raising out of the pool.

            The job failing and the REPORT failing are separated deliberately. One `try` around
            both meant anything that went wrong while emitting `done` was shown to the user as the
            job having failed.
            """
            try:
                result = self._call()
            except Exception as exc:  # noqa: BLE001 - a worker must never raise into the pool.
                self._emit(self.signals.failed, str(exc))
                return
            self._emit(self.signals.done, result)

        @staticmethod
        def _emit(signal: Any, payload: Any) -> None:
            """Property 2. Emitting into a deleted source is a shutdown condition, not an error."""
            try:
                signal.emit(payload)
            except RuntimeError:
                # The window closed and Qt deleted the signal source before this finished. Nothing
                # is listening any more, so there is nowhere to report and nothing to report to.
                pass

    class Job(_BaseJob):
        """A zero-argument callable run off the UI thread."""

        def __init__(self, fn: Any) -> None:
            super().__init__()
            self.fn = fn

        def _call(self) -> Any:
            return self.fn()

    class StreamingJob(_BaseJob):
        """A callable handed a `progress` reporter, for work long enough to need one.

        The reporter is `self.signals.progress.emit`, wrapped so a shutdown mid-run is silent for
        the same reason the final emit is. A progress line raising `RuntimeError` out of a worker
        thread would be the crash-at-close described in this module's docstring, moved earlier.
        """

        def __init__(self, fn: Any) -> None:
            super().__init__()
            self.fn = fn

        def _report(self, line: str) -> None:
            self._emit(self.signals.progress, line)

        def _call(self) -> Any:
            return self.fn(self._report)

else:  # pragma: no cover - no desktop extra installed

    JobSignals = None  # type: ignore[assignment,misc]
    Job = None  # type: ignore[assignment,misc]
    StreamingJob = None  # type: ignore[assignment,misc]
