"""Small PySide6 desktop shell for the first RE-call workflow."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

from recall.desktop.models import RuntimeProfile
from recall.desktop.runtime import RuntimeErrorBase


_qt_widgets: Any = None
try:
    _qt_widgets = importlib.import_module("PySide6.QtWidgets")
except ImportError:  # pragma: no cover, exercised on environments without the desktop extra
    pass

QApplication: Any = getattr(_qt_widgets, "QApplication", None)


if QApplication is not None:
    from recall.desktop.main_window import (  # re-exported: moved out of this module
        _project_name as _project_name,
        _format_summary as _format_summary,
        _DropZone as _DropZone,
        _FocuslessItemDelegate as _FocuslessItemDelegate,
        _FileHeader as _FileHeader,
        _HiddenSortItem as _HiddenSortItem,
        _TableWatermark as _TableWatermark,
        _PageWatermark as _PageWatermark,
        _CLOSE_WAIT_MS as _CLOSE_WAIT_MS,
        MainWindow as MainWindow,
    )


def application_and_window(build_window: Callable[[], Any]) -> tuple[Any, Any]:
    """Create the `QApplication` FIRST, then build the window, and return both.

    ⛔ **The parameter is a FACTORY, and that is the whole point of this function.** It used to take
    an already-constructed window, which reads harmlessly and is not:

        run_window(InstallerWindow(default_root=...))

    Python evaluates the argument before the call, so the window was built before this function had
    made the application. Qt answers a widget constructed with no `QApplication` by printing
    `QWidget: Must construct a QApplication before a QWidget` and aborting the process on its fatal
    handler: no traceback, no window, exit `0xC0000409`. That shipped in the 0.9.7 installer and was
    reported by the first person to run it. A factory moves the construction inside, where the
    ordering is decided, so no call site can get it wrong again.

    ⛔ **One launcher, because there are now two windows.** The graphical installer
    (`recall/desktop/install_ui.py`) needs exactly this and nothing else, and a second copy of it
    would be a second place that decides whether to reuse an existing `QApplication`, which is the
    line that decides whether the installer can be opened from inside the running desktop app or
    crashes with "A QApplication instance already exists".

    ⚠️ **Split out of `run_window` so the self-test can exercise the ordering without an event
    loop.** The self-test used to build an application and a window itself, in the right order,
    which is why it passed against the bundle that could not start: it rehearsed a sequence nothing
    shipped. It now comes through here.
    """
    if QApplication is None:
        raise RuntimeErrorBase('The desktop extra is required. Install with: pip install "recall-rag[desktop]"')
    if not callable(build_window):
        # An already-built widget IS the defect: it was constructed at the call site, before any
        # application existed. Naming it beats letting Qt abort the process.
        raise TypeError(
            "run_window takes a callable that builds the window, not a window. Pass "
            "`lambda: MyWindow(...)` so the QApplication is created first."
        )
    app = QApplication.instance() or QApplication([])
    return app, build_window()


def run_window(build_window: Callable[[], Any]) -> int:
    """Build the window inside a live `QApplication`, show it, and run until it closes."""
    app, window = application_and_window(build_window)
    window.show()
    return int(app.exec())


def run_app(profile: RuntimeProfile, profile_path: Path | None = None) -> int:
    """The main desktop window, saving its settings to `profile_path` (None: the default file)."""
    if QApplication is None:
        raise RuntimeErrorBase('The desktop extra is required. Install with: pip install "recall-rag[desktop]"')
    return run_window(lambda: MainWindow(profile, profile_path=profile_path))
