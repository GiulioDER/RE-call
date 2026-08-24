"""The order a window and its `QApplication` are created in, which is what a double-click does.

⛔ **This file exists because the published 0.9.7 installer could not start at all.** `install_main`
launched the window like this:

    run_window(InstallerWindow(default_root=...))

Python evaluates the argument before the call, so the window was constructed BEFORE `run_window`
created the application. Qt answers that with `QWidget: Must construct a QApplication before a
QWidget` and aborts the process on its fatal handler: no traceback, no window, exit `0xC0000409`.
Reported from the shipped binary by the first person to run it, reproduced here.

⚠️ **Every other desktop test in this repository, and the selftest the release workflow runs, made
a `QApplication` first and a window second.** That is the correct order, and it is not the order the
entry point used, so the whole suite passed against a bundle that could not open its own window. A
test that rehearses a launch sequence of its own invention tests the rehearsal.

⚠️ **An in-process test cannot reproduce the defect by simply omitting the application.** Once any
test in the session has constructed a `QApplication` it cannot be unmade, so a later test sees one
whether or not the code under test made it. The tests here therefore assert the SEQUENCE OF EVENTS,
which is the real invariant, and one of them spawns a fresh interpreter, which is the only thing
that reproduces what a person double-clicking the exe actually gets.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def _recording_application(events: list[str]) -> type:
    """A stand-in for `QApplication` that appends to `events` when it is constructed.

    Deliberately not a Qt object. The property under test is the sequence "application, then
    window", and asserting it against a fake keeps these tests meaningful on a machine without the
    desktop extra, where the defect is just as real. A fresh class per call, so one test's instance
    is never visible to another through `instance()`.
    """

    class App:
        _instance: object | None = None

        def __init__(self, argv: list[str]) -> None:
            events.append("application")
            App._instance = self

        @classmethod
        def instance(cls) -> object | None:
            return cls._instance

        def exec(self) -> int:
            events.append("exec")
            return 0

    return App


def test_run_window_creates_the_application_before_it_builds_the_window() -> None:
    """The launcher owns both, which is the only reason it can order them.

    That is what makes the parameter a factory rather than a widget: a caller handed a widget has
    already lost, because the construction happened at the call site.
    """
    import recall.desktop.ui as ui

    events: list[str] = []
    app_class = _recording_application(events)

    class Window:
        def __init__(self) -> None:
            events.append("window")

        def show(self) -> None:
            events.append("show")

    original = ui.QApplication
    ui.QApplication = app_class
    try:
        assert ui.run_window(lambda: Window()) == 0
    finally:
        ui.QApplication = original

    assert events == ["application", "window", "show", "exec"], (
        "the window must be constructed after the QApplication exists; any other order aborts the "
        f"process on a real Qt build, and this one was {events}"
    )


def test_run_window_refuses_an_already_constructed_widget() -> None:
    """⛔ Passing a widget is the defect itself, so it must not be quietly accepted.

    By the time such a call reaches the launcher the damage is done: the widget was built at the
    call site, before any application existed. Accepting it would preserve the exact shape of the
    bug for every future caller, and the failure would once again be a process abort rather than a
    message naming the mistake.
    """
    import recall.desktop.ui as ui

    events: list[str] = []
    app_class = _recording_application(events)

    class Window:
        def show(self) -> None:  # pragma: no cover - never reached
            events.append("show")

    original = ui.QApplication
    ui.QApplication = app_class
    try:
        with pytest.raises(TypeError, match="callable"):
            ui.run_window(Window())
    finally:
        ui.QApplication = original


def test_install_main_does_not_build_the_window_before_the_application(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """⛔ **The regression. This is what the published 0.9.7 installer shipped with.**

    Asserted at the entry point rather than at the launcher, because the defect was never inside
    `run_window`: it was in how `run_window` was CALLED, and a test of the launcher alone passes
    against a broken call site.
    """
    import recall.desktop.install_ui as install_ui
    import recall.desktop.main as desktop_main
    import recall.desktop.ui as ui

    events: list[str] = []
    app_class = _recording_application(events)

    class Window:
        def __init__(self, **kwargs: Any) -> None:
            events.append("window")

        def show(self) -> None:
            events.append("show")

    monkeypatch.setattr(install_ui, "InstallerWindow", Window)
    monkeypatch.setattr(ui, "QApplication", app_class)

    assert desktop_main.install_main(["--data-root", str(tmp_path)]) == 0
    assert events and events[0] == "application", (
        "the installer built its window before any QApplication existed, which is the exact "
        f"sequence Qt aborts on: {events}"
    )


def test_run_app_does_not_build_the_window_before_the_application(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The main desktop window was launched the same way, so it carried the same defect.

    Worth its own test rather than trusting that one fix covered both. They are two call sites, and
    the one that is not shipped as a frozen binary is the one whose regression nobody would notice.
    """
    import recall.desktop.ui as ui
    from recall.desktop.models import RuntimeMode, RuntimeProfile

    events: list[str] = []
    app_class = _recording_application(events)

    class Window:
        def __init__(self, profile: Any) -> None:
            events.append("window")

        def show(self) -> None:
            events.append("show")

    monkeypatch.setattr(ui, "QApplication", app_class)
    # ⚠️ **`raising=False`, because without PySide6 there is no `MainWindow` to replace.**
    # `MainWindow` is defined inside `if QApplication is not None:`, so on a machine with no
    # desktop extra the attribute does not exist at all and a plain `setattr` raises
    # `AttributeError` before the test can assert anything. `InstallerWindow` has an explicit
    # `= None` fallback and its sibling test does not need this; `MainWindow` does not.
    # Caught by Linux CI, where the extra is absent, after passing on a Windows box where it is
    # installed: exactly the platform-shaped blindness this file was written about.
    monkeypatch.setattr(ui, "MainWindow", Window, raising=False)

    profile = RuntimeProfile(mode=RuntimeMode.LOCAL_DATABASE, dsn="postgresql://x@127.0.0.1/x")

    assert ui.run_app(profile) == 0
    assert events and events[0] == "application", (
        f"same defect as the installer, in the other window: {events}"
    )


def test_the_real_installer_starts_in_a_process_that_has_no_application(tmp_path: Path) -> None:
    """⛔ **The only check here that reproduces what a person double-clicking the exe gets.**

    Everything else in this file replaces Qt with a fake, and everything else in the suite runs
    inside a process that already has a `QApplication`. This spawns one that does not, drives the
    REAL entry point against REAL widgets offscreen, and lets the event loop return immediately. A
    wrong construction order kills the child on Qt's fatal handler, and the exit code says so.
    """
    pytest.importorskip("PySide6")

    script = f"""
import recall.desktop.ui as ui

# Subclassed rather than patched: the event loop method on the Shiboken type is not assignable, and
# reaching the REAL QApplication constructor is half of what this test is for.
class _App(ui.QApplication):
    def exec(self):
        return 0

ui.QApplication = _App

from recall.desktop.main import install_main

raise SystemExit(install_main(["--data-root", {str(tmp_path)!r}]))
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        timeout=300,
    )

    assert result.returncode == 0, (
        "the installer did not survive being started in a process with no QApplication, which is "
        f"exactly what a person double-clicking the exe does. exit={result.returncode}\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_the_selftest_starts_the_window_through_the_shared_launcher(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """⛔ **Why the release workflow was green for a bundle that could not start.**

    The selftest constructed its own application and then its own window, in that order. Correct,
    and not the order the entry point used, so the check passed on precisely the bundle it existed
    to reject.

    Asserted by observation rather than by reading the source: the launcher is replaced with a
    watcher, and the selftest has to have gone through it.
    """
    pytest.importorskip("PySide6")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.chdir(tmp_path)

    import recall.desktop.main as desktop_main
    import recall.desktop.ui as ui

    used: list[str] = []
    real = ui.application_and_window

    def watched(build_window: Any) -> tuple[Any, Any]:
        used.append("application_and_window")
        return real(build_window)

    monkeypatch.setattr(ui, "application_and_window", watched)
    monkeypatch.setattr(desktop_main, "_model_cache_exists", lambda: False)

    assert desktop_main._selftest() == 0
    assert used == ["application_and_window"], (
        "the selftest must build its window through the same code the entry point orders its "
        "launch with, or it proves an order that nothing ships"
    )
