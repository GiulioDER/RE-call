"""CLI entry point for the RE-call Windows desktop application."""

from __future__ import annotations

import argparse
import sys
from typing import Any

from recall.desktop.models import RuntimeMode, RuntimeProfile
from recall.desktop.profiles import load_profile
from recall.desktop.ui import run_app


def install_main(argv: list[str] | None = None) -> int:
    """Open the graphical installer.

    The window is built in `recall/desktop/install_ui.py` and launched through `ui.run_window`, the
    same launcher the main desktop window uses. That matters for one specific reason: it is the code
    that decides whether to reuse an existing `QApplication`, so the installer can be opened from
    inside the running desktop app instead of crashing with "A QApplication instance already
    exists".
    """
    parser = argparse.ArgumentParser(prog="recall-install")
    parser.add_argument(
        "--data-root",
        default=None,
        help="the folder suggested for recall's data (default: ~/.recall)",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="build the window off-screen, exercise the engine's imports and exit. Proves a "
        "packaged build is complete without a person having to click anything.",
    )
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    from pathlib import Path

    from recall.desktop.install_ui import InstallerWindow
    from recall.desktop.ui import run_window

    if InstallerWindow is None:
        raise SystemExit(
            "the graphical installer needs PySide6. Install it with "
            '`pip install "recall-rag[desktop]"`, or run `recall wizard` to be asked the same '
            "questions in the terminal."
        )
    return run_window(
        InstallerWindow(default_root=Path(args.data_root) if args.data_root else None)
    )


def _selftest() -> int:
    """Prove a packaged build can actually do the thing it exists to do, without a person.

    ⛔ **A frozen bundle's characteristic failure is `ModuleNotFoundError` at the moment of use.**
    PyInstaller finds imports by static analysis, and this codebase imports lazily nearly
    everywhere — `from recall.wizard.headless import run_headless` inside a function, and so on down
    the whole engine. So a bundle missing half of recall starts fine, draws its window, and dies the
    instant somebody presses Install. `--help` proves nothing about that: it exits at argparse,
    before any of it is reached.

    This reaches everything the first click reaches: the window, the question plan, the config
    builder, and the engine's own entry points. It stops short of `app.exec()`, so it can run
    unattended in CI, and short of provisioning anything, so it changes nothing.
    """
    import os
    import tempfile
    from pathlib import Path

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    failures: list[str] = []

    try:
        from recall.desktop.install_ui import InstallerWindow
        from recall.desktop.ui import run_window  # noqa: F401 - the launcher must be importable
        from recall.wizard.questions import build_config, question_plan
        # The engine, reached only from inside a callback in normal use, which is exactly why a
        # bundle can be missing it and look healthy.
        from recall.wizard.headless import load_config, run_headless  # noqa: F401
        from recall.wizard.pipeline import run_corpus  # noqa: F401
        from recall.wizard.uninstall import plan_uninstall  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - the whole point is to name what is missing
        print(f"selftest: import failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    try:
        from PySide6.QtWidgets import QApplication

        QApplication.instance() or QApplication([])
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            window = InstallerWindow(default_root=root)
            try:
                plan = question_plan(default_root=root)
                if set(window._fields) != {question.key for question in plan}:
                    failures.append("the form does not render the question plan")
                document = build_config(window._answers())
                for key in ("docs_root", "code_root", "memory_root", "corpus_version"):
                    if key not in document:
                        failures.append(f"the config it builds is missing {key}")
            finally:
                window.close()
    except Exception as exc:  # noqa: BLE001
        print(f"selftest: building the window failed: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1

    if failures:
        for failure in failures:
            print(f"selftest: {failure}", file=sys.stderr)
        return 1
    print("selftest: ok")
    return 0


def _qt_confirm(title: str, text: str, detail: str) -> bool:
    """Ask in a dialog, defaulting to No. The plan goes in verbatim, never summarised."""
    from PySide6.QtWidgets import QApplication, QMessageBox

    QApplication.instance() or QApplication([])
    box = QMessageBox()
    box.setWindowTitle(title)
    box.setText(text)
    box.setDetailedText(detail)
    box.setStandardButtons(QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
    box.setDefaultButton(QMessageBox.StandardButton.No)
    return bool(getattr(box, "exec")() == QMessageBox.StandardButton.Yes)


def _qt_notify(title: str, text: str) -> None:
    """Report an outcome, falling back to stderr when there is no desktop extra to report in."""
    try:
        from PySide6.QtWidgets import QApplication, QMessageBox

        QApplication.instance() or QApplication([])
        QMessageBox.information(None, title, text)
    except Exception:  # noqa: BLE001 - a missing desktop extra must not hide the reason
        print(f"{title}: {text}", file=sys.stderr)


def uninstall_main(
    argv: list[str] | None = None,
    *,
    confirm: Any = None,
    notify: Any = None,
) -> int:
    """Show what an uninstall would remove, and do it only after somebody agrees.

    ⚠️ **The plan is rendered into the dialog, not summarised.** This removes containers and
    rewrites the MCP client's configuration; a confirmation that says "are you sure?" without
    saying what it is sure about is a confirmation of nothing. `UninstallPlan.render` produces the
    exact text the terminal prints, so the two surfaces cannot come to disagree about what is going
    and what is staying — including that the folders being indexed are kept, which on a default
    install is the fact the person most needs to see.

    `confirm` and `notify` are injected for the same reason `InstallerWindow` takes its
    collaborators: the decision logic here is what needs testing, and it should not require a
    dialog to exercise. The Qt implementations are the defaults.
    """
    parser = argparse.ArgumentParser(prog="recall-uninstall")
    parser.add_argument(
        "--data-root",
        required=True,
        help="the data folder chosen during installation, recorded in its wizard.json",
    )
    parser.add_argument(
        "--purge-data",
        action="store_true",
        help="also remove the database volume holding the built indexes",
    )
    args = parser.parse_args(argv)

    from pathlib import Path

    from recall.wizard.uninstall import UninstallRefusal, execute, plan_uninstall

    ask = confirm or _qt_confirm
    tell = notify or _qt_notify

    try:
        plan = plan_uninstall(
            data_root=Path(args.data_root).expanduser(), purge_data=args.purge_data
        )
    except UninstallRefusal as exc:
        # ⚠️ Shown, not raised. Pointing this at the wrong folder is the ordinary mistake, and a
        # frozen binary has nowhere to print a traceback that anybody will ever read.
        tell("Cannot uninstall", str(exc))
        return 1

    if not ask("Uninstall recall", "Remove this recall install?", plan.render()):
        return 0
    report = execute(plan, purge_data=args.purge_data)
    tell("Uninstall finished", report.render())
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="recall-desktop")
    parser.add_argument("--profile", help="path to a runtime profile JSON")
    args = parser.parse_args(argv)
    profile = load_profile()
    if args.profile:
        from pathlib import Path

        from recall.desktop.profiles import load_profile as load_from_path

        profile = load_from_path(Path(args.profile))
    if profile is None:
        profile = RuntimeProfile(mode=RuntimeMode.DOCKER, compose_file="docker-compose.desktop.yml")
    return run_app(profile)


if __name__ == "__main__":
    raise SystemExit(main())
