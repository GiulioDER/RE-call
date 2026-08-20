"""CLI entry point for the RE-call Windows desktop application."""

from __future__ import annotations

import argparse

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
    args = parser.parse_args(argv)

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
