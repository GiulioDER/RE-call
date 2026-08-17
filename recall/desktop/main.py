"""CLI entry point for the RE-call Windows desktop application."""

from __future__ import annotations

import argparse

from recall.desktop.models import RuntimeMode, RuntimeProfile
from recall.desktop.profiles import load_profile
from recall.desktop.ui import run_app


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
