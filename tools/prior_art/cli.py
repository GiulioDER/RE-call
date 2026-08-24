"""Command line interface for the prior art evidence corpus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from urllib.request import Request, urlopen

from .convention import check_prior_work_declarations
from .loader import load_dataset
from .render import render_files
from .validate import validate_dataset


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate and render RE-call prior art evidence")
    parser.add_argument("--root", type=Path, default=None, help="prior art data directory")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("validate")
    render = subparsers.add_parser("render")
    render.add_argument("--check", action="store_true")
    subparsers.add_parser("report")
    experiments = subparsers.add_parser("check-experiments")
    experiments.add_argument("paths", nargs="+", type=Path)
    links = subparsers.add_parser("check-links")
    links.add_argument("--timeout", type=float, default=10.0)
    return parser


def _check_links(dataset: dict[str, object], timeout: float) -> int:
    failures = 0
    for source in dataset["sources"]:  # type: ignore[index]
        url = source["canonical_url"]  # type: ignore[index]
        request = Request(str(url), method="HEAD", headers={"User-Agent": "RE-call-prior-art/1"})
        try:
            with urlopen(request, timeout=timeout) as response:
                print(f"{response.status} {url}")
        except Exception as exc:  # pragma: no cover, network is environment dependent
            failures += 1
            print(f"FAIL {url}: {exc}", file=sys.stderr)
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    dataset = load_dataset(args.root) if args.root else load_dataset()
    if args.command == "validate":
        errors = validate_dataset(dataset)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print("prior art corpus valid")
        return 0
    if args.command in {"render", "report"}:
        try:
            changed = render_files(dataset, check=getattr(args, "check", False))
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            return 1
        if changed:
            for path in changed:
                print(f"generated file is stale: {path}", file=sys.stderr)
            return 1
        print("prior art reports are current")
        return 0
    if args.command == "check-experiments":
        errors = check_prior_work_declarations(args.paths)
        if errors:
            print("\n".join(errors), file=sys.stderr)
            return 1
        print("prior work declarations are present")
        return 0
    return _check_links(dataset, args.timeout)


if __name__ == "__main__":
    raise SystemExit(main())
