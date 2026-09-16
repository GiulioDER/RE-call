"""Verify that a release tag names the version the package will publish.

The release workflow is the last reversible point before a tag can publish to PyPI. This check
keeps a stale tag, or a tag cut before a version bump was completed, from producing an artifact
whose package metadata names a different version.
"""

from __future__ import annotations

import argparse
import pathlib
import re
import tomllib

REPO = pathlib.Path(__file__).resolve().parent.parent
VERSION_RE = re.compile(r"^v(\d+\.\d+\.\d+)$")


class Refusal(RuntimeError):
    """A release tag does not describe the package being built."""


def project_version(root: pathlib.Path = REPO) -> str:
    """Return the version declared in the package metadata."""
    document = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    return str(document["project"]["version"])


def verify(tag: str, root: pathlib.Path = REPO) -> str:
    """Return the tag version, or refuse a malformed or mismatched release tag."""
    match = VERSION_RE.fullmatch(tag)
    if match is None:
        raise Refusal(f"{tag!r} is not a vX.Y.Z release tag")

    tagged_version = match.group(1)
    declared_version = project_version(root)
    if tagged_version != declared_version:
        raise Refusal(
            f"tag {tag} names {tagged_version}, but pyproject.toml declares {declared_version}"
        )
    return tagged_version


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("tag", help="the release tag, such as v0.14.0")
    args = parser.parse_args(argv)
    try:
        version = verify(args.tag)
    except (OSError, KeyError, tomllib.TOMLDecodeError, Refusal) as exc:
        parser.error(str(exc))
    print(f"release tag {args.tag} matches package version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
