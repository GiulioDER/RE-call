"""Single source of truth for the package version."""

from __future__ import annotations

__version__ = "0.10.0"

#: Bumped by scripts/bump_version.py on every release. Keep the tuple in step: several
#: callers compare it rather than parsing the string.
VERSION_INFO = (0, 10, 0)


def version_string() -> str:
    """Return the version as it is printed by the CLI."""

    return __version__


def version_tuple() -> tuple[int, int, int]:
    """Return the version as a comparable tuple."""

    return VERSION_INFO


def is_at_least(major: int, minor: int, patch: int = 0) -> bool:
    """True when this build is at or beyond the given version."""

    return VERSION_INFO >= (major, minor, patch)
