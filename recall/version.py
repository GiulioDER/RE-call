"""Single source of truth for the package version."""

from __future__ import annotations

__version__ = "0.14.0"

#: Derived from ``__version__`` so the comparable form cannot drift from the published version.
_major, _minor, _patch = (int(part) for part in __version__.split("."))
VERSION_INFO = (_major, _minor, _patch)


def version_string() -> str:
    """Return the version as it is printed by the CLI."""

    return __version__


def version_tuple() -> tuple[int, int, int]:
    """Return the version as a comparable tuple."""

    return VERSION_INFO


def is_at_least(major: int, minor: int, patch: int = 0) -> bool:
    """True when this build is at or beyond the given version."""

    return VERSION_INFO >= (major, minor, patch)
