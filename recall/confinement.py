"""Whether a path really lies under a root: the one containment check for reads and writes (M3).

CCA's `access-control` class holds 35 confirmed RE-call findings, 20 of them P1 (measured
2026-11-04), and about half are one shape: each reader checked containment its own way, and some
way was wrong. The check ran on a link's name rather than its target, or after `Path.resolve()` had
already opened a UNC path (on Windows that contacts the host and sends the user's credentials), or
without refusing a drive-relative name such as `C:x`. Every reader under a root calls this module
instead, and the ast-grep rule `containment-via-confined` refuses a hand-written `is_relative_to`.

- `confined(root, name)`: a NAME a caller supplies, joined under the root. Anything that names its
  own root, drive or share is refused before any filesystem call.
- `within(root, path)`: a path already under the root by construction (a walk, a glob), or an
  absolute path a caller supplies. A drive-relative path, or a UNC share other than the root's, is
  refused before resolving, and the decision is made on the resolved target, so a symlink or
  junction that leaves the root is out.

Both return the RESOLVED path, or None. Read the returned path, never the one you passed in: a
decision made on a link's name and a read through its target is the confused deputy this exists to
prevent.
"""

from __future__ import annotations

from pathlib import Path, PurePath, PurePosixPath, PureWindowsPath


def anchored(text: str) -> bool:
    """Whether a path names its own root, drive or share, so joining it would leave the root.

    Checked before any filesystem call: on Windows `resolve()` opens a UNC path such as
    `\\\\host\\share\\x` (or `//host/share/x`) to follow links, which contacts that host and sends the
    user's credentials, so a containment check that follows the resolve comes too late.
    """
    for flavour in (PureWindowsPath, PurePosixPath):
        candidate = flavour(text)
        if candidate.anchor or candidate.drive:
            return True
    return False


def remote_elsewhere(root: PurePath | str, path: PurePath | str) -> bool:
    """Whether resolving `path` would leave `root`'s machine or drive: a UNC share other than the
    one `root` is on (resolving it contacts that host), or a drive-relative name such as `C:x`
    (another drive's working directory). Refused even where absolute paths are allowed. A walk under
    a root on a share stays on that share, so its paths are not refused."""
    windows = PureWindowsPath(str(path))
    if windows.drive and not windows.root:
        return True
    return windows.drive.startswith("\\\\") and (
        windows.drive.casefold() != PureWindowsPath(str(root)).drive.casefold()
    )


def within(root: Path, path: PurePath | str) -> Path | None:
    """`path` resolved, if it lies under `root` resolved; None otherwise or when it cannot resolve."""
    if remote_elsewhere(root, path):
        return None
    try:
        resolved = Path(path).resolve()
        boundary = root.resolve()
    except (OSError, RuntimeError, ValueError):  # RuntimeError: a link loop before Python 3.13
        return None
    return resolved if resolved.is_relative_to(boundary) else None


def confined(root: Path, name: PurePath | str) -> Path | None:
    """The file a caller-supplied `name` names under `root`, resolved, or None if it leaves it."""
    if anchored(str(name)):
        return None
    return within(root, root / name)
