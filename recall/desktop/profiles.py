"""Persist non secret desktop runtime settings."""

from __future__ import annotations

import json
import os
from pathlib import Path

from recall.desktop.models import RuntimeProfile


def profile_path() -> Path:
    root = os.environ.get("APPDATA") or os.environ.get("LOCALAPPDATA")
    if root:
        return Path(root) / "RE-call" / "runtime.json"
    return Path.home() / ".recall" / "runtime.json"


def load_profile(path: Path | None = None) -> RuntimeProfile | None:
    target = path or profile_path()
    # ⚠️ **Construction is INSIDE the guard, and that was the bug.** Widening the catch to
    # `ValueError` covered `json.loads`, but `RuntimeProfile.from_dict` sat outside it — and it
    # raises `ValueError` too: an unknown `RuntimeMode`, a docker profile with no compose file, an
    # empty `default_tenant`. So a file that PARSED but did not validate escaped, out through the
    # unguarded call in `main.py`, and the application would not open, which a bad settings
    # file must never cause.
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
        if not isinstance(value, dict):
            return None
        return RuntimeProfile.from_dict(value)
    except (OSError, ValueError, TypeError):
        return None


def _write_json_atomically(target: Path, payload: object) -> Path:
    """Write `payload` to `target` through `recall.atomic_write` (LF, bytes, fsync'd).

    One implementation, because there were two and they had already diverged: `save_profile` wrote
    platform newlines and the pipeline writer (removed 2026-11-19) pinned LF, so on Windows two files in the same directory
    written by the same module disagreed. LF wins, matching every writer in the install path.

    Delegating to `atomic_write_bytes` fixed two more defects the local version carried: the
    temp name was `target.with_suffix(".tmp")`, which REPLACES the extension, so every save of
    `runtime.json` shared one `runtime.tmp` and two concurrent saves could promote each other's
    half-written JSON; and nothing was fsync'd, so a crash could publish an empty rename.
    """
    from recall.atomic_write import atomic_write_bytes

    target.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_bytes(target, (json.dumps(payload, indent=2) + "\n").encode("utf-8"))
    return target


def save_profile(profile: RuntimeProfile, path: Path | None = None) -> Path:
    return _write_json_atomically(path or profile_path(), profile.to_dict())


def read_token(profile: RuntimeProfile) -> str | None:
    """Read a token through keyring when configured, without persisting it in runtime.json."""
    if not profile.token_key:
        return None
    try:
        import keyring
    except ImportError:
        return None
    try:
        value = keyring.get_password("recall", profile.token_key)
        return str(value) if value is not None else None
    except Exception:  # BROAD-CATCH: fail-open
        return None
