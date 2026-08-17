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
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (FileNotFoundError, OSError, json.JSONDecodeError):
        return None
    if not isinstance(value, dict):
        return None
    return RuntimeProfile.from_dict(value)


def save_profile(profile: RuntimeProfile, path: Path | None = None) -> Path:
    target = path or profile_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(profile.to_dict(), indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target


def read_token(profile: RuntimeProfile) -> str | None:
    """Read a token through keyring when configured, without persisting it in runtime.json."""
    if not profile.token_key:
        return None
    try:
        import keyring
    except ImportError:
        return None
    try:
        return keyring.get_password("recall", profile.token_key)
    except Exception:
        return None

