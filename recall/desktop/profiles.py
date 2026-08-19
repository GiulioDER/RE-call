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
    except (OSError, ValueError):  # ValueError also covers UnicodeDecodeError; see load_pipelines
        return None
    if not isinstance(value, dict):
        return None
    return RuntimeProfile.from_dict(value)


def _write_json_atomically(target: Path, payload: object) -> Path:
    """Write `payload` to `target` through a sibling temporary, then rename over it.

    One implementation, because there were two and they had already diverged: `save_profile` wrote
    platform newlines and `save_pipelines` pinned LF, so on Windows two files in the same directory
    written by the same module disagreed. LF wins, matching every writer in the install path.
    """
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n")
    temporary.replace(target)
    return target


def save_profile(profile: RuntimeProfile, path: Path | None = None) -> Path:
    return _write_json_atomically(path or profile_path(), profile.to_dict())


def pipeline_path() -> Path:
    """Where the per-corpus pipeline choices live, beside `runtime.json`.

    A separate file rather than a field on `RuntimeProfile`, because they answer different
    questions: the profile says how to REACH recall (mode, compose file, endpoint, tenant), and this
    says what the retrieval stack should BE. The wizard writes the profile and knows nothing about
    embedder or reranker choices; keeping them apart means neither has to load the other's concerns.
    """
    return profile_path().with_name("pipeline.json")


def load_pipelines(path: Path | None = None) -> dict[str, dict[str, object]]:
    """The saved pipeline choices, or an empty mapping.

    Empty for anything unreadable, on the same reasoning as the wizard's state file: a corrupt
    settings file must not stop the application starting. The worst case of ignoring it is the user
    re-entering their choices; the worst case of refusing is an app that will not open.
    """
    target = path or pipeline_path()
    try:
        value = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        # ValueError, not json.JSONDecodeError. UnicodeDecodeError is a ValueError and NOT a
        # JSONDecodeError, so a file that is not valid UTF-8 — a torn write, a file saved as
        # UTF-16 — used to propagate out of `MainWindow.__init__` and stop the app opening, which
        # is the exact outcome the docstring above promises it prevents. JSONDecodeError is a
        # ValueError too, so this is strictly wider.
        return {}
    if not isinstance(value, dict):
        return {}
    return {
        str(source_type): dict(settings)
        for source_type, settings in value.items()
        if isinstance(settings, dict)
    }


def save_pipelines(configs: dict[str, dict[str, object]], path: Path | None = None) -> Path:
    """Persist the pipeline choices, atomically.

    ⚠️ **This exists because the UI told the user "Configuration saved" and did not save.** The
    choices lived in an in-memory dict, so the status line asserted a state nothing had created and
    reopening the app restored the defaults. Demonstrated before fixing: set the embedder model,
    press Save, close the window, reopen, and the field reads `hashing-64` again.
    """
    return _write_json_atomically(path or pipeline_path(), configs)


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
    except Exception:
        return None
