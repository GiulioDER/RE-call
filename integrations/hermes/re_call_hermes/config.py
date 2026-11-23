"""The provider's settings: `<hermes_home>/re-call/config.json`, and the child's environment.

The default Lite store is shared by every agent of the user, in `<XDG_CONFIG_HOME, else ~/.config>/re-call/`.
The config file, the logs, the mirror journal and the embedding cache stay in `<hermes_home>/re-call/`."""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import tempfile
import time
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

DATA_DIR = "re-call"
STORE_FILE = "memory.db"
MEMO_DIR = "memos"
CONFIG_FILE = "config.json"
ASSISTANT_SUBFOLDER = "hermes"

DEFAULTS: dict[str, Any] = {
    "recall_command": "recall-mcp",
    "dsn": "",
    "memo_folder": "",
    "k": 5,
    "mirror_builtin": True,
}

# What the child needs from this environment: the OS and cache variables docs/LITE.md lists, plus
# PATH, TMPDIR and the locale; the model cache and the RE-call model settings `recall setup` used,
# so the server finds the model setup cached; and the proxy and CA variables, so a first download
# works behind a proxy; and the source policy `recall setup` applied, so the server's refresh and
# search keep denied files out. Nothing else of Hermes's environment, its provider keys included,
# reaches the child.
PASSTHROUGH_ENV = (
    "PATH", "SYSTEMROOT", "SystemRoot", "APPDATA", "LOCALAPPDATA", "TEMP", "TMP", "TMPDIR",
    "USERPROFILE", "HOMEDRIVE", "HOMEPATH", "HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME", "LANG", "LC_ALL",
    "HF_HOME", "HF_HUB_OFFLINE", "FASTEMBED_CACHE_PATH",
    "RECALL_MODEL_CACHE", "RECALL_MODEL_SHA256", "RECALL_EMBED_PROFILE", "RECALL_SOURCE_POLICY_FILE",
    "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY", "https_proxy", "http_proxy", "no_proxy",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
)
# The embedding cache holds vectors of the memo text; it stays in the profile's data folder, with the
# logs, even when the store is shared.
EMBED_CACHE_FILE = "embeddings.sqlite"
# What a non-Lite store's server registers (RE-call's `search` preset), and so all it is offered.
SEARCH_TOOLS = ("recall_search", "recall_evidence")
_TRUE, _FALSE = {"true", "1", "yes", "on"}, {"false", "0", "no", "off"}
# A password in a URL query (`?password=`) or in a libpq keyword DSN (`password=...`).
_DSN_PASSWORD = re.compile(r"(^|[\s?&])password\s*=", re.IGNORECASE)
logger = logging.getLogger(__name__)
# Old profile stores already reported by this process: settings are loaded on every probe, and each
# notice is worth one line.
_NOTIFIED: set[Path] = set()


def atomic_json_write(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=path.name, suffix=".tmp")
    with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    # Windows refuses to replace a file another process has open for reading; that lasts moments.
    for attempt in range(50):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 49:
                raise
            time.sleep(0.02)


def _which_on_path(name: str) -> str | None:
    """`name` in PATH's absolute entries only. Windows' own search, and any relative PATH entry,
    would try the current directory first, so a project Hermes runs in could plant the server."""
    exts = [""]
    if os.name == "nt":
        pathext = [e for e in os.environ.get("PATHEXT", ".COM;.EXE;.BAT;.CMD").split(";") if e]
        if Path(name).suffix.upper() not in {e.upper() for e in pathext}:  # `python3.12` still needs `.exe`
            exts = pathext
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if not entry or not os.path.isabs(entry):
            continue
        for ext in exts:
            candidate = Path(entry) / (name + ext)
            if candidate.is_file() and os.access(candidate, os.X_OK):
                return str(candidate)
    return None


def _as_bool(value: Any) -> bool:
    """A JSON bool, or the string `hermes memory setup` saves for it."""
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in _TRUE or text in _FALSE:
        return text in _TRUE
    raise ValueError(f"not a boolean: {value!r}")


def _absolute(path: Path, base: Path) -> Path:
    """A user path made absolute against the provider's data folder, which is the child's cwd too."""
    path = path.expanduser()
    return path if path.is_absolute() else base / path


def shared_dir() -> Path:
    """The per-user store directory every agent uses by default. A relative XDG_CONFIG_HOME is
    ignored, as the XDG specification says."""
    xdg = os.environ.get("XDG_CONFIG_HOME", "")
    path = Path(xdg)
    return (path if path.is_absolute() else Path.home() / ".config") / DATA_DIR


def _shared_dir_or_none() -> Path | None:
    """shared_dir(), or None where no home directory can be found (a service with no HOME): the
    shared store then has no path, which problem() reports instead of raising."""
    try:
        return shared_dir()
    except RuntimeError:
        return None


NO_HOME = ("no home directory was found, so the shared store has no path; set XDG_CONFIG_HOME to an "
           "absolute path, or set dsn")


def _store_dir(data_dir: Path) -> Path:
    """Where the default store lives: the shared directory, unless only this profile's old store exists."""
    shared, old = shared_dir(), data_dir / STORE_FILE
    if not old.is_file():
        return shared
    if (shared / STORE_FILE).is_file():
        _notify(old, f"re-call: using the shared store {shared / STORE_FILE}; this profile's old store {old} "
                     f"is no longer read (set dsn to sqlite:///{old.as_posix()} to keep using it)")
        return shared
    _notify(old, f"re-call: using this profile's old store {old}, not the shared {shared / STORE_FILE}; to "
                 f"share it, close every agent, then move {STORE_FILE} with its {STORE_FILE}-wal and "
                 f"{STORE_FILE}-shm sidecars and {MEMO_DIR}/ from {data_dir} to {shared}")
    return data_dir


def normal_path(path: Path) -> str:
    """A path as two of the provider's checks compare it: resolved and case-normalized."""
    return os.path.normcase(str(path.resolve()))


def _same_path(a: Path, b: Path) -> bool:
    return normal_path(a) == normal_path(b)


def _notify(old: Path, line: str) -> None:
    if old not in _NOTIFIED:
        _NOTIFIED.add(old)
        logger.warning(line)


def read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        return {}
    return value if isinstance(value, dict) else {}


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    recall_command: str
    dsn: str
    memo_folder: Path
    k: int
    mirror_builtin: bool
    config_error: str = ""

    @property
    def lite(self) -> bool:
        return self.dsn.startswith("sqlite:///")

    @property
    def store_path(self) -> Path | None:
        return Path(self.dsn[len("sqlite:///"):]) if self.lite else None

    @property
    def assistant_folder(self) -> Path:
        return self.memo_folder / ASSISTANT_SUBFOLDER

    def resolve_argv(self) -> list[str] | None:
        """The server's argv with an absolute executable, or None when it cannot be found.

        `recall_command` is one executable (`recall-mcp`) or an interpreter with arguments
        (`/path/to/python -m recall_mcp.server`, the form docs/LITE.md gives for other clients).
        """
        raw = self.recall_command.strip()
        whole = Path(raw).expanduser()
        if whole.is_absolute() and whole.is_file():
            return [str(whole)]
        argv = [part.strip('"') for part in shlex.split(raw, posix=os.name != "nt")]
        if not argv:
            return None
        head = Path(argv[0]).expanduser()
        if head.is_absolute():
            found = str(head) if head.is_file() else None
        elif len(head.parts) > 1:
            found = None  # a relative path would resolve against whatever directory Hermes runs in
        else:
            found = _which_on_path(argv[0])
        return [found, *argv[1:]] if found else None

    def problem(self) -> str:
        """Why this configuration cannot start, or "" when it can. Reads the disk, never the network."""
        if self.config_error:
            return self.config_error
        if self.resolve_argv() is None:
            return (
                f"the RE-call server command {self.recall_command!r} was not found; install "
                "recall-rag[fastembed,mcp]>=0.15.0 and set recall_command to its recall-mcp"
            )
        if not self.lite and (urlsplit(self.dsn).password or _DSN_PASSWORD.search(self.dsn)):
            return "the dsn holds a password; keep it in ~/.pgpass instead of the Hermes config"
        store = self.store_path
        shared = _shared_dir_or_none()
        if store is not None and shared is not None and _same_path(store, shared / STORE_FILE) != _same_path(
                self.memo_folder, shared / MEMO_DIR):
            return (
                f"the shared store {shared / STORE_FILE} and the shared memo folder {shared / MEMO_DIR} go "
                f"together, but this configuration pairs {store} with {self.memo_folder}; leave dsn and "
                "memo_folder blank to share, or set both to a store and memo folder of their own"
            )
        if store is not None and not store.is_file():
            return (
                f"no RE-call store at {store}; create it with: recall --embedder fastembed setup "
                f"--lite --memory {self.memo_folder} --store {store} --no-register"
            )
        return ""

    def child_env(self, base: Mapping[str, str]) -> dict[str, str]:
        env = {k: base[k] for k in PASSTHROUGH_ENV if k in base}
        env.update(
            PYTHONSAFEPATH="1",
            PYTHONIOENCODING="utf-8",
            RECALL_SERVING_DSN=self.dsn,
            RECALL_EMBEDDER="fastembed",
            RECALL_TENANT="default",
            RECALL_ENV="development",
            RECALL_TRUST_MODE="strict",
            RECALL_TRANSPORT="stdio",
            RECALL_EMBED_CACHE=str(self.data_dir / EMBED_CACHE_FILE),
        )
        if self.lite:
            env.update(
                RECALL_INDEX_ROOT=str(self.memo_folder),
                RECALL_INDEX_MODE="legacy",
                RECALL_ASSISTANT_MEMOS=str(self.assistant_folder),
                RECALL_MCP_TOOLS="assistant",
            )
        else:
            env["RECALL_MCP_TOOLS"] = " ".join(SEARCH_TOOLS)
        return env


def load_settings(hermes_home: str | Path) -> Settings:
    data_dir = Path(hermes_home) / DATA_DIR
    error = ""
    try:
        values = {**DEFAULTS, **read_json(data_dir / CONFIG_FILE)}
        k = int(values["k"])
        mirror_builtin = _as_bool(values["mirror_builtin"])
    except (ValueError, TypeError) as exc:  # reported by problem(), so activation refuses with a reason
        error = f"{data_dir / CONFIG_FILE} is not valid: {exc}"
        values, k, mirror_builtin = dict(DEFAULTS), int(DEFAULTS["k"]), bool(DEFAULTS["mirror_builtin"])
    dsn = str(values["dsn"] or "")
    # The shared store indexes only <shared>/memos, so a custom memo folder keeps a store under the
    # profile, as does an explicit dsn for its memos.
    memos = _absolute(Path(values["memo_folder"]), data_dir) if values["memo_folder"] else None
    shared = _shared_dir_or_none()
    custom_memos = memos is not None and (shared is None or not _same_path(memos, shared / MEMO_DIR))
    if shared is None and not dsn and not custom_memos:
        error = error or NO_HOME
        store_dir = data_dir  # a placeholder path; problem() refuses with NO_HOME before it is used
    else:
        store_dir = data_dir if dsn or custom_memos else _store_dir(data_dir)
    if not dsn or dsn.startswith("sqlite:///"):
        store = _absolute(Path(dsn[len("sqlite:///"):]), data_dir) if dsn else store_dir / STORE_FILE
        dsn = "sqlite:///" + store.as_posix()
        if shared is not None and _same_path(store, shared / STORE_FILE):
            store_dir = store.parent  # a dsn naming the shared store gets its memo root too
    memo_folder = memos or store_dir / MEMO_DIR
    return Settings(
        data_dir=data_dir,
        recall_command=str(values["recall_command"] or DEFAULTS["recall_command"]),
        dsn=dsn,
        memo_folder=memo_folder,
        k=max(1, min(k, 20)),
        mirror_builtin=mirror_builtin,
        config_error=error,
    )


def save_settings(values: Mapping[str, Any], hermes_home: str | Path) -> None:
    path = Path(hermes_home) / DATA_DIR / CONFIG_FILE
    merged = {**read_json(path), **{k: v for k, v in values.items() if k in DEFAULTS}}
    atomic_json_write(path, merged)
