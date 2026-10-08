"""`recall setup --lite`: memory for this project in one SQLite file, with no server and no Docker.

Four steps, each reported, none hidden:

1. open (or create) the store at `<project>/.recall/memory.db`;
2. index the memory folder into it, the same `Indexer` every other path uses;
3. calibrate it automatically, so strict search can answer as soon as there is enough text;
4. register the MCP server for THIS project only (Claude Code's local scope), unless asked not to.

Local scope, for the reasons `recall.wizard.wiring.register_local_scope` gives: no approval
prompt, and no server answering about this memory inside some other repository. The entry
launches the interpreter running this command, so the client cannot pick up a different Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, cast

from recall.lite import LITE_DSN_PREFIX, LiteStore

if TYPE_CHECKING:
    from recall.embeddings import Embedder
    from recall.store import PgVectorStore
    from recall.wizard.wiring import LocalScopeRegistration

__all__ = ["LiteSetupReport", "SERVER_NAME", "default_store_path", "lite_dsn", "run_lite_setup"]

#: The MCP server name this registers. One per project, since local scope is keyed by project.
SERVER_NAME = "recall-memory"


#: What the server needs from the OS when a client starts it with ONLY its registered `env`, which
#: some clients do (measured with `claude -p --mcp-config`): without APPDATA, Python cannot find the
#: user site-packages ("No module named 'anyio'"); without SystemRoot, Windows sockets fail
#: (WinError 10106); the temp, cache and home variables are where an embedding model and the
#: embedding cache live. Measured 2026-10-08 by starting the registered server with exactly this env
#: and listing its tools. PATH is deliberately NOT here: the server runs no programs, and a PATH
#: frozen at setup time would go stale. Where the client merges instead, these are the same values.
SERVER_OS_ENV = (
    "APPDATA", "LOCALAPPDATA", "SystemRoot", "TEMP", "TMP", "USERPROFILE", "HOMEDRIVE", "HOMEPATH",
    "HOME", "XDG_CACHE_HOME", "XDG_DATA_HOME",
)


def server_os_env() -> dict[str, str]:
    """The variables in `SERVER_OS_ENV` that are set here, plus PYTHONSAFEPATH.

    PYTHONSAFEPATH=1 keeps the session's working directory off `sys.path`: a client starts the
    server in the session's directory, and a project that holds its own `recall/` package would
    otherwise shadow the installed one.
    """
    import os

    found = {name: os.environ[name] for name in SERVER_OS_ENV if os.environ.get(name)}
    return {**found, "PYTHONSAFEPATH": "1"}


def default_store_path(project_root: Path) -> Path:
    return project_root / ".recall" / "memory.db"


def lite_dsn(path: Path) -> str:
    return LITE_DSN_PREFIX + path.resolve().as_posix()


@dataclass
class LiteSetupReport:
    store: Path
    memory: Path
    files: int = 0
    chunks: int = 0
    skipped: int = 0
    calibration: str = ""
    calibration_reason: str = ""
    registration: LocalScopeRegistration | None = None
    notes: list[str] = field(default_factory=list)

    def render(self) -> str:
        lines = [
            f"store        {self.store}",
            f"dsn          {lite_dsn(self.store)}   (the plugin's Database DSN, if you use the plugin)",
            f"memory       {self.memory}",
            f"indexed      {self.chunks} chunk(s) from {self.files} file(s); {self.skipped} unchanged",
            f"calibration  {self.calibration}: {self.calibration_reason}",
        ]
        reg = self.registration
        if reg is None:
            lines.append("claude code  not registered (--no-register)")
        elif reg.recorded:
            lines.append(f"claude code  registered {', '.join(reg.registered)} for this project in {reg.config_path}")
        elif reg.conflicts:
            names = ", ".join(f"{name} ({owner})" for name, owner in reg.conflicts)
            lines.append(f"claude code  NOT registered: {names} already exists and was left alone")
        else:
            lines.append(f"claude code  NOT registered: {reg.skipped_reason}")
        lines.extend(f"note         {note}" for note in self.notes)
        return "\n".join(lines)


def run_lite_setup(
    *,
    project_root: Path,
    memory: Path,
    embedder: Embedder,
    embedder_name: str,
    store_path: Path | None = None,
    register: bool = True,
    config_path: Path | None = None,
    interpreter: str | None = None,
) -> LiteSetupReport:
    """Build, calibrate and register a lite store for `project_root`. See the module docstring."""
    from recall.index import Indexer
    from recall.lite.calibration import ensure_calibrated

    project_root = project_root.resolve()
    memory = memory.resolve()
    if not memory.is_dir():
        raise NotADirectoryError(f"the memory folder {memory} does not exist")
    path = (store_path or default_store_path(project_root)).resolve()
    report = LiteSetupReport(store=path, memory=memory)
    with LiteStore(path, dim=embedder.dim) as store:
        # Typed as the Postgres store, as everywhere the shared Indexer takes a lite one: it calls
        # only the indexing methods `LiteStore` implements (see `recall.backends`).
        stats = Indexer(cast("PgVectorStore", store), embedder).index_path(memory)
        report.files, report.chunks, report.skipped = stats.files, stats.chunks, stats.skipped
        outcome = ensure_calibrated(store, embedder)
        report.calibration, report.calibration_reason = outcome.status.value, outcome.reason
    if register:
        from recall.wizard.wiring import ServerBlock, register_local_scope

        block = ServerBlock(
            name=SERVER_NAME,
            tenant="default",
            env={
                **server_os_env(),
                "RECALL_DSN": lite_dsn(path),
                "RECALL_EMBEDDER": embedder_name,
                "RECALL_INDEX_ROOT": str(memory),
                "RECALL_TENANT": "default",
            },
            rationale="a lite store: one SQLite file in this project, calibrated by indexing",
        )
        report.registration = register_local_scope(
            (block,), project_root=project_root, config_path=config_path, interpreter=interpreter
        )
    if path.is_relative_to(project_root):
        ignore = project_root / ".gitignore"
        listed = ignore.exists() and any(
            line.strip() in {".recall", ".recall/", "/.recall", "/.recall/"}
            for line in ignore.read_text(encoding="utf-8", errors="replace").splitlines()
        )
        if not listed:
            report.notes.append("add `.recall/` to .gitignore: the store holds your memory's text and vectors")
    return report
