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

import os
import shlex
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
#: some clients do: without APPDATA, Python cannot find the
#: user site-packages ("No module named 'anyio'"); without SystemRoot, Windows sockets fail
#: (WinError 10106); the temp, cache and home variables are where an embedding model and the
#: embedding cache live. Measured 2026-10-08 by starting the registered server with exactly this env
#: and listing its tools. PATH is deliberately NOT here: the server runs no programs, and a PATH
#: frozen at setup time would go stale. Claude Code 2.1.238 and 2.1.293 were measured that day to
#: merge the block over their own environment; the replacement demonstration used a direct
#: process started with only the registered environment. Keep passthrough as insurance.
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


def _embedding_environment(
    embedder: Embedder, embedder_name: str, *, registration_required: bool
) -> dict[str, str] | None:
    """Snapshot reproducible model settings, without rendering credentials in the report."""
    from recall.embeddings import artifact_is_pinned, embedder_is_hosted, embedding_profile
    from recall.embedding_registry import find_registered_profile, registered_profile

    identity = embedding_profile(embedder)
    selected = os.environ.get("RECALL_EMBED_PROFILE", "").strip()
    values = {"RECALL_EMBEDDER": embedder_name, "RECALL_EMBED_PROFILE": selected}
    resolved_entry = find_registered_profile(identity.profile_id)
    if resolved_entry is not None and resolved_entry.backend == "voyage-multimodal":
        if registration_required:
            raise ValueError("multimodal profile registration is unavailable in lite setup; use a text profile")
        return None
    if not selected:
        if (resolved_entry is not None and artifact_is_pinned(identity)) or embedder_is_hosted(embedder):
            if registration_required:
                raise ValueError("set the resolved RECALL_EMBED_PROFILE and its required settings before registration")
            return None  # a library caller may index a supplied model without registering it
        return values
    try:
        if selected != identity.profile_id:
            raise ValueError("RECALL_EMBED_PROFILE does not match the resolved embedder")
        entry = registered_profile(selected)
        if entry.hosted:
            credential = os.environ.get(entry.api_key_env)
            if not credential:
                raise ValueError(f"{entry.api_key_env} is required to reproduce this profile in the server")
            values[entry.api_key_env] = credential
        else:
            location = os.environ.get(entry.artifact_path_env)
            digest = os.environ.get("RECALL_MODEL_SHA256")
            if not location or digest != identity.artifact_digest:
                raise ValueError("the resolved profile needs matching artifact path and RECALL_MODEL_SHA256")
            values[entry.artifact_path_env] = location
            values["RECALL_MODEL_SHA256"] = digest
    except ValueError:
        if registration_required:
            raise
        return None  # indexing an explicit model does not require reproducible server settings
    return values


def _followup_commands(env: dict[str, str], memory: Path) -> list[str]:
    """Render commands for the current shell family; credentials stay in client configuration."""
    if os.name == "nt":
        assignments = []
        for key, value in env.items():
            quoted = value.replace("'", "''")
            assignments.append(f"$env:{key} = '{quoted}'")
        quoted_memory = str(memory).replace("'", "''")
        return assignments + ["recall doctor", f"recall dashboard --root '{quoted_memory}'"]
    commands = ["recall doctor", f"recall dashboard --root {shlex.quote(str(memory))}"]
    prefix = "env " + " ".join(shlex.quote(f"{key}={value}") for key, value in env.items())
    return [f"{prefix} {command}" for command in commands]


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
    #: Only non-secret settings needed by the printed CLI followups.
    followup_env: dict[str, str] = field(default_factory=dict)

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
        if self.followup_env:
            lines.append("\nFrom this project, check the store and open its read-only dashboard:")
            lines.extend(f"  {command}" for command in _followup_commands(self.followup_env, self.memory))
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
    from recall.cache import default_cache
    from recall.context import context_policy_for_profile
    from recall.embeddings import embedding_profile_id
    from recall.lite.calibration import ensure_calibrated
    from recall.lite.writer_lock import lite_lane

    project_root = project_root.resolve()
    memory = memory.resolve()
    if not memory.is_dir():
        raise NotADirectoryError(f"the memory folder {memory} does not exist")
    path = (store_path or default_store_path(project_root)).resolve()
    report = LiteSetupReport(store=path, memory=memory)
    model_env = _embedding_environment(embedder, embedder_name, registration_required=register)
    local_env = {
        "RECALL_DSN": lite_dsn(path), "RECALL_SERVING_DSN": lite_dsn(path),
        "RECALL_INDEX_ROOT": str(memory), "RECALL_TENANT": "default",
        "RECALL_ENV": "development", "RECALL_INDEX_MODE": "legacy",
        "RECALL_TRANSPORT": "stdio", "RECALL_TRUST_MODE": "strict",
    }
    if model_env is not None:
        local_env.update(model_env)
        safe_keys = {
            "RECALL_DSN", "RECALL_SERVING_DSN", "RECALL_INDEX_ROOT", "RECALL_TENANT",
            "RECALL_ENV", "RECALL_INDEX_MODE", "RECALL_TRANSPORT",
            "RECALL_TRUST_MODE", "RECALL_EMBEDDER", "RECALL_EMBED_PROFILE", "RECALL_MODEL_CACHE",
            "RECALL_MODEL_SHA256", "RECALL_QWEN_MODEL_PATH",
        }
        report.followup_env = {key: value for key, value in local_env.items() if key in safe_keys}
    with LiteStore(path, dim=embedder.dim) as store:
        existing_model = store.generation_binding().get("embedder_model")
        if existing_model and existing_model != embedder.name:
            raise ValueError("this store was calibrated with another embedder; choose a different --store file")
        # Typed as the Postgres store, as everywhere the shared Indexer takes a lite one: it calls
        # only the indexing methods `LiteStore` implements (see `recall.backends`).
        with lite_lane(store.path, purpose="setup"):
            with default_cache() as cache:
                stats = Indexer(
                    cast("PgVectorStore", store), embedder, cache=cache,
                    context_policy=context_policy_for_profile(embedding_profile_id(embedder)),
                ).index_path(memory)
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
                **local_env,
            },
            rationale="a lite store: one SQLite file in this project, calibrated by indexing",
        )
        report.registration = register_local_scope(
            (block,), project_root=project_root, config_path=config_path, interpreter=interpreter
        )
    if path.is_relative_to(project_root):
        ignore = project_root / ".gitignore"
        relative = path.relative_to(project_root).as_posix()
        in_default_directory = path.is_relative_to(project_root / ".recall")
        literal = "".join("\\" + char if char in r"\*?[" else char for char in relative)
        if literal.startswith(("!", "#")):
            literal = "\\" + literal
        target = ".recall/" if in_default_directory else f"{literal}*"
        patterns = {".recall", ".recall/", "/.recall", "/.recall/"} if in_default_directory else {target, f"/{target}"}
        listed = ignore.exists() and any(
            line.strip() in patterns
            for line in ignore.read_text(encoding="utf-8", errors="replace").splitlines()
        )
        if not listed:
            report.notes.append(
                f"add `{target}` to .gitignore: the store holds your memory's text and vectors; "
                "include its SQLite -wal and -shm sidecars"
            )
    return report
