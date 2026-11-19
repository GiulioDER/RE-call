"""Stable data contracts shared by the desktop UI and runtime adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from recall._env import strict_bool


class RuntimeMode(StrEnum):
    VPS_MCP = "vps_mcp"
    DOCKER = "docker"
    #: A PostgreSQL the user already runs, with no container anywhere. The MCP servers are ordinary
    #: local processes pointed at that database, which is what the wizard already registers for
    #: Claude Code — the desktop simply had no way to say it.
    LOCAL_DATABASE = "local_database"


class SourceCategory(StrEnum):
    DOCUMENTS = "documents"
    CODE = "code"
    MEMORY = "memory"


@dataclass(frozen=True)
class RuntimeProfile:
    mode: RuntimeMode
    endpoint: str | None = None
    compose_file: str | None = None
    compose_project: str | None = None
    #: The database for `LOCAL_DATABASE`. Held here rather than read from the environment so the
    #: UI shows what it will actually connect to, and so two profiles on one machine cannot end up
    #: fighting over one `RECALL_DSN`.
    dsn: str | None = None
    default_tenant: str = "default"
    shared_profile: str = "user"
    #: The embedder a `LOCAL_DATABASE` server resolves, handed to it as `RECALL_EMBEDDER`. Without
    #: it the server fell back to its own default (`fastembed`) whatever the database was built
    #: with. Docker carries the embedder in its compose file and a remote server in its own config.
    #: There is no `pinned_version`: nothing read or wrote it, so a `runtime.json` that still names
    #: one loads, and the key is dropped on the next save.
    embedder: str | None = None
    #: False stops the release checks the app makes by itself (on start and every 6 hours); the
    #: Settings page's "Check for updates" still works. There is no update channel: the check
    #: always reads the latest GitHub release, so a `runtime.json` that still names one loads,
    #: and the key is dropped on the next save.
    update_enabled: bool = True
    token_key: str | None = None

    def __post_init__(self) -> None:
        if self.mode is RuntimeMode.VPS_MCP and not self.endpoint:
            raise ValueError("a VPS MCP profile needs an endpoint")
        if self.mode is RuntimeMode.DOCKER and not self.compose_file:
            raise ValueError("a Docker profile needs a compose file")
        if self.mode is RuntimeMode.LOCAL_DATABASE and not self.dsn:
            raise ValueError("a local database profile needs a dsn")
        if not self.default_tenant.strip():
            raise ValueError("default_tenant must be non empty")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mode"] = self.mode.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuntimeProfile":
        raw_mode = value.get("mode", RuntimeMode.DOCKER)
        # It gates the app's own network calls: a hand-written "false" read through bool() was
        # True, and the release checks kept running (CCA ENV-004). A value that is not a boolean
        # turns them off rather than raising, because `load_profile` answers a ValueError by
        # replacing the whole profile with a default that has the checks on.
        raw_update = value.get("update_enabled", True)
        if isinstance(raw_update, bool):
            update_enabled = raw_update
        else:
            try:
                update_enabled = strict_bool(str(raw_update), name="update_enabled")
            except ValueError:
                update_enabled = False
        return cls(
            mode=RuntimeMode(raw_mode),
            endpoint=value.get("endpoint"),
            compose_file=value.get("compose_file"),
            compose_project=value.get("compose_project"),
            dsn=value.get("dsn"),
            default_tenant=str(value.get("default_tenant", "default")),
            shared_profile=str(value.get("shared_profile", "user")),
            embedder=value.get("embedder"),
            update_enabled=update_enabled,
            token_key=value.get("token_key"),
        )


#: Which corpus kind each category belongs in. The wizard's own vocabulary
#: (`recall.wizard.corpora.CorpusKind`) is docs/code/memory, and this is the desktop's half of that
#: agreement; `tests/test_desktop.py` pins the two together so a fourth kind cannot drift in on one
#: side only.
_KIND_BY_CATEGORY = {
    SourceCategory.DOCUMENTS: "docs",
    SourceCategory.CODE: "code",
    SourceCategory.MEMORY: "memory",
}


@dataclass(frozen=True)
class SourceSelection:
    category: SourceCategory
    paths: tuple[Path, ...]
    tenant: str
    shared: bool = False
    #: The scope the "all projects" choice maps to. Carried rather than hardcoded, because
    #: `RuntimeProfile.shared_profile` is configurable and everything ELSE in the UI already reads
    #: it from there; a literal here silently ingested into `user-*` for a profile that named a
    #: different shared scope, so display and calibration used one tenant while ingest used another.
    shared_profile: str = "user"

    def __post_init__(self) -> None:
        if not self.paths:
            raise ValueError("at least one source path is required")
        if not self.tenant.strip():
            raise ValueError("tenant must be non empty")

    @property
    def physical_tenant(self) -> str:
        """`{scope}-{kind}`, matching `recall.wizard.corpora.tenant_for`.

        ⚠️ **MEMORY used to map to `-docs`, and that sent the user's memory into the wrong corpus.**
        The wizard builds THREE corpora per project — docs, code and memory — and the memory one is
        the writable, never-calibrated scope that memory notes belong in. Mapping MEMORY onto
        `-docs` put them in the corpus that is production-routed, strict-trust and calibrated, which
        is both the wrong destination and the one place a stray write does the most harm.

        The wizard has always provisioned `<project>-memory`; nothing addressed it.
        """
        suffix = _KIND_BY_CATEGORY[self.category]
        base = self.shared_profile if self.shared else self.tenant
        return f"{base}-{suffix}"


@dataclass(frozen=True)
class JobStatus:
    job_id: str
    state: str
    message: str = ""
    files: int = 0
    chunks: int = 0
    progress: float | None = None
    error: str | None = None


@dataclass(frozen=True)
class CalibrationSnapshot:
    tenant: str
    generation_id: str | None
    status: str
    calibration_id: str | None = None
    threshold: float | None = None
    separability: float | None = None
    answerable: int | None = None
    unanswerable: int | None = None
    certified: bool | None = None
    message: str = ""
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ReleaseInfo:
    version: str
    url: str
    sha256: str | None = None
    asset_name: str | None = None

