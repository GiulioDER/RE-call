"""Stable data contracts shared by the desktop UI and runtime adapters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any


class RuntimeMode(StrEnum):
    VPS_MCP = "vps_mcp"
    DOCKER = "docker"


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
    default_tenant: str = "default"
    shared_profile: str = "user"
    pinned_version: str | None = None
    update_channel: str = "stable"
    update_enabled: bool = True
    token_key: str | None = None

    def __post_init__(self) -> None:
        if self.mode is RuntimeMode.VPS_MCP and not self.endpoint:
            raise ValueError("a VPS MCP profile needs an endpoint")
        if self.mode is RuntimeMode.DOCKER and not self.compose_file:
            raise ValueError("a Docker profile needs a compose file")
        if not self.default_tenant.strip():
            raise ValueError("default_tenant must be non empty")

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["mode"] = self.mode.value
        return value

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RuntimeProfile":
        raw_mode = value.get("mode", RuntimeMode.DOCKER)
        return cls(
            mode=RuntimeMode(raw_mode),
            endpoint=value.get("endpoint"),
            compose_file=value.get("compose_file"),
            compose_project=value.get("compose_project"),
            default_tenant=str(value.get("default_tenant", "default")),
            shared_profile=str(value.get("shared_profile", "user")),
            pinned_version=value.get("pinned_version"),
            update_channel=str(value.get("update_channel", "stable")),
            update_enabled=bool(value.get("update_enabled", True)),
            token_key=value.get("token_key"),
        )


@dataclass(frozen=True)
class SourceSelection:
    category: SourceCategory
    paths: tuple[Path, ...]
    tenant: str
    shared: bool = False

    def __post_init__(self) -> None:
        if not self.paths:
            raise ValueError("at least one source path is required")
        if not self.tenant.strip():
            raise ValueError("tenant must be non empty")

    @property
    def physical_tenant(self) -> str:
        suffix = "code" if self.category is SourceCategory.CODE else "docs"
        base = "user" if self.shared else self.tenant
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

