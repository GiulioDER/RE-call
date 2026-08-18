"""Runtime adapters for VPS MCP and local Docker deployments."""

from __future__ import annotations

import asyncio
import base64
import json
import os
import subprocess
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Protocol

from recall.desktop.models import CalibrationSnapshot, JobStatus, ReleaseInfo, RuntimeProfile, SourceSelection
from recall.desktop.profiles import read_token
from recall.desktop.updates import latest_release


class RuntimeErrorBase(RuntimeError):
    """Base error surfaced by a runtime adapter."""


class ToolGateway(Protocol):
    def call(self, name: str, arguments: dict[str, Any]) -> Any: ...

    def close(self) -> None: ...


class SdkMcpGateway:
    """Synchronous facade over the MCP Python SDK, suitable for worker threads."""

    def __init__(self, profile: RuntimeProfile, command: list[str] | None = None) -> None:
        self.profile = profile
        self.command = command
        self._closed = False

    def call(self, name: str, arguments: dict[str, Any]) -> Any:
        if self._closed:
            raise RuntimeErrorBase("MCP session is closed")
        return asyncio.run(self._call(name, arguments))

    async def _call(self, name: str, arguments: dict[str, Any]) -> Any:
        from mcp.client.session import ClientSession

        if self.command is not None:
            from mcp.client.stdio import StdioServerParameters, stdio_client

            params = StdioServerParameters(command=self.command[0], args=self.command[1:], env=os.environ.copy())
            async with stdio_client(params) as streams:
                async with ClientSession(*streams) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments)
                    return _tool_result(result)

        from mcp.client.streamable_http import streamable_http_client

        headers = {}
        token = read_token(self.profile)
        if token:
            headers["Authorization"] = f"Bearer {token}"
        try:
            import httpx2

            http_client = httpx2.AsyncClient(headers=headers)
        except ImportError:
            http_client = None
        async with streamable_http_client(
            self.profile.endpoint or "", http_client=http_client
        ) as streams:
            async with ClientSession(*streams) as session:
                await session.initialize()
                result = await session.call_tool(name, arguments)
                return _tool_result(result)

    def close(self) -> None:
        self._closed = True


def _tool_result(result: Any) -> Any:
    if getattr(result, "is_error", False):
        raise RuntimeErrorBase(str(result))
    pieces = getattr(result, "content", None) or []
    for piece in pieces:
        text = getattr(piece, "text", None)
        if text is not None:
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                return text
    return result


class RuntimeManager(ABC):
    def __init__(self, profile: RuntimeProfile, gateway: ToolGateway | None = None) -> None:
        self.profile = profile
        self.gateway = gateway
        self._job_tenants: dict[str, str] = {}
        self._jobs: dict[str, JobStatus] = {}

    @abstractmethod
    def start(self) -> None: ...

    @abstractmethod
    def stop(self) -> None: ...

    @abstractmethod
    def health(self) -> dict[str, Any]: ...

    def list_tenants(self) -> list[str]:
        result = self._call("recall_tenants", {})
        if isinstance(result, dict):
            tenants = result.get("tenants", [])
        else:
            tenants = result
        return [str(value) for value in tenants]

    def start_ingest(self, selection: SourceSelection) -> JobStatus:
        payload = {
            "category": selection.category.value,
            "tenant": selection.physical_tenant,
            "shared": selection.shared,
            "files": [_encoded_file(path) for path in selection.paths],
        }
        physical_tenant = selection.physical_tenant
        result = self._call_for(physical_tenant, "recall_ingest", payload)
        job = _job_from_result(result, physical_tenant)
        self._job_tenants[job.job_id] = physical_tenant
        self._jobs[job.job_id] = job
        return job

    def job_status(self, job_id: str) -> JobStatus:
        result = self._call_for(self._job_tenants.get(job_id, self.profile.default_tenant), "recall_job_status", {"job_id": job_id})
        status = _job_from_result(result, job_id)
        if status.state == "unknown" and job_id in self._jobs:
            return self._jobs[job_id]
        self._jobs[job_id] = status
        return status

    def calibration_status(self, tenant: str) -> CalibrationSnapshot:
        result = self._call_for(tenant, "recall_calibration_status", {"tenant": tenant})
        if not isinstance(result, dict):
            return CalibrationSnapshot(tenant=tenant, generation_id=None, status="unknown", message=str(result))
        return CalibrationSnapshot(
            tenant=tenant,
            generation_id=result.get("generation_id"),
            status=str(result.get("status", "unknown")),
            calibration_id=result.get("calibration_id"),
            threshold=result.get("threshold"),
            separability=result.get("separability"),
            answerable=result.get("answerable"),
            unanswerable=result.get("unanswerable"),
            certified=result.get("certified"),
            message=str(result.get("message", "")),
            raw=result,
        )

    def run_calibration(self, tenant: str) -> CalibrationSnapshot:
        result = self._call_for(tenant, "recall_calibration_run", {"tenant": tenant})
        return self.calibration_status(tenant) if result is None else _calibration_from_result(tenant, result)

    def publish_calibration(self, tenant: str, calibration_id: str) -> CalibrationSnapshot:
        self._call_for(tenant, "recall_calibration_publish", {"tenant": tenant, "calibration_id": calibration_id})
        return self.calibration_status(tenant)

    def check_update(self) -> ReleaseInfo:
        """Read signed release metadata through the desktop update channel."""
        return latest_release()

    def apply_update(self, release: ReleaseInfo) -> dict[str, Any]:
        """Apply a managed local update, or fail closed for an unconfigured VPS coordinator."""
        del release
        if isinstance(self, VpsMcpRuntime):
            raise RuntimeErrorBase(
                "VPS updates require an explicitly configured deployment coordinator; MCP read/write "
                "credentials cannot deploy a server release."
            )
        raise RuntimeErrorBase("Docker update must be applied by the Docker runtime adapter")

    def _call(self, name: str, arguments: dict[str, Any]) -> Any:
        if self.gateway is None:
            raise RuntimeErrorBase("runtime is not started")
        return self.gateway.call(name, arguments)

    def _call_for(self, tenant: str, name: str, arguments: dict[str, Any]) -> Any:
        del tenant
        return self._call(name, arguments)


class VpsMcpRuntime(RuntimeManager):
    def start(self) -> None:
        self.gateway = self.gateway or SdkMcpGateway(self.profile)
        self.health()

    def stop(self) -> None:
        if self.gateway:
            self.gateway.close()
        self.gateway = None

    def health(self) -> dict[str, Any]:
        result = self._call("recall_stats", {})
        return result if isinstance(result, dict) else {"status": "ready", "details": result}


_LEGACY_SERVICES = {
    "default-docs": "recall-docs",
    "default-code": "recall-code",
    "user-docs": "recall-user-docs",
    "user-code": "recall-user-code",
}
"""Service names used by the hand-written `docker-compose.desktop.yml` that shipped before the
wizard. The wizard's generated stack uses `recall-<tenant>` throughout; these are the aliases that
keep an existing install working."""

_SERVICE_TENANTS = {service: tenant for tenant, service in _LEGACY_SERVICES.items()}


def _tenant_for_service(service: str) -> str | None:
    """The tenant scope a compose service serves, or None if it serves none.

    The legacy names are checked first, because `recall-docs` would otherwise read as the scope
    `docs` and invent a project by that name in the scope selector.
    """
    if service in _SERVICE_TENANTS:
        return _SERVICE_TENANTS[service]
    if service.startswith("recall-"):
        return service[len("recall-") :]
    return None


class DockerRuntime(RuntimeManager):
    def __init__(self, profile: RuntimeProfile, gateway: ToolGateway | None = None) -> None:
        super().__init__(profile, gateway)
        self._process: subprocess.Popen[str] | None = None
        self._gateways: dict[str, ToolGateway] = {}
        self._services: frozenset[str] | None = None

    def _service_names(self) -> frozenset[str]:
        """The services this compose file actually defines, asked of Compose itself.

        Not parsed here, for two reasons. Compose is the authority on what its own file declares,
        so anything this module parsed could disagree with what `exec` will accept; and the
        alternative needs a YAML parser, which the `desktop` extra does not install — the wizard
        writes JSON, but `docker-compose.desktop.yml` is real YAML.

        Cached, because this sits under every tool call and a subprocess per call is not free.
        `start()` clears it, which is also the hook for a project added after the window opened.
        """
        if self._services is None:
            result = self._compose("config", "--services")
            self._services = frozenset(
                line.strip() for line in result.stdout.splitlines() if line.strip()
            )
        return self._services

    def _service_for_tenant(self, tenant: str) -> str:
        """Map a tenant scope onto a compose service, by asking the file rather than a literal map.

        ⚠️ **This was a hardcoded four-entry dict, and it made the UI unable to drive the stack the
        wizard installs.** The wizard names a service `recall-<tenant>`, so `recall-default-docs`;
        the dict asked for `recall-docs`, which that file does not contain. Measured against a
        generated document: every scope missed, including the default one, so the mismatch was not
        confined to projects the user added.

        `_LEGACY_SERVICES` keeps `docker-compose.desktop.yml` working. Note that only the `default`
        project differs between the two schemes — `user-docs` is `recall-user-docs` under both.
        """
        names = self._service_names()
        for candidate in (f"recall-{tenant}", _LEGACY_SERVICES.get(tenant)):
            if candidate and candidate in names:
                return candidate
        offered = ", ".join(sorted(names - {"db"})) or "none"
        raise RuntimeErrorBase(
            f"the compose file defines no MCP service for tenant scope {tenant!r}; it offers "
            f"{offered}. A project has to be provisioned by the wizard before it can be served."
        )

    def _gateway_for(self, tenant: str) -> ToolGateway:
        if self.gateway is not None:
            return self.gateway
        if tenant in self._gateways:
            return self._gateways[tenant]
        compose = self.profile.compose_file
        if not compose:
            raise RuntimeErrorBase("Docker compose file is not configured")
        command = ["docker", "compose", "-f", compose]
        if self.profile.compose_project:
            command.extend(["-p", self.profile.compose_project])
        command.extend(["exec", "-T", self._service_for_tenant(tenant), "python", "-m", "recall_mcp.server"])
        gateway = SdkMcpGateway(self.profile, command=command)
        self._gateways[tenant] = gateway
        return gateway

    def _compose(self, *args: str) -> subprocess.CompletedProcess[str]:
        compose = self.profile.compose_file
        if not compose:
            raise RuntimeErrorBase("Docker compose file is not configured")
        command = ["docker", "compose", "-f", compose]
        if self.profile.compose_project:
            command.extend(["-p", self.profile.compose_project])
        command.extend(args)
        try:
            return subprocess.run(command, check=True, capture_output=True, text=True, timeout=120)
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeErrorBase(f"Docker runtime failed: {exc}") from exc

    def start(self) -> None:
        self._compose("up", "-d", "--wait")
        self._services = None  # the file may have gained services since the window opened
        for service in sorted(self._service_names() - {"db"}):
            self._compose("exec", "-T", service, "recall", "schema", "apply")
        default_scope = self.profile.default_tenant
        if not default_scope.endswith(("-docs", "-code")):
            default_scope = f"{default_scope}-docs"
        self._call_for(default_scope, "recall_stats", {})
        self.health()

    def stop(self) -> None:
        if self.gateway:
            self.gateway.close()
        for gateway in self._gateways.values():
            gateway.close()
        self._gateways.clear()
        self.gateway = None

    def health(self) -> dict[str, Any]:
        result = self._compose("ps", "--format", "json")
        return {"status": "ready", "compose": result.stdout, "mcp": "ready" if self._gateways or self.gateway else "starting"}

    def list_tenants(self) -> list[str]:
        """The projects the managed local stack can actually serve.

        Read from the compose file, so a project the wizard provisioned appears here and one it did
        not is never offered. It used to be derived from a literal `("default-docs", "default-code")`
        pair, which meant the answer was `["default"]` whatever the stack contained: a project added
        through the UI vanished on the next start, and one the wizard had genuinely built never
        showed up at all.

        The shared profile stays out because the desktop adds it as an explicit "all projects" entry.

        An enumeration failure falls back to the default project rather than propagating. This runs
        while the window is being built, and being unable to ask is not the same as there being
        nothing: refusing here would leave the user with an app that will not open, when every
        operation that needs a service still raises on its own.
        """
        names = {self.profile.default_tenant}
        try:
            services = self._service_names()
        except RuntimeErrorBase:
            return sorted(names)
        for service in services:
            scope = _tenant_for_service(service)
            if scope and scope.endswith(("-docs", "-code")):
                names.add(scope.rsplit("-", 1)[0])
        names.discard(self.profile.shared_profile)
        return sorted(names)

    def _call_for(self, tenant: str, name: str, arguments: dict[str, Any]) -> Any:
        return self._gateway_for(tenant).call(name, arguments)

    def apply_update(self, release: ReleaseInfo) -> dict[str, Any]:
        """Pull the pinned compose images, restart the managed stack, and recheck readiness."""
        del release
        self._compose("pull")
        self._compose("up", "-d", "--wait")
        return self.health()


def _encoded_file(path: Path) -> dict[str, str]:
    try:
        data = path.read_bytes()
    except OSError as exc:
        raise RuntimeErrorBase(f"cannot read {path.name}: {exc}") from exc
    return {"name": path.name, "content_b64": base64.b64encode(data).decode("ascii")}


def _job_from_result(result: Any, fallback: str) -> JobStatus:
    value = result if isinstance(result, dict) else {"message": str(result)}
    raw_progress = value.get("progress")
    progress = float(raw_progress) if isinstance(raw_progress, (int, float)) else None
    return JobStatus(
        job_id=str(value.get("job_id", fallback)),
        state=str(value.get("state", "completed")),
        message=str(value.get("message", "")),
        files=int(value.get("files", 0)),
        chunks=int(value.get("chunks", 0)),
        progress=progress,
        error=value.get("error"),
    )


def _calibration_from_result(tenant: str, result: Any) -> CalibrationSnapshot:
    value = result if isinstance(result, dict) else {}
    return CalibrationSnapshot(
        tenant=tenant,
        generation_id=value.get("generation_id"),
        status=str(value.get("status", "draft")),
        calibration_id=value.get("calibration_id"),
        threshold=value.get("threshold"),
        separability=value.get("separability"),
        answerable=value.get("answerable"),
        unanswerable=value.get("unanswerable"),
        certified=value.get("certified"),
        message=str(value.get("message", "")),
        raw=value,
    )


def create_runtime(profile: RuntimeProfile, gateway: ToolGateway | None = None) -> RuntimeManager:
    if profile.mode.value == "vps_mcp":
        return VpsMcpRuntime(profile, gateway)
    return DockerRuntime(profile, gateway)
