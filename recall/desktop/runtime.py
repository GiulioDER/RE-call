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

_STACK_MUTATING_VERBS = frozenset({"up", "down", "pull"})
"""Compose verbs after which the running stack may no longer match what the caches describe."""

_SLOW_VERBS = frozenset({"up", "pull", "build"})
"""Verbs that can legitimately take minutes, so they must not share the quick budget.

⚠️ **One 120s timeout governed every verb, and it could not cover the work this stack now does.**
Two independent reasons, either sufficient: the generated compose gives each tenant service a
`build:` stanza and nothing in the install path ever builds it, so the desktop's first `up` builds
LibreOffice plus a PyPI install; and the database healthcheck carries `start_period: 180s`, already
longer than the old cap, so `up --wait` was killed while Compose was still legitimately waiting.
Five auditors reached this from five directions.
"""

#: Sized above `start_period` (180s) + `interval x retries` (60s) with room for a cold image build.
#: `stack.bring_up` uses 300s for the database alone; this has to cover a build as well.
_SLOW_VERB_TIMEOUT = 1800

#: `config`, `ps`, `exec` — a read or a command inside a running container.
_QUICK_VERB_TIMEOUT = 120

_CORPUS_SUFFIXES = ("-docs", "-code", "-memory")
"""The corpus kinds a tenant scope can end in.

⚠️ **This must equal `get_args(recall.wizard.corpora.CorpusKind)`**, and it is pinned to it by
`tests/test_desktop.py::test_the_corpus_suffixes_match_the_wizards_kinds` rather than imported,
so the desktop package keeps no import dependency on the wizard and a divergence still fails a test.

It is a constant because the first version of this change spelled `("-docs", "-code")` inline in
three places and MISSED `-memory` in all three, which every install provisions.
"""


def _tenant_for_service(service: str) -> str | None:
    """The tenant scope a compose service serves, or None if it serves none.

    The legacy names are checked first because a naive prefix strip turns `recall-docs` into the
    scope `docs`, which carries no corpus suffix and would therefore be discarded by every caller:
    a legacy install would silently lose its `default` project from the scope list and never have
    its schema applied. (An earlier version of this comment claimed the strip would "invent a
    project called docs" instead. It cannot — the suffix filter drops it — and the real consequence
    is the opposite one.)
    """
    if service in _SERVICE_TENANTS:
        return _SERVICE_TENANTS[service]
    if service.startswith("recall-"):
        return service[len("recall-") :]
    return None


def _corpus_scope(service: str) -> str | None:
    """The tenant scope this service serves, only when it is a real corpus scope.

    One predicate, so `start()`, `list_tenants()` and the refusal message cannot disagree about
    what counts as a tenant-serving service. They did.
    """
    scope = _tenant_for_service(service)
    return scope if scope and scope.endswith(_CORPUS_SUFFIXES) else None


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
        # Built from the SAME predicate the rest of the class uses, not from `names - {"db"}`.
        # The loose form would advertise any sidecar the compose file carries as something this
        # call accepts, and it answered in service names when the argument refused was a scope.
        offered = ", ".join(sorted(filter(None, (_corpus_scope(name) for name in names)))) or "none"
        raise RuntimeErrorBase(
            f"the compose file defines no MCP service for tenant scope {tenant!r}; it serves "
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
        verb = args[0] if args else ""
        if verb in _STACK_MUTATING_VERBS:
            self._invalidate_topology()
        timeout = _SLOW_VERB_TIMEOUT if verb in _SLOW_VERBS else _QUICK_VERB_TIMEOUT
        try:
            return subprocess.run(
                command, check=True, capture_output=True, text=True, timeout=timeout
            )
        except subprocess.TimeoutExpired as exc:
            # Named separately, because "timed out" and "failed" need different answers from the
            # user and the old message could not tell them apart.
            raise RuntimeErrorBase(
                f"`docker compose {verb}` did not finish within {timeout}s. A first start builds "
                f"the recall image, which takes minutes; it may still be running in Docker."
            ) from exc
        except subprocess.CalledProcessError as exc:
            # ⚠️ **Include the stderr we already captured.** `str(CalledProcessError)` is only
            # "Command '[...]' returned non-zero exit status 1", so a failed image build, a pull
            # denial, a port collision and "dependency failed to start: container is unhealthy"
            # all reached the user as one identical sentence — on exactly the paths that now build
            # images and provision projects. `stack.bring_up` already extracts it; this copies it.
            detail = (exc.stderr or exc.stdout or "").strip()
            raise RuntimeErrorBase(
                f"`docker compose {verb}` failed: {detail[-400:] or exc}"
            ) from exc
        except (OSError, subprocess.SubprocessError) as exc:
            raise RuntimeErrorBase(f"Docker runtime failed: {exc}") from exc

    def _invalidate_topology(self) -> None:
        """Forget both caches, because the stack may no longer be the one they describe.

        ⚠️ **`_gateways` has to go too, and the first version of this change forgot it.** A gateway
        memoises a fully resolved `docker compose exec <service> ...` argv, not a lookup, so
        clearing only `_services` leaves the two disagreeing: re-running the wizard while the window
        is open and then reconnecting would refresh the topology while every already-connected
        tenant kept shelling out to a service that no longer exists, producing an opaque stdio
        failure instead of the deliberate refusal above.

        It lives in `_compose` rather than in `start()` because `apply_update()` restarts the stack
        too and was missed for exactly that reason. A third restart path should not be able to
        repeat it.
        """
        self._services = None
        for gateway in self._gateways.values():
            gateway.close()
        self._gateways.clear()

    def start(self) -> None:
        self._compose("up", "-d", "--wait")  # also invalidates the topology caches
        # ONE `schema apply`, not one per service. `schema apply` migrates a DATABASE, not a tenant
        # — `recall/cli.py` calls `apply_migrations(migration_dsn, table=..., dim=...)` with no
        # tenant argument — and every service in both stacks shares one migration DSN, set
        # unconditionally at `recall/wizard/stack.py::compose_document`. So the second and later
        # applies were byte-identical redundant work, and because the loop is sequential and each
        # iteration pays a container exec plus a CLI start, startup had become O(projects): roughly
        # 3-6s per service, so five projects turned a ~20s start into ~60s that grew with every
        # project the user added.
        migrated = sorted(filter(None, (_corpus_scope(name) for name in self._service_names())))
        if not migrated:
            # Not a clean start. The old hardcoded loop failed loudly here because `check=True` hit
            # a missing service; deriving the list would instead apply nothing and report ready.
            raise RuntimeErrorBase(
                "the compose file defines no tenant-serving MCP service, so no schema was applied; "
                "the stack cannot serve anything and needs to be provisioned by the wizard"
            )
        first = self._service_for_tenant(migrated[0])
        self._compose("exec", "-T", first, "recall", "schema", "apply")
        default_scope = self.profile.default_tenant
        if not default_scope.endswith(_CORPUS_SUFFIXES):
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

        Read from the compose file, so a project the wizard provisioned appears here. It used to be
        derived from a literal `("default-docs", "default-code")` pair, which meant the answer was
        `["default"]` whatever the stack contained: a project added through the UI vanished on the
        next start, and one the wizard had genuinely built never showed up at all.

        **The profile's default project is always offered, provisioned or not.** The selector needs
        one entry and `_populate_scopes` reinserts it regardless, so withholding it would only make
        the two disagree. Every OTHER project has to be in the compose file. An earlier version of
        this docstring said a project the wizard did not provision "is never offered", which the
        unconditional seed below contradicts.

        The shared profile stays out because the desktop adds it as an explicit "all projects" entry.

        An enumeration failure falls back to the default project rather than propagating, so an app
        that cannot ask still opens. Note this branch is close to unreachable through the UI: the
        only caller runs `start()` first, and `start()` calls `_service_names()` unguarded, so a
        real enumeration failure surfaces there with the compose error attached. (The rationale
        here used to be "this runs while the window is being built", which was never true of a
        caller — it runs on a worker thread, after `start()`.)
        """
        names = {self.profile.default_tenant}
        try:
            services = self._service_names()
        except RuntimeErrorBase:
            return sorted(names)
        for service in services:
            if scope := _corpus_scope(service):
                names.add(scope.rsplit("-", 1)[0])
        if self.profile.shared_profile != self.profile.default_tenant:
            # Guarded, because a profile that names the same scope for both would otherwise leave
            # the list EMPTY. The window recovers, since `_populate_scopes` reinserts the default,
            # but a runtime that answers "no projects" for a stack that has one is not something to
            # rely on a caller to paper over.
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
