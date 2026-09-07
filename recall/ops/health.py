"""Small, dependency safe health controllers for ECS and load balancers."""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

from recall.observability import get_logger

_log = get_logger("ops.health")


@dataclass
class HealthController:
    """Own startup and cached readiness state without exposing corpus data or secrets."""

    cache_seconds: float = 5.0
    started: bool = False
    startup_error: str | None = None
    runtime_state: dict[str, object] | None = None
    _last_check: float = field(default=0.0, init=False, repr=False)
    _last_ready: bool = field(default=False, init=False, repr=False)
    _last_detail: dict[str, Any] = field(default_factory=dict, init=False, repr=False)

    def mark_starting(self) -> None:
        self.started = False
        self.startup_error = None

    def mark_started(self, state: dict[str, object]) -> None:
        self.runtime_state = state
        self.started = True
        self.startup_error = None
        self._last_check = 0.0

    def mark_failed(self, exc: BaseException) -> None:
        self.started = False
        self.startup_error = type(exc).__name__
        self.runtime_state = None

    def mark_stopped(self) -> None:
        self.started = False
        self.runtime_state = None

    def liveness(self) -> tuple[int, dict[str, object]]:
        return 200, {"status": "ok"}

    def startup(self) -> tuple[int, dict[str, object]]:
        if self.started:
            versions = {}
            if self.runtime_state is not None:
                raw_versions = self.runtime_state.get("secret_versions", {})
                if isinstance(raw_versions, dict):
                    versions = dict(raw_versions)
            return 200, {"status": "started", "secret_versions": versions}
        detail: dict[str, object] = {"status": "starting"}
        if self.startup_error is not None:
            detail.update({"status": "failed", "error": self.startup_error})
            return 503, detail
        return 503, detail

    def readiness(self) -> tuple[int, dict[str, object]]:
        if not self.started or self.runtime_state is None:
            return 503, {"status": "starting", "database": "not_checked"}
        now = time.monotonic()
        if now - self._last_check < self.cache_seconds:
            return (200 if self._last_ready else 503), dict(self._last_detail)

        state = self.runtime_state
        probe = state.get("health_probe")
        checks: dict[str, str] = {"database": "ok", "schema": "ok", "rls": "ok"}
        failures: list[str] = []
        try:
            if probe is None:
                raise RuntimeError("no database probe")
            check_schema = getattr(probe, "check_schema")
            check_schema()
            check_rls = getattr(probe, "check_rls_effective")
            if not check_rls():
                checks["rls"] = "failed"
                failures.append("rls")
            if state.get("generation_mode") and not state.get("active_generation"):
                checks["active_generation"] = "failed"
                failures.append("active_generation")
            if state.get("enterprise_readiness_ok") is False:
                checks["calibration"] = "failed"
                failures.append("calibration")
            else:
                checks["calibration"] = "ok"
        except Exception as exc:  # BROAD-CATCH: fail-closed
            checks["database"] = "failed"
            checks["schema"] = "unknown"
            failures.append("database")
            _log.warning("readiness probe failed: %s", type(exc).__name__)

        # Redis is deliberately reported but never gates database readiness. Reads have a bounded
        # fallback budget and writes fail closed when the shared limiter is unavailable.
        limiter = state.get("limiter")
        checks["rate_limiter"] = "configured" if limiter is not None else "disabled"
        detail: dict[str, object] = {
            "status": "ok" if not failures else "failed",
            "checks": checks,
            "failures": failures,
        }
        self._last_check = now
        self._last_ready = not failures
        self._last_detail = detail
        return (200 if self._last_ready else 503), detail


def route_response(controller: HealthController, name: str) -> tuple[int, dict[str, object]]:
    """Return a route result for tests and the MCP custom route adapter."""
    if name == "livez":
        return controller.liveness()
    if name == "startupz":
        return controller.startup()
    if name == "readyz":
        return controller.readiness()
    raise ValueError(f"unknown health route: {name}")
