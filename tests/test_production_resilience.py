from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from recall.ops.backup import receipt_from_metadata
from recall.ops.health import HealthController, route_response
from recall.ops.restore import CutoverGuard, validate_restored_database
from recall.ops.restore_drill import _validation_dsn
from recall.ops.secrets import AwsSecretsManagerProvider
from recall_mcp.limits import Rate, RateLimited, RateLimiter, RateLimiterUnavailable, RedisRateLimiter


class _Probe:
    def __init__(self, *, rls: bool = True) -> None:
        self.rls = rls

    def check_schema(self) -> None:
        return None

    def check_rls_effective(self) -> bool:
        return self.rls

    def active_generation_id(self) -> str:
        return "generation"


def test_health_routes_keep_liveness_independent_and_redis_non_gating() -> None:
    controller = HealthController()
    assert route_response(controller, "livez") == (200, {"status": "ok"})
    assert route_response(controller, "readyz")[0] == 503
    state = {"health_probe": _Probe(), "limiter": object()}
    controller.mark_started(state)
    status, payload = route_response(controller, "readyz")
    assert status == 200
    assert payload["checks"]["rate_limiter"] == "configured"


def test_health_database_failure_is_unready_but_liveness_stays_up() -> None:
    controller = HealthController()
    controller.mark_started({"health_probe": None, "limiter": None})
    assert route_response(controller, "readyz")[0] == 503
    assert route_response(controller, "livez")[0] == 200


def test_health_readiness_checks_every_configured_tenant_probe() -> None:
    controller = HealthController()
    controller.mark_started(
        {
            "health_probes": [_Probe(), _Probe(rls=False)],
            "generation_mode": True,
            "enterprise_readiness_ok": True,
            "limiter": None,
        }
    )
    status, payload = route_response(controller, "readyz")
    assert status == 503
    assert payload["checks"]["tenants"] == "2"
    assert "rls:1" in payload["failures"]


def test_redis_limiter_uses_hashed_tenant_keys_and_local_read_fallback(monkeypatch: pytest.MonkeyPatch) -> None:
    local = RateLimiter({"read": Rate(1, 1)})
    asyncio.run(local.check_async("tenant", "read"))
    with pytest.raises(RateLimited):
        local.check("tenant", "read")
    limiter = RedisRateLimiter("redis://invalid", {"read": Rate(2, 1)}, fallback_read_budget=1)
    bucket, idem = limiter._keys("tenant/a", "read", "request-1")
    assert "tenant/a" not in bucket
    assert "request-1" not in idem
    assert bucket.startswith("recall:rate:default:")


def test_redis_limiter_zero_read_fallback_fails_closed() -> None:
    class BrokenRedis:
        async def script_load(self, _script: str) -> str:
            raise OSError("redis unavailable")

    limiter = RedisRateLimiter(
        "redis://invalid", {"read": Rate(2, 1)}, fallback_read_budget=0, redis_client=BrokenRedis()
    )
    with pytest.raises(RateLimiterUnavailable):
        asyncio.run(limiter.check("tenant", "read", read_only=True))


def test_redis_limiter_rejects_nonfinite_configuration() -> None:
    with pytest.raises(ValueError):
        RedisRateLimiter("redis://invalid", {"read": Rate(1, 1)}, fallback_read_budget=float("inf"))


def test_backup_receipt_never_contains_secret_values() -> None:
    receipt = receipt_from_metadata(
        database_identifier="cluster",
        backup_identifier="snapshot",
        backup_type="pitr",
        region="eu-west-1",
        active_generation="generation",
        row_counts={"chunks": 10},
        checksums={"chunks": "abc"},
        configuration={"dsn": "redacted"},
    )
    payload = json.loads(receipt.json())
    assert payload["configuration_fingerprint"] != "redacted"
    assert "password" not in receipt.json().lower()


def test_cutover_requires_freeze_and_retains_rollback_target() -> None:
    guard = CutoverGuard(active_target="production")
    with pytest.raises(RuntimeError):
        guard.cutover("restore", confirmation="CUTOVER_RESTORED_CLUSTER")
    guard.freeze_writes()
    guard.cutover("restore", confirmation="CUTOVER_RESTORED_CLUSTER")
    guard.rollback(confirmation="ROLLBACK_RESTORE")
    assert guard.active_target == "production"


def test_secret_provider_returns_version_without_logging_or_transforming_value() -> None:
    class Client:
        def get_secret_value(self, **kwargs: str) -> dict[str, str]:
            return {"SecretString": "value", "VersionId": "v2", "ARN": "arn:secret"}

    value = AwsSecretsManagerProvider(client=Client()).get("db")
    assert value.value == "value"
    assert value.version_id == "v2"


def test_restore_validation_reports_structural_failures() -> None:
    class Cursor:
        def __init__(self, sql: str) -> None:
            self.sql = sql

        def __enter__(self) -> "Cursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str) -> None:
            self.sql = sql

        def fetchone(self) -> tuple[object]:
            if "max(version)" in self.sql:
                return ("0001",)
            return (False,)

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor("")

    result = validate_restored_database(Connection(), expected_schema_version="0002")
    assert not result.passed
    assert "schema" in result.failures


def test_restore_validation_requires_forced_rls_and_real_checksum_provider() -> None:
    class Cursor:
        def __init__(self) -> None:
            self.sql = ""

        def __enter__(self) -> "Cursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str) -> None:
            self.sql = sql

        def fetchone(self) -> tuple[object]:
            if "max(version)" in self.sql:
                return ("0002",)
            if "relforcerowsecurity" in self.sql:
                return (True,)
            return (True,)

    class Connection:
        def cursor(self) -> Cursor:
            return Cursor()

    result = validate_restored_database(
        Connection(),
        expected_schema_version="0002",
        expected_checksums={"chunks": "expected"},
        checksum_provider=lambda _connection: {"chunks": "actual"},
    )
    assert not result.passed
    assert "checksums" in result.failures


def test_restore_validation_runs_tenant_bound_serving_checks() -> None:
    class Cursor:
        def __init__(self, connection: "Connection") -> None:
            self.connection = connection
            self.sql = ""

        def __enter__(self) -> "Cursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: tuple[object, ...] = ()) -> None:
            self.sql = sql
            self.connection.calls.append((sql, params))

        def fetchone(self) -> tuple[object]:
            if "set_config" in self.sql:
                return ("tenant-a",)
            if "current_user" in self.sql and "pg_roles" not in self.sql:
                return ("recall_server",)
            if "max(version)" in self.sql:
                return ("0023",)
            if "extname = 'vector'" in self.sql:
                return (True,)
            if "relforcerowsecurity" in self.sql:
                return (True,)
            if "pg_indexes" in self.sql:
                return (True,)
            if "active_generation_id" in self.sql:
                return ("gen-a",)
            return (True,)

    class Connection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def cursor(self) -> Cursor:
            return Cursor(self)

    connection = Connection()
    result = validate_restored_database(
        connection,
        expected_schema_version="0023",
        expected_tenant="tenant-a",
        expected_generation="gen-a",
        expected_role="recall_server",
        representative_chunk_id="chunk-a",
    )

    assert result.passed
    assert result.checks == {
        "role": True,
        "grants": True,
        "schema": True,
        "pgvector": True,
        "rls": True,
        "indexes": True,
        "tenant": True,
        "active_generation": True,
        "calibration": True,
        "authenticated_search": True,
        "representative_retrieval": True,
    }
    generation_calls = [call for call in connection.calls if "active_generation_id" in call[0]]
    assert generation_calls
    assert generation_calls[0][1] == ("tenant-a",)
    assert "WHERE s.tenant_id = %s" in generation_calls[0][0]


def test_restore_validation_rejects_cross_tenant_generation_match() -> None:
    class Cursor:
        def __init__(self, connection: "Connection") -> None:
            self.connection = connection
            self.sql = ""

        def __enter__(self) -> "Cursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: tuple[object, ...] = ()) -> None:
            self.sql = sql
            self.connection.calls.append((sql, params))

        def fetchone(self) -> tuple[object]:
            if "set_config" in self.sql:
                return ("tenant-a",)
            if "current_user" in self.sql and "pg_roles" not in self.sql:
                return ("recall_server",)
            if "max(version)" in self.sql:
                return ("0023",)
            if "active_generation_id" in self.sql:
                return (None,)
            return (True,)

    class Connection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def cursor(self) -> Cursor:
            return Cursor(self)

    result = validate_restored_database(
        Connection(),
        expected_schema_version="0023",
        expected_tenant="tenant-a",
        expected_generation="gen-from-tenant-b",
        representative_chunk_id="chunk-a",
    )

    assert not result.passed
    assert "active_generation" in result.failures
    assert "calibration" in result.failures
    assert "authenticated_search" in result.failures
    assert "representative_retrieval" in result.failures


def test_restore_validation_dsn_is_bound_to_returned_cluster(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECALL_RESTORE_VALIDATION_DSN", "postgresql://user:pass@{host}:{port}/recall")
    assert _validation_dsn({"endpoint": "restored.example", "port": 5433}) == (
        "postgresql://user:pass@restored.example:5433/recall"
    )
    monkeypatch.setenv("RECALL_RESTORE_VALIDATION_DSN", "postgresql://user:pass@fixed/recall")
    with pytest.raises(RuntimeError, match=r"contain \{host\}"):
        _validation_dsn({"endpoint": "restored.example", "port": 5432})


def test_terraform_reference_contains_private_two_az_resilience_stack() -> None:
    root = Path(__file__).parents[1] / "infra" / "aws"
    assert (root / "rds.tf").read_text(encoding="utf-8").count("aws_rds_cluster_instance") >= 1
    assert "backup_retention_period         = 35" in (root / "rds.tf").read_text(encoding="utf-8")
    assert "deployment_circuit_breaker" in (root / "ecs.tf").read_text(encoding="utf-8")
    assert "object_lock_enabled" in (root / "s3.tf").read_text(encoding="utf-8")
