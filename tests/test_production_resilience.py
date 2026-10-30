from __future__ import annotations

import asyncio
import json
import threading
import uuid
from pathlib import Path

import pytest

from recall.ops.backup import receipt_from_metadata
from recall.ops.health import HealthController, route_response
from recall.ops.restore import CutoverGuard, validate_restored_database
from recall.ops.restore_drill import (
    ChecksumPolicy,
    _checksum_policy,
    _checksum_provider,
    _validation_dsn,
)
from recall.ops.secrets import AwsSecretsManagerProvider
from recall.errors import IdempotencyConflict
from recall_mcp.server import _durable_replay, _mutation_fingerprint, _record_mutation_result
from recall_mcp.limits import (
    IdempotencyResultMissing,
    Rate,
    RateLimited,
    RateLimiter,
    RateLimiterUnavailable,
    RedisRateLimiter,
)
from tests.conftest import requires_db


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
    assert payload["checks"]["tenant_probes"] == "2"
    assert "rls:1" in payload["failures"]


def test_health_readiness_checks_shared_control_plane_once() -> None:
    class ControlPlaneProbe:
        def __init__(self) -> None:
            self.calls = 0

        def check_readiness(self) -> None:
            self.calls += 1

    control_plane = ControlPlaneProbe()
    controller = HealthController(cache_seconds=0)
    controller.mark_started(
        {
            "control_plane": control_plane,
            "health_probes": [_Probe(), _Probe()],
            "limiter": None,
        }
    )
    status, payload = route_response(controller, "readyz")
    assert status == 200
    assert control_plane.calls == 1
    assert payload["checks"]["control_plane"] == "ok"
    assert payload["checks"]["tenant_probes"] == "2"


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


def test_idempotency_identity_is_shared_across_budgets() -> None:
    limiter = RedisRateLimiter("redis://unused", {"read": Rate(2, 1), "write": Rate(2, 1)})
    assert limiter._keys("tenant/a", "read", "request-1")[1] == limiter._keys(
        "tenant/a", "write", "request-1"
    )[1]


def test_idempotency_key_reuse_across_operations_is_a_conflict() -> None:
    class Redis:
        async def script_load(self, _script: str) -> str:
            return "sha"

        async def evalsha(self, _sha: str, _keys: int, *_args: object) -> list[int]:
            return [0, 0, 2]

        async def get(self, key: str) -> bytes | None:
            if ":idempotency-result:" in key:
                return None
            return b'{"operation":"recall_index","request_fingerprint":"old"}'

        async def mget(self, keys: list[str]) -> list[bytes | None]:
            return [await self.get(key) for key in keys]

    limiter = RedisRateLimiter(
        "redis://unused", {"write": Rate(2, 1)}, redis_client=Redis()
    )
    with pytest.raises(IdempotencyConflict):
        asyncio.run(
            limiter.check(
                "tenant",
                "write",
                idempotency_key="request-1",
                idempotency_operation="recall_forget",
                idempotency_fingerprint="new",
            )
        )


def test_redis_limiter_zero_read_fallback_fails_closed() -> None:
    class BrokenRedis:
        async def script_load(self, _script: str) -> str:
            raise OSError("redis unavailable")

    limiter = RedisRateLimiter(
        "redis://invalid", {"read": Rate(2, 1)}, fallback_read_budget=0, redis_client=BrokenRedis()
    )
    with pytest.raises(RateLimiterUnavailable):
        asyncio.run(limiter.check("tenant", "read", read_only=True))


def test_redis_limiter_exposes_missing_mutation_result_for_durable_recovery() -> None:
    class Redis:
        async def script_load(self, _script: str) -> str:
            return "sha"

        async def evalsha(self, _sha: str, _keys: int, *_args: object) -> list[int]:
            return [0, 0, 2]

        async def get(self, _key: str) -> None:
            return None

        async def mget(self, _keys: list[str]) -> list[None]:
            return [None, None]

    limiter = RedisRateLimiter(
        "redis://unused", {"write": Rate(2, 1)}, redis_client=Redis()
    )
    with pytest.raises(IdempotencyResultMissing):
        asyncio.run(limiter.check("tenant", "write", idempotency_key="request-1"))


def test_redis_and_local_limiters_share_permanent_oversized_cost_semantics() -> None:
    """An impossible cost is rejected before either backend can start a retry cycle.

    Invariant: local and Redis limiters report the same permanent refusal with no retry delay.
    Failure mode: Redis reaches Lua, receives a zero wait, and the caller retries forever at the
    one millisecond floor. Red proof: this node is intended to fail against the current
    ``RedisRateLimiter.check`` until its preflight guard is restored; the production symbol
    targeted is that method's backend dispatch before ``script_load``.
    """
    local = RateLimiter({"write": Rate(2, 1)})
    with pytest.raises(RateLimited) as local_error:
        local.check("tenant", "write", cost=3)

    class Redis:
        calls = 0

        async def script_load(self, _script: str) -> str:
            self.calls += 1
            return "sha"

        async def evalsha(self, _sha: str, _keys: int, *_args: object) -> list[int]:
            self.calls += 1
            # A stale Lua script can still return the old zero wait. The Python boundary must
            # reject the impossible request before reaching that script.
            return [0, -1, 0]

    client = Redis()
    redis = RedisRateLimiter("redis://unused", {"write": Rate(2, 1)}, redis_client=client)
    with pytest.raises(RateLimited) as redis_error:
        asyncio.run(redis.check("tenant", "write", cost=3))

    assert local_error.value.retry_after_seconds == 0.0
    assert redis_error.value.retry_after_seconds == 0.0
    assert "can never succeed" in str(local_error.value)
    assert str(redis_error.value) == str(local_error.value)
    assert client.calls == 0


@requires_db
def test_postgres_receipt_recovers_after_redis_result_write_failure(make_store) -> None:
    """A committed receipt must recover the response when the Redis cache write fails.

    BUG-002 (CCA run 20261030T101709Z) changed the contract this test pins: the cache write is best
    effort once the durable receipt exists, so the committed mutation no longer surfaces the Redis
    error. Red proof: failed on 3ada1afd0, where recall_mcp.server._record_mutation_result awaited
    the cache write bare, with the injected OSError raised out of _record_mutation_result.
    """

    class Redis:
        def __init__(self) -> None:
            self.markers: set[str] = set()
            self.result_write_attempts = 0

        async def script_load(self, _script: str) -> str:
            return "sha"

        async def evalsha(self, _sha: str, _numkeys: int, _bucket: str, idem: str, *_args: object):
            if idem and idem in self.markers:
                return [0, 0, 2]
            if idem:
                self.markers.add(idem)
            return [1, 0, 0]

        async def get(self, _key: str) -> None:
            return None

        async def mget(self, _keys: list[str]) -> list[None]:
            return [None, None]

        async def set(self, key: str, _value: str, *, px: int) -> None:
            assert ":idempotency-result:" in key
            assert px > 0
            self.result_write_attempts += 1
            raise OSError("injected Redis result-write failure")

    store = make_store(64)
    redis = Redis()
    limiter = RedisRateLimiter(
        "redis://unused", {"write": Rate(10, 1)}, redis_client=redis
    )
    operation = "recall_index"
    key = "receipt-boundary-" + uuid.uuid4().hex
    fingerprint = _mutation_fingerprint(operation, {"path": "memory"})
    result = '{"files": 1, "chunks": 2}'
    state = {"store": store, "limiter": limiter}

    asyncio.run(
        limiter.check(
            store.tenant,
            "write",
            idempotency_key=key,
            idempotency_operation=operation,
            idempotency_fingerprint=fingerprint,
        )
    )
    asyncio.run(
        _record_mutation_result(
            state,
            store.tenant,
            key,
            result,
            idempotency_operation=operation,
            idempotency_fingerprint=fingerprint,
        )
    )
    assert redis.result_write_attempts == 1

    with pytest.raises(IdempotencyResultMissing):
        asyncio.run(
            limiter.check(
                store.tenant,
                "write",
                idempotency_key=key,
                idempotency_operation=operation,
                idempotency_fingerprint=fingerprint,
            )
        )
    assert (
        asyncio.run(_durable_replay(store, key, operation, fingerprint)) == result
    )


def test_durable_replay_moves_synchronous_receipt_reads_off_the_event_loop() -> None:
    caller_thread = threading.get_ident()

    class Store:
        def get_operation_receipt(
            self, _key: str, *, operation: str, request_fingerprint: str
        ) -> str:
            assert operation == "recall_index"
            assert request_fingerprint == "fingerprint"
            assert threading.get_ident() != caller_thread
            return "result"

    assert asyncio.run(_durable_replay(Store(), "key", "recall_index", "fingerprint")) == "result"


def test_redis_replay_rejects_a_different_request_fingerprint() -> None:
    class Redis:
        def __init__(self) -> None:
            self.value: str | None = None

        async def set(self, _key: str, value: str, *, px: int) -> None:
            self.value = value

        async def get(self, _key: str) -> str | None:
            return self.value

    redis = Redis()
    limiter = RedisRateLimiter(
        "redis://unused", {"write": Rate(2, 1)}, redis_client=redis
    )
    asyncio.run(
        limiter.store_idempotency_result(
            "tenant",
            "key",
            "result",
            operation="recall_index",
            request_fingerprint="fingerprint-a",
        )
    )
    with pytest.raises(IdempotencyConflict):
        asyncio.run(
            limiter.get_idempotency_result(
                "tenant",
                "key",
                operation="recall_index",
                request_fingerprint="fingerprint-b",
            )
        )


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


def test_secret_bootstrap_applies_values_and_returns_only_versions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import recall.ops.secrets as secrets

    class Provider:
        def __init__(self, **_kwargs: object) -> None:
            self.calls: list[str] = []

        def resolve_env(self, mapping):
            self.calls.extend(mapping.values())
            return {
                env_name: secrets.SecretValue(env_name, "secret-value", f"version-{index}")
                for index, env_name in enumerate(mapping, start=1)
            }

    provider = Provider()
    env = {
        "RECALL_AWS_SECRET_MAPPING": (
            '{"RECALL_SERVING_DSN":"database-secret","RECALL_REDIS_URL":"redis-secret"}'
        )
    }
    versions = secrets.apply_aws_secret_mapping(env, provider=provider)

    assert provider.calls == ["database-secret", "redis-secret"]
    assert versions == {
        "RECALL_SERVING_DSN": "version-1",
        "RECALL_REDIS_URL": "version-2",
    }
    assert env["RECALL_SERVING_DSN"] == "secret-value"
    assert env["RECALL_REDIS_URL"] == "secret-value"


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


def test_restore_checksum_defaults_to_bounded_primary_key_prefix(monkeypatch: pytest.MonkeyPatch) -> None:
    """Bound checksum work while preserving deterministic source and restore digests.

    Invariant: the default restore drill checksum query must carry a bound and primary key order.
    Failure mode: a growing corpus silently turns recovery validation back into an unbounded full
    table scan. Red proof: after this implementation, the production symbol ``_checksum_provider``
    was temporarily mutated to always use ``_CHECKSUM_QUERIES``; this node then failed its
    ``LIMIT %s`` assertion. The bounded implementation passes the same node.
    """
    monkeypatch.delenv("RECALL_RESTORE_CHECKSUM_MODE", raising=False)
    monkeypatch.delenv("RECALL_RESTORE_CHECKSUM_LIMIT", raising=False)
    assert _checksum_policy() == ChecksumPolicy(mode="bounded", max_rows_per_table=10000)

    class Cursor:
        def __init__(self, connection: "Connection") -> None:
            self.connection = connection
            self.sql = ""
            self.params: tuple[object, ...] = ()

        def __enter__(self) -> "Cursor":
            return self

        def __exit__(self, *args: object) -> None:
            return None

        def execute(self, sql: str, params: tuple[object, ...] = ()) -> None:
            self.sql = sql
            self.params = params
            self.connection.calls.append((sql, params))

        def fetchone(self) -> tuple[str]:
            return ("digest",)

    class Connection:
        def __init__(self) -> None:
            self.calls: list[tuple[str, tuple[object, ...]]] = []

        def cursor(self) -> Cursor:
            return Cursor(self)

    connection = Connection()
    durations: dict[str, float] = {}
    result = _checksum_provider(
        connection,
        {"recall_chunks_v1": "digest"},
        policy=_checksum_policy(),
        durations_ms=durations,
    )(connection)

    assert result == {"recall_chunks_v1": "digest"}
    assert "ORDER BY tenant_id, generation_id, chunk_id LIMIT %s" in connection.calls[0][0]
    assert connection.calls[0][1] == (10000,)
    assert durations.keys() == {"recall_chunks_v1"}


def test_restore_validation_runs_tenant_bound_serving_checks() -> None:
    """The restore receipt names the probe as direct database validation, never authenticated HTTP.

    Invariant: the validator's receipt must distinguish a direct SQL probe from the MCP HTTP path.
    Failure mode: an operator could treat a successful database query as proof of OIDC and HTTP
    behavior. Red proof: test node ``tests/test_production_resilience.py::test_restore_validation_runs_tenant_bound_serving_checks``
    was run against the pre fix implementation at commit ``8ec55226`` with the expectation
    changed to ``direct_database_search``; it failed because the production symbol
    ``validate_restored_database`` emitted ``authenticated_search``. The restored implementation
    passes the same node.
    """
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
        "direct_database_search": True,
        "representative_retrieval": True,
    }
    generation_calls = [call for call in connection.calls if "active_generation_id" in call[0]]
    assert generation_calls
    assert generation_calls[0][1] == ("tenant-a",)
    assert "WHERE s.tenant_id = %s" in generation_calls[0][0]
    rls_calls = [call for call in connection.calls if "relforcerowsecurity" in call[0]]
    assert rls_calls
    assert "pg_attribute" in rls_calls[0][0]
    assert "pg_policy" in rls_calls[0][0]
    assert "recall_chunks_v1" not in rls_calls[0][0]
    grant_calls = [call for call in connection.calls if "has_table_privilege" in call[0]]
    assert grant_calls
    assert "pg_attribute" in grant_calls[0][0]


def test_restore_validation_rejects_cross_tenant_generation_match() -> None:
    """A failed direct database probe uses the same non authenticated receipt vocabulary.

    Red proof: test node ``tests/test_production_resilience.py::test_restore_validation_rejects_cross_tenant_generation_match``
    failed against commit ``8ec55226`` after the expected field was changed to
    ``direct_database_search``; the validator returned ``authenticated_search`` instead. The
    current implementation passes the same node.
    """
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
    assert "direct_database_search" in result.failures
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


def test_public_alb_has_pre_auth_source_and_request_abuse_controls() -> None:
    """The public entry point must bound source abuse before ECS authentication work."""
    root = Path(__file__).parents[1] / "infra" / "aws"
    waf = (root / "waf.tf").read_text(encoding="utf-8")
    alarms = (root / "cloudwatch.tf").read_text(encoding="utf-8")
    variables = (root / "variables.tf").read_text(encoding="utf-8")

    assert 'resource "aws_wafv2_web_acl" "alb"' in waf
    assert 'resource "aws_wafv2_web_acl_association" "alb"' in waf
    assert 'resource "aws_wafv2_ip_set" "alb_allowed_sources"' in waf
    assert 'resource "aws_wafv2_ip_set" "alb_blocked_sources"' in waf
    assert 'aggregate_key_type = "IP"' in waf
    assert 'search_string         = "/mcp"' in waf
    assert "AWSManagedRulesCommonRuleSet" in waf
    assert "AWSManagedRulesKnownBadInputsRuleSet" in waf
    assert "waf_rate_limit_per_5m" in variables
    assert "waf_allowed_source_cidrs" in variables
    assert "waf_blocked_source_cidrs" in variables
    assert 'resource "aws_sns_topic" "alerts"' in alarms
    assert "aws_sns_topic.alerts.arn" in alarms
    assert 'resource "aws_cloudwatch_metric_alarm" "alb_waf_blocked"' in alarms
    assert 'resource "aws_cloudwatch_metric_alarm" "alb_http_4xx"' in alarms


def test_restore_drill_isolated_from_serving_task_and_role() -> None:
    root = Path(__file__).parents[1] / "infra" / "aws"
    iam = (root / "iam.tf").read_text(encoding="utf-8")
    ecs = (root / "ecs.tf").read_text(encoding="utf-8")
    drill = (root / "restore_drill.tf").read_text(encoding="utf-8")

    serving_role = iam.split('resource "aws_iam_role" "restore_drill_execution"', 1)[0]
    serving_task = ecs.split('resource "aws_ecs_task_definition" "this"', 1)[1].split(
        'resource "aws_ecs_service" "this"', 1
    )[0]
    forbidden = (
        "rds:RestoreDBClusterToPointInTime",
        "rds:DeleteDBCluster",
        "rds:CreateDBInstance",
        "rds:DeleteDBInstance",
        "restore_validation_dsn_secret_arn",
        "RECALL_RESTORE_",
    )
    for capability in forbidden:
        assert capability not in serving_role
        assert capability not in serving_task
    assert 'resource "aws_iam_role" "ecs_task"' not in serving_role
    assert "aws_iam_role.ecs_task.arn" not in serving_task
    assert 'Action = ["ecs:TagResource"]' not in serving_role

    assert 'resource "aws_iam_role" "restore_drill_task"' in iam
    assert 'resource "aws_ecs_task_definition" "restore_drill"' in drill
    assert "aws_iam_role.restore_drill_task.arn" in drill
    assert "aws_ecs_task_definition.restore_drill.arn" in drill
    assert "RECALL_RESTORE_CHECKSUM_MODE" in drill
    assert "RECALL_RESTORE_CHECKSUM_LIMIT" in drill
    assert "RECALL_RESTORE_SMOKE_EMBEDDER" in drill
    assert "restore_smoke_embedder" in (root / "variables.tf").read_text(encoding="utf-8")
    assert "aws_ecs_task_definition.this.arn" not in drill
    assert "rds:RestoreDBClusterToPointInTime" in iam
    assert "rds:DeleteDBCluster" in iam
    assert "rds:CreateDBInstance" in iam
    assert "rds:DeleteDBInstance" in iam
    cluster_policy = iam.split('resource "aws_iam_role_policy" "restore_drill_task"', 1)[1]
    cluster_actions = cluster_policy.split("Resource = [local.rds_cluster_arn_pattern]", 1)[0]
    assert '"rds:CreateDBInstance"' in cluster_actions


def test_idempotency_reservation_outlives_the_bucket_window() -> None:
    """BUG-001: a reservation must last as long as the durable receipt it guards.

    With the bucket TTL (twice the refill window, 120 s for a per-minute budget) a retry after a
    crash or during a long mutation found no reservation and no receipt, and ran the mutation again.
    Red proof: failed on 3ada1afd0 (recall_mcp.limits.RedisRateLimiter.check passed six script
    arguments and the script reused the bucket TTL) with AssertionError on the argument count.
    """
    from recall.constants import IDEMPOTENCY_RECEIPT_RETENTION_SECONDS
    from recall_mcp.limits import _token_bucket_lua

    calls: list[tuple[object, ...]] = []

    class Redis:
        async def script_load(self, _script: str) -> str:
            return "sha"

        async def evalsha(self, _sha: str, _numkeys: int, *args: object) -> list[int]:
            calls.append(args)
            return [1, 0, 0]

    limiter = RedisRateLimiter("redis://unused", {"write": Rate(20, 20 / 60)}, redis_client=Redis())
    asyncio.run(
        limiter.check(
            "tenant-a", "write", idempotency_key="k-1", idempotency_operation="recall_index"
        )
    )
    script_args = calls[0][2:]
    assert len(script_args) == 7
    assert script_args[6] == IDEMPOTENCY_RECEIPT_RETENTION_SECONDS * 1000
    assert "redis.call('SET', KEYS[2], ARGV[6], 'PX', tonumber(ARGV[7]))" in _token_bucket_lua()


@requires_db
def test_a_replay_answers_only_the_principal_that_made_the_request(make_store, monkeypatch) -> None:
    """SEC-004 (owner decision 2026-10-30): receipts are keyed by tenant and key, not principal.

    Without binding, another principal of the same tenant sending the same key and arguments was
    handed the first principal's stored response. Red proof: by mutation, since the helper is new:
    making recall_mcp.server._principal_bound_fingerprint return the bare fingerprint failed this
    test with "DID NOT RAISE" for the second principal.
    """
    from types import SimpleNamespace

    from mcp.server.mcpserver.exceptions import ToolError

    import recall_mcp.server as srv

    store = make_store(64)
    operation = "recall_index"
    key = "principal-bound-" + uuid.uuid4().hex
    fingerprint = _mutation_fingerprint(operation, {"path": "memory"})

    def as_principal(name: str) -> None:
        monkeypatch.setattr(
            srv,
            "get_access_token",
            lambda: SimpleNamespace(claims={"principal": name, "tenant": store.tenant}, client_id=name),
        )

    as_principal("alice")
    asyncio.run(
        _record_mutation_result(
            {"store": store},
            store.tenant,
            key,
            '{"files": 1}',
            idempotency_operation=operation,
            idempotency_fingerprint=fingerprint,
        )
    )
    assert asyncio.run(
        _durable_replay(store, key, operation, srv._principal_bound_fingerprint(fingerprint))
    ) == '{"files": 1}'

    as_principal("bob")
    with pytest.raises(ToolError, match="idempotency_conflict"):
        asyncio.run(
            _durable_replay(store, key, operation, srv._principal_bound_fingerprint(fingerprint))
        )


def _rds_stubber():
    boto3 = pytest.importorskip("boto3")
    from botocore.stub import Stubber

    client = boto3.client(
        "rds", region_name="eu-west-1", aws_access_key_id="x", aws_secret_access_key="y"
    )
    return client, Stubber(client)


def test_drill_cluster_cleanup_sends_a_request_rds_accepts() -> None:
    """Drill janitor review P1: DeleteDBCluster has no DeletionProtection parameter.

    The Stubber validates parameters against the botocore service model, so this crosses the
    seam the fake drill manager cannot. Red proof: failed on e60e79883
    (recall.ops.backup.BackupManager.delete_cluster passed DeletionProtection=False) with
    botocore StubAssertionError on the DeleteDBCluster parameters.
    """
    from recall.ops.backup import BackupManager

    client, stubber = _rds_stubber()
    stubber.add_response(
        "delete_db_cluster",
        {"DBCluster": {"DBClusterIdentifier": "src-drill-old"}},
        {"DBClusterIdentifier": "src-drill-old", "SkipFinalSnapshot": True},
    )
    stubber.add_client_error(
        "describe_db_clusters",
        service_error_code="DBClusterNotFoundFault",
        expected_params={"DBClusterIdentifier": "src-drill-old"},
    )
    with stubber:
        BackupManager(rds_client=client, s3_client=object()).delete_cluster(
            "src-drill-old", confirmation="DELETE_RESTORE_DRILL_CLUSTER"
        )
    stubber.assert_no_pending_responses()


def test_stale_drill_listing_reads_the_real_cluster_shape() -> None:
    """Drill janitor: the listing works on the response shape RDS returns, paginated.

    Red proof: by mutation, since the method is new: reading the tag list from "Tags" instead of
    "TagList" in recall.ops.backup.BackupManager.stale_drill_clusters failed this test with
    AssertionError (nothing selected).
    """
    from datetime import UTC, datetime

    from recall.ops.backup import BackupManager

    client, stubber = _rds_stubber()
    stubber.add_response(
        "describe_db_clusters",
        {
            "DBClusters": [
                {
                    "DBClusterIdentifier": "src-drill-old",
                    "ClusterCreateTime": datetime(2026, 10, 1, tzinfo=UTC),
                    "TagList": [{"Key": "recall:restore-drill", "Value": "true"}],
                    "DBClusterMembers": [{"DBInstanceIdentifier": "src-drill-old-writer"}],
                },
                {"DBClusterIdentifier": "src", "ClusterCreateTime": datetime(2026, 1, 1, tzinfo=UTC)},
            ]
        },
        {},
    )
    with stubber:
        stale = BackupManager(rds_client=client, s3_client=object()).stale_drill_clusters(
            "src-drill-", older_than=datetime(2026, 10, 30, tzinfo=UTC)
        )
    assert stale == [{"cluster_identifier": "src-drill-old", "instances": ["src-drill-old-writer"]}]


def test_check_sync_off_an_anyio_worker_uses_a_client_of_its_own() -> None:
    """PERF-001 review P3: a plain thread is not an AnyIO worker; the isolated path must run.

    Red proof: by mutation: re-raising every RuntimeError in
    recall_mcp.limits.RedisRateLimiter.check_sync (dropping the AnyIO message filter) failed this
    test with AnyIO's "Not running inside an AnyIO worker thread".
    """
    calls: list[tuple[object, ...]] = []

    class Redis:
        async def script_load(self, _script: str) -> str:
            return "sha"

        async def evalsha(self, _sha: str, _numkeys: int, *args: object) -> list[int]:
            calls.append(args)
            return [1, 0, 0]

    limiter = RedisRateLimiter("redis://unused", {"index_bytes": Rate(1e9, 1e6)}, redis_client=Redis())
    errors: list[BaseException] = []

    def worker() -> None:
        try:
            limiter.check_sync("tenant-a", "index_bytes", 1.0)
        except BaseException as exc:  # noqa: BLE001 - reported below
            errors.append(exc)

    thread = threading.Thread(target=worker)
    thread.start()
    thread.join(10)
    assert errors == []
    assert len(calls) == 1


@requires_db
def test_the_tool_path_binds_the_principal_on_both_sides(make_store, monkeypatch) -> None:
    """SEC-004 review P3: `_require` and the receipt write must bind the same principal.

    Drives the real `recall_forget` tool on the authenticated path. Red proof: by mutation:
    deleting the `_principal_bound_fingerprint` call in recall_mcp.server `_require` failed this
    test at alice's own replay (ToolError idempotency_conflict instead of the first response).
    """
    from types import SimpleNamespace

    from mcp.server.mcpserver.exceptions import ToolError

    import recall_mcp.server as srv
    from recall.embeddings import HashingEmbedder
    from recall.types import Chunk
    from recall_mcp.stores import StoreRegistry

    store = make_store(64)
    store.upsert([Chunk("a", "f.md", "the caching decision was adopted")], [[1.0] + [0.0] * 63])

    class Registry(StoreRegistry):
        def __init__(self) -> None:  # noqa: D107 - a one-store test double
            self._allowed = frozenset({store.tenant})
            self._control_plane = None

        def get(self, tenant: str):  # type: ignore[override]
            return store

        def get_shadow(self, tenant: str):  # type: ignore[override]
            return None

    state = {"stores": Registry(), "embedder": HashingEmbedder(dim=64), "calibration": None}
    server = srv.build_server()
    tool = {t.name: t for t in server._tool_manager.list_tools()}["recall_forget"]
    ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))
    key = "bound-" + uuid.uuid4().hex

    def call_as(principal: str) -> str:
        monkeypatch.setattr(
            srv,
            "get_access_token",
            lambda: SimpleNamespace(
                scopes=[srv.SCOPE_READ, srv.SCOPE_WRITE, srv.SCOPE_FORGET],
                claims={"tenant": store.tenant, "principal": principal},
                client_id=principal,
            ),
        )
        return asyncio.run(tool.fn(ctx=ctx, sources=["f.md"], idempotency_key=key))

    first = call_as("alice")
    assert json.loads(first)["chunks_removed"] == 1
    assert call_as("alice") == first
    with pytest.raises(ToolError, match="idempotency_conflict"):
        call_as("bob")
