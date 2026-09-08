"""Scheduled restore drill entry point.

The task creates a new cluster through Aurora PITR, emits a receipt, and leaves cutover to the
operator. Environment variables are identifiers only and must be supplied through ECS secrets or
task configuration, never through source control.
"""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from time import monotonic

from recall.ops.backup import BackupManager
from recall.ops.restore import validate_restored_database


_CHECKSUM_QUERIES = {
    "recall_chunks_v1": (
        "SELECT md5(COALESCE(string_agg(md5(row_to_json(t)::text), '' "
        "ORDER BY md5(row_to_json(t)::text)), '')) FROM (SELECT * FROM recall_chunks_v1) t"
    ),
    "recall_generations": (
        "SELECT md5(COALESCE(string_agg(md5(row_to_json(t)::text), '' "
        "ORDER BY md5(row_to_json(t)::text)), '')) FROM (SELECT * FROM recall_generations) t"
    ),
}

_BOUNDED_CHECKSUM_QUERIES = {
    "recall_chunks_v1": (
        "SELECT md5(COALESCE(string_agg(md5(row_to_json(t)::text), '' "
        "ORDER BY md5(row_to_json(t)::text)), '')) FROM ("
        "SELECT * FROM recall_chunks_v1 "
        "ORDER BY tenant_id, generation_id, chunk_id LIMIT %s) t"
    ),
    "recall_generations": (
        "SELECT md5(COALESCE(string_agg(md5(row_to_json(t)::text), '' "
        "ORDER BY md5(row_to_json(t)::text)), '')) FROM ("
        "SELECT * FROM recall_generations "
        "ORDER BY tenant_id, generation_id LIMIT %s) t"
    ),
}


@dataclass(frozen=True)
class ChecksumPolicy:
    """Select the restore checksum coverage and its bounded work budget."""

    mode: str
    max_rows_per_table: int | None


def _checksum_policy() -> ChecksumPolicy:
    mode = os.environ.get("RECALL_RESTORE_CHECKSUM_MODE", "bounded").strip().lower()
    if mode == "full":
        return ChecksumPolicy(mode="full", max_rows_per_table=None)
    if mode != "bounded":
        raise RuntimeError("RECALL_RESTORE_CHECKSUM_MODE must be bounded or full")

    raw_limit = os.environ.get("RECALL_RESTORE_CHECKSUM_LIMIT", "10000").strip()
    try:
        limit = int(raw_limit)
    except ValueError as exc:
        raise RuntimeError("RECALL_RESTORE_CHECKSUM_LIMIT must be an integer") from exc
    if not 1 <= limit <= 1_000_000:
        raise RuntimeError("RECALL_RESTORE_CHECKSUM_LIMIT must be between 1 and 1000000")
    return ChecksumPolicy(mode="bounded", max_rows_per_table=limit)


def _validation_dsn(cluster: dict[str, object]) -> str:
    """Bind the validation credential to the endpoint returned for this restore."""
    template = os.environ.get("RECALL_RESTORE_VALIDATION_DSN", "").strip()
    if not template:
        raise RuntimeError(
            "RECALL_RESTORE_VALIDATION_DSN is required for a successful restore drill"
        )
    endpoint = str(cluster.get("endpoint") or "").strip()
    if not endpoint:
        raise RuntimeError("restored cluster did not publish a writer endpoint")
    if "{host}" not in template:
        raise RuntimeError(
            "RECALL_RESTORE_VALIDATION_DSN must contain {host} so validation binds to the restored cluster"
        )
    return template.format(host=endpoint, port=cluster.get("port", 5432))


def _expected_checksums() -> dict[str, str] | None:
    raw = os.environ.get("RECALL_RESTORE_EXPECTED_CHECKSUMS", "").strip()
    if not raw:
        return None
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError("RECALL_RESTORE_EXPECTED_CHECKSUMS must be a JSON object") from exc
    if not isinstance(value, dict) or not all(
        isinstance(key, str)
        and isinstance(item, str)
        and key in {"recall_chunks_v1", "recall_generations"}
        for key, item in value.items()
    ):
        raise RuntimeError("restore checksum keys must be recall_chunks_v1 or recall_generations")
    return value


def _checksum_provider(
    connection: object,
    expected: dict[str, str],
    *,
    policy: ChecksumPolicy | None = None,
    durations_ms: dict[str, float] | None = None,
) -> Callable[[object], dict[str, str]]:
    selected_policy = policy or ChecksumPolicy(mode="full", max_rows_per_table=None)

    def checksum(_connection: object) -> dict[str, str]:
        result: dict[str, str] = {}
        for table in expected:
            started = monotonic()
            with connection.cursor() as cursor:  # type: ignore[attr-defined]
                if selected_policy.mode == "bounded":
                    cursor.execute(
                        _BOUNDED_CHECKSUM_QUERIES[table],
                        (selected_policy.max_rows_per_table,),
                    )
                else:
                    cursor.execute(_CHECKSUM_QUERIES[table])
                row = cursor.fetchone()
            result[table] = str(row[0] if row else "")
            if durations_ms is not None:
                durations_ms[table] = round((monotonic() - started) * 1000, 3)
        return result

    return checksum


def _calibration_check(dsn: str, tenant: str, generation: str) -> Callable[[object], bool]:
    def check(_connection: object) -> bool:
        from recall.calibration_v2 import CalibrationRepository, CalibrationStatus

        resolution = CalibrationRepository(dsn, tenant, actor="restore-drill").resolve(generation)
        return resolution.status is CalibrationStatus.CERTIFIED

    return check


def run() -> dict[str, object]:
    source = os.environ["RECALL_RESTORE_SOURCE_CLUSTER"]
    target = os.environ.get(
        "RECALL_RESTORE_TARGET_CLUSTER", f"{source}-drill-{uuid.uuid4().hex[:10]}"
    )
    instance = os.environ.get("RECALL_RESTORE_TARGET_INSTANCE", f"{target}-writer")
    manager = BackupManager(region=os.environ.get("AWS_REGION"))
    created = False
    instance_created = False
    try:
        result = manager.restore_pitr(
            source,
            target,
            subnet_group_name=os.environ["RECALL_RESTORE_SUBNET_GROUP"],
            kms_key_id=os.environ["RECALL_RESTORE_KMS_KEY_ID"],
            confirmation="RESTORE_NEW_CLUSTER",
        )
        created = True
        manager.wait_for_cluster_available(target)
        manager.create_restore_instance(
            target,
            instance,
            instance_class=os.environ["RECALL_RESTORE_INSTANCE_CLASS"],
            subnet_group_name=os.environ["RECALL_RESTORE_SUBNET_GROUP"],
        )
        instance_created = True
        restored_writer = manager.wait_for_instance_available(instance)
        dsn = _validation_dsn(restored_writer)
        import psycopg

        with psycopg.connect(dsn, connect_timeout=10) as connection:
            expected_checksums = _expected_checksums()
            checksum_policy = _checksum_policy()
            checksum_durations_ms: dict[str, float] = {}
            schema_version = os.environ.get("RECALL_RESTORE_SCHEMA_VERSION", "").strip()
            if not schema_version:
                raise RuntimeError(
                    "RECALL_RESTORE_SCHEMA_VERSION is required for restore validation"
                )
            expected_generation = os.environ.get("RECALL_RESTORE_EXPECTED_GENERATION", "").strip()
            if not expected_generation:
                raise RuntimeError(
                    "RECALL_RESTORE_EXPECTED_GENERATION is required for tenant generation validation"
                )
            expected_role = os.environ.get("RECALL_RESTORE_EXPECTED_ROLE", "").strip()
            if not expected_role:
                raise RuntimeError(
                    "RECALL_RESTORE_EXPECTED_ROLE is required for serving role validation"
                )
            if expected_checksums is None or set(expected_checksums) != {
                "recall_chunks_v1",
                "recall_generations",
            }:
                raise RuntimeError(
                    "RECALL_RESTORE_EXPECTED_CHECKSUMS must contain both restore table checksums"
                )
            tenant = os.environ.get("RECALL_RESTORE_TENANT", "").strip()
            if not tenant:
                raise RuntimeError(
                    "RECALL_RESTORE_TENANT is required for tenant scoped restore validation"
                )
            representative_chunk_id = os.environ.get(
                "RECALL_RESTORE_REPRESENTATIVE_CHUNK_ID", ""
            ).strip()
            if not representative_chunk_id:
                raise RuntimeError(
                    "RECALL_RESTORE_REPRESENTATIVE_CHUNK_ID is required for representative retrieval validation"
                )
            validation = validate_restored_database(
                connection,
                expected_schema_version=schema_version,
                expected_tenant=tenant,
                expected_generation=expected_generation,
                expected_role=expected_role,
                representative_chunk_id=representative_chunk_id,
                expected_checksums=expected_checksums,
                checksum_provider=(
                    _checksum_provider(
                        connection,
                        expected_checksums,
                        policy=checksum_policy,
                        durations_ms=checksum_durations_ms,
                    )
                    if expected_checksums
                    else None
                ),
                calibration_check=_calibration_check(dsn, tenant, expected_generation),
            )
        if not validation.passed:
            raise RuntimeError(f"restore validation failed: {', '.join(validation.failures)}")
        receipt = {
            "drill": True,
            "source": source,
            "target": target,
            "restore": result,
            "validation": validation.to_dict(),
            "checksum_validation": {
                "mode": checksum_policy.mode,
                "max_rows_per_table": checksum_policy.max_rows_per_table,
                "durations_ms": checksum_durations_ms,
            },
        }
        print(json.dumps(receipt, sort_keys=True, default=str))
        return receipt
    finally:
        if instance_created:
            try:
                manager.delete_instance(instance, confirmation="DELETE_RESTORE_DRILL_INSTANCE")
            finally:
                if created:
                    manager.delete_cluster(target, confirmation="DELETE_RESTORE_DRILL_CLUSTER")
        elif created:
            manager.delete_cluster(target, confirmation="DELETE_RESTORE_DRILL_CLUSTER")


if __name__ == "__main__":
    run()
