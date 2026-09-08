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


def _validation_dsn(cluster: dict[str, object]) -> str:
    """Bind the validation credential to the endpoint returned for this restore."""
    template = os.environ.get("RECALL_RESTORE_VALIDATION_DSN", "").strip()
    if not template:
        raise RuntimeError("RECALL_RESTORE_VALIDATION_DSN is required for a successful restore drill")
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
        isinstance(key, str) and isinstance(item, str) and key in {"recall_chunks_v1", "recall_generations"}
        for key, item in value.items()
    ):
        raise RuntimeError("restore checksum keys must be recall_chunks_v1 or recall_generations")
    return value


def _checksum_provider(
    connection: object, expected: dict[str, str]
) -> Callable[[object], dict[str, str]]:
    def checksum(_connection: object) -> dict[str, str]:
        result: dict[str, str] = {}
        for table in expected:
            with connection.cursor() as cursor:  # type: ignore[attr-defined]
                cursor.execute(_CHECKSUM_QUERIES[table])
                row = cursor.fetchone()
            result[table] = str(row[0] if row else "")
        return result

    return checksum


def run() -> dict[str, object]:
    source = os.environ["RECALL_RESTORE_SOURCE_CLUSTER"]
    target = os.environ.get("RECALL_RESTORE_TARGET_CLUSTER", f"{source}-drill-{uuid.uuid4().hex[:10]}")
    manager = BackupManager(region=os.environ.get("AWS_REGION"))
    created = False
    try:
        result = manager.restore_pitr(
            source,
            target,
            subnet_group_name=os.environ["RECALL_RESTORE_SUBNET_GROUP"],
            kms_key_id=os.environ["RECALL_RESTORE_KMS_KEY_ID"],
            confirmation="RESTORE_NEW_CLUSTER",
        )
        created = True
        restored_cluster = manager.wait_for_cluster_available(target)
        dsn = _validation_dsn(restored_cluster)
        import psycopg

        with psycopg.connect(dsn, connect_timeout=10) as connection:
            expected_checksums = _expected_checksums()
            validation = validate_restored_database(
                connection,
                expected_schema_version=os.environ.get("RECALL_RESTORE_SCHEMA_VERSION", ""),
                expected_generation=os.environ.get("RECALL_RESTORE_EXPECTED_GENERATION") or None,
                expected_checksums=expected_checksums,
                checksum_provider=(
                    _checksum_provider(connection, expected_checksums) if expected_checksums else None
                ),
            )
        if not validation.passed:
            raise RuntimeError(f"restore validation failed: {', '.join(validation.failures)}")
        receipt = {"drill": True, "source": source, "target": target, "restore": result, "validation": validation.to_dict()}
        print(json.dumps(receipt, sort_keys=True, default=str))
        return receipt
    finally:
        if created:
            manager.delete_cluster(target, confirmation="DELETE_RESTORE_DRILL_CLUSTER")


if __name__ == "__main__":
    run()
