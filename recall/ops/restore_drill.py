"""Scheduled restore drill entry point.

The task creates a new cluster through Aurora PITR, emits a receipt, and leaves cutover to the
operator. Environment variables are identifiers only and must be supplied through ECS secrets or
task configuration, never through source control.
"""

from __future__ import annotations

import json
import os
import uuid

from recall.ops.backup import BackupManager
from recall.ops.restore import validate_restored_database


def run() -> dict[str, object]:
    source = os.environ["RECALL_RESTORE_SOURCE_CLUSTER"]
    target = os.environ.get("RECALL_RESTORE_TARGET_CLUSTER", f"{source}-drill-{uuid.uuid4().hex[:10]}")
    manager = BackupManager(region=os.environ.get("AWS_REGION"))
    result = manager.restore_pitr(
        source,
        target,
        subnet_group_name=os.environ["RECALL_RESTORE_SUBNET_GROUP"],
        kms_key_id=os.environ["RECALL_RESTORE_KMS_KEY_ID"],
        confirmation="RESTORE_NEW_CLUSTER",
    )
    manager.wait_for_cluster_available(target)
    dsn = os.environ.get("RECALL_RESTORE_VALIDATION_DSN", "").strip()
    if not dsn:
        raise RuntimeError("RECALL_RESTORE_VALIDATION_DSN is required for a successful restore drill")
    import psycopg

    with psycopg.connect(dsn, connect_timeout=10) as connection:
        validation = validate_restored_database(
            connection,
            expected_schema_version=os.environ.get("RECALL_RESTORE_SCHEMA_VERSION", ""),
            expected_generation=os.environ.get("RECALL_RESTORE_EXPECTED_GENERATION") or None,
        )
    if not validation.passed:
        raise RuntimeError(f"restore validation failed: {', '.join(validation.failures)}")
    receipt = {"drill": True, "source": source, "target": target, "restore": result, "validation": validation.to_dict()}
    print(json.dumps(receipt, sort_keys=True, default=str))
    return receipt


if __name__ == "__main__":
    run()
