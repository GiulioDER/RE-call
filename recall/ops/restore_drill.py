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


def run() -> dict[str, object]:
    source = os.environ["RECALL_RESTORE_SOURCE_CLUSTER"]
    target = os.environ.get("RECALL_RESTORE_TARGET_CLUSTER", f"{source}-drill-{uuid.uuid4().hex[:10]}")
    result = BackupManager(region=os.environ.get("AWS_REGION")).restore_pitr(
        source,
        target,
        subnet_group_name=os.environ["RECALL_RESTORE_SUBNET_GROUP"],
        kms_key_id=os.environ["RECALL_RESTORE_KMS_KEY_ID"],
        confirmation="RESTORE_NEW_CLUSTER",
    )
    receipt = {"drill": True, "source": source, "target": target, "restore": result}
    print(json.dumps(receipt, sort_keys=True, default=str))
    return receipt


if __name__ == "__main__":
    run()

