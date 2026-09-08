"""Aurora backup orchestration and immutable, secret free receipts."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass, asdict
from datetime import UTC, datetime
from typing import Any

from recall.schema import load_migrations


@dataclass(frozen=True)
class BackupReceipt:
    database_identifier: str
    backup_identifier: str
    backup_timestamp: str
    backup_type: str
    region: str
    schema_version: str
    active_generation: str | None
    row_counts: dict[str, int]
    checksums: dict[str, str]
    configuration_fingerprint: str
    verified: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    def json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


def configuration_fingerprint(values: dict[str, str]) -> str:
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(canonical).hexdigest()


class BackupManager:
    """Use native RDS APIs and S3 object lock for receipts.

    The manager is deliberately lazy. Constructing it performs no AWS call, and restore methods
    require an explicit confirmation token before they mutate cloud state.
    """

    def __init__(self, *, region: str | None = None, rds_client: Any | None = None, s3_client: Any | None = None) -> None:
        self.region = region or os.environ.get("AWS_REGION", "")
        self._rds = rds_client
        self._s3 = s3_client

    def _clients(self) -> tuple[Any, Any]:
        if self._rds is None or self._s3 is None:
            try:
                import boto3
            except ImportError as exc:  # pragma: no cover, optional AWS extra
                raise RuntimeError("boto3 is required for AWS backup operations") from exc
            session = boto3.session.Session(region_name=self.region or None)
            self._rds = self._rds or session.client("rds")
            self._s3 = self._s3 or session.client("s3")
        return self._rds, self._s3

    def status(self, cluster_identifier: str) -> dict[str, object]:
        rds, _ = self._clients()
        cluster = rds.describe_db_clusters(DBClusterIdentifier=cluster_identifier)["DBClusters"][0]
        return {
            "cluster_identifier": cluster_identifier,
            "status": cluster.get("Status"),
            "engine": cluster.get("Engine"),
            "engine_version": cluster.get("EngineVersion"),
            "backup_retention_days": cluster.get("BackupRetentionPeriod"),
            "latest_restorable_time": cluster.get("LatestRestorableTime"),
            "earliest_restorable_time": cluster.get("EarliestRestorableTime"),
            "storage_encrypted": cluster.get("StorageEncrypted"),
            "endpoint": cluster.get("Endpoint"),
            "reader_endpoint": cluster.get("ReaderEndpoint"),
            "port": cluster.get("Port", 5432),
        }

    def create_snapshot(self, cluster_identifier: str, snapshot_identifier: str) -> dict[str, object]:
        rds, _ = self._clients()
        response = rds.create_db_cluster_snapshot(
            DBClusterIdentifier=cluster_identifier,
            DBClusterSnapshotIdentifier=snapshot_identifier,
            Tags=[{"Key": "recall:backup", "Value": "true"}],
        )
        snapshot = response["DBClusterSnapshot"]
        return {
            "snapshot_identifier": snapshot.get("DBClusterSnapshotIdentifier"),
            "status": snapshot.get("Status"),
            "cluster_identifier": snapshot.get("DBClusterIdentifier"),
            "snapshot_create_time": snapshot.get("SnapshotCreateTime"),
            "encrypted": snapshot.get("StorageEncrypted"),
        }

    def verify_snapshot(self, snapshot_identifier: str) -> dict[str, object]:
        rds, _ = self._clients()
        snapshot = rds.describe_db_cluster_snapshots(
            DBClusterSnapshotIdentifier=snapshot_identifier
        )["DBClusterSnapshots"][0]
        return {
            "snapshot_identifier": snapshot.get("DBClusterSnapshotIdentifier"),
            "status": snapshot.get("Status"),
            "cluster_identifier": snapshot.get("DBClusterIdentifier"),
            "encrypted": snapshot.get("StorageEncrypted"),
            "engine": snapshot.get("Engine"),
            "engine_version": snapshot.get("EngineVersion"),
        }

    def wait_for_cluster_available(self, cluster_identifier: str) -> dict[str, object]:
        """Wait for a restored cluster to become usable before reporting drill success."""
        rds, _ = self._clients()
        rds.get_waiter("db_cluster_available").wait(DBClusterIdentifier=cluster_identifier)
        return self.status(cluster_identifier)

    def delete_cluster(self, cluster_identifier: str, *, confirmation: str | None = None) -> None:
        """Delete only an isolated drill cluster after an explicit confirmation token."""
        if confirmation != "DELETE_RESTORE_DRILL_CLUSTER":
            raise ValueError("drill cleanup requires confirmation=DELETE_RESTORE_DRILL_CLUSTER")
        rds, _ = self._clients()
        rds.delete_db_cluster(
            DBClusterIdentifier=cluster_identifier,
            SkipFinalSnapshot=True,
            DeletionProtection=False,
        )
        rds.get_waiter("db_cluster_deleted").wait(DBClusterIdentifier=cluster_identifier)

    def create_restore_instance(
        self,
        cluster_identifier: str,
        instance_identifier: str,
        *,
        instance_class: str,
        subnet_group_name: str,
    ) -> dict[str, object]:
        """Create the temporary writer required before an Aurora cluster accepts connections."""
        rds, _ = self._clients()
        response = rds.create_db_instance(
            DBInstanceIdentifier=instance_identifier,
            DBInstanceClass=instance_class,
            Engine="aurora-postgresql",
            DBClusterIdentifier=cluster_identifier,
            DBSubnetGroupName=subnet_group_name,
            PubliclyAccessible=False,
            Tags=[{"Key": "recall:restore-drill", "Value": "true"}],
        )
        instance = response["DBInstance"]
        return {
            "instance_identifier": instance.get("DBInstanceIdentifier"),
            "status": instance.get("DBInstanceStatus"),
            "endpoint": (instance.get("Endpoint") or {}).get("Address"),
        }

    def wait_for_instance_available(self, instance_identifier: str) -> dict[str, object]:
        """Wait until the temporary writer is accepting connections."""
        rds, _ = self._clients()
        rds.get_waiter("db_instance_available").wait(DBInstanceIdentifier=instance_identifier)
        response = rds.describe_db_instances(DBInstanceIdentifier=instance_identifier)
        instance = response["DBInstances"][0]
        endpoint = instance.get("Endpoint") or {}
        return {
            "instance_identifier": instance.get("DBInstanceIdentifier"),
            "status": instance.get("DBInstanceStatus"),
            "endpoint": endpoint.get("Address"),
            "port": endpoint.get("Port", 5432),
        }

    def delete_instance(self, instance_identifier: str, *, confirmation: str | None = None) -> None:
        """Delete only the temporary drill writer after an explicit confirmation token."""
        if confirmation != "DELETE_RESTORE_DRILL_INSTANCE":
            raise ValueError("drill cleanup requires confirmation=DELETE_RESTORE_DRILL_INSTANCE")
        rds, _ = self._clients()
        rds.delete_db_instance(DBInstanceIdentifier=instance_identifier, SkipFinalSnapshot=True)
        rds.get_waiter("db_instance_deleted").wait(DBInstanceIdentifier=instance_identifier)

    def restore_pitr(
        self,
        source_cluster_identifier: str,
        target_cluster_identifier: str,
        *,
        restore_time: datetime | None = None,
        subnet_group_name: str,
        kms_key_id: str,
        confirmation: str | None = None,
    ) -> dict[str, object]:
        if confirmation != "RESTORE_NEW_CLUSTER":
            raise ValueError("restore requires confirmation=RESTORE_NEW_CLUSTER")
        rds, _ = self._clients()
        params: dict[str, object] = {
            "DBClusterIdentifier": target_cluster_identifier,
            "SourceDBClusterIdentifier": source_cluster_identifier,
            "DBSubnetGroupName": subnet_group_name,
            "StorageEncrypted": True,
            "KmsKeyId": kms_key_id,
            "DeletionProtection": False,
            "CopyTagsToSnapshot": True,
        }
        if restore_time is not None:
            params["RestoreType"] = "copy-on-write"
            params["RestoreToTime"] = restore_time.astimezone(UTC)
        cluster = rds.restore_db_cluster_to_point_in_time(**params)["DBCluster"]
        return {
            "cluster_identifier": cluster.get("DBClusterIdentifier"),
            "status": cluster.get("Status"),
            "restore_type": "pitr" if restore_time is not None else "latest",
            "source_cluster_identifier": source_cluster_identifier,
        }

    def put_receipt(self, bucket: str, key: str, receipt: BackupReceipt) -> None:
        _, s3 = self._clients()
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=receipt.json().encode("utf-8"),
            ContentType="application/json",
            ServerSideEncryption="aws:kms",
            BucketKeyEnabled=True,
            Metadata={"schema-version": receipt.schema_version, "verified": str(receipt.verified).lower()},
        )


def receipt_from_metadata(
    *,
    database_identifier: str,
    backup_identifier: str,
    backup_type: str,
    region: str,
    active_generation: str | None,
    row_counts: dict[str, int],
    checksums: dict[str, str],
    configuration: dict[str, str],
    verified: bool = False,
) -> BackupReceipt:
    return BackupReceipt(
        database_identifier=database_identifier,
        backup_identifier=backup_identifier,
        backup_timestamp=datetime.now(UTC).isoformat(),
        backup_type=backup_type,
        region=region,
        schema_version=str(load_migrations()[-1].version),
        active_generation=active_generation,
        row_counts=dict(row_counts),
        checksums=dict(checksums),
        configuration_fingerprint=configuration_fingerprint(configuration),
        verified=verified,
    )
