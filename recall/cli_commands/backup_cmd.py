"""`recall backup`: inspect, create, verify, and stage Aurora recovery operations."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = sub.add_parser("backup", help="manage Aurora backups and immutable receipts")
    parser.set_defaults(func=_cmd_backup)
    commands = parser.add_subparsers(dest="backup_cmd", required=True)

    status = commands.add_parser("status", help="show continuous backup and PITR status")
    status.add_argument("--cluster", required=True, dest="cluster_identifier")
    status.add_argument("--region", default=None)

    create = commands.add_parser("create", help="create an encrypted cluster snapshot")
    create.add_argument("--cluster", required=True, dest="cluster_identifier")
    create.add_argument("--snapshot", required=True, dest="snapshot_identifier")
    create.add_argument("--region", default=None)
    create.add_argument("--receipt-bucket", default=None)
    create.add_argument("--receipt-key", default=None)

    verify = commands.add_parser("verify", help="verify a receipt and snapshot metadata")
    verify.add_argument("--cluster", required=True, dest="cluster_identifier")
    verify.add_argument("--snapshot", required=True, dest="snapshot_identifier")
    verify.add_argument("--region", default=None)

    restore = commands.add_parser("restore", help="restore into a new isolated Aurora cluster")
    restore.add_argument("--source-cluster", required=True)
    restore.add_argument("--target-cluster", required=True)
    restore.add_argument("--subnet-group", required=True)
    restore.add_argument("--kms-key-id", required=True)
    restore.add_argument("--region", default=None)
    restore.add_argument("--restore-time", default=None, help="UTC ISO 8601 point in time")
    restore.add_argument("--confirm", choices=["RESTORE_NEW_CLUSTER"], default=None)


def _manager(region: str | None):
    from recall.ops.backup import BackupManager

    return BackupManager(region=region)


def _cmd_backup(args: argparse.Namespace) -> None:
    if args.backup_cmd == "status":
        print(json.dumps(_manager(args.region).status(args.cluster_identifier), indent=2, default=str))
        return
    if args.backup_cmd == "create":
        manager = _manager(args.region)
        snapshot = manager.create_snapshot(args.cluster_identifier, args.snapshot_identifier)
        result: dict[str, object] = {"snapshot": snapshot}
        if args.receipt_bucket:
            from recall.ops.backup import receipt_from_metadata

            receipt = receipt_from_metadata(
                database_identifier=args.cluster_identifier,
                backup_identifier=args.snapshot_identifier,
                backup_type="manual_snapshot",
                region=manager.region,
                active_generation=None,
                row_counts={},
                checksums={},
                configuration={"cluster": args.cluster_identifier},
            )
            manager.put_receipt(args.receipt_bucket, args.receipt_key or f"receipts/{args.snapshot_identifier}.json", receipt)
            result["receipt"] = receipt.to_dict()
        print(json.dumps(result, indent=2, default=str))
        return
    if args.backup_cmd == "verify":
        manager = _manager(args.region)
        snapshot = manager.verify_snapshot(args.snapshot_identifier)
        passed = bool(snapshot.get("encrypted")) and snapshot.get("status") in {"available", "creating"}
        print(json.dumps({"verified": passed, "snapshot": snapshot}, indent=2, default=str))
        if not passed:
            raise SystemExit(1)
        return
    if args.backup_cmd == "restore":
        restore_time = None
        if args.restore_time:
            restore_time = datetime.fromisoformat(args.restore_time.replace("Z", "+00:00"))
            if restore_time.tzinfo is None:
                restore_time = restore_time.replace(tzinfo=UTC)
        result = _manager(args.region).restore_pitr(
            args.source_cluster,
            args.target_cluster,
            restore_time=restore_time,
            subnet_group_name=args.subnet_group,
            kms_key_id=args.kms_key_id,
            confirmation=args.confirm,
        )
        print(json.dumps(result, indent=2, default=str))
        return
    raise SystemExit(f"unknown backup subcommand: {args.backup_cmd}")
