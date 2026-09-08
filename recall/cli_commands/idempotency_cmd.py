"""`recall idempotency`: repair a missing replay result without rerunning a mutation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from recall.errors import IdempotencyConflict
from recall.store import PgVectorStore


_CONFIRMATION = "RECONCILE_IDEMPOTENCY"


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = sub.add_parser(
        "idempotency",
        help="inspect or reconcile an idempotent mutation receipt",
        description=(
            "Record an operator verified mutation result in PostgreSQL after the Redis replay "
            "result was lost. This command never executes the mutation."
        ),
    )
    parser.set_defaults(_opens_db=True, func=_cmd_idempotency)
    commands = parser.add_subparsers(dest="idempotency_cmd", required=True)
    reconcile = commands.add_parser(
        "reconcile",
        help="record a verified result so the original mutation key can be replayed",
        description=(
            "Persist a verified JSON response for an idempotency key. Inspect the mutation's "
            "side effect first. Without --confirm this is a dry run."
        ),
    )
    reconcile.add_argument("--key", required=True, help="the original idempotency key")
    reconcile.add_argument("--operation", required=True, help="the original mutation tool")
    reconcile.add_argument(
        "--fingerprint",
        required=True,
        help="the original canonical request fingerprint",
    )
    reconcile.add_argument(
        "--result-file",
        required=True,
        help="UTF-8 JSON response file produced or verified by the operator",
    )
    reconcile.add_argument(
        "--confirm",
        default=None,
        help=f"write only when this equals {_CONFIRMATION}",
    )
    reconcile.add_argument(
        "--dim",
        type=int,
        default=1,
        help="store dimension needed only to open the receipt store (default: 1)",
    )


def _read_result(path_text: str) -> str:
    path = Path(path_text)
    try:
        result = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise SystemExit(f"idempotency reconcile: cannot read result file: {exc}") from exc
    if not result.strip():
        raise SystemExit("idempotency reconcile: result file is empty")
    try:
        json.loads(result)
    except json.JSONDecodeError as exc:
        raise SystemExit("idempotency reconcile: result file must contain valid JSON") from exc
    return result


def _cmd_idempotency(args: argparse.Namespace) -> None:
    if args.idempotency_cmd != "reconcile":  # noqa: S105, parser subcommand, not a credential
        raise SystemExit(f"unknown idempotency subcommand: {args.idempotency_cmd}")
    if args.dim < 1:
        raise SystemExit("idempotency reconcile: --dim must be at least 1")

    with PgVectorStore(
        args.dsn,
        dim=args.dim,
        table=args.table,
        tenant=args.tenant,
    ) as store:
        try:
            existing = store.get_operation_receipt(
                args.key,
                operation=args.operation,
                request_fingerprint=args.fingerprint,
            )
        except IdempotencyConflict as exc:
            raise SystemExit(
                "idempotency reconcile: the durable receipt conflicts with the supplied "
                "operation or fingerprint"
            ) from exc
        if existing is not None:
            print(
                json.dumps(
                    {
                        "status": "already_reconciled",
                        "idempotency_key": args.key,
                        "operation": args.operation,
                    },
                    sort_keys=True,
                )
            )
            return

        if args.confirm != _CONFIRMATION:
            print(
                json.dumps(
                    {
                        "status": "dry_run",
                        "idempotency_key": args.key,
                        "operation": args.operation,
                        "request_fingerprint": args.fingerprint,
                        "result_file": str(Path(args.result_file)),
                        "message": (
                            f"re-run with --confirm {_CONFIRMATION} after verifying the side effect"
                        ),
                    },
                    sort_keys=True,
                )
            )
            return

        result = _read_result(args.result_file)
        store.record_operation_receipt(
            args.key,
            result,
            operation=args.operation,
            request_fingerprint=args.fingerprint,
        )
        recorded = store.get_operation_receipt(
            args.key,
            operation=args.operation,
            request_fingerprint=args.fingerprint,
        )
        if recorded != result:
            raise SystemExit(
                "idempotency reconcile: another receipt already exists with a different result"
            )
        print(
            json.dumps(
                {
                    "status": "reconciled",
                    "idempotency_key": args.key,
                    "operation": args.operation,
                    "message": "durable receipt recorded; the mutation was not executed",
                },
                sort_keys=True,
            )
        )
