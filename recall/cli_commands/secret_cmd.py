"""`recall secret`: verify staged ECS task secret versions without printing values."""

from __future__ import annotations

import argparse
import json


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = sub.add_parser("secret", help="verify runtime secret rotation state")
    parser.set_defaults(func=_cmd_secret)
    commands = parser.add_subparsers(dest="secret_cmd", required=True)
    verify = commands.add_parser("verify", help="verify every running task has intended versions")
    verify.add_argument("--cluster", required=True)
    verify.add_argument("--service", required=True)
    verify.add_argument("--versions", required=True, help="JSON object mapping secret names to version IDs")
    verify.add_argument("--region", default=None)


def _cmd_secret(args: argparse.Namespace) -> None:
    if args.secret_cmd != "verify":  # noqa: S105 - parser subcommand, not a credential
        raise SystemExit(f"unknown secret subcommand: {args.secret_cmd}")
    try:
        versions = json.loads(args.versions)
    except json.JSONDecodeError as exc:
        raise SystemExit("--versions must be a JSON object") from exc
    if not isinstance(versions, dict) or not all(isinstance(key, str) and isinstance(value, str) for key, value in versions.items()):
        raise SystemExit("--versions must be a JSON object mapping names to version IDs")
    from recall.ops.rotation import EcsSecretRotator

    checks = EcsSecretRotator(region=args.region).verify_tasks(args.cluster, args.service, versions)
    print(json.dumps([check.to_dict() for check in checks], indent=2))
    if not checks or any(not check.healthy for check in checks):
        raise SystemExit(1)
