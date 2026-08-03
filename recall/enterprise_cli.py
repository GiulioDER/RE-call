"""Operator CLI for versioned enterprise index generations."""
from __future__ import annotations

import argparse
import os

from recall.control_plane import ControlPlane
from recall.store import PgVectorStore, require_secure_dsn


def _dsn() -> str:
    dsn = os.environ.get("RECALL_DSN")
    if not dsn:
        raise SystemExit("RECALL_DSN is required")
    require_secure_dsn(dsn)
    return dsn


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="recall-enterprise")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("migrate")
    create = commands.add_parser("create-generation")
    create.add_argument("generation_id")
    create.add_argument("table")
    create.add_argument("profile")
    create.add_argument("dimension", type=int)
    create.add_argument("--tenant", default="default")
    ready = commands.add_parser("mark-ready")
    ready.add_argument("generation_id")
    ready.add_argument("--chunks", type=int, required=True)
    ready.add_argument("--sources", type=int, required=True)
    route = commands.add_parser("set-route")
    route.add_argument("tenant")
    route.add_argument("active_generation")
    route.add_argument("--shadow-generation")
    cutover = commands.add_parser("cutover")
    cutover.add_argument("tenant")
    return parser


def main() -> None:
    args = _parser().parse_args()
    dsn = _dsn()
    control = ControlPlane(dsn)
    if args.command == "migrate":
        control.apply_migrations()
    elif args.command == "create-generation":
        control.register_generation(
            args.generation_id, args.table, args.profile, args.dimension
        )
        store = PgVectorStore(
            dsn, args.dimension, table=args.table, tenant=args.tenant,
            generation_id=args.generation_id,
        )
        try:
            store.ensure_schema()
        except Exception:
            control.set_generation_state(args.generation_id, "failed")
            raise
        finally:
            store.close()
    elif args.command == "mark-ready":
        control.set_generation_state(
            args.generation_id, "ready",
            chunk_count=args.chunks, source_count=args.sources,
        )
    elif args.command == "set-route":
        control.set_route(
            args.tenant, args.active_generation, args.shadow_generation
        )
    elif args.command == "cutover":
        control.cutover(args.tenant)


if __name__ == "__main__":
    main()
