"""`recall schema`: inspect or apply versioned database migrations."""

from __future__ import annotations

import argparse

from recall.cli_commands._shared import _make_embedder


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_schema = sub.add_parser("schema", help="inspect or apply versioned database migrations")

    p_schema.set_defaults(_opens_db=True, func=_cmd_schema)
    p_schema.add_argument(
        "--dim",
        type=int,
        default=None,
        help="embedding dimension (default: infer from --embedder)",
    )
    schema_sub = p_schema.add_subparsers(dest="schema_cmd", required=True)
    schema_sub.add_parser("status", help="show installed and required schema versions")
    schema_sub.add_parser("plan", help="show pending migrations without changing the database")
    schema_sub.add_parser("apply", help="apply pending migrations with the migration role")
    p_schema_grants = schema_sub.add_parser(
        "grants",
        help="print the GRANT statements a serving role needs (prints SQL, runs none)",
    )
    p_schema_grants.add_argument("--role", required=True, help="the serving role name")
    p_schema_grants.add_argument(
        "--enterprise",
        action="store_true",
        help="also grant the enterprise control-plane tables and their sequence",
    )


def _cmd_schema(args: argparse.Namespace) -> None:
    from recall.schema import apply_migrations, schema_plan, schema_status

    if args.schema_cmd == "grants":
        # Prints SQL for an operator to run as the object owner; touches no database, so
        # it needs neither a DSN nor an embedder.
        from recall.schema import serving_grants

        for statement in serving_grants(
            args.role, table=args.table, enterprise=args.enterprise
        ):
            print(statement)
        return
    dim = args.dim if args.dim is not None else _make_embedder(args.embedder).dim
    inspect_dsn = args.migration_dsn or args.dsn
    if args.schema_cmd == "status":
        status = schema_status(inspect_dsn, table=args.table, dim=dim)
        print(f"table: {status.table}")
        print(f"current: {status.current_version or 'none'}")
        print(f"required: {status.required_version}")
        print(f"compatible: {'yes' if status.compatible else 'no'}")
        for migration in status.migrations:
            print(f"{migration.version} {migration.state:<7} {migration.filename}")
        if not status.compatible:
            raise SystemExit(1)
        return
    if args.schema_cmd == "plan":
        pending = schema_plan(inspect_dsn, table=args.table, dim=dim)
        if not pending:
            print("schema is current; no changes planned")
        else:
            for migration in pending:
                print(f"would apply {migration.version} {migration.filename}")
        return
    if not args.migration_dsn:
        raise SystemExit(
            "schema apply requires --migration-dsn or RECALL_MIGRATION_DSN; "
            "the serving DSN is never used for DDL"
        )
    applied = apply_migrations(args.migration_dsn, table=args.table, dim=dim)
    if not applied:
        print("schema is current; nothing applied")
    else:
        for migration in applied:
            print(f"applied {migration.version} {migration.filename}")
