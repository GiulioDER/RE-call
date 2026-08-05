"""Operator CLI for versioned enterprise index generations.

Five of these subcommands exist because the machinery behind them did not: `replay_pending`,
`validate_generation_parity` and `check_enterprise_readiness` were reachable only from Python, and
a migration outbox with no drain command is a deadlock with a documented recovery nobody can run.
`ControlPlane.replay_pending` had exactly one reference in the whole repository, its own
definition.

The rule every subcommand here obeys: a physical table name is never taken from an argument. It is
read from `recall_index_generations` and revalidated against the identifier allowlist on the way
out (`recall.control_plane.validate_table_name`). An operator names a GENERATION; the registry
names the table.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys

import psycopg

from recall.control_plane import SERVABLE_STATES, ControlPlane, IndexGeneration
from recall.migration import validate_generation_parity
from recall.store import PgVectorStore, require_secure_dsn


def _dsn() -> str:
    dsn = os.environ.get("RECALL_DSN")
    if not dsn:
        raise SystemExit("RECALL_DSN is required")
    require_secure_dsn(dsn)
    return dsn


def _require_generation(control: ControlPlane, generation_id: str) -> IndexGeneration:
    generation = control.generation(generation_id)
    if generation is None:
        raise SystemExit(f"unknown generation: {generation_id}")
    return generation


def _open(dsn: str, generation: IndexGeneration, tenant: str) -> PgVectorStore:
    """Open a store on a generation's registered table. `physical_table` is already validated.

    `ControlPlane._generation` runs `validate_table_name` on every row it builds, so a table name
    that reached the registry by some other route than `create-generation` is refused here rather
    than interpolated into a query.
    """
    if generation.state not in SERVABLE_STATES:
        # `retire_generation` delegates its real protection to "the serving path refuses a
        # retired generation". This is a write path (`replay` calls `replace_sources` through
        # it) and it was not covered by that refusal, so retiring a generation still left a
        # command that would write into it. A `failed` generation is one whose DDL did not
        # finish, so the target may be half-built.
        raise SystemExit(
            f"generation {generation.generation_id!r} is {generation.state!r}; refusing to open "
            f"it. Route the tenant at a servable generation first."
        )
    return PgVectorStore(
        dsn,
        dim=generation.dimension,
        table=generation.physical_table,
        tenant=tenant,
        generation_id=generation.generation_id,
    )


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

    replay = commands.add_parser(
        "replay", help="drain this tenant's pending migration outbox"
    )
    replay.add_argument("tenant")
    parity = commands.add_parser(
        "parity", help="compare the active and shadow generations before cutover"
    )
    parity.add_argument("tenant")
    readiness = commands.add_parser(
        "readiness", help="run the enterprise readiness checks for one tenant"
    )
    readiness.add_argument("tenant")
    readiness.add_argument(
        "--profile",
        help="embedding profile id to construct; defaults to the active generation's profile",
    )
    readiness.add_argument(
        "--allow-legacy-profile",
        action="store_true",
        help="accept an unpinned model artifact and rows without profile metadata",
    )
    status = commands.add_parser(
        "status", help="generations, routes and outbox depth"
    )
    status.add_argument("--tenant", help="also report this tenant's route and pending events")
    status.add_argument("--json", action="store_true", dest="as_json")
    retire = commands.add_parser(
        "retire", help="mark a generation retired once it is no longer routed"
    )
    retire.add_argument("generation_id")
    retire.add_argument(
        "--tenant",
        required=True,
        help="tenant whose route must not reference the generation; routes are tenant isolated, "
        "so retirement is confirmed one tenant at a time",
    )
    return parser


def _cmd_replay(control: ControlPlane, dsn: str, tenant: str) -> int:
    """Open exactly the generations the pending events name, then drain in sequence order."""
    # Summaries, not payloads. Each payload carries every chunk's text AND embedding for a
    # batch, and all this needs is two generation ids per event; pulling the full outbox to read
    # them moved megabytes per event, three times over, in the command whose whole purpose is to
    # run when the outbox is at its largest.
    summaries = control.pending_event_summaries(tenant)
    if not summaries:
        print(f"no pending migration events for tenant {tenant!r}")
        return 0
    wanted: set[str] = set()
    for summary in summaries:
        for key in ("active_generation", "shadow_generation"):
            value = summary.get(key)
            if isinstance(value, str) and value:
                wanted.add(value)
    stores: dict[str, PgVectorStore] = {}
    try:
        for generation_id in sorted(wanted):
            generation = _require_generation(control, generation_id)
            stores[generation_id] = _open(dsn, generation, tenant)
        completed = control.replay_pending(tenant, stores)
    finally:
        for store in stores.values():
            store.close()
    remaining = control.pending_count(tenant)
    print(f"replayed {completed} event(s) for tenant {tenant!r}; {remaining} still pending")
    return 0 if remaining == 0 else 1


def _cmd_parity(control: ControlPlane, dsn: str, tenant: str) -> int:
    route = control.route(tenant)
    if route is None:
        raise SystemExit(f"tenant {tenant!r} has no route")
    if route.shadow is None:
        raise SystemExit(f"tenant {tenant!r} has no shadow generation to compare")
    # ExitStack, because opening the shadow can fail (bad table, dimension mismatch, timeout)
    # and the active store is already open by then. The previous `finally` covered only the
    # comparison, which was the one call that could not leak.
    with contextlib.ExitStack() as stack:
        active = stack.enter_context(_open(dsn, route.active, tenant))
        shadow = stack.enter_context(_open(dsn, route.shadow, tenant))
        parity = validate_generation_parity(active, shadow)
    print(f"active chunks: {parity.active_chunks}\nshadow chunks: {parity.shadow_chunks}")
    for label, values in (
        ("missing from shadow", parity.missing_sources),
        ("extra in shadow", parity.extra_sources),
        ("hash mismatch", parity.hash_mismatches),
    ):
        if values:
            print(f"{label}: {len(values)}")
    if parity.valid:
        print("parity: OK")
        return 0
    for failure in parity.failures:
        print(f"parity FAILED: {failure}", file=sys.stderr)
    return 1


def _cmd_readiness(
    control: ControlPlane, dsn: str, tenant: str, profile_id: str | None, allow_legacy: bool
) -> int:
    from recall.calibration import load_for as calibration_load_for
    from recall.embeddings import embedding_profile_id
    from recall.readiness import check_enterprise_readiness
    from recall_mcp.service import make_profile_embedder

    route = control.route(tenant)
    if route is None:
        raise SystemExit(f"tenant {tenant!r} has no route")
    embedder = make_profile_embedder(profile_id or route.active.embedding_profile)
    store = _open(dsn, route.active, tenant)
    try:
        result = check_enterprise_readiness(
            store,
            embedder,
            control_plane=control,
            calibration=calibration_load_for(embedding_profile_id(embedder)),
            allow_legacy_profile=allow_legacy,
        )
    finally:
        store.close()
    for warning in result.warnings:
        print(f"degraded: {warning}")
    for failure in result.failures:
        print(f"failed: {failure}", file=sys.stderr)
    print(f"ready: {result.ready}\ndegraded: {result.degraded}")
    return 0 if result.ready else 1


def _cmd_status(control: ControlPlane, tenant: str | None, as_json: bool) -> int:
    generations = control.generations()
    report: dict[str, object] = {
        "control_plane_ledger": control.ledger_state().describe(),
        "generations": [
            {
                "generation_id": g.generation_id,
                "physical_table": g.physical_table,
                "embedding_profile": g.embedding_profile,
                "dimension": g.dimension,
                "state": g.state,
                "chunk_count": g.chunk_count,
                "source_count": g.source_count,
            }
            for g in generations
        ],
    }
    if tenant:
        route = control.route(tenant)
        pending = control.pending_event_summaries(tenant)
        report["tenant"] = tenant
        report["route"] = None if route is None else {
            "active": route.active.generation_id,
            "shadow": None if route.shadow is None else route.shadow.generation_id,
            "updated_at": route.updated_at.isoformat(),
        }
        # Operation ids and counts only, and the QUERY projects them too. Saying "we do not
        # print the payload" while selecting it still moves the corpus text out of the
        # database and into this process; an operator status command is not a retrieval path,
        # so it must not read one.
        report["pending_events"] = [
            {k: event[k] for k in
             ("sequence_id", "operation_id", "operation_kind", "active_count", "shadow_count")}
            for event in pending
        ]
    if as_json:
        print(json.dumps(report, indent=2, sort_keys=True))
        return 0
    print(f"control plane ledger: {report['control_plane_ledger']}")
    for g in generations:
        print(
            f"  {g.generation_id}  {g.state:<9} {g.physical_table}  "
            f"{g.embedding_profile} dim={g.dimension} chunks={g.chunk_count}"
        )
    if tenant:
        route_report = report["route"]
        print(f"tenant {tenant}: route={route_report}")
        events = report["pending_events"]
        assert isinstance(events, list)
        print(f"  pending events: {len(events)}")
        for event in events:
            print(f"    {event}")
    return 0


def main() -> None:
    args = _parser().parse_args()
    dsn = _dsn()
    control = ControlPlane(dsn)
    if args.command == "migrate":
        control.apply_migrations()
    elif args.command == "create-generation":
        # PgVectorStore registers the vector codec as soon as it connects.  On PostgreSQL
        # installations where pgvector is not a trusted extension, creating it requires a
        # superuser.  Never solve that by elevating the migration role: require the database
        # operator to install the extension once, then keep generation DDL restricted here.
        with psycopg.connect(dsn, autocommit=True) as conn:
            vector_type = conn.execute("SELECT to_regtype('vector')").fetchone()
        if vector_type is None or vector_type[0] is None:
            raise RuntimeError(
                "pgvector is not installed in this database; a database operator must run "
                "CREATE EXTENSION vector before creating an index generation"
            )
        control.register_generation(
            args.generation_id, args.table, args.profile, args.dimension
        )
        store: PgVectorStore | None = None
        try:
            store = PgVectorStore(
                dsn, args.dimension, table=args.table, tenant=args.tenant,
                generation_id=args.generation_id,
            )
            store.ensure_schema()
        except Exception:
            control.set_generation_state(args.generation_id, "failed")
            raise
        finally:
            if store is not None:
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
    elif args.command == "replay":
        raise SystemExit(_cmd_replay(control, dsn, args.tenant))
    elif args.command == "parity":
        raise SystemExit(_cmd_parity(control, dsn, args.tenant))
    elif args.command == "readiness":
        raise SystemExit(
            _cmd_readiness(
                control, dsn, args.tenant, args.profile, args.allow_legacy_profile
            )
        )
    elif args.command == "status":
        raise SystemExit(_cmd_status(control, args.tenant, args.as_json))
    elif args.command == "retire":
        control.retire_generation(args.generation_id, args.tenant)
        print(f"retired {args.generation_id}")


if __name__ == "__main__":
    main()
