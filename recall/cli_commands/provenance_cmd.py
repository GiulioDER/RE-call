"""The ``recall provenance`` command group."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any, cast

from recall.provenance_controller import (
    FactApplicationRequest,
    ProvenanceController,
    evidence_card_from_payload,
    source_digest,
)
from recall.types import AtomicFact, EvidenceCard

if TYPE_CHECKING:
    from recall.lite import LiteStore
    from recall.store import PgVectorStore


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = sub.add_parser(
        "provenance",
        help="apply structured facts through the deterministic provenance controller",
        description=(
            "Apply structured facts through server-created evidence cards, or inspect the current "
            "tenant-scoped deterministic provenance projection."
        ),
    )
    parser.set_defaults(_opens_db=True, func=_cmd_provenance)
    commands = parser.add_subparsers(dest="provenance_cmd", required=True)

    apply = commands.add_parser("apply", help="apply a claim using server-created evidence cards")
    apply.add_argument("--claim", required=True, help="JSON file containing one atomic fact")
    apply.add_argument("--cards", required=True, help="JSON file containing evidence cards")
    apply.add_argument("--request-id", required=True)
    apply.add_argument("--generation", required=True)
    apply.add_argument("--writer", default="cli")
    apply.add_argument("--sqlite-path", default=None, metavar="PATH")
    apply.add_argument("--source-root", default=None, metavar="PATH")

    current = commands.add_parser("current", help="read the current fact projection")
    current.add_argument("--as-of", default=None)
    current.add_argument("--sqlite-path", default=None, metavar="PATH")

    rebuild = commands.add_parser(
        "rebuild",
        help="rebuild the materialized current facts from the fact ledger",
        description=(
            "Rebuild the tenant's materialized current facts (recall.fact_current) from its "
            "immutable fact ledger. They are derived state: a read replays the ledger whenever "
            "they lag it, so this changes how fast a read is answered, never what it answers. "
            "--check reports without writing; --reconcile materializes only the ledger events "
            "the tables lack, through the materialization outbox on PostgreSQL."
        ),
    )
    mode = rebuild.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true", help="report whether the tables are current; write nothing")
    mode.add_argument("--reconcile", action="store_true", help="materialize only the events the tables lack")
    rebuild.add_argument("--sqlite-path", default=None, metavar="PATH")


def _read_cards(path: str) -> tuple[EvidenceCard, ...]:
    payload: Any = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        payload = payload.get("cards", [])
    if not isinstance(payload, list):
        raise SystemExit("--cards must contain a JSON array or an evidence result object")
    try:
        return tuple(evidence_card_from_payload(item) for item in payload)
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"invalid evidence card payload: {exc}") from exc


def _decision_payload(decision: Any) -> dict[str, object]:
    return {
        "allowed": decision.allowed,
        "decision_code": str(decision.code),
        "request_id": decision.request_id,
        "fact_id": decision.fact_id,
        "retried": decision.retried,
        "detail": decision.detail,
        "event_id": decision.event.event_id if decision.event else None,
        "evidence_card_ids": [card.card_id for card in decision.cards],
    }


def _lite_file(path: Path) -> tuple[dict[str, str] | None, set[str]]:
    """A lite file's `meta` table (None when it has none) and its table names, read-only."""
    import sqlite3

    try:
        conn = sqlite3.connect(f"{path.resolve().as_uri()}?mode=ro", uri=True)
    except sqlite3.Error:
        return None, set()
    try:
        tables = {str(row[0]) for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        if "meta" not in tables:
            return None, tables
        return {str(key): str(value) for key, value in conn.execute("SELECT key, value FROM meta").fetchall()}, tables
    except sqlite3.Error:
        return None, set()
    finally:
        conn.close()


def _lite_target(args: argparse.Namespace) -> tuple[Path, dict[str, str], set[str]]:
    """The lite file a `sqlite:///` DSN names, checked without writing to it.

    The generation route is refused by name; a file that does not exist, or is not a lite store,
    is refused untouched. Opening a `LiteStore` on either would create it, or stamp it with a
    tenant and a width, which a later `recall index` at the real width then refuses.
    """
    from recall.backends import lite_route_refusal
    from recall.lite import LITE_DSN_PREFIX
    from recall.runtime_route import resolve_runtime_route

    route = getattr(args, "_runtime_route", None) or resolve_runtime_route()
    if (refusal := lite_route_refusal(args.dsn, route)) is not None:
        raise SystemExit(refusal)
    path = Path(args.dsn[len(LITE_DSN_PREFIX):])
    if not path.exists():
        raise SystemExit(f"{path} does not exist; nothing was read or created (build it with `recall setup --lite`)")
    meta, tables = _lite_file(path)
    if meta is None or "dim" not in meta:
        raise SystemExit(f"{path} is not a lite store (it has no RE-call meta table); nothing was read or written")
    return path, meta, tables


def _open_lite(args: argparse.Namespace, dim: int) -> LiteStore:
    """The lite store at the width `dim`, a mismatch refused by name instead of a traceback."""
    from recall.backends import open_legacy_store
    from recall.lite import LiteStoreError

    try:
        store = open_legacy_store(args.dsn, dim=dim, table=args.table, tenant=args.tenant)
    except LiteStoreError as exc:
        raise SystemExit(f"provenance: {exc}") from exc
    return cast("LiteStore", store)


def _rebuild_postgres(args: argparse.Namespace, action: str) -> tuple[Any, int]:
    from recall.fact_current import PostgresCurrentFacts
    from recall.fact_ledger import PostgresMaterializationOutbox
    from recall.provenance_controller import MaterializationRecovery

    facts = PostgresCurrentFacts(args.dsn, tenant_id=args.tenant)
    if action == "check":
        return facts.status(), 0
    if action == "rebuild":
        return facts.rebuild(), 0
    # Crash recovery: the events the tables lack get outbox rows and are delivered through them,
    # a thousand at a time, until none is left or a batch delivers nothing (its failures are in
    # the outbox's last_error).
    recovery = MaterializationRecovery(
        tenant_id=args.tenant,
        outbox=PostgresMaterializationOutbox(args.dsn, tenant_id=args.tenant),
        materializer=facts,
    )
    reconciled = 0
    while missing := facts.unfiled_events():
        delivered = recovery.reconcile(missing[:1000], limit=1000)
        if not delivered:
            break
        reconciled += delivered
    return facts.status(), reconciled


def _rebuild_lite(args: argparse.Namespace, action: str) -> tuple[Any, int]:
    from recall.fact_current import CURRENT_EVENTS_TABLE, CurrentFactsStatus, SQLiteCurrentFacts
    from recall.fact_ledger import FACT_LEDGER_TABLE
    from recall.lite.facts import ensure_fact_tables

    path, meta, tables = _lite_target(args)
    if meta.get("tenant") != args.tenant:
        raise SystemExit(f"provenance rebuild: {path} holds tenant {meta.get('tenant')!r}; pass --tenant {meta.get('tenant')}")
    if FACT_LEDGER_TABLE not in tables:
        # No fact was ever applied: nothing to rebuild, and opening the ledger would create it.
        return CurrentFactsStatus(args.tenant, 0, 0, 0, "table", None), 0
    with _open_lite(args, int(meta["dim"])) as store:
        if action == "check" and CURRENT_EVENTS_TABLE not in tables:
            (count,), = store._metadata_values(
                "SELECT count(*) FROM recall_fact_ledger_events WHERE tenant_id = ?", (args.tenant,)
            )
            reason = "the file has no current-facts tables yet"
            return CurrentFactsStatus(args.tenant, int(count), 0, 0, "table", reason), 0
        ensure_fact_tables(store)
        facts = SQLiteCurrentFacts(store._conn, tenant_id=args.tenant, write=store._write)
        if action == "check":
            return facts.status(), 0
        if action == "rebuild":
            return facts.rebuild(), 0
        # A lite file has no outbox: its appends materialize in their own transaction, so what
        # the tables lack was written by an older build, and is filed directly.
        reconciled = facts.reconcile()
        return facts.status(), reconciled


def _cmd_rebuild(args: argparse.Namespace, lite: bool) -> None:
    if args.sqlite_path:
        raise SystemExit(
            "provenance rebuild: the --sqlite-path adapter replays its file in memory and keeps no "
            "current-facts tables; pass the store's DSN instead (PostgreSQL, or a sqlite:/// lite store)"
        )
    action = "check" if args.check else "reconcile" if args.reconcile else "rebuild"
    status, reconciled = _rebuild_lite(args, action) if lite else _rebuild_postgres(args, action)
    print(json.dumps(
        {"action": action, "reconciled": reconciled, **status.to_payload()},
        ensure_ascii=False, indent=2, sort_keys=True,
    ))


def _cmd_provenance(args: argparse.Namespace) -> None:
    from recall.backends import is_lite_dsn
    from recall.fact_ledger import PostgresFactLedger, SQLiteFactLedger

    # A `sqlite:///` DSN is a lite store: its own fact ledger and cards, the same output as
    # PostgreSQL. `--sqlite-path` stays the separate local adapter it always was.
    lite = not args.sqlite_path and is_lite_dsn(args.dsn)
    if args.provenance_cmd == "rebuild":
        _cmd_rebuild(args, lite)
        return
    if args.provenance_cmd == "current":
        instant = (
            datetime.fromisoformat(args.as_of.replace("Z", "+00:00"))
            if args.as_of
            else datetime.now(UTC)
        )
        if args.sqlite_path:
            with SQLiteFactLedger(args.sqlite_path, tenant_id=args.tenant) as ledger:
                events = ledger.current(tenant_id=args.tenant, now=instant)
        elif lite:
            from recall.fact_ledger import FACT_LEDGER_TABLE
            from recall.lite.facts import LiteFactLedger

            # A read needs no embedder: the width is the one the file records.
            path, meta, tables = _lite_target(args)
            if meta.get("tenant") != args.tenant:
                # As on PostgreSQL, a tenant with no rows reads no facts; a lite file holds one.
                print(f"{path} holds tenant {meta.get('tenant')!r}; pass --tenant {meta.get('tenant')} to read it", file=sys.stderr)
                events = ()
            elif FACT_LEDGER_TABLE not in tables:
                events = ()  # no fact was ever applied; opening the ledger would create its tables
            else:
                with _open_lite(args, int(meta["dim"])) as lite_store:
                    events = LiteFactLedger(lite_store).current(tenant_id=args.tenant, now=instant)
        else:
            events = PostgresFactLedger(args.dsn, tenant_id=args.tenant).current(
                tenant_id=args.tenant, now=instant
            )
        print(json.dumps({
            "tenant_id": args.tenant,
            "as_of": instant.isoformat(),
            "facts": [
                {
                    "event_id": event.event_id,
                    "fact_id": event.fact_id,
                    "fact": event.fact.to_payload() if event.fact else None,
                    "evidence_card_ids": [card.card_id for card in event.evidence_cards],
                    "generation_id": event.generation_id,
                    "writer": event.writer,
                    "asserted_at": event.created_at.isoformat(),
                }
                for event in events
            ],
        }, ensure_ascii=False, indent=2, sort_keys=True))
        return

    if args.provenance_cmd != "apply":
        raise SystemExit(f"unknown provenance command {args.provenance_cmd!r}")
    claim_payload = json.loads(Path(args.claim).read_text(encoding="utf-8"))
    cards = _read_cards(args.cards)
    claim = AtomicFact.from_payload(claim_payload)
    request = FactApplicationRequest(claim, tuple(card.card_id for card in cards), args.request_id)

    if args.sqlite_path:
        if not args.source_root:
            raise SystemExit("--source-root is required with --sqlite-path")
        root = Path(args.source_root).resolve()

        def current_digest(card: EvidenceCard) -> str | None:
            path = (root / card.source).resolve()
            try:
                path.relative_to(root)
                return source_digest(path.read_text(encoding="utf-8"))
            except (OSError, UnicodeError, ValueError):
                return None

        from recall.provenance_cards import SQLiteEvidenceCardStore

        with (
            SQLiteEvidenceCardStore(args.sqlite_path, tenant_id=args.tenant) as card_store,
            SQLiteFactLedger(args.sqlite_path, tenant_id=args.tenant) as ledger,
        ):
            card_store.put(cards)
            decision = ProvenanceController(
                tenant_id=args.tenant,
                generation_id=args.generation,
                cards=card_store,
                ledger=ledger,
                source_digest_for=current_digest,
                writer=args.writer,
            ).apply_fact(request)
        print(json.dumps(_decision_payload(decision), ensure_ascii=False, indent=2, sort_keys=True))
        return

    from recall.cli_commands._shared import _make_embedder
    from recall.store import PgVectorStore

    embedder = _make_embedder(args.embedder)
    if lite:
        path, meta, _tables = _lite_target(args)
        if meta.get("tenant") != args.tenant:
            raise SystemExit(f"provenance apply: {path} holds tenant {meta.get('tenant')!r}; pass --tenant {meta.get('tenant')}")
        if int(meta["dim"]) != embedder.dim:
            raise SystemExit(
                f"provenance apply: {path} was built at width {meta['dim']}, but the embedder "
                f"({args.embedder}) has width {embedder.dim}; use the embedder the store was built with"
            )
        with _open_lite(args, embedder.dim) as lite_store:
            # A lite store's generation is the digest of what it holds, so it cannot take one
            # from the command line. `apply_fact_memory` uses the store's own generation, so
            # without this check a stale --generation would be silently ignored.
            if args.generation != lite_store.generation_id:
                raise SystemExit(
                    f"provenance apply: --generation {args.generation!r} is not this lite store's; "
                    f"it is {lite_store.generation_id} now (it moves with every write to the store)"
                )
            _print_applied(cast("PgVectorStore", lite_store), embedder, claim_payload, cards, args)
        return
    with PgVectorStore(
        args.dsn,
        dim=embedder.dim,
        table=args.table,
        tenant=args.tenant,
        generation_id=args.generation,
    ) as store:
        _print_applied(store, embedder, claim_payload, cards, args)


def _print_applied(
    store: PgVectorStore,
    embedder: Any,
    claim_payload: Any,
    cards: Sequence[EvidenceCard],
    args: argparse.Namespace,
) -> None:
    """Apply the claim through the MCP path on `store` and print its decision, for either backend."""
    from recall_mcp.provenance import apply_fact_memory

    print(json.dumps(apply_fact_memory(
        store,
        embedder,
        claim=claim_payload,
        evidence_card_ids=[card.card_id for card in cards],
        request_id=args.request_id,
        writer=args.writer,
    ), ensure_ascii=False, indent=2, sort_keys=True))
