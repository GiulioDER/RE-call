"""`recall records`: audit records by exact key, the CLI twin of the `recall_records` MCP tool.

`--all` is the full dump CCA's round-trip check compares its ledger against
(`docs/AUDIT_MEMORY_DESIGN.md`, "Round-trip contract"), so it reads every record in one lookup
and one snapshot and prints one complete envelope, never a truncated one. Served on a lite store (`sqlite:///`) as on
PostgreSQL; reads no embedder and no vectors.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from recall.backends import lite_route_refusal, open_legacy_store
from recall.lite import is_lite_dsn, lite_dsn_path
from recall.records import RECORD_KINDS, RecordsQueryError, lookup_records
from recall.runtime_route import RouteConfigurationError, RuntimeRoute, resolve_runtime_route
from recall.store import DEFAULT_TABLE, PgVectorStore
from recall.trust import terminal_safe


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_records = sub.add_parser(
        "records",
        help="look up audit records (bugs, false positives, lessons) by anchor or id",
        description=(
            "Exact lookup of the `recall_record` objects memos carry, with no embedder and no "
            "query: by anchors (with --repo), by --id, or --all for a full dump. Each record is "
            "returned exactly as authored, with its match to the anchors asked for. Superseded "
            "and expired records are listed and marked, never hidden."
        ),
    )
    p_records.set_defaults(_opens_db=True, func=_cmd_records)
    form = p_records.add_mutually_exclusive_group(required=True)
    form.add_argument(
        "--anchors",
        metavar="FILE",
        help="a JSON list of {file, symbol, code_sha} objects (at most 200); `-` reads stdin",
    )
    form.add_argument(
        "--id", dest="ids", action="append", metavar="ID", help="a record id; repeat for several"
    )
    form.add_argument(
        "--all", dest="all_records", action="store_true", help="every record in the tenant"
    )
    p_records.add_argument("--repo", help="the repository anchors are paths in (owner/name)")
    p_records.add_argument(
        "--kind",
        dest="kinds",
        action="append",
        choices=sorted(RECORD_KINDS),
        help="only records of this kind; repeat for several",
    )
    p_records.add_argument(
        "--json", action="store_true", help="print the full JSON envelope (the MCP tool's output)"
    )


def _read_anchors(spec: str) -> Any:
    try:
        text = sys.stdin.read() if spec == "-" else Path(spec).read_text(encoding="utf-8")
        return json.loads(text)
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"recall records: cannot read anchors from {spec!r}: {exc}") from exc


def _lookup(store: Any, args: argparse.Namespace, anchors: Any) -> dict[str, Any]:
    if not args.all_records:
        return lookup_records(store, repo=args.repo, anchors=anchors, ids=args.ids, kinds=args.kinds)
    # One lookup, one snapshot, every record. Paging by separate calls read a separate snapshot per
    # page, and a legacy store has no generation id to tell them apart, so an index run landing
    # between pages duplicated or skipped records with `truncated` still false. A round-trip check
    # reads every record such a dump lacks as missing.
    return lookup_records(store, repo=args.repo, kinds=args.kinds, all_records=True, limit=None)


def _cmd_records(args: argparse.Namespace) -> None:
    anchors = _read_anchors(args.anchors) if args.anchors is not None else None
    try:
        route: RuntimeRoute = getattr(args, "_runtime_route", None) or resolve_runtime_route()
    except RouteConfigurationError as exc:
        raise SystemExit(f"recall records: {exc}") from exc
    if (refusal := lite_route_refusal(args.dsn, route)) is not None:
        raise SystemExit(refusal)
    # No embedder: the lookup reads metadata only, so the width is a placeholder on PostgreSQL and
    # the width the file records on a lite store.
    dimension = 1
    if is_lite_dsn(args.dsn):
        from recall.lite.inspect import lite_file_meta

        if args.table != DEFAULT_TABLE:
            raise SystemExit(
                f"recall records: a lite store ({args.dsn}) has one table, '{DEFAULT_TABLE}'; "
                f"--table {args.table!r} cannot be honoured. Use another sqlite:/// file instead"
            )
        path = lite_dsn_path(args.dsn)
        if not path.exists():
            # An empty corpus, as an empty PostgreSQL table reads; a lookup must not create the file.
            _print({"tenant_id": args.tenant, "generation_id": None, "truncated": False, "records": []}, args)
            return
        meta = lite_file_meta(path)
        if meta is None or "dim" not in meta:
            raise SystemExit(f"{path} is not a lite store (it has no RE-call meta table); nothing was read or written")
        if meta.get("tenant") != args.tenant:
            held = terminal_safe(meta.get("tenant"))
            raise SystemExit(f"recall records: {path} holds tenant {held!r}; pass --tenant {held} to read it")
        dimension = int(meta["dim"])
    if route.uses_generation:
        from recall.generation_store import GenerationStore

        store_context: PgVectorStore = GenerationStore(args.dsn, dimension, tenant=args.tenant)
    else:
        store_context = open_legacy_store(args.dsn, dim=dimension, table=args.table, tenant=args.tenant)
    with store_context as store:
        try:
            result = _lookup(store, args, anchors)
        except RecordsQueryError as exc:
            raise SystemExit(f"recall records: {exc}") from exc
    _print(result, args)


def _print(result: dict[str, Any], args: argparse.Namespace) -> None:
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return
    records = result["records"]
    if not records:
        print("no records match")
        return
    for record in records:
        marks = []
        if record["superseded_by"]:
            marks.append(f"superseded by {terminal_safe(record['superseded_by'])}")
        if record["valid_until"]:
            marks.append(f"valid until {terminal_safe(record['valid_until'])}")
        if record["caused"]:
            marks.append(f"caused {len(record['caused'])}")
        suffix = f"  ({'; '.join(marks)})" if marks else ""
        print(
            f"{record['anchor_match']:<14} {record['kind']:<25} {terminal_safe(record['id'])}"
            f"  {terminal_safe(record['source'])}{suffix}"
        )
    print(f"{len(records)} record(s), generation {terminal_safe(result['generation_id'])}")
