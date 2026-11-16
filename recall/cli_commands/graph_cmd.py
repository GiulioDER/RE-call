"""`recall graph`: inspect or rebuild the derived semantic evidence graph.

On PostgreSQL the graph belongs to a generation (`recall.generations`); on a lite store it belongs
to the file's one corpus (`recall.lite.graph`), so `--generation` is optional there and, when
given, must name that corpus.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from contextlib import closing
from typing import TYPE_CHECKING

from recall.lite import is_lite_dsn, lite_dsn_path

if TYPE_CHECKING:
    from recall.semantic_graph import GraphReadiness


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = sub.add_parser(
        "graph", help="inspect or rebuild the derived semantic evidence graph"
    )
    parser.set_defaults(_opens_db=True, func=_cmd_graph)
    graph_sub = parser.add_subparsers(dest="graph_cmd", required=True)
    rebuild = graph_sub.add_parser(
        "rebuild", help="rebuild deterministic graph rows for an existing generation"
    )
    rebuild.add_argument(
        "--generation", help="the generation to rebuild: required on PostgreSQL; a lite store's own corpus by default"
    )
    status = graph_sub.add_parser("status", help="show the graph build marker and its recorded counts")
    status.add_argument("--generation", help="the generation to report: the active one by default")


def _cmd_graph(args: argparse.Namespace) -> None:
    if is_lite_dsn(args.dsn):
        readiness = _lite_graph(args)
    elif args.graph_cmd == "rebuild":
        from recall.generations import GenerationManager

        if not args.generation:
            raise SystemExit("recall graph rebuild: --generation is required on PostgreSQL")
        readiness = GenerationManager(args.dsn, args.tenant).rebuild_graph(args.generation)
    else:
        from recall.generation_store import GenerationStore
        from recall.generation_types import NoActiveGeneration

        # Reads the readiness marker only, so the width is a placeholder.
        store = GenerationStore(args.dsn, 1, tenant=args.tenant)
        try:
            readiness = store.graph_readiness(args.generation)
        except NoActiveGeneration as exc:  # a message, not a traceback (CCA BUG-005)
            raise SystemExit(f"recall graph status: {exc}; pass --generation to read another") from exc
        finally:
            store.close()
    print(json.dumps(readiness.__dict__, indent=2, default=str))


def _lite_graph(args: argparse.Namespace) -> GraphReadiness:
    from recall.backends import write_session
    from recall.cli_commands._lite import lite_read_meta
    from recall.lite import LiteStore
    from recall.trust import terminal_safe

    command = f"recall graph {args.graph_cmd}"
    meta = lite_read_meta(command, args)
    if meta is None:
        raise SystemExit(f"{command}: no lite store at {lite_dsn_path(args.dsn)}; `recall index <memory folder>` creates it")
    if meta.get("tenant") != args.tenant:
        held = terminal_safe(meta.get("tenant"))
        raise SystemExit(f"{command}: {lite_dsn_path(args.dsn)} holds tenant {held!r}; pass --tenant {held} to read it")

    def refuse_another(current: str) -> None:
        if args.generation and args.generation != current:
            raise SystemExit(
                f"{command}: a lite store has one corpus, {current}; --generation {terminal_safe(args.generation)} is not it"
            )

    if args.graph_cmd == "status":
        from recall.lite.graph import read_marker, readiness
        from recall.lite.store import corpus_fingerprint_of, generation_id_for

        # A diagnostic must also work before graph tables exist, without creating them.
        with closing(sqlite3.connect(f"{lite_dsn_path(args.dsn).resolve().as_uri()}?mode=ro", uri=True)) as conn:
            conn.execute("BEGIN")
            if conn.execute("SELECT value FROM meta WHERE key = 'tenant'").fetchone() != (args.tenant,):
                raise SystemExit(f"{command}: the store tenant changed during inspection; retry")
            marker = read_marker(conn)
            generation = str(marker["generation_id"]) if marker is not None else generation_id_for(
                corpus_fingerprint_of(conn.execute("SELECT id, text, embedding FROM chunks"))
            )
            result = readiness(marker, args.tenant, generation)
            refuse_another(result.generation_id)
            return result

    # `lite_read_meta` has refused a table other than the file's one.
    with LiteStore.from_dsn(args.dsn, dim=int(meta["dim"]), tenant=args.tenant) as store:
        # Hash before the lane so an unchanged corpus is a cache hit under it (CCA PERF-001).
        store.lite_generation_id()
        with write_session(store, "graph rebuild"):
            refuse_another(store.lite_generation_id())
            graph = store.rebuild_semantic_graph()
            refuse_another(graph.generation_id)
            with store.snapshot():
                result = store.graph_readiness()
                if result != graph.readiness():
                    raise SystemExit(f"{command}: the corpus changed during the graph rebuild; retry")
            return result
