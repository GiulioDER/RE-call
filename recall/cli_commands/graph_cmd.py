"""`recall graph`: inspect or rebuild the derived semantic evidence graph."""

from __future__ import annotations

import argparse
import json

from recall.generations import GenerationManager


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = sub.add_parser(
        "graph", help="inspect or rebuild the derived semantic evidence graph"
    )
    parser.set_defaults(_opens_db=True, func=_cmd_graph)
    graph_sub = parser.add_subparsers(dest="graph_cmd", required=True)
    rebuild = graph_sub.add_parser(
        "rebuild", help="rebuild deterministic graph rows for an existing generation"
    )
    rebuild.add_argument("--generation", required=True)


def _cmd_graph(args: argparse.Namespace) -> None:
    manager = GenerationManager(args.dsn, args.tenant)
    if args.graph_cmd == "rebuild":
        readiness = manager.rebuild_graph(args.generation)
        print(json.dumps(readiness.__dict__, indent=2, default=str))
        return
    raise SystemExit(f"unknown graph subcommand: {args.graph_cmd}")
