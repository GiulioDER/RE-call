"""`recall lite`: commands for the one-file local store (`recall setup --lite`).

`recall lite refresh` brings the store up to date with its memo folder; the MCP server does the same
on its own (`recall_mcp.lite_refresh`), so this is for a person, a script, or a scheduler. `--check`
says what a refresh would find and changes nothing. See `recall.lite.refresh`.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

#: Exit codes. 3 is the embedding queue's "busy", so a script can treat both alike.
EXIT_OK, EXIT_FAILED, EXIT_BUSY = 0, 1, 3


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_lite = sub.add_parser("lite", help="the one-file local store: keep it current with its memo folder")
    lite_sub = p_lite.add_subparsers(dest="lite_cmd", required=True)
    p_refresh = lite_sub.add_parser(
        "refresh",
        help="index what changed in the memo folder and recalibrate; --check reports only",
        description=(
            "Bring a lite store up to date with its memo folder: skip at once when no memo changed, "
            "otherwise index under the store's write lane (a refresh already queued makes this one "
            "unnecessary) and recalibrate. Exit 0 refreshed, unchanged or coalesced; 1 failed; 3 busy. "
            "With --check: write nothing, exit 0 when current and 1 when a refresh would index something."
        ),
    )
    p_refresh.set_defaults(func=_cmd_refresh)
    p_refresh.add_argument("--store", type=Path, default=None,
                           help="the SQLite file (default: .recall/memory.db in the current directory)")
    p_refresh.add_argument("--memory", type=Path, action="append", default=None,
                           help="a memo folder; repeatable (default: RECALL_INDEX_ROOT, else ./memory)")
    p_refresh.add_argument("--embedder", default=None,
                           help="the embedder the store was built with (default: RECALL_EMBEDDER, else "
                                "the one the store recorded)")
    p_refresh.add_argument("--force", action="store_true", help="refresh even when no memo changed")
    p_refresh.add_argument("--wait", type=float, default=None,
                           help="seconds to queue for the write lane (default: RECALL_LITE_WRITE_WAIT, 300)")
    p_refresh.add_argument("--check", action="store_true", help="report what a refresh would find; change nothing")
    p_refresh.add_argument("--json", action="store_true", help="print the report as JSON")


def _roots(args: argparse.Namespace) -> list[Path]:
    if args.memory:
        return list(args.memory)
    return [Path(os.environ.get("RECALL_INDEX_ROOT", "").strip() or "memory")]


def _cmd_refresh(args: argparse.Namespace) -> None:
    from recall.lite.refresh import Outcome, check, refresh

    db = (args.store or Path.cwd() / ".recall" / "memory.db").resolve()
    roots = _roots(args)
    if args.check:
        report = check(db, roots)
        print(json.dumps(report.to_dict(), indent=2, default=str) if args.json else report.render())
        raise SystemExit(EXIT_OK if report.current else EXIT_FAILED)
    if not db.exists():
        raise SystemExit(f"no lite store at {db}; create one with `recall setup --lite`")

    from recall.embeddings import resolve_embedder
    from recall.lite import LiteStore
    from recall.lite.inspect import inspect_lite_file

    name = args.embedder or os.environ.get("RECALL_EMBEDDER", "").strip() or inspect_lite_file(db).embedder_model
    if not name:
        # A wrong embedder of the same width does not raise: it ranks nonsense. So never guess.
        raise SystemExit(
            f"which embedder built {db}? Pass --embedder or set RECALL_EMBEDDER (the store has not "
            "recorded one yet)"
        )
    embedder = resolve_embedder(name)
    with LiteStore(db, dim=embedder.dim) as store:
        result = refresh(store, roots, embedder, force=args.force, wait_s=args.wait)
    print(json.dumps(result.to_dict(), indent=2, default=str) if args.json else result.render())
    if result.outcome is Outcome.FAILED:
        raise SystemExit(EXIT_FAILED)
    if result.outcome is Outcome.BUSY:
        raise SystemExit(EXIT_BUSY)
