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
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from recall.security_policy import AccessContext, SourceSecurityPolicy

#: Exit codes. 3 is the embedding queue's "busy", so a script can treat both alike.
EXIT_OK, EXIT_FAILED, EXIT_BUSY = 0, 1, 3


def register(sub: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    p_lite = sub.add_parser(
        "lite",
        help="the one-file local store: keep it current with its memo folder",
        description=(
            "Commands for the one-file local store that `recall setup --lite` creates. `refresh` brings "
            "it up to date with its memo folder; the MCP server does the same on its own."
        ),
    )
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
                           help="a memo folder; repeatable (default: the folders the store was last "
                                "refreshed from, else RECALL_INDEX_ROOT, else the current directory, "
                                "as `recall setup --lite` defaults)")
    p_refresh.add_argument("--embedder", default=None,
                           help="the embedder the store was built with (default: RECALL_EMBEDDER, else "
                                "the one the store recorded)")
    p_refresh.add_argument("--force", action="store_true", help="refresh even when no memo changed")
    p_refresh.add_argument("--allow-prune", action="store_true",
                           help="accept removing the rows of more than half the indexed memos when "
                                "their files are gone (the guard refuses it otherwise); never automatic")
    p_refresh.add_argument("--wait", type=float, default=None,
                           help="seconds to queue for the write lane (default: RECALL_LITE_WRITE_WAIT, 300)")
    p_refresh.add_argument("--check", action="store_true", help="report what a refresh would find; change nothing")
    p_refresh.add_argument("--json", action="store_true", help="print the report as JSON")


def _roots(args: argparse.Namespace, db: Path) -> list[Path]:
    """The memo folders: as given, else those the store was last refreshed from, else as setup.

    The recorded folders come first because `recall setup --lite` writes RECALL_INDEX_ROOT into the
    MCP server's environment, not the shell's: a plain `recall lite refresh` must still find the
    folder the server refreshes, or the two would fight over one state file (CCA ENV-002).
    """
    from recall.lite.refresh import read_state

    if args.memory:
        return list(args.memory)
    recorded = read_state(db).get("roots")
    if isinstance(recorded, list) and recorded and all(isinstance(r, str) for r in recorded):
        return [Path(r) for r in recorded]
    return [Path(os.environ.get("RECALL_INDEX_ROOT", "").strip() or Path.cwd())]


def _policy(tenant: str) -> tuple[SourceSecurityPolicy | None, AccessContext | None]:
    """The operator's source policy and the refresh context, or (None, None).

    The context is the server's (`refresh_access_context`), not `recall index`'s: both write one
    state file, so they must decide alike.
    """
    from recall.lite.refresh import refresh_access_context
    from recall.security_policy import load_source_policy

    policy = load_source_policy()
    if policy is None:
        return None, None
    return policy, refresh_access_context(tenant)


def _cmd_refresh(args: argparse.Namespace) -> None:
    from recall.lite.refresh import Outcome, check, refresh

    from recall.lite.inspect import inspect_lite_file
    from recall.lite.refresh import policy_key

    db = (args.store or Path.cwd() / ".recall" / "memory.db").resolve()
    roots = _roots(args, db)
    if args.check:
        policy, context = _policy(inspect_lite_file(db).tenant or "default")
        report = check(db, roots, policy=policy_key(policy, context))
        print(json.dumps(report.to_dict(), indent=2, default=str) if args.json else report.render())
        raise SystemExit(EXIT_OK if report.current else EXIT_FAILED)
    if not db.exists():
        raise SystemExit(f"no lite store at {db}; create one with `recall setup --lite`")

    from recall.embeddings import resolve_embedder
    from recall.lite import LiteStore

    name = args.embedder or os.environ.get("RECALL_EMBEDDER", "").strip() or inspect_lite_file(db).embedder_model
    if not name:
        # A wrong embedder of the same width does not raise: it ranks nonsense. So never guess.
        raise SystemExit(
            f"which embedder built {db}? Pass --embedder or set RECALL_EMBEDDER (the store has not "
            "recorded one yet)"
        )
    embedder = resolve_embedder(name)
    with LiteStore(db, dim=embedder.dim) as store:
        policy, context = _policy(store.tenant)
        result = refresh(
            store,
            roots,
            embedder,
            force=args.force,
            wait_s=args.wait,
            allow_prune=args.allow_prune,
            security_policy=policy,
            security_context=context,
        )
    print(json.dumps(result.to_dict(), indent=2, default=str) if args.json else result.render())
    if result.outcome is Outcome.FAILED:
        raise SystemExit(EXIT_FAILED)
    if result.outcome is Outcome.BUSY:
        raise SystemExit(EXIT_BUSY)
