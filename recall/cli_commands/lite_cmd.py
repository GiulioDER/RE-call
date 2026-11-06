"""`recall lite`: commands for the one-file local store (`recall setup --lite`).

`recall lite refresh` brings the store up to date with its memo folder; the MCP server does the same
on its own (`recall_mcp.lite_refresh`), so this is for a person, a script, or a scheduler. `--check`
says what a refresh would find and changes nothing. See `recall.lite.refresh`.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import TYPE_CHECKING

from recall.lite.writer_lock import DEFAULT_WAIT_S, WAIT_ENV

if TYPE_CHECKING:
    from recall.index_budget import IndexBudget
    from recall.lite.refresh import Snapshot
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
            "With --check: write nothing, exit 0 when current and 1 otherwise (changed, never "
            "refreshed, or the memo folder or the store missing)."
        ),
    )
    p_refresh.set_defaults(func=_cmd_refresh)
    p_refresh.add_argument("--store", type=Path, default=None,
                           help="the SQLite file (default: the file a sqlite:/// --serving-dsn, "
                                "RECALL_SERVING_DSN or RECALL_DSN names, else .recall/memory.db in "
                                "the current directory)")
    p_refresh.add_argument("--memory", type=Path, action="append", default=None,
                           help="a memo folder; repeatable (default: the folders the store was last "
                                "refreshed from, else RECALL_INDEX_ROOT, else the current directory, "
                                "as `recall setup --lite` defaults)")
    from recall.cli import StoreGiven

    # SUPPRESS, not None: a default here would overwrite `recall --embedder X lite refresh`'s X.
    p_refresh.add_argument("--embedder", action=StoreGiven, default=argparse.SUPPRESS,
                           help="the embedder to refresh with (default: RECALL_EMBEDDER, else the one "
                                "the store recorded). A RECALL_EMBEDDER that differs from the recorded "
                                "one is refused without --force; a different value typed here switches "
                                "the store to it, with a warning, and re-embeds the memo folders")
    p_refresh.add_argument("--force", action="store_true",
                           help="refresh even when no memo changed; also confirms switching to a "
                                "RECALL_EMBEDDER that differs from the embedder the store recorded, "
                                "which re-embeds every memo under the memo folders")
    p_refresh.add_argument("--allow-prune", action="store_true",
                           help="accept removing the rows of more than half the indexed memos when "
                                "their files are gone (the guard refuses it otherwise); never automatic")
    p_refresh.add_argument("--wait", type=_finite_seconds, default=None,
                           help=f"seconds to queue for the write lane (default: {WAIT_ENV['write']}, "
                                f"{DEFAULT_WAIT_S['write']:.0f})")
    p_refresh.add_argument("--check", action="store_true", help="report what a refresh would find; change nothing")
    p_refresh.add_argument("--json", action="store_true", help="print the report as JSON")


def _finite_seconds(value: str) -> float:
    """`--wait` as a number of seconds, refusing nan and inf as `writer_lock._wait_default` refuses
    them from the environment: nan would read as "try once" and inf as "wait forever", both
    silently (CCA NUM-007). A negative value is clamped to 0 downstream, as from the environment."""
    seconds = float(value)
    if not math.isfinite(seconds):
        raise argparse.ArgumentTypeError(f"must be a finite number of seconds, got {value!r}")
    return seconds


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


def _key_root(db: Path) -> Path | None:
    """The folder stored names are relative to: the server's (recorded in the state file), else
    RECALL_INDEX_ROOT, else none (each memo folder names its own files)."""
    from recall.lite.refresh import read_state

    recorded = read_state(db).get("key_root")
    if isinstance(recorded, str) and recorded:
        return Path(recorded)
    configured = os.environ.get("RECALL_INDEX_ROOT", "").strip()
    return Path(configured) if configured else None


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


def _budget() -> IndexBudget | None:
    """The configured budget, or None (with a warning) when it is malformed: it never stops a refresh."""
    import sys

    from recall.index_budget import budget_from_env

    try:
        return budget_from_env(os.environ)
    except ValueError as exc:
        print(f"warning: {exc}; the budget is not checked", file=sys.stderr)
        return None


def _warn_if_over_budget(snap: Snapshot, budget: IndexBudget | None) -> None:
    """A person may index past the budget; the server will not, so say so (CCA SEC-002)."""
    import sys

    from recall.lite.refresh import over_budget

    if budget is None:
        return
    refusal = over_budget(snap, budget)
    if refusal is not None:
        print(
            f"warning: {refusal.removesuffix(' Nothing was indexed.')} This refresh ran anyway, "
            "because a person asked for it; the MCP server refuses to refresh a folder this large "
            "on its own.",
            file=sys.stderr,
        )


def _cmd_refresh(args: argparse.Namespace) -> None:
    from recall.lite.refresh import Outcome, check, policy_key, refresh, snapshot

    from recall.lite.inspect import inspect_lite_file

    from recall.lite import is_lite_dsn, lite_dsn_path
    from recall.lite.setup import default_store_path

    # The store `recall search` and `recall index` follow when no --store is typed: the lite DSN
    # `recall setup --lite` prints for the shell, else the default in the current directory
    # (CCA ENV-006).
    db = (
        args.store
        or (lite_dsn_path(args.dsn) if is_lite_dsn(args.dsn) else default_store_path(Path.cwd()))
    ).resolve()
    roots = _roots(args, db)
    budget = _budget()
    if args.check:
        # "default" is `LiteStore`'s own default tenant, which is a literal there too.
        policy, context = _policy(inspect_lite_file(db).tenant or "default")
        report = check(db, roots, policy=policy_key(policy, context), budget=budget)
        print(json.dumps(report.to_dict(), indent=2, default=str) if args.json else report.render())
        raise SystemExit(EXIT_OK if report.current else EXIT_FAILED)
    if not db.exists():
        raise SystemExit(f"no lite store at {db}; create one with `recall setup --lite`")

    from recall.cli_commands import _shared
    from recall.lite import LiteStore

    info = inspect_lite_file(db)
    spec, profile, switched = _shared._lite_embedder_spec(
        args, info.embedder_spec, db, force=args.force,
        how_to_switch="pass --force to re-embed every memo with it",
        recorded_profile=info.embedder_profile,
    )
    name = spec or info.embedder_model
    ask = f"which embedder built {db}? Pass --embedder or set RECALL_EMBEDDER"
    if not name:
        # A wrong embedder of the same width does not raise: it ranks nonsense. So never guess.
        raise SystemExit(f"{ask} (the store has not recorded one yet)")
    try:
        # The shared helper, so refresh translates an unusable embedder exactly as every other
        # command does (CCA CODE-029); it raises SystemExit chained to the resolver's error.
        embedder = _shared._make_embedder(name)
    except SystemExit as exc:
        if not spec and isinstance(exc.__cause__, ValueError):
            # A store set up before the spec was recorded holds only the model's own name, such
            # as BAAI/bge-small-en-v1.5, which names no provider; guessing one is the nonsense above.
            raise SystemExit(
                f"{ask} (the store recorded the model {name!r}, which names no embedder)"
            ) from None
        raise
    _shared._refuse_unbound_embedder(embedder, info.embedder_model, db, args, switched=switched)
    # A typed embedder the calibration is not bound to must re-embed too, or the refusal's own
    # advice (pass --embedder) would end `unchanged` on an unchanged folder.
    rebind = info.embedder_model is not None and embedder.name != info.embedder_model
    # A store that recorded no spec (built by an older `recall index`) records this one, which only
    # a run that reaches the write lane does; an unchanged folder would return before it (CCA BUG-003).
    # A store that recorded a spec before profiles were recorded adopts its profile the same way,
    # or every profile guard would skip it for good (CCA STAKES-001).
    adopt = bool(spec) and (not info.embedder_spec or info.embedder_profile is None)
    # One walk serves both the refresh's change check and the budget warning below (CCA
    # PERF-004): the memo files are the same before and after, and the walk is the cost.
    snap = snapshot(roots)
    with LiteStore(db, dim=embedder.dim) as store:
        policy, context = _policy(store.tenant)
        result = refresh(
            store,
            roots,
            embedder,
            # A switch must reach the indexer: the no-change check looks at the memos only (CCA
            # BUG-001), so without this a typed switch on an unchanged folder did nothing.
            force=args.force or switched or rebind or adopt,
            wait_s=args.wait,
            allow_prune=args.allow_prune,
            security_policy=policy,
            security_context=context,
            key_root=_key_root(db),
            before=snap,
            # Also on a store that recorded no spec, so the guard engages from now on, and on a
            # rebind, so the binding names the new vectors.
            record_spec=spec if spec and (switched or rebind or adopt) else None,
            record_profile=profile,
        )
    print(json.dumps(result.to_dict(), indent=2, default=str) if args.json else result.render())
    _warn_if_over_budget(snap, budget)
    if result.outcome is Outcome.FAILED:
        raise SystemExit(EXIT_FAILED)
    if result.outcome is Outcome.BUSY:
        raise SystemExit(EXIT_BUSY)
