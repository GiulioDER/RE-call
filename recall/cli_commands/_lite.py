"""The lite-store preflight shared by the metadata-only read commands (`recall scopes`,
`recall records`): one statement of what such a read refuses before it opens a store."""

from __future__ import annotations

import argparse

from recall.lite import lite_dsn_path
from recall.store import DEFAULT_TABLE


def lite_read_meta(command: str, args: argparse.Namespace) -> dict[str, str] | None:
    """What the lite file at `args.dsn` records (its width and tenant), or None when it does not
    exist, which a read answers as an empty corpus, as an empty PostgreSQL table reads.

    Refuses `--table` other than the default, since a lite file has one table and answering for
    the real one would describe memory the caller asked to keep out of view, and refuses a file
    that is not a lite store. Nothing is opened, created or stamped: opening a store at a guessed
    width stamped a file that is not a lite store with one, and broke every later index run. The
    tenant check is the caller's, because each command decides what a wrong tenant answers.
    """
    from recall.lite.inspect import lite_file_meta

    if args.table != DEFAULT_TABLE:
        raise SystemExit(
            f"{command}: a lite store ({args.dsn}) has one table, '{DEFAULT_TABLE}'; --table "
            f"{args.table!r} cannot be honoured. Use another sqlite:/// file instead"
        )
    path = lite_dsn_path(args.dsn)
    if not path.exists():
        return None
    meta = lite_file_meta(path)
    if meta is None or "dim" not in meta:
        raise SystemExit(f"{path} is not a lite store (it has no RE-call meta table); nothing was read or written")
    return meta
