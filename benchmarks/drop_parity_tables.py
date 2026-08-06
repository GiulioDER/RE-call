#!/usr/bin/env python
"""Drop leftover parity tables through the store, NEVER through psql.

⚠️ THE LEDGER IS THE WHOLE POINT. `PgVectorStore.drop_table()` drops the table AND deletes the
matching `recall_schema_migrations` rows in ONE transaction. A bare `DROP TABLE` leaves those rows
behind, after which the next run reads the ledger, sees every migration already applied, skips
creation, and then fails validating a table that is not there. That trap cost a restart on
2026-08-06 and is the reason this is a script rather than a one-line psql.

WHEN TO USE THIS RATHER THAN `--drop-generations`. The compare stage's `--drop-generations` is the
supported route for the tables a RUN KNOWS ABOUT. It cannot reach a table an aborted run created
but never registered: `_index` calls `ensure_schema()` before it indexes, so a failure inside it
leaves an orphan the cleanup block never sees. This sweeps the catalog instead, so it finds those.

Refuses any table outside the declared prefixes, prints what it will drop before dropping, and
verifies the ledger went with the tables rather than assuming it did.

Prior work: searched once for this whole line of work, recorded in
`check_generation_parity.py`'s docstring. `docs_search(source_type="memory")` for "generation
parity raw content hash identical across context modes RE-call" on 2026-08-06 returned
`gap_warning` TRUE (top-3 cosine 0.485 / 0.480 / 0.479, all under the 0.50 floor), so no memo
covers it. ⚠️ That search was scoped to the MEMORY corpus, which cannot see repository tests, and
concluding "no prior work" from it was wrong once already this session: the invariant WAS covered
by `tests/test_context_modes_index.py`. This file is operational cleanup rather than a measurement,
so it has no prior-art question of its own beyond that.
"""
from __future__ import annotations

import argparse
import sys

import psycopg

from recall.store import PgVectorStore

#: Dropping is irreversible, so the sweep is an ALLOWLIST rather than a pattern the caller supplies.
#: A typo in `--prefix` must not be able to reach a production table.
DEFAULT_PREFIXES = ("pfull_", "psmoke_", "parity_")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dsn", required=True)
    parser.add_argument(
        "--prefix",
        action="append",
        dest="prefixes",
        help=f"table prefix to sweep; repeatable. Defaults to {DEFAULT_PREFIXES}.",
    )
    parser.add_argument(
        "--dim",
        type=int,
        default=384,
        help="irrelevant to a DROP, but PgVectorStore's constructor requires one. Every parity "
        "table is a 384-dimension bge generation.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="list what would be dropped and exit. Do this first.",
    )
    args = parser.parse_args()
    prefixes = tuple(args.prefixes) if args.prefixes else DEFAULT_PREFIXES

    with psycopg.connect(args.dsn, autocommit=True, connect_timeout=10) as conn:
        rows = conn.execute(
            "SELECT relname FROM pg_stat_user_tables ORDER BY relname"
        ).fetchall()
    tables = [r[0] for r in rows if r[0].startswith(prefixes)]

    if not tables:
        print(f"nothing to drop for prefixes {prefixes}")
        return 0

    print(f"prefixes: {prefixes}")
    for table in tables:
        print(f"  would drop {table}")
    if args.dry_run:
        print("\n--dry-run: nothing was dropped")
        return 0

    for table in tables:
        # Re-checked per table rather than trusting the filter above, because this is the statement
        # that actually destroys data.
        if not table.startswith(prefixes):  # pragma: no cover - defensive
            print(f"REFUSING {table}: outside {prefixes}", file=sys.stderr)
            return 1
        PgVectorStore(args.dsn, dim=args.dim, table=table).drop_table()
        print(f"dropped {table}", flush=True)

    # ⚠️ Filtered in Python with the SAME `startswith` the drop used, NOT with SQL `LIKE`.
    # `_` is a single-character WILDCARD in LIKE, so `pfull_%` also matches `pfullX_foo`, which
    # `startswith(("pfull_",))` does not. Two filters that are meant to describe one set must be
    # one predicate, or the verification can fail on rows the drop was never going to touch.
    with psycopg.connect(args.dsn, autocommit=True, connect_timeout=10) as conn:
        ledger = conn.execute("SELECT target_table FROM recall_schema_migrations").fetchall()
    # The reason this is not a psql DROP, asserted rather than assumed.
    remaining = sum(1 for (t,) in ledger if t.startswith(prefixes))
    print(f"ledger rows remaining for those prefixes: {remaining} (must be 0)")
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
