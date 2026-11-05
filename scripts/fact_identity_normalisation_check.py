"""Count the stored facts and cards whose identity a fact normalisation would change (read only).

The owner decided on 2026-11-04 (NUM-002, NUM-003) that a fact's validity bound with an offset is
its UTC instant and a whole-number float is an int, so `01:00+01:00` and `00:00Z`, or `20.0` and
`20`, are one fact. That normalisation ships together with the migration that re-keys what is
stored, after the AML window: applied on read alone it changed the identity of every fact stored
the other way, broke the id check of every card holding one (one such row made a tenant's whole
ledger unreadable and its forgets impossible), and orphaned the stored ids that name them. This is
the inventory that migration starts from (POST_AML_DEPLOY.md, step 1).

    python scripts/fact_identity_normalisation_check.py --sqlite PATH
    RECALL_FACT_WRITE_DSN=... python scripts/fact_identity_normalisation_check.py --tenant memory [--tenant ...]

It prints one count per table and exits 1 when any stored row would change, 0 when none would.
PostgreSQL's tables enforce tenant isolation, so each tenant is read under its own setting, and a
tenant not named is not read. The DSN comes from `--dsn-env` (default RECALL_FACT_WRITE_DSN), so
its password stays out of the process table; `--dsn` is an explicit override. Values are decoded as
the ledger decodes them (`recall.provenance_json.decode_jsonb`), so a number PostgreSQL's jsonb
rewrote (an exponent float) is judged as written.
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import Any

from recall.db_constants import TENANT_GUC
from recall.fact_retraction import FACT_WRITE_DSN_ENV
from recall.provenance_json import decode_jsonb

TABLES = (
    ("recall_fact_ledger_events", "fact", "fact"),
    ("recall_fact_ledger_events", "evidence_cards", "cards"),
    ("recall_evidence_cards", "card", "card"),
)


def _time_changes(value: object) -> bool:
    """Whether a stored bound names an instant with an offset other than UTC."""
    if not isinstance(value, str):
        return False
    try:
        instant = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return False
    return instant.tzinfo is not None and instant.utcoffset() != UTC.utcoffset(None)


def _number_changes(value: object) -> bool:
    """Whether a stored value holds a float that normalisation turns into an int."""
    if isinstance(value, float):
        return value.is_integer()
    if isinstance(value, Mapping):
        return any(_number_changes(item) for item in value.values())
    if isinstance(value, list):
        return any(_number_changes(item) for item in value)
    return False


def fact_changes(payload: object) -> bool:
    """Whether a stored fact payload decodes to another identity under normalisation."""
    if not isinstance(payload, Mapping) or payload.get("redacted") is True:
        return False
    return (
        _time_changes(payload.get("valid_from"))
        or _time_changes(payload.get("valid_until"))
        or _number_changes(payload.get("object"))
        or _number_changes(payload.get("context"))
    )


def card_changes(payload: object) -> bool:
    """Whether a stored card holds a fact that normalisation changes (its id would not match)."""
    if not isinstance(payload, Mapping) or payload.get("redacted") is True:
        return False
    facts = payload.get("structured_facts") or ()
    return isinstance(facts, list) and any(fact_changes(item) for item in facts)


def _decoded(value: object) -> object:
    """A stored column as the ledger reads it: JSON text parsed, jsonb companions applied."""
    parsed = json.loads(value) if isinstance(value, (str, bytes)) else value
    if isinstance(parsed, list):
        return [decode_jsonb(item) if isinstance(item, Mapping) else item for item in parsed]
    return decode_jsonb(parsed) if isinstance(parsed, Mapping) else parsed


def count(kind: str, values: Iterable[object]) -> int:
    """How many stored `values` of `kind` (fact, cards or card) normalisation changes."""
    total = 0
    for raw in values:
        value = _decoded(raw)
        if kind == "fact":
            total += fact_changes(value)
        elif kind == "cards":
            total += isinstance(value, list) and any(card_changes(card) for card in value)
        else:
            total += card_changes(value)
    return total


def _sqlite_counts(path: str) -> dict[str, int]:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        names = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
        return {
            f"{table}.{column}": count(kind, (row[0] for row in conn.execute(f"SELECT {column} FROM {table}")))
            for table, column, kind in TABLES
            if table in names
        }
    finally:
        conn.close()


def _postgres_counts(dsn: str, tenants: list[str]) -> dict[str, int]:
    import psycopg

    counts: dict[str, int] = {}
    with psycopg.connect(dsn, autocommit=True, connect_timeout=10) as conn:
        conn.execute("SET default_transaction_read_only = on")
        for tenant in tenants:
            conn.execute("SELECT set_config(%s, %s, false)", (TENANT_GUC, tenant))
            for table, column, kind in TABLES:
                rows = conn.execute(f"SELECT {column} FROM {table}").fetchall()
                key = f"{tenant}:{table}.{column}"
                counts[key] = count(kind, (row[0] for row in rows))
    return counts


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--sqlite", help="a lite store or SQLite ledger file")
    parser.add_argument("--dsn", help="a PostgreSQL DSN; prefer --dsn-env, which keeps it out of argv")
    parser.add_argument(
        "--dsn-env", default=FACT_WRITE_DSN_ENV,
        help=f"the variable holding the PostgreSQL DSN (default {FACT_WRITE_DSN_ENV})",
    )
    parser.add_argument("--tenant", action="append", default=[], help="a tenant to read (PostgreSQL)")
    args = parser.parse_args(argv)
    if args.sqlite:
        counts: dict[str, Any] = _sqlite_counts(args.sqlite)
        if not counts:
            # A file holding none of the tables (the wrong file, or a store from before them)
            # printed 0 and exited 0, which reads as nothing to re-key (2026-11-05, ENV-004).
            tables = ", ".join(sorted({table for table, _column, _kind in TABLES}))
            print(f"{args.sqlite}: none of {tables} found; not a fact ledger", file=sys.stderr)
            return 2
    else:
        dsn = args.dsn or os.environ.get(args.dsn_env, "").strip()
        if not dsn:
            parser.error(f"pass --sqlite, or set {args.dsn_env} (or --dsn) for PostgreSQL")
        if not args.tenant:
            parser.error("PostgreSQL needs at least one --tenant")
        counts = _postgres_counts(dsn, args.tenant)
    for key, value in counts.items():
        print(f"{key}: {value}")
    changed = sum(counts.values())
    print(f"rows whose identity changes: {changed}")
    return 1 if changed else 0


if __name__ == "__main__":
    sys.exit(main())
