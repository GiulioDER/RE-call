"""Move an indexed chunk table between hosts, and PROVE the vectors survived.

::

    # on the machine that did the encoding (e.g. a rented GPU box)
    python -m benchmarks.beam.transfer_index dump --dsn "$SRC" --table bench_beam_bge_v2 \
        --out /workspace/bge_v2.copy.gz

    # on the destination, AFTER the table exists via the normal schema path
    python -m benchmarks.beam.transfer_index restore --dsn "$DST" --table bench_beam_bge_v2 \
        --in bge_v2.copy.gz

    # either side, any time
    python -m benchmarks.beam.transfer_index checksum --dsn "$DSN" --table bench_beam_bge_v2

Prior work: searched with ``docs_search(source_type="memory", ...)``.
[[feedback_prefer_gpu_rent_for_inference]] is the standing decision this serves (rent a GPU rather
than burn VPS CPU on embedding; measured 360x on a real case). Its workflow says to scp results to
the VPS *before* authorising shutdown, and [[project-neural-pricer-5m-vast]] repeats it. Nothing
existed to move an indexed table, so this is new.

Why DATA-ONLY, and not `pg_dump` of the table
---------------------------------------------
A full dump carries the DDL: the RLS policy, `FORCE ROW LEVEL SECURITY`, ownership, and the index
definitions. Restoring that onto another host means reconciling roles that may not exist there and
a migration ledger that will not know about the table. So the schema is authored at the destination
by RE-call's own `ensure_schema`, exactly as a local run would author it, and only ROWS move. The
destination stays a normal RE-call table that `check_schema` recognises.

Why `COPY` through psycopg and not the `pg_dump` binary
-------------------------------------------------------
A rented GPU image is not guaranteed to ship postgres client binaries, and matching `pg_dump`'s
version to the server's is one more thing to get wrong across two hosts running different pgvector
builds (0.8.4 here, 0.8.2 on VPS2). `COPY ... TO STDOUT` needs only the driver that is already a
dependency.

The generated column
--------------------
`tsv` is `GENERATED ALWAYS ... STORED`. It cannot be inserted into, and it does not need to be:
postgres recomputes it from `text` on insert. Columns are therefore DISCOVERED and generated ones
excluded, rather than hardcoded — a hardcoded list silently drops any column a later migration adds,
and the rows would restore looking complete.

The checksum is the point
-------------------------
An RLS-forced table read by a role without `BYPASSRLS` returns ZERO rows rather than erroring, so a
transfer can "succeed" and move nothing. Row counts alone would not catch a vector that lost
precision in transit either. `checksum` therefore digests the ordered
`(tenant_id, id, embedding)` triples, so source and destination must agree exactly or the transfer
is not accepted.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import psycopg

#: Columns postgres computes itself. Discovered per table rather than assumed; this is only the
#: catalog predicate used to find them.
_GENERATED = "a.attgenerated <> ''"


def transferable_columns(conn: psycopg.Connection, table: str) -> list[str]:
    """The column names a COPY may write: every real column that is not generated.

    Ordered by `attnum` so dump and restore agree on ordering without either side stating it.
    """
    rows = conn.execute(
        "SELECT a.attname FROM pg_attribute a "
        "WHERE a.attrelid = %s::regclass AND a.attnum > 0 AND NOT a.attisdropped "
        f"AND NOT ({_GENERATED}) ORDER BY a.attnum",
        (table,),
    ).fetchall()
    if not rows:
        raise SystemExit(f"table {table!r} has no transferable columns (does it exist?)")
    return [r[0] for r in rows]


def _tenant_key(conn: psycopg.Connection, table: str) -> str:
    return "tenant_id" if "tenant_id" in set(transferable_columns(conn, table)) else "tenant"


def _scope(conn: psycopg.Connection, tenant: str | None) -> None:
    """Make one tenant's rows visible to a role without RLS bypass.

    `SET` does not take a bound parameter, and interpolating a tenant name into DDL-ish SQL is how
    an injection gets in, so `set_config` is used — it is a normal function call and takes the name
    as a value.
    """
    if tenant is not None:
        conn.execute("SELECT set_config(%s, %s, false)", ("recall.tenant_id", tenant))


def _digest_one(
    conn: psycopg.Connection, table: str, key: str, tenant: str | None = None
) -> tuple[int, str]:
    """Rows and digest, filtered by an EXPLICIT predicate rather than by row-level security.

    Relying on RLS to do this filtering is a trap, and it bit this tool during its own round-trip
    test: `set_config` is a no-op for a role that BYPASSES RLS, so a "per-tenant" read through a
    superuser silently returned the WHOLE table for every tenant and the totals came back at 3x
    the truth. RLS is a containment boundary, not a WHERE clause — where the filtering is
    load-bearing, the query has to say so itself.
    """
    where = f" WHERE {key} = %(t)s" if tenant is not None else ""
    row = conn.execute(
        f"SELECT count(*), md5(coalesce(string_agg({key} || '|' || id || '|' || "
        f"embedding::text, E'\\n' ORDER BY {key}, id), '')) FROM {table}{where}",
        {"t": tenant} if tenant is not None else None,
    ).fetchone()
    assert row is not None
    return int(row[0]), str(row[1])


def checksum(dsn: str, table: str, tenants: list[str] | None = None) -> dict[str, object]:
    """Row count and a digest over the ordered (tenant, id, embedding) triples.

    The digest is computed IN the database so the vectors are never rendered through a client's
    float formatting on the way to being compared — two hosts with different client libraries would
    otherwise be able to disagree about identical data, or agree about different data.

    `tenants` scopes the read per tenant, which is REQUIRED for a role without RLS bypass. These
    tables carry `FORCE ROW LEVEL SECURITY` with a policy keyed on `current_setting`, so such a
    role reads **zero rows and no error** — measured on VPS2, where `recall_beam` saw 0 rows in a
    table holding 108,015. A per-tenant digest map is also strictly more useful than one global
    digest: a mismatch names the tenant instead of just existing.
    """
    with psycopg.connect(dsn) as conn:
        key = _tenant_key(conn, table)
        pgv = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        bypass_row = conn.execute(
            "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
        bypass = bool(bypass_row and bypass_row[0])

        per_tenant: dict[str, dict[str, object]] = {}
        if tenants:
            total = 0
            for tenant in tenants:
                _scope(conn, tenant)
                n, digest = _digest_one(conn, table, key, tenant)
                per_tenant[tenant] = {"rows": n, "digest": digest}
                total += n
            rows, n_tenants = total, len([t for t in tenants if per_tenant[t]["rows"]])
            # Order-independent roll-up of the per-tenant digests, so the top-level number does not
            # depend on the order tenants happened to be listed in.
            combined = hashlib.md5(
                "\n".join(f"{t}:{per_tenant[t]['digest']}" for t in sorted(per_tenant)).encode()
            ).hexdigest()
        else:
            _scope(conn, None)
            rows, combined = _digest_one(conn, table, key)
            n_row = conn.execute(f"SELECT count(DISTINCT {key}) FROM {table}").fetchone()
            n_tenants = int(n_row[0]) if n_row else 0

    return {
        "table": table,
        "rows": rows,
        "tenants": n_tenants,
        "digest": combined,
        "per_tenant": per_tenant or None,
        "pgvector": pgv[0] if pgv else None,
        "reader_bypasses_rls": bypass,
        "scoped": bool(tenants),
    }


def dump(dsn: str, table: str, out: Path, tenants: list[str] | None = None) -> dict[str, object]:
    before = checksum(dsn, table, tenants)
    if before["rows"] == 0:
        raise SystemExit(
            f"refusing to dump an EMPTY {table!r}. If the table is not really empty, the reader "
            f"lacks RLS bypass (reader_bypasses_rls={before['reader_bypasses_rls']}) and is seeing "
            f"nothing through a forced row-security policy — pass --tenant for each tenant, or use "
            f"a role with bypass. This refusal exists because the alternative is a 'successful' "
            f"transfer of zero rows."
        )
    with psycopg.connect(dsn) as conn:
        cols = transferable_columns(conn, table)
        collist = ", ".join(f'"{c}"' for c in cols)
        key = _tenant_key(conn, table)
        out.parent.mkdir(parents=True, exist_ok=True)
        # ONE FILE PER TENANT when scoped. A single concatenated stream cannot be restored under
        # row-level security: `COPY ... FROM` is checked row by row against the policy's WITH
        # CHECK, only one tenant can be in scope at a time, and the first row belonging to any
        # other tenant aborts the load. Separate files also make a partial restore resumable.
        written: dict[str, str] = {}
        for tenant in (tenants or [None]):
            _scope(conn, tenant)
            target = out if tenant is None else out.with_name(f"{out.name}.{tenant}")
            where = f" WHERE {key} = %(t)s" if tenant is not None else ""
            stmt = f"COPY (SELECT {collist} FROM {table}{where}) TO STDOUT"
            params = {"t": tenant} if tenant is not None else None
            with gzip.open(target, "wb") as fh, conn.cursor().copy(stmt, params) as copy:
                for block in copy:
                    fh.write(bytes(block))
            written[tenant or "*"] = target.name
    total_bytes = sum(
        (out.parent / name).stat().st_size for name in written.values()
    )
    meta = {
        "source": before,
        "columns": cols,
        "tenants": tenants,
        "files": written,
        "path": str(out),
        "bytes": total_bytes,
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(
        json.dumps(meta, indent=2), encoding="utf-8"
    )
    return meta


def restore(dsn: str, table: str, src: Path, *, truncate: bool = True) -> dict[str, object]:
    meta_path = src.with_suffix(src.suffix + ".meta.json")
    if not meta_path.exists():
        raise SystemExit(
            f"{meta_path} is missing. It carries the SOURCE checksum, and without it a restore "
            f"cannot be verified — only assumed."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    with psycopg.connect(dsn) as conn:
        # Postgres refuses `COPY FROM` outright for a role subject to row-level security
        # ("COPY FROM not supported with row-level security. HINT: Use INSERT statements
        # instead"). That is a server limitation, not a policy this tool can scope around: unlike
        # the READ side, where `set_config` makes one tenant visible, there is no per-tenant COPY
        # that becomes legal. So the restore requires a role that bypasses RLS, and says so up
        # front rather than failing several hundred megabytes into a transfer.
        bypass_row = conn.execute(
            "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
        if not (bypass_row and bypass_row[0]):
            user = conn.execute("SELECT current_user").fetchone()
            raise SystemExit(
                f"restore needs a role that bypasses row-level security; {user[0]!r} does not. "
                f"Postgres does not allow COPY FROM under RLS at all. Re-run the restore as a "
                f"superuser or a BYPASSRLS role (on VPS2: the local `postgres` role over the unix "
                f"socket). The DUMP side does not need this — it can be scoped with --tenant."
            )
        cols = transferable_columns(conn, table)
        if cols != meta["columns"]:
            raise SystemExit(
                f"column mismatch: dump has {meta['columns']}, destination has {cols}. The two "
                f"tables were authored by different schema versions; migrate the destination "
                f"first rather than forcing rows into a different shape."
            )
        collist = ", ".join(f'"{c}"' for c in cols)
        key = _tenant_key(conn, table)
        tenants = meta.get("tenants")
        files = meta.get("files") or {"*": src.name}
        if truncate:
            # A scoped restore must replace only the tenants it was given. TRUNCATE cannot express
            # that (it is not row-filtered at all), and neither can an unqualified DELETE here —
            # this restore runs as a role that BYPASSES RLS, so the policy filters nothing for it
            # and one DELETE per tenant would wipe the table N times over.
            if tenants:
                for tenant in tenants:
                    conn.execute(f"DELETE FROM {table} WHERE {key} = %s", (tenant,))
            else:
                conn.execute(f"TRUNCATE {table}")
        for tenant in (tenants or [None]):
            _scope(conn, tenant)
            name = files.get(tenant or "*")
            if name is None:
                raise SystemExit(f"dump has no file for tenant {tenant!r}")
            path = src.parent / name
            if not path.exists():
                raise SystemExit(f"{path} is missing — the dump is incomplete")
            with gzip.open(path, "rb") as fh, conn.cursor().copy(
                f"COPY {table} ({collist}) FROM STDIN"
            ) as copy:
                while chunk := fh.read(1 << 20):
                    copy.write(chunk)
        conn.commit()

    after = checksum(dsn, table, meta.get("tenants"))
    source = meta["source"]
    ok = after["digest"] == source["digest"] and after["rows"] == source["rows"]
    result = {"source": source, "destination": after, "verified": ok}
    if not ok:
        raise SystemExit(
            "TRANSFER NOT VERIFIED — the destination does not match the source.\n"
            f"  rows   : {source['rows']} -> {after['rows']}\n"
            f"  digest : {source['digest']} -> {after['digest']}\n"
            "The rows are in place but must not be used as if they were the encoded corpus."
        )
    return result


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m benchmarks.beam.transfer_index")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("checksum", help="report rows, tenants and the vector digest")
    c.add_argument("--dsn", required=True)
    c.add_argument("--table", required=True)
    c.add_argument("--tenant", action="append", dest="tenants")

    d = sub.add_parser("dump", help="write the table's rows to a gzipped COPY stream")
    d.add_argument("--dsn", required=True)
    d.add_argument("--table", required=True)
    d.add_argument("--out", type=Path, required=True)
    d.add_argument("--tenant", action="append", dest="tenants",
                   help="repeatable; REQUIRED for a role without RLS bypass")

    r = sub.add_parser("restore", help="load rows and verify against the source checksum")
    r.add_argument("--dsn", required=True)
    r.add_argument("--table", required=True)
    r.add_argument("--in", dest="src", type=Path, required=True)
    r.add_argument("--append", action="store_true", help="do NOT truncate first")

    args = p.parse_args(argv)
    if args.cmd == "checksum":
        print(json.dumps(checksum(args.dsn, args.table, args.tenants), indent=2))
    elif args.cmd == "dump":
        print(json.dumps(dump(args.dsn, args.table, args.out, args.tenants), indent=2))
    else:
        print(json.dumps(restore(args.dsn, args.table, args.src, truncate=not args.append), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
