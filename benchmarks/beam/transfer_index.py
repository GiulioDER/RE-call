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
precision in transit either. `checksum` therefore digests the ordered `(tenant_id, id, embedding)` triples, so source and
destination must agree on those or the transfer is not accepted.

⚠️ SCOPE, stated because "the checksum is the point" invites over-reading: the digest proves the
VECTORS and their (tenant, id) keys survived. COPY moves every non-generated column, so `text`,
`source`, `metadata`, `indexed_at` and `first_indexed_at` are transferred but NOT digested — a
corruption confined to those verifies clean. It also proves TRANSIT, not tampering: the source
digest travels in the same unsigned sidecar as the data it certifies.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
from pathlib import Path

import psycopg

#: Columns postgres computes itself. Discovered per table rather than assumed; this is only the
#: catalog predicate used to find them.
_GENERATED = "a.attgenerated <> ''"


def _validated_table(table: str) -> str:
    """Reject anything that is not a plain identifier, BEFORE it reaches an interpolated statement.

    `{table}` is f-string interpolated into TRUNCATE, DELETE, COPY and the digest. Today that is
    not exploitable, but only because every entry point happens to call `transferable_columns`
    first, which binds the name as `%s::regclass` and so rejects a payload — a side effect of
    column discovery that nothing names as validation and no test covers. Reordering one call, or
    adding a fourth subcommand, would silently turn it into TRUNCATE injection. The repo already
    owns the right control: `recall/control_plane.validate_table_name`, whose own comment reads
    "Physical table identifiers are interpolated into SQL, so this is an allowlist, not a filter".
    """
    from recall.control_plane import validate_table_name

    return str(validate_table_name(table))


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


def _pin_copy_gucs(conn: psycopg.Connection) -> None:
    """Pin the session settings that decide how COPY TEXT renders non-vector columns.

    `pg_dump` pins these for the same reason. `timestamptz` renders per the session's DateStyle, so
    a source host set to `SQL, DMY` writes `06/08/2026` where a destination on `ISO, MDY` reads
    8 June — both valid, silently different, and invisible because the digest covers only
    (tenant, id, embedding). The vector column is NOT at risk: pgvector's `vector_out` uses
    shortest-round-trip decimal and ignores `extra_float_digits`.
    """
    conn.execute("SET datestyle = 'ISO, YMD'")
    conn.execute("SET intervalstyle = 'iso_8601'")
    conn.execute("SET extra_float_digits = 3")
    conn.execute("SET timezone = 'UTC'")


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
) -> tuple[int, int, str]:
    """Rows, embedded rows, and a digest over the ordered (tenant, id, embedding) triples.

    Three things here are deliberate and each one fixes a defect this tool shipped with.

    **Per-row md5, then aggregate.** The original hashed one `string_agg` of every row's full
    vector text. At the 108,015-row table this module was written for that is ~0.5 GB at 384
    dimensions and ~1.37 GB at 1024 — past PostgreSQL's 1 GB limit on a single value, so the
    verification raised `out of memory` instead of verifying, at exactly the scale it exists for.
    Hashing each row first bounds the aggregate at 33 bytes per row (~3.5 MB) with the same
    collision properties.

    **NULL is a value, not an annihilator.** `x || NULL` is NULL and `string_agg` SKIPS NULL
    inputs, so a row whose embedding was never written contributed NOTHING: two tables differing
    only in their unembedded rows digested identically, and a wholly unembedded table digested to
    md5(''). `rows_embedded` is returned alongside so a partially-encoded corpus is a number the
    operator sees rather than an absence.

    **`COLLATE "C"`.** The ordering ran under each server's `lc_collate`, and this tool compares
    two independent installs by design, so byte-identical data could digest differently and be
    reported as corruption. A false alarm on the tool's only guarantee is what teaches an operator
    to bypass it.

    Filtering is an EXPLICIT predicate, never row-level security: `set_config` is a no-op for a
    role that bypasses RLS, which once made a per-tenant read return the whole table and report 3x
    the true row count.
    """
    where = f" WHERE {key} = %(t)s" if tenant is not None else ""
    row = conn.execute(
        f"SELECT count(*), count(embedding), "
        f"md5(coalesce(string_agg("
        f"  md5({key} || '|' || id || '|' || coalesce(embedding::text, '<NULL>')), "
        f"  ',' ORDER BY {key} COLLATE \"C\", id COLLATE \"C\"), '')) "
        f"FROM {table}{where}",
        {"t": tenant} if tenant is not None else None,
    ).fetchone()
    assert row is not None
    return int(row[0]), int(row[1]), str(row[2])


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
    table = _validated_table(table)
    with psycopg.connect(dsn) as conn:
        # Defence in depth, and NOT a fix for a live defect — an earlier version of this comment
        # claimed it was. The digest covers `tenant_id`, `id` and `embedding` only: the first two
        # are TEXT, and `_pin_copy_gucs`'s own docstring records that pgvector's `vector_out`
        # ignores `extra_float_digits`. None of these GUCs can move any of them, and the one thing
        # that WAS host-dependent — ordering — is handled by `COLLATE "C"`, which is not a GUC.
        # The pin earns its place only if a future column enters the digest.
        _pin_copy_gucs(conn)
        key = _tenant_key(conn, table)
        pgv = conn.execute(
            "SELECT extversion FROM pg_extension WHERE extname = 'vector'"
        ).fetchone()
        bypass_row = conn.execute(
            "SELECT rolsuper OR rolbypassrls FROM pg_roles WHERE rolname = current_user"
        ).fetchone()
        bypass = bool(bypass_row and bypass_row[0])

        per_tenant: dict[str, dict[str, object]] = {}
        embedded = 0
        if tenants:
            # Deduplicated ONCE, order preserved. `total` used to sum the raw list while
            # `per_tenant` overwrote by key, so `--tenant a --tenant a` reported twice the rows
            # it held and the meta claimed twice what its single file contained.
            tenants = list(dict.fromkeys(tenants))
            total = 0
            for tenant in tenants:
                _scope(conn, tenant)
                n, n_emb, digest = _digest_one(conn, table, key, tenant)
                per_tenant[tenant] = {"rows": n, "rows_embedded": n_emb, "digest": digest}
                total += n
                embedded += n_emb
            rows, n_tenants = total, len([t for t in tenants if per_tenant[t]["rows"]])
            # Order-independent roll-up of the per-tenant digests, so the top-level number does not
            # depend on the order tenants happened to be listed in.
            combined = hashlib.md5(
                "\n".join(f"{t}:{per_tenant[t]['digest']}" for t in sorted(per_tenant)).encode()
            ).hexdigest()
        else:
            _scope(conn, None)
            rows, embedded, combined = _digest_one(conn, table, key)
            n_row = conn.execute(f"SELECT count(DISTINCT {key}) FROM {table}").fetchone()
            n_tenants = int(n_row[0]) if n_row else 0

    return {
        "table": table,
        "rows": rows,
        "rows_embedded": embedded,
        "tenants": n_tenants,
        "digest": combined,
        "per_tenant": per_tenant or None,
        "pgvector": pgv[0] if pgv else None,
        "reader_bypasses_rls": bypass,
        "scoped": bool(tenants),
    }


def dump(dsn: str, table: str, out: Path, tenants: list[str] | None = None) -> dict[str, object]:
    table = _validated_table(table)
    tenants = list(dict.fromkeys(tenants)) if tenants else None
    before = checksum(dsn, table, tenants)
    # Per SLICE, not just in total. The refusal below tested the SUM while the restore's DELETE
    # operates per tenant, so one empty slice (a crashed ingest, a mistyped --tenant) passed the
    # guard, deleted that tenant at the destination, loaded nothing back, and still VERIFIED —
    # md5('') equals md5('') on both sides. A guard has to sit at the granularity of the damage.
    _per_tenant: dict[str, dict[str, object]] = before.get("per_tenant") or {}  # type: ignore[assignment]
    empty = [t for t, v in _per_tenant.items() if not v.get("rows")]
    if empty:
        raise SystemExit(
            f"refusing to dump empty tenant slice(s) {empty}: restoring them would DELETE those "
            f"tenants at the destination, load nothing back, and still report verified. Check the "
            f"spelling, or drop them from --tenant."
        )
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
        # 0700/0600: this file holds the corpus VERBATIM (the `text` column travels with the
        # vectors) and the documented workflow writes it on a rented third-party host. SECURITY.md
        # calls the corpus the asset and notes it may contain "a secret pasted into prose".
        # The confidentiality guarantee is on the FILE (0600, below), not on the directory.
        #
        # An earlier version of this chmod'd `out.parent` unconditionally to 0700, which was worse
        # than the no-op it replaced: `--out dump.gz` resolves the parent to `.`, so it re-moded
        # the process's working directory, and `--out /tmp/x.gz` would have stripped /tmp's sticky
        # bit. It also raises PermissionError on a directory owned by someone else, turning a
        # working dump into a crash. A tool does not get to re-permission a directory it did not
        # create. `mode=` here still applies on creation, and is simply inert otherwise.
        out.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        # ONE FILE PER TENANT when scoped, for the reason that actually holds: a reader WITHOUT
        # RLS bypass can only see one tenant at a time, so per-tenant is the only dump such a role
        # can produce, and a per-tenant digest names which tenant mismatched instead of merely
        # reporting that something did.
        #
        # ⚠️ An earlier version of this comment justified the split by row-by-row `WITH CHECK`
        # evaluation during `COPY FROM`, and claimed the files make a partial restore resumable.
        # Both were wrong: Postgres refuses `COPY FROM` under RLS outright (see `restore`), and
        # the restore is a single transaction with no subset option, so there is no partial state
        # to resume from.
        written: dict[str, str] = {}
        _pin_copy_gucs(conn)
        scoped: list[str | None] = list(tenants) if tenants else [None]
        for tenant in scoped:
            _scope(conn, tenant)
            target = out if tenant is None else out.with_name(f"{out.name}.{tenant}")
            where = f" WHERE {key} = %(t)s" if tenant is not None else ""
            stmt = f"COPY (SELECT {collist} FROM {table}{where}) TO STDOUT"
            params = {"t": tenant} if tenant is not None else None
            # NOT `touch(mode=0o600, exist_ok=True)`: when the file already exists — a re-run, or
            # a dump written before this fix — pathlib's `os.utime` succeeds and the function
            # returns BEFORE reaching the `os.open` that would apply the mode, and `gzip.open`
            # then truncates in place and inherits the old permissions. Open the fd ourselves so
            # the mode is enforced on every run, not only on first creation, and before any bytes
            # are written rather than after.
            fd = os.open(target, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            # `raw` is BOUND and closed in a finally. `gzip.GzipFile` closes a fileobj only when
            # it opened the path itself, so `gzip.open(os.fdopen(fd, "wb"))` left the underlying
            # handle open with the compressed tail still buffered: measured, the target was 0
            # bytes on disk at the point `stat().st_size` reads it below, and the dump was correct
            # only because CPython refcounting finalised an unnamed temporary. That is not a
            # guarantee — it is an implementation detail of one interpreter.
            raw = os.fdopen(fd, "wb")
            try:
                # The open mode covers only a file this call CREATES (umask can only clear bits,
                # so 0600 is an upper bound there). An EXISTING file keeps the permissions it
                # already had — the re-run case, and the reason the `touch(mode=…)` this replaced
                # was a no-op. chmod through the fd we now own, so there is no path race.
                if os.name != "nt":
                    os.chmod(target, 0o600)
                with gzip.GzipFile(fileobj=raw, mode="wb") as fh, conn.cursor().copy(
                    stmt, params
                ) as copy:
                    for block in copy:
                        fh.write(bytes(block))
            finally:
                raw.close()
            written[tenant or "*"] = target.name
    total_bytes = sum(
        (out.parent / name).stat().st_size for name in written.values()
    )
    meta: dict[str, object] = {
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


def restore(dsn: str, table: str, src: Path, *, truncate: bool = False) -> dict[str, object]:
    # FIRST statement, matching `checksum` and `dump`. This is the subcommand `_validated_table`'s
    # own docstring is about: `restore` is the only path that runs TRUNCATE and DELETE, and it was
    # the one call site the validation fix skipped. Relying on the `%s::regclass` bind inside
    # `transferable_columns` is precisely the incidental protection that docstring says not to
    # trust. It also keeps dump and restore agreeing on what a legal table name is.
    table = _validated_table(table)
    meta_path = src.with_suffix(src.suffix + ".meta.json")
    if not meta_path.exists():
        raise SystemExit(
            f"{meta_path} is missing. It carries the SOURCE checksum, and without it a restore "
            f"cannot be verified — only assumed."
        )
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    # Validate the untrusted sidecar BEFORE opening anything. The documented workflow produces it
    # on a rented third-party box, and it supplies both the DELETE targets and the filenames that
    # get joined to the dump directory. Verified in this checkout: `Path(base) / "/etc/shadow"`
    # discards the base entirely, and `../` walks out. Checking after connecting would mean the
    # first thing a hostile sidecar meets is a live database handle.
    for _name in (meta.get("files") or {}).values():
        if Path(str(_name)).name != str(_name):
            raise SystemExit(
                f"the dump names a file with a path separator or an absolute path ({_name!r}). "
                f"A restore reads only plain filenames beside the archive it was given."
            )

    with psycopg.connect(dsn) as conn:
        # The PARSE side of COPY TEXT. `COPY ... FROM STDIN` interprets the incoming text under
        # the destination's DateStyle/IntervalStyle/timezone, so pinning only the writer converts
        # a formatting mismatch into a silently different stored value rather than preventing it.
        # Both sides, or neither: this is the half that was missing.
        _pin_copy_gucs(conn)
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
            user_row = conn.execute("SELECT current_user").fetchone()
            user = (user_row[0] if user_row else "<unknown>",)
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
        # The chunk table is not the whole index. `recall_sparse_v1` is keyed on
        # (tenant_id, chunk_table, profile_id, id) with NO foreign key by design (migration 0012:
        # "the parent table is a COLUMN VALUE, not a schema reference"), so nothing cascades. A
        # truncating restore therefore leaves stale sparse rows whose ids no longer match the
        # chunks, and `store.sparse_row_count()` stays > 0 — which is precisely the condition that
        # SUPPRESSES the guard for an unencoded corpus. The digest cannot see any of it.
        # Named from the store rather than duplicated as a literal, so the guard and the writer
        # can never drift apart on WHICH table this is (`store.sparse_row_count` reads the same
        # constant). Resolution is deliberately left to `search_path`, matching the store: the
        # guard has to look where the writer writes, not at a schema of its own choosing.
        from recall.store import SPARSE_TABLE

        sparse = conn.execute("SELECT to_regclass(%s) IS NOT NULL", (SPARSE_TABLE,)).fetchone()
        if sparse and sparse[0]:
            # Scoped to the tenants BEING RESTORED. `store.sparse_row_count` filters on
            # (tenant_id, chunk_table, profile_id), so another tenant's sparse rows cannot
            # suppress the guard for this one and are therefore not a reason to refuse. Counting
            # them anyway made a scoped restore of tenant A refuse because tenant B was encoded,
            # and the remedy in the message ("clear them first") would have destroyed B's index.
            # An unscoped dump still checks the whole table, which is the correct width for it.
            if tenants:
                stale = conn.execute(
                    f"SELECT count(*) FROM {SPARSE_TABLE} "
                    f"WHERE chunk_table = %s AND tenant_id = ANY(%s)",
                    (table, list(tenants)),
                ).fetchone()
                scope = f"for {len(tenants)} restored tenant(s)"
            else:
                stale = conn.execute(
                    f"SELECT count(*) FROM {SPARSE_TABLE} WHERE chunk_table = %s", (table,)
                ).fetchone()
                scope = "across all tenants"
            if stale and stale[0]:
                raise SystemExit(
                    f"the destination holds {stale[0]} learned-sparse rows {scope} for "
                    f"{table!r} in {SPARSE_TABLE}, which this transfer does not move and cannot "
                    f"verify. Restoring would leave them orphaned against the new chunk ids while "
                    f"sparse_row_count() stays non-zero, suppressing the very guard that exists "
                    f"to catch an unencoded corpus. Clear those tenants' rows first, or re-encode "
                    f"after."
                )

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
            # The sidecar is the LEAST trusted input here: the documented workflow produces it
            # on a rented third-party box. `Path(base) / "/etc/shadow"` discards the base
            # entirely and `../` walks out of the dump directory — both verified in this checkout.
            if Path(name).name != name:
                raise SystemExit(
                    f"the dump names a file with a path separator or an absolute path ({name!r}). "
                    f"A restore reads only plain filenames beside the archive it was given."
                )
            path = src.parent / name
            if not path.exists():
                raise SystemExit(f"{path} is missing — the dump is incomplete")
            with gzip.open(path, "rb") as fh, conn.cursor().copy(
                f"COPY {table} ({collist}) FROM STDIN"
            ) as copy:
                while chunk := fh.read(1 << 20):
                    copy.write(chunk)
        # Verify INSIDE the transaction, BEFORE the commit. Previously the commit happened first
        # and the check ran afterwards on a fresh connection, so a mismatch — including a spurious
        # one from a collation or pgvector-version difference — left the destination with its old
        # rows destroyed and the new rows unverified, in a table `check_schema` accepts as a
        # normal index. Nothing in this module takes a backup, so that state was unrecoverable.
        # Raising here lets psycopg's context manager roll the whole load back.
        source = meta["source"]
        after_rows = 0
        after_digest = ""
        per_tenant_after: dict[str, dict[str, object]] = {}
        for tenant in (tenants or [None]):
            _scope(conn, tenant)
            n, n_emb, digest = _digest_one(conn, table, key, tenant)
            after_rows += n
            if tenant is None:
                after_digest = digest
            else:
                per_tenant_after[tenant] = {"rows": n, "rows_embedded": n_emb, "digest": digest}
        if tenants:
            after_digest = hashlib.md5(
                "\n".join(
                    f"{t}:{per_tenant_after[t]['digest']}" for t in sorted(per_tenant_after)
                ).encode()
            ).hexdigest()
        if after_digest != source["digest"] or after_rows != source["rows"]:
            raise SystemExit(
                "TRANSFER NOT VERIFIED — the destination does not match the source, so the load "
                "was ROLLED BACK and the destination is unchanged.\n"
                f"  rows   : {source['rows']} -> {after_rows}\n"
                f"  digest : {source['digest']} -> {after_digest}\n"
                f"  source pgvector: {source.get('pgvector')} — a version or collation difference "
                f"produces this too, so rule those out before assuming data loss."
            )
        conn.commit()

    return {
        "source": meta["source"],
        "destination": checksum(dsn, table, tenants),
        "verified": True,
    }


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
    r.add_argument("--replace", action="store_true",
                   help="DELETE the rows this dump covers before loading. OFF by default: every "
                        "other unsafe path in this package is an explicit opt-in, and an unscoped "
                        "replace is a bare TRUNCATE touching every tenant, not only those being "
                        "restored.")

    args = p.parse_args(argv)
    if args.cmd == "checksum":
        print(json.dumps(checksum(args.dsn, args.table, args.tenants), indent=2))
    elif args.cmd == "dump":
        print(json.dumps(dump(args.dsn, args.table, args.out, args.tenants), indent=2))
    else:
        print(json.dumps(restore(args.dsn, args.table, args.src, truncate=args.replace), indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
