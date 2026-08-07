"""Red→green proofs for the audit findings in the index transfer tool.

Findings pinned:
  DAT-001 / PERF-001 / BUG-003  the digest aggregated every vector into ONE Postgres value, which
                                exceeds the 1 GB varlena limit at the table size the module cites
  DAT-002 / STAKES-002 / BUG-007  an empty tenant slice DELETEs the destination tenant, restores
                                nothing, and verifies clean because md5('') == md5('')
  DAT-003 / BUG-008             restore committed before verifying, so a mismatch left the old
                                rows destroyed and the new rows unverified
  DAT-012 / NUM-005             a NULL embedding annihilated its row out of the digest entirely
  DAT-007 / BUG-005             the digest ordering was collation-dependent across hosts
  NUM-007                       a repeated --tenant double-counted rows
  SEC-002                       restore took DELETE targets and file paths from an untrusted sidecar

The DB-backed tests skip when no Postgres is reachable; the pure-logic ones always run.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

psycopg = pytest.importorskip("psycopg")

from benchmarks.beam import transfer_index as ti  # noqa: E402

#: Admin DSN used only to create the throwaway database these tests own.
ADMIN_DSN = os.environ.get("RECALL_TEST_DSN", "postgresql://recall:recall@localhost:55432/recall")
TEST_DB = "xfer_guard_db"
DSN = ADMIN_DSN.rsplit("/", 1)[0] + "/" + TEST_DB
TABLE = "xfer_guard_chunks"
DIM = 8


def _pg_available() -> bool:
    try:
        with psycopg.connect(ADMIN_DSN, connect_timeout=3):
            return True
    except Exception:
        return False


pg = pytest.mark.skipif(not _pg_available(), reason="no Postgres at RECALL_TEST_DSN")


@pytest.fixture()
def table():
    """A throwaway DATABASE, not just a table.

    The tool applies RE-call's migration ledger, and current master refuses to migrate an
    evaluation table until the global generation migrations exist in that database — so these
    tests cannot borrow a database that some other run has half-configured.
    """
    from recall.schema import apply_migrations
    from recall.store import PgVectorStore

    with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")
        conn.execute(f"CREATE DATABASE {TEST_DB}")
    with psycopg.connect(DSN, autocommit=True) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector")

    apply_migrations(DSN, table="chunks", dim=DIM)
    with PgVectorStore(DSN, dim=DIM, tenant="t-a", table=TABLE) as store:
        store.ensure_schema()
    yield TABLE
    with psycopg.connect(ADMIN_DSN, autocommit=True) as conn:
        conn.execute(f"DROP DATABASE IF EXISTS {TEST_DB} WITH (FORCE)")


def _insert(conn, tenant: str, rows: list[tuple[str, list[float] | None]]) -> None:
    for chunk_id, vec in rows:
        conn.execute(
            f"INSERT INTO {TABLE} (tenant_id, id, source, text, metadata, embedding) "
            f"VALUES (%s, %s, %s, %s, %s::jsonb, %s)",
            (tenant, chunk_id, "s.md", f"text {chunk_id}", "{}",
             None if vec is None else "[" + ",".join(str(x) for x in vec) + "]"),
        )


@pg
def test_a_null_embedding_does_not_vanish_from_the_digest(table) -> None:
    """DAT-012: `x || NULL` is NULL and string_agg SKIPS NULLs, so those rows left no trace."""
    with psycopg.connect(DSN) as conn:
        _insert(conn, "t-a", [("a", [1.0] * DIM), ("b", None)])
        conn.commit()
    first = ti.checksum(DSN, table, ["t-a"])

    with psycopg.connect(DSN) as conn:
        conn.execute(f"DELETE FROM {table} WHERE id = 'b'")
        _insert(conn, "t-a", [("zzz", None)])       # different row, still NULL vector
        conn.commit()
    second = ti.checksum(DSN, table, ["t-a"])

    assert first["rows"] == second["rows"] == 2
    assert first["digest"] != second["digest"], (
        "a row whose embedding is NULL contributed nothing, so two different tables digest alike"
    )


@pg
def test_an_all_null_table_is_not_reported_as_verified(table) -> None:
    """DAT-012: a wholly unembedded corpus digested to md5('') on both sides."""
    with psycopg.connect(DSN) as conn:
        _insert(conn, "t-a", [("a", None), ("b", None)])
        conn.commit()
    report = ti.checksum(DSN, table, ["t-a"])
    assert report["rows"] == 2
    assert report.get("rows_embedded") == 0, "the unembedded count must be visible, not implied"
    assert report["digest"] != "d41d8cd98f00b204e9800998ecf8427e"


@pg
def test_dump_refuses_a_named_tenant_that_holds_nothing(table, tmp_path: Path) -> None:
    """DAT-002: an empty slice DELETEs the destination tenant and then verifies clean."""
    with psycopg.connect(DSN) as conn:
        _insert(conn, "t-a", [("a", [1.0] * DIM)])
        conn.commit()
    with pytest.raises(SystemExit, match="(?i)t-typo|empty|no rows"):
        ti.dump(DSN, table, tmp_path / "x.copy.gz", ["t-a", "t-typo"])


@pg
def test_repeated_tenant_does_not_double_count(table) -> None:
    """NUM-007: `total` summed the raw list while per_tenant deduplicated."""
    with psycopg.connect(DSN) as conn:
        _insert(conn, "t-a", [(f"r{i}", [float(i)] * DIM) for i in range(5)])
        conn.commit()
    report = ti.checksum(DSN, table, ["t-a", "t-a"])
    assert report["rows"] == 5, f"repeated --tenant reported {report['rows']} rows for 5"


@pg
def test_digest_is_computable_on_a_wide_table(table) -> None:
    """DAT-001: the old digest built one Postgres value holding every vector's full text."""
    with psycopg.connect(DSN) as conn:
        _insert(conn, "t-a", [(f"r{i}", [float(i)] * DIM) for i in range(200)])
        conn.commit()
    report = ti.checksum(DSN, table, ["t-a"])
    assert report["rows"] == 200
    # The aggregated value must be bounded per row, not proportional to the vector width.
    with psycopg.connect(DSN) as conn:
        agg = conn.execute(
            f"SELECT length(string_agg(md5(tenant_id || '|' || id || '|' || "
            f"coalesce(embedding::text, '<NULL>')), ',')) FROM {table}"
        ).fetchone()[0]
    assert agg is not None and agg < 200 * 40, "digest input is not per-row bounded"


@pg
def test_a_failed_verification_does_not_leave_the_table_destroyed(table, tmp_path: Path) -> None:
    """DAT-003: restore committed, THEN verified, so a mismatch left the old rows gone."""
    with psycopg.connect(DSN) as conn:
        _insert(conn, "t-a", [("orig", [9.0] * DIM)])
        conn.commit()
    out = tmp_path / "x.copy.gz"
    ti.dump(DSN, table, out, ["t-a"])

    # Corrupt the recorded source digest so verification must fail.
    meta_path = out.with_suffix(out.suffix + ".meta.json")
    doc = json.loads(meta_path.read_text(encoding="utf-8"))
    doc["source"]["digest"] = "0" * 32
    doc["source"]["per_tenant"]["t-a"]["digest"] = "0" * 32
    meta_path.write_text(json.dumps(doc), encoding="utf-8")

    # --replace is the path where a failed verification could destroy data, so that is the one
    # the rollback proof has to exercise.
    with pytest.raises(SystemExit, match="(?i)not verified|rolled back"):
        ti.restore(DSN, table, out, truncate=True)

    with psycopg.connect(DSN) as conn:
        surviving = conn.execute(f"SELECT count(*) FROM {table}").fetchone()[0]
    assert surviving == 1, "a failed verification must roll back, not leave a destroyed table"


def test_restore_refuses_a_sidecar_filename_that_escapes_its_directory(tmp_path: Path) -> None:
    """SEC-002: `Path(base) / '/etc/shadow'` discards the base entirely."""
    out = tmp_path / "x.copy.gz"
    out.write_bytes(b"")
    meta = {
        "source": {"rows": 1, "digest": "x", "per_tenant": {"t-a": {"rows": 1, "digest": "x"}}},
        "columns": ["tenant_id"], "tenants": ["t-a"],
        "files": {"t-a": "/etc/shadow"}, "path": str(out), "bytes": 0,
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")
    with pytest.raises(SystemExit, match="(?i)filename|outside|escape|separator"):
        ti.restore("postgresql://unused/db", "t", out)


def test_restore_is_non_destructive_by_default() -> None:
    """ENV-004: every other unsafe path in this package is an explicit opt-in; this one was not."""
    import inspect

    assert inspect.signature(ti.restore).parameters["truncate"].default is False


def test_restore_validates_the_table_name_before_it_can_reach_truncate(tmp_path: Path) -> None:
    """BUG-001: the identifier fix reached `checksum` and `dump` but skipped `restore`.

    `restore` is the ONLY subcommand that runs TRUNCATE and DELETE, so it was the one call site
    where interpolation actually matters. It was protected solely by the `%s::regclass` bind
    inside `transferable_columns` — a side effect of column discovery that `_validated_table`'s
    own docstring names as the thing not to rely on.

    RED before the fix: validation happened nowhere on this path, so the call got as far as
    opening a connection and failed on the DSN instead of on the identifier.
    """
    out = tmp_path / "x.copy.gz"
    out.write_bytes(b"")
    meta = {
        "source": {"rows": 0, "digest": "d", "per_tenant": {}},
        "columns": ["tenant_id"], "tenants": None, "files": {"*": out.name},
        "path": str(out), "bytes": 0,
    }
    out.with_suffix(out.suffix + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # An unreachable DSN on purpose: if validation is correctly FIRST, the identifier is refused
    # before any connection is attempted, so the DSN never gets a chance to produce the error.
    with pytest.raises(ValueError) as exc:
        ti.restore("postgresql://unused/db", 'evil"; TRUNCATE users; --', out)
    assert "connect" not in str(exc.value).lower(), (
        "reached the database before validating the identifier"
    )


def test_validation_runs_before_the_sidecar_is_even_read(tmp_path: Path) -> None:
    """Ordering is the property: a bad identifier must not depend on a well-formed sidecar."""
    missing = tmp_path / "does-not-exist.copy.gz"

    with pytest.raises(ValueError) as exc:
        ti.restore("postgresql://unused/db", "bad-name-with-dashes", missing)
    assert "meta.json" not in str(exc.value), (
        "the missing-sidecar error won the race; the identifier check must come first"
    )


@pg
def test_the_sparse_orphan_refusal_names_the_profile_not_just_the_tenant(table, tmp_path) -> None:
    """The remedy has to be as narrow as the harm.

    `store.sparse_row_count` keys on (tenant_id, chunk_table, profile_id), so the suppression this
    guard exists to prevent is PER PROFILE. A refusal that says only "clear those tenants' rows"
    points the operator at a DELETE wide enough to destroy a second profile's encoding — the same
    harm as the cross-tenant version that was fixed earlier, one key narrower.

    RED before this change: the message carried a bare total and named no profile at all.
    """
    from recall.store import SPARSE_TABLE

    src = tmp_path / "x.copy.gz"
    src.write_bytes(b"")
    # Real columns, read from the destination. A hand-written list trips the column-mismatch
    # guard first, and this test would then pass or fail for a reason it is not about.
    with psycopg.connect(DSN) as _c:
        real_columns = ti.transferable_columns(_c, TABLE)
    meta = {
        "source": {"rows": 1, "digest": "d", "per_tenant": {"t-a": {"rows": 1, "digest": "d"}}},
        "columns": real_columns, "tenants": ["t-a"], "files": {"t-a": src.name},
        "path": str(src), "bytes": 0,
    }
    src.with_suffix(src.suffix + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # Two profiles for ONE tenant, plus a second tenant that must not be named in the remedy.
    with psycopg.connect(DSN, autocommit=True) as conn:
        for tenant, profile, n in (("t-a", "bge-small-v1", 2), ("t-a", "splade-v3", 1), ("t-b", "bge-small-v1", 5)):
            for i in range(n):
                conn.execute(
                    f"INSERT INTO {SPARSE_TABLE} (tenant_id, chunk_table, profile_id, id, vec, nnz) "
                    f"VALUES (%s, %s, %s, %s, '{{1:1}}/30522', 1) ON CONFLICT DO NOTHING",
                    (tenant, TABLE, profile, f"{profile}-{i}"),
                )

    with pytest.raises(SystemExit) as exc:
        ti.restore(DSN, TABLE, src)
    message = str(exc.value)

    assert "bge-small-v1" in message and "splade-v3" in message, (
        "the refusal must name every profile whose rows the operator is being told to clear"
    )
    assert "t-b" not in message, "a tenant that is not being restored must not be in the remedy"
    # NOT `"3" in message` — that was satisfied by the "3" inside `splade-v3` no matter what total
    # the guard computed, so it would have passed against `len(stale_rows)` or a constant. Anchor
    # the number to the phrase only the total can produce.
    assert "holds 3 learned-sparse rows" in message, (
        "the total must be the sum across both of the restored tenant's profiles"
    )


@pg
def test_the_truncated_breakdown_hands_over_a_query_scoped_like_the_refusal(table, tmp_path) -> None:
    """The overflow branch had no test, which is how it shipped contradicting itself.

    The message tells the operator that "deleting wider than this list destroys another profile's
    index" and then, past 20 groups, hands them a query to get the rest. An earlier version built
    that query WITHOUT the tenant predicate the guard itself applied — so following the remedy
    would have cleared tenants that were never being restored: the precise harm the scoping fixed,
    printed as the fix.
    """
    from recall.store import SPARSE_TABLE

    src = tmp_path / "y.copy.gz"
    src.write_bytes(b"")
    with psycopg.connect(DSN) as _c:
        real_columns = ti.transferable_columns(_c, TABLE)
    meta = {
        "source": {"rows": 1, "digest": "d", "per_tenant": {"t-a": {"rows": 1, "digest": "d"}}},
        "columns": real_columns, "tenants": ["t-a"], "files": {"t-a": src.name},
        "path": str(src), "bytes": 0,
    }
    src.with_suffix(src.suffix + ".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    # 25 profiles for the restored tenant (past the cap of 20), plus one for a tenant that is not.
    with psycopg.connect(DSN, autocommit=True) as conn:
        for i in range(25):
            conn.execute(
                f"INSERT INTO {SPARSE_TABLE} (tenant_id, chunk_table, profile_id, id, vec, nnz) "
                f"VALUES (%s, %s, %s, %s, '{{1:1}}/30522', 1)",
                ("t-a", TABLE, f"profile-{i:02d}", f"id-{i}"),
            )
        conn.execute(
            f"INSERT INTO {SPARSE_TABLE} (tenant_id, chunk_table, profile_id, id, vec, nnz) "
            f"VALUES (%s, %s, %s, %s, '{{1:1}}/30522', 1)",
            ("t-elsewhere", TABLE, "profile-00", "id-x"),
        )

    with pytest.raises(SystemExit) as exc:
        ti.restore(DSN, TABLE, src)
    message = str(exc.value)

    assert "beyond the first 20" in message, "past the cap the message must say it truncated"
    assert "tenant_id IN ('t-a')" in message, (
        "the handed-over query must carry the SAME tenant scope as the refusal"
    )
    assert "t-elsewhere" not in message, "a tenant not being restored must not appear anywhere"
    # The total counts every group, not just the 20 rendered.
    assert "holds 25 learned-sparse rows" in message
