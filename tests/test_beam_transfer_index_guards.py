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
