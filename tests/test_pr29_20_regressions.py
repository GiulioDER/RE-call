"""BUG-001/BUG-002/NUM-001/DAT-001 regression boundaries.

Red proof: baseline 418511e98, Linux Python 3.12, 2026-11-03. Each node below
failed its intended assertion: loop aborted scan, IPv6 host was lost, negative
margin was zero, or a late metadata commit left the reader's graph unchanged.
"""
import math
from contextlib import contextmanager

import psycopg
from psycopg.conninfo import conninfo_to_dict
import pytest

from recall.calibration import best_threshold
from recall.index import candidate_files
from recall.store import PgVectorStore
from tests import test_tenancy
from tests.conftest import TEST_DSN, requires_db
from tests.test_supersession_freshness import DIM, _chunk, shared_table  # noqa: F401


def test_negative_only_threshold_has_strict_margin():
    edge = math.nextafter(0.01, 0.0)
    assert best_threshold([], [edge] * 20) > edge


def test_looping_file_does_not_abort_candidate_scan(tmp_path):
    good = tmp_path / "good.md"
    good.write_text("valid memory", encoding="utf8")
    try:
        (tmp_path / "loop.md").symlink_to("loop.md")
    except OSError as exc:
        pytest.skip(f"cannot create a file symlink here: {exc}")
    failure = None
    try:
        paths = candidate_files(tmp_path)
    except RuntimeError as exc:
        failure = exc
        paths = []
    assert failure is None, "a looping file aborted the complete scan"
    assert paths == [good]


def test_rls_fixture_preserves_ipv6_connection_parameters(monkeypatch):
    monkeypatch.setattr(test_tenancy, "TEST_DSN", "postgresql://admin:example@[::1]:5544/db?application_name=control")
    monkeypatch.setattr(test_tenancy, "_role_can_bypass_rls", lambda dsn: True)

    class Connection:
        def execute(self, *args, **kwargs):
            pass

    @contextmanager
    def connect(*args, **kwargs):
        yield Connection()

    monkeypatch.setattr(test_tenancy.psycopg, "connect", connect)
    fixture = test_tenancy.unprivileged_dsn.__wrapped__("fixture_table")
    try:
        role, dsn = next(fixture)
        try:
            parsed = conninfo_to_dict(dsn)
        except psycopg.ProgrammingError:
            parsed = {}
        assert parsed.get("host") == "::1", "RLS fixture corrupted the IPv6 destination"
        assert parsed["port"] == "5544"
        assert parsed["dbname"] == "db"
        assert parsed["application_name"] == "control"
        assert parsed["user"] == role
    finally:
        fixture.close()


@requires_db
def test_late_commit_invalidates_another_readers_supersession_cache(shared_table):  # noqa: F811
    stores = [PgVectorStore(TEST_DSN, dim=DIM, table=shared_table) for _ in range(3)]
    reader, older, newer = stores
    try:
        reader.ensure_schema()
        reader.upsert([_chunk(n, n + ".md", n) for n in ("a", "b", "c")], [[1.0] * DIM] * 3)
        with older._direct.transaction():
            older._direct.execute("SELECT now()")
            newer.upsert([_chunk("c", "c.md", "newer transaction")], [[1.0] * DIM])
            assert reader.supersession_map() == {}
            older.upsert([_chunk("b", "b.md", "late commit", "a.md")], [[1.0] * DIM])
        assert reader.supersession_map() == {"a.md": "b.md"}, "late metadata commit left reader cache stale"
    finally:
        for store in stores:
            store.close()
