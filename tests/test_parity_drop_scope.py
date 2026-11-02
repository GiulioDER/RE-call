"""Parity cleanup cannot broaden the fixed destructive namespaces."""

import sys

import pytest

from benchmarks import drop_parity_tables as cleanup


@pytest.fixture
def cleanup_transport(monkeypatch):
    touched = []

    class Connection:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def execute(self, query):
            self.query = query
            return self

        def fetchall(self):
            return [('chunks',), ('pfull_one',), ('parity_two',)] if 'pg_stat_user_tables' in self.query else []

    def connect(*args, **kwargs):
        touched.append('connect')
        return Connection()

    class Store:
        def __init__(self, *args, table, **kwargs):
            self.table = table

        def drop_table(self):
            touched.append(self.table)

    monkeypatch.setattr(cleanup.psycopg, 'connect', connect)
    monkeypatch.setattr(cleanup, 'PgVectorStore', Store)
    return touched


@pytest.mark.parametrize('prefix', ['', 'chunks', 'p', 'parit'])
def test_cleanup_rejects_unapproved_prefix_before_database(monkeypatch, cleanup_transport, prefix):
    """SEC-001: baseline returns success and can drop chunks; assertions fail for all four."""
    monkeypatch.setattr(sys, 'argv', ['drop-parity', '--dsn', 'unused', '--prefix', prefix])
    try:
        result = cleanup.main()
    except SystemExit as exc:
        result = exc.code
    assert result == 2, 'unsafe cleanup prefix must be rejected'
    assert cleanup_transport == [], 'rejection must precede all database access'


@pytest.mark.parametrize('prefix,expected', [
    (None, ['pfull_one', 'parity_two']), ('pfull_', ['pfull_one']),
    ('pfull_one', ['pfull_one']),
])
def test_cleanup_preserves_default_and_narrower_prefixes(
    monkeypatch, cleanup_transport, prefix, expected,
):
    """Control: rejecting all prefixes must fail; retain the explicitly narrower operation."""
    argv = ['drop-parity', '--dsn', 'unused']
    if prefix is not None:
        argv += ['--prefix', prefix]
    monkeypatch.setattr(sys, 'argv', argv)
    try:
        result = cleanup.main()
    except SystemExit as exc:
        result = exc.code
    assert result == 0, 'approved cleanup scopes must remain usable'
    assert [item for item in cleanup_transport if item != 'connect'] == expected
