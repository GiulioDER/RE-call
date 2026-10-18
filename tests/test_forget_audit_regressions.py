"""Recovery regressions for absolute-source retries and concurrent forget plans."""
from __future__ import annotations
import sqlite3
from recall.lite import LiteStore
from recall.lite.facts import LiteFactLedger
from recall_mcp.lifecycle import forget_memory
from tests.test_fact_retraction import remember, apply, current, DOOR
from tests.fact_fixtures import NOW

def test_bug001_absolute_forget_retry_retracts(tmp_path, monkeypatch):
    with LiteStore(tmp_path / 'retry.db', dim=8) as store:
        remember(store, 'door.md')
        apply(store, DOOR, 'door.md', 'request-1')
        def fail(*a, **kw):
            raise sqlite3.OperationalError('synthetic failure')
        with monkeypatch.context() as patch:
            patch.setattr(LiteFactLedger, 'retract_forgotten', fail)
            result = forget_memory(store, ['/notes/door.md'])
        assert result.facts_retracted == -1
        result = forget_memory(store, ['/notes/door.md'])
        assert current(store) == [], 'absolute-source retry left the assertion live'
        assert result.facts_retracted == 1

def test_dat001_two_prepared_forgets_retract_the_last_support(make_store):
    """DAT-001: two pre-delete plans must observe both committed deletions.

    Original PR 931 head a2a075a7 fails on the bare ledger assertion after both callbacks: each
    frozen plan incorrectly counts the other deleted source as surviving.
    """
    from uuid import uuid4
    from recall.fact_ledger import PostgresFactLedger
    from recall_mcp.lifecycle import _forget_retractions
    from recall.provenance_cards import PostgresEvidenceCardStore
    from tests.conftest import TEST_DSN
    from tests.test_fact_retraction_postgres import _pg, _remember, _apply
    tenant = f"retraction-race-{uuid4().hex}"
    with _pg(make_store, tenant) as store:
        ledger = PostgresFactLedger(TEST_DSN, tenant_id=tenant)
        cards = PostgresEvidenceCardStore(TEST_DSN, tenant_id=tenant)
        for name in ('a.md', 'b.md'):
            _remember(store, name)
            _apply(store, cards, ledger, name, f'apply-{name}')
        a = _forget_retractions(store, None, ['a.md'], ['/notes/a.md'])
        b = _forget_retractions(store, None, ['b.md'], ['/notes/b.md'])
        store.delete_sources(['/notes/a.md'])
        store.delete_sources(['/notes/b.md'])
        a()
        b()
        assert not ledger.current(tenant_id=tenant, now=NOW), 'both deleted supports escaped permanent retraction'
        _remember(store, 'a.md')
        assert not ledger.current(tenant_id=tenant, now=NOW), 'restoring a memo revived explicitly forgotten facts'
