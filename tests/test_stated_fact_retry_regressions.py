"""Behavioral regressions from the independently verified fact audit.

The original combined nodes failed their intended assertions before repairs.
Splitting by feature ownership preserves the exact test bodies for the PR stack.
"""
from __future__ import annotations
from recall.lite import LiteStore
from recall.lite.facts import LiteFactLedger
from recall.lite.refresh import refresh
from tests.test_stated_facts import memo, listed
from tests.test_lite_calibration import ContentWordEmbedder, _write_memos, DIM

def test_bug002_stated_append_retries_without_file_change(tmp_path, monkeypatch):
    monkeypatch.setenv('RECALL_EMBED_CACHE', 'off')
    monkeypatch.delenv('RECALL_STATED_FACT_SOURCES', raising=False)
    folder = tmp_path / 'memory'
    _write_memos(folder, 4)
    memo(folder, 'door.md', facts=[{'subject':'user','predicate':'door_code','object':'4512'}])
    with LiteStore(tmp_path / 'memory.db', dim=DIM) as store:
        def fail(*a, **kw):
            raise RuntimeError('synthetic failure')
        with monkeypatch.context() as patch:
            patch.setattr(LiteFactLedger, 'apply_assertion', fail)
            failed = refresh(store, [folder], ContentWordEmbedder(), force=True)
        assert failed.facts_stated == -1
        retried = refresh(store, [folder], ContentWordEmbedder())
        assert listed(store) == ['4512'], f'unchanged refresh did not retry: {retried.outcome}'


def test_bug002_auto_refresh_retries_failed_stating(tmp_path, monkeypatch):
    """BUG-002: an unchanged folder retries a failed fact append after backoff.

    Baseline 9beca141 fails at the final current-fact assertion because its auto refresher
    adopts a failed reconciliation as a successful fingerprint and never retries it.
    """
    from recall_mcp.lite_refresh import LiteAutoRefresher
    from tests.test_lite_auto_refresh_mcp import Clock
    monkeypatch.setenv('RECALL_EMBED_CACHE', 'off')
    monkeypatch.delenv('RECALL_STATED_FACT_SOURCES', raising=False)
    root = tmp_path / 'memory'
    _write_memos(root, 4)
    memo(root, 'door.md', facts=[{'subject':'user','predicate':'door_code','object':'4512'}])
    clock = Clock()
    with LiteStore(tmp_path / 'auto.db', dim=DIM) as store:
        refresher = LiteAutoRefresher(store, ContentWordEmbedder(), [root], interval_s=0, wait_s=30, clock=clock)
        def fail(*args, **kwargs):
            raise RuntimeError('synthetic ledger failure')
        with monkeypatch.context() as patch:
            patch.setattr(LiteFactLedger, 'apply_assertion', fail)
            refresher._run()
        clock.now += 1000
        try:
            refresher.before_retrieval()
            assert listed(store) == ['4512'], 'auto refresh permanently adopted an incomplete stating run'
        finally:
            refresher.stop()
