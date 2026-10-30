"""PR570/572/579 sync regressions. Red proof uses baseline 9807af5f9.

Expected assertion failures: preplanning errors produce no notice; unchanged
normalized Markdown reuploads against raw-byte inventory; legacy memo edits add
another source instead of replacing the old value. All transport credentials
and content are synthetic, with a real disposable test database for replacement.
"""
from __future__ import annotations

import hashlib
import os
import uuid

import pytest

import recall_hooks
from recall_hooks import credentials, hosted

from .conftest import TEST_DSN, requires_db


CONFIG = {'endpoint': 'https://memory.example.test/mcp', 'account': 'sync@example.test'}


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(tmp_path / 'config'))
    monkeypatch.delenv('RECALL_HOOK_CONFIG_HOME', raising=False)


@pytest.mark.parametrize('failure', ['auth', 'inventory'])
def test_preplanning_failure_reports_pending_memos(failure, tmp_path, monkeypatch):
    root = tmp_path / 'memory'
    root.mkdir()
    (root / 'memo.md').write_text('# Memo\n\nOrdinary project knowledge.\n')
    hosted.write_manifest(CONFIG, {'pending': {}})
    def auth(_config):
        if failure == 'auth':
            raise credentials.AuthError('synthetic credential unavailable')
        return {}
    def inventory(*args):
        raise hosted.SyncError('network', 'synthetic inventory unavailable')
    monkeypatch.setattr(credentials, 'headers', auth)
    monkeypatch.setattr(hosted, 'remote_inventory', inventory)
    hosted.sync_memory_roots([('worktree', root)], CONFIG)
    assert len(hosted.read_manifest(CONFIG)['pending']) == 1
    assert '1 memo(s)' in recall_hooks._unsynced_notice(CONFIG)


@pytest.mark.parametrize('raw', [b'# Memo\r\n\r\nClean prose.\r\n', b'\xef\xbb\xbf# Memo\n\nClean prose.\n', b'# Memo\n\nClean\x00 prose.\n'])
def test_unchanged_markdown_matches_raw_upload_inventory(raw, tmp_path, monkeypatch):
    from recall.desktop.uploads import stage_uploads
    root = tmp_path / 'memory'
    root.mkdir()
    path = root / 'memo.md'
    path.write_bytes(raw)
    remote = {}
    monkeypatch.setattr(credentials, 'headers', lambda _config: {})
    monkeypatch.setattr(hosted, 'remote_inventory', lambda *_args: dict(remote))
    def upload(_endpoint, _head, _tool, params):
        _job, staged, _size = stage_uploads('digest-' + uuid.uuid4().hex, params['files'])
        for path in staged.rglob('*'):
            if path.is_file():
                remote[path.as_uri()] = hashlib.sha256(path.read_bytes()).hexdigest()
        return {'state': 'completed'}
    monkeypatch.setattr(hosted, 'call_tool', upload)
    first = hosted.sync_memory_roots([('worktree', root)], CONFIG)
    second = hosted.sync_memory_roots([('worktree', root)], CONFIG)
    assert first.uploaded == 1
    assert second.uploaded == 0 and second.unchanged == 1
    assert path.read_bytes() == raw


@requires_db
def test_legacy_sync_replaces_edited_memo_and_preserves_other_project(tmp_path, monkeypatch):
    from recall.desktop.uploads import stage_uploads
    from recall.embeddings import HashingEmbedder
    from recall.store import PgVectorStore
    from recall_mcp.server import ingest_into_serving_store
    from recall_mcp.settings import Settings

    monkeypatch.delenv('RECALL_ENV', raising=False)
    monkeypatch.setenv('RECALL_LOCAL_ALLOWLIST', os.environ['RECALL_INDEX_ROOT'])
    store = PgVectorStore(TEST_DSN, dim=64, tenant='sync-regression-' + uuid.uuid4().hex)
    store.check_schema()
    embedder = HashingEmbedder(dim=64)
    state = {'embedder': embedder, 'settings': Settings.from_env(dict(os.environ)), 'generation_mode': False}
    monkeypatch.setattr(credentials, 'headers', lambda _config: {})
    monkeypatch.setattr(hosted, 'remote_inventory', lambda *_args: store.source_raw_hashes())
    def upload(_endpoint, _head, _tool, params):
        _job, staged, _size = stage_uploads(store.tenant, params['files'])
        namespace = params.get('source_namespace')
        kwargs = {'source_namespace': namespace} if namespace is not None else {}
        ingest_into_serving_store(state, store, str(staged), params['category'], **kwargs)
        return {'state': 'completed'}
    monkeypatch.setattr(hosted, 'call_tool', upload)
    roots = [tmp_path / 'project-a', tmp_path / 'project-b']
    bodies = ['# Memo\n\nOriginal policy alpha.\n', '# Memo\n\nIndependent policy beta.\n']
    try:
        for root, body in zip(roots, bodies, strict=True):
            root.mkdir()
            (root / 'memo.md').write_text(body)
            assert hosted.sync_memory_roots([('worktree', root)], CONFIG).uploaded == 1
        updated = '# Memo\n\nCorrected policy gamma.\n'
        (roots[0] / 'memo.md').write_text(updated)
        assert hosted.sync_memory_roots([('worktree', roots[0])], CONFIG).uploaded == 1
        hashes = store.source_raw_hashes()
        assert len(hashes) == 2, 'an edited memo must replace its source, retaining the other project'
        assert set(hashes.values()) == {hashlib.sha256(text.encode()).hexdigest() for text in [updated, bodies[1]]}
        assert hosted.sync_memory_roots([('worktree', roots[0])], CONFIG).unchanged == 1
    finally:
        store.close()
