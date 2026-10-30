"""CCA repair regressions for FIX-005 and FIX-006.

Red proof targets the first repair tree 6ac03365c692f838ef8f75d855cfcccaf6c72600:
publication before index_memory admission overwrites denied/busy sources; ancestry
matching assigns the wrong root; raw-size-only scan admits normalized-empty files.
Every node fails its final behavior assertion on that pre-correction implementation.
"""
from __future__ import annotations

import base64
from contextlib import contextmanager
from types import SimpleNamespace

import pytest

from recall_hooks import credentials, hosted

CONFIG={'endpoint':'https://memory.example.test/mcp','account':'repair@example.test'}


@pytest.fixture(autouse=True)
def isolate(tmp_path,monkeypatch):
    monkeypatch.setenv('CLAUDE_CONFIG_DIR',str(tmp_path/'config'))
    monkeypatch.delenv('RECALL_HOOK_CONFIG_HOME',raising=False)
    monkeypatch.setattr(credentials,'headers',lambda _config:{})
    monkeypatch.setattr(hosted,'remote_inventory',lambda *_args:{})


@pytest.mark.parametrize('raw',[b'\xef\xbb\xbf',b'\x00',b'\xef\xbb\xbf\x00'])
def test_normalized_empty_file_does_not_refuse_the_batch(raw,tmp_path,monkeypatch):
    from recall.desktop.uploads import UploadError,stage_uploads
    root=tmp_path/'memory'
    root.mkdir()
    (root/'empty.md').write_bytes(raw)
    (root/'good.md').write_bytes(b'# Memo\n\nUseful prose.\n')
    (root/'spaces.md').write_bytes(b' \n')
    sent=[]
    def upload(_endpoint,_head,_tool,params):
        try:
            stage_uploads('repair-empty',params['files'])
        except UploadError as exc:
            raise hosted.SyncError('refusal',str(exc)) from exc
        sent.extend(item['name'] for item in params['files'])
        return {'state':'completed'}
    monkeypatch.setattr(hosted,'call_tool',upload)
    result=hosted.sync_memory_roots([('worktree',root)],CONFIG)
    assert result.uploaded==2 and sorted(sent)==['worktree/good.md','worktree/spaces.md']
    assert hosted.read_manifest(CONFIG)['skipped']['worktree/empty.md']==hosted.SKIP_EMPTY


def test_source_identity_uses_the_exact_producing_root(tmp_path,monkeypatch):
    root=tmp_path/'memory'
    sub=root/'sub'
    sub.mkdir(parents=True)
    (sub/'a.md').write_text('# Note\n\nSafe prose.\n')
    sent=[]
    def upload(_endpoint,_head,_tool,params):
        sent.append({item['name']:item.get('source_key') for item in params['files']})
        return {'state':'completed'}
    monkeypatch.setattr(hosted,'call_tool',upload)
    hosted.sync_memory_roots([('worktree',root),('worktree',sub)],CONFIG)
    hosted.sync_memory_roots([('worktree',sub)],CONFIG)
    assert sent[0]['worktree/a.md']==sent[1]['worktree/a.md']
    assert sent[0]['worktree/sub/a.md']!=sent[0]['worktree/a.md']


def existing_upload(tmp_path,monkeypatch):
    from recall.desktop.uploads import promote_uploads,stage_uploads
    from recall_mcp.settings import Settings
    monkeypatch.setenv('RECALL_INDEX_ROOT',str(tmp_path))
    tenant='repair-admission'
    def payload(body):return [{'name':'blocked/memo.md','content_b64':base64.b64encode(body).decode()}]
    _job,staged,_size=stage_uploads(tenant,payload(b'old source'))
    stable=promote_uploads(tenant,staged,'sync-memory-review')
    _job,pending,_size=stage_uploads(tenant,payload(b'new source'))
    store=SimpleNamespace(tenant=tenant,_dsn=None)
    state={'embedder':object(),'settings':Settings.from_env({'RECALL_INDEX_ROOT':str(tmp_path)}),'generation_mode':False}
    return store,state,pending,stable/'blocked/memo.md'


def test_denied_namespace_upload_does_not_overwrite_existing_source(tmp_path,monkeypatch):
    from recall.security_policy import AccessContext,SourceSecurityPolicy
    from recall_mcp.indexing import IndexPreflightError
    from recall_mcp.server import ingest_into_serving_store
    store,state,pending,old=existing_upload(tmp_path,monkeypatch)
    state['source_security_policy']=SourceSecurityPolicy(())
    context=AccessContext(tenant=store.tenant,principal='restricted',purpose='test',clearance='public')
    with pytest.raises(IndexPreflightError):
        ingest_into_serving_store(state,store,str(pending),'memory',access_context=context,source_namespace='review')
    assert old.read_bytes()==b'old source'


def test_busy_writer_does_not_overwrite_existing_source(tmp_path,monkeypatch):
    from recall import backends
    from recall.lite.writer_lock import LiteWriterBusy
    from recall_mcp.server import ingest_into_serving_store
    store,state,pending,old=existing_upload(tmp_path,monkeypatch)
    @contextmanager
    def busy(*_args,**_kwargs):
        raise LiteWriterBusy('write','ingest',0.0,1,None)
        yield
    monkeypatch.setattr(backends,'write_session',busy)
    with pytest.raises(LiteWriterBusy):
        ingest_into_serving_store(state,store,str(pending),'memory',source_namespace='review')
    assert old.read_bytes()==b'old source'
