"""Behavioral regressions from the independently verified fact audit.

The original combined nodes failed their intended assertions before repairs.
Splitting by feature ownership preserves the exact test bodies for the PR stack.
"""
from __future__ import annotations
import asyncio
import json
import pytest
from types import SimpleNamespace
from tests.test_stated_facts_postgres import manager as manager, body, promote, uri, served

def test_sec002_current_facts_respects_source_policy(manager, monkeypatch):
    from recall.embeddings import HashingEmbedder
    from recall.security_policy import AccessContext, SourceRule, SourceSecurityPolicy
    from recall_mcp import server as srv
    from recall_mcp.auth import SCOPE_READ
    from recall_mcp.settings import Settings
    from tests.conftest import TEST_DSN
    marker = 'SEC002_SYNTHETIC_RESTRICTED_VALUE'
    monkeypatch.setenv('RECALL_STATED_FACT_SOURCES', uri(manager, 'memory'))
    monkeypatch.setenv('RECALL_FACT_WRITE_DSN', TEST_DSN)
    monkeypatch.delenv('RECALL_ASSISTANT_MEMOS', raising=False)
    promote(manager, {'memory/private.md': body([{'subject':'test-user','predicate':'private-value','object':marker}]),'docs/other.md':body(text='Unrelated public background.')}, 'sec002-v1')
    policy = SourceSecurityPolicy((SourceRule('memory', classification='restricted', principals=frozenset({'owner'})),))
    caller = AccessContext('reader', manager.tenant_id, clearance='public')
    assert not policy.decide('memory/private.md', caller).allowed
    token = SimpleNamespace(scopes=[SCOPE_READ], claims={'tenant':manager.tenant_id,'principal':'reader','clearance':'public'},client_id='sec002-reader')
    monkeypatch.setattr(srv, 'get_access_token', lambda: token)
    settings = Settings.from_env({'RECALL_TRANSPORT':'stdio','RECALL_EMBEDDER':'hashing','RECALL_MCP_TOOLS':'recall_current_facts'})
    mcp = srv.build_server(settings)
    tool = next(t for t in mcp._tool_manager.list_tools() if t.name == 'recall_current_facts')
    with served(manager) as store:
        class Registry:
            def get(self, tenant):
                assert tenant == manager.tenant_id
                return store
        state = {'embedder':HashingEmbedder(dim=64),'stores':Registry(),'source_security_policy':policy,'settings':settings}
        ctx = SimpleNamespace(request_context=SimpleNamespace(lifespan_context=state))
        result = json.loads(asyncio.run(tool.fn(ctx=ctx)))
    assert marker not in json.dumps(result), 'restricted fact leaked through the real MCP tool'


@pytest.mark.parametrize('admitted', ['first', 'second'])
def test_sec003_duplicate_names_do_not_borrow_source_admission(make_store, tmp_path, monkeypatch, admitted):
    """An excluded absolute source cannot borrow admission from an allowed alias.

    Red baseline 9beca141 plus the preceding audit repairs: the final assertion fails
    because note.md combines the excluded source's hash with its admitted namesake.
    The fixture uses two normal recall_index calls and verifies both absolute sources.
    The first-source preservation control goes red against the rejected all-sources
    admission repair, which lets an excluded namesake hide authorized content.
    """
    from uuid import uuid4
    from recall_mcp.indexing import index_memory
    from tests.test_generations import _Embedder
    from tests.test_stated_facts_postgres import DOOR, listed
    from tests.conftest import TEST_DSN
    root = tmp_path / 'root'
    first, second = root / 'first', root / 'second'
    first.mkdir(parents=True)
    second.mkdir()
    (first / 'note.md').write_bytes(body([DOOR]))
    (second / 'note.md').write_bytes(body(text='Unrelated background.'))
    monkeypatch.setenv('RECALL_INDEX_ROOT', str(root))
    monkeypatch.setenv('RECALL_STATED_FACT_SOURCES', f'{first} {second}')
    monkeypatch.setenv('RECALL_FACT_WRITE_DSN', TEST_DSN)
    monkeypatch.setenv('RECALL_EMBED_CACHE', 'off')
    store = make_store(64)
    store._tenant = 'alias-admission-' + uuid4().hex
    for folder in (first, second):
        index_memory(store, _Embedder(1), str(folder), env={'RECALL_INDEX_ROOT':str(root), 'RECALL_FACT_WRITE_DSN':TEST_DSN})
    assert len(store.sources_for_identifiers(['note.md'])['note.md']) == 2
    assert listed(store) == [('4512', ('note.md',))], 'fixture did not state the first source'
    monkeypatch.setenv('RECALL_STATED_FACT_SOURCES', str(root / admitted))
    expected = [('4512', ('note.md',))] if admitted == 'first' else []
    assert listed(store) == expected, 'source admission and content came from different namesakes'


def test_sec003_excluded_namesake_cannot_supersede(make_store, tmp_path, monkeypatch):
    """An excluded claimant cannot borrow its admitted namesake's authority.

    Red against the original supersession admission (ANY resolved source admitted):
    the final assertion loses 4512 after indexing an excluded supersedes claimant.
    The namesake and target controls are indexed through the public legacy entry point.
    """
    from uuid import uuid4
    from recall_mcp.indexing import index_memory
    from tests.test_generations import _Embedder
    from tests.test_stated_facts_postgres import DOOR, listed
    from tests.conftest import TEST_DSN
    root = tmp_path / 'root'
    allowed, excluded, target = (root / name for name in ('allowed', 'excluded', 'target'))
    for folder in (allowed, excluded, target):
        folder.mkdir(parents=True)
    (allowed / 'note.md').write_bytes(body(text='An unrelated authorized memo.'))
    (excluded / 'note.md').write_bytes(body(text='An unauthorized claimant.', supersedes='old.md'))
    (target / 'old.md').write_bytes(body([DOOR]))
    monkeypatch.setenv('RECALL_INDEX_ROOT', str(root))
    monkeypatch.setenv('RECALL_STATED_FACT_SOURCES', f'{allowed} {target}')
    monkeypatch.setenv('RECALL_FACT_WRITE_DSN', TEST_DSN)
    monkeypatch.setenv('RECALL_EMBED_CACHE', 'off')
    store = make_store(64)
    store._tenant = 'alias-supersession-' + uuid4().hex
    for folder in (allowed, target):
        index_memory(store, _Embedder(1), str(folder), env={'RECALL_INDEX_ROOT':str(root), 'RECALL_FACT_WRITE_DSN':TEST_DSN})
    assert listed(store) == [('4512', ('old.md',))], 'fixture did not state the authorized target'
    index_memory(store, _Embedder(1), str(excluded), env={'RECALL_INDEX_ROOT':str(root), 'RECALL_FACT_WRITE_DSN':TEST_DSN})
    assert len(store.sources_for_identifiers(['note.md'])['note.md']) == 2
    assert listed(store) == [('4512', ('old.md',))], 'excluded claimant borrowed an admitted namesake authority'
