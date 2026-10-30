"""PR579 credential boundaries, exercised only with synthetic credentials.

Red proof: unchanged tests against baseline 9807af5f9 on a Linux test host target
credentials._account and _config_home. Expected assertions reject cross-context
cache/store reads and cwd-relative credential paths. Exact nodes are recorded
in .claude/audits red/green receipts before this file enters the suite.
"""
from __future__ import annotations

import hashlib
import json

import pytest

import recall_hooks
from recall_hooks import credentials as cred


CONFIG = {
    'endpoint': 'https://memory.example.test/mcp',
    'account': 'owner@example.test',
    'auth': {'token_url': 'https://id.example.test/token', 'client_id': 'client-a', 'scopes': 'read write'},
}


@pytest.fixture(autouse=True)
def isolate(tmp_path, monkeypatch):
    monkeypatch.setenv('CLAUDE_CONFIG_DIR', str(tmp_path / 'config'))
    monkeypatch.delenv('RECALL_HOOK_CONFIG_HOME', raising=False)
    monkeypatch.setattr(cred, '_keyring', lambda: None)


def changed_config(field):
    if field == 'endpoint':
        return {**CONFIG, 'endpoint': 'https://other-memory.example.test/mcp'}
    return {**CONFIG, 'auth': {**CONFIG['auth'], field: {
        'token_url': 'https://other-id.example.test/token',
        'client_id': 'client-b', 'scopes': 'read',
    }[field]}}


@pytest.mark.parametrize('field', ['endpoint', 'token_url', 'client_id', 'scopes'])
def test_access_token_cannot_cross_credential_context(field):
    cred.store_access_token(CONFIG, cred.Credential('<synthetic-access>', 1000))
    assert cred.cached_access_token(changed_config(field), now=0) is None
    assert cred.cached_access_token(CONFIG, now=0) is not None


@pytest.mark.parametrize('field', ['endpoint', 'token_url', 'client_id', 'scopes'])
@pytest.mark.parametrize('keychain', [False, True])
def test_refresh_token_cannot_cross_credential_context(field, keychain, monkeypatch):
    class Ring:
        def __init__(self):
            self.values = {}
        def get_password(self, service, account):
            return self.values.get((service, account))
        def set_password(self, service, account, value):
            self.values[service, account] = value
        def delete_password(self, service, account):
            del self.values[service, account]
    ring = Ring()
    if keychain:
        monkeypatch.setattr(cred, '_keyring', lambda: ring)
    cred.write_refresh_token(CONFIG, '<synthetic-refresh>')
    assert cred.read_refresh_token(changed_config(field)) is None
    assert cred.read_refresh_token(CONFIG) == '<synthetic-refresh>'
    cred.logout(changed_config(field))
    assert cred.read_refresh_token(CONFIG) == '<synthetic-refresh>'


def test_legacy_unbound_tokens_are_not_reused():
    path = cred.fallback_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({CONFIG['account']: '<legacy-refresh>'}))
    legacy = path.parent / ('recall-hook-token-' + hashlib.sha256(CONFIG['account'].encode()).hexdigest()[:16] + '.json')
    legacy.write_text(json.dumps({'account': CONFIG['account'], 'access_token': '<legacy-access>', 'expires_at': 1000}))
    assert cred.read_refresh_token(CONFIG) is None
    assert cred.cached_access_token(CONFIG, now=0) is None


@pytest.mark.parametrize('kind', ['tilde', 'padded', 'hook-home'])
def test_credential_home_matches_hook_home(kind, tmp_path, monkeypatch):
    home = tmp_path / 'home'
    home.mkdir()
    monkeypatch.setenv('HOME', str(home))
    monkeypatch.setenv('USERPROFILE', str(home))
    if kind == 'tilde':
        monkeypatch.setenv('CLAUDE_CONFIG_DIR', '~/.claude-alt')
    elif kind == 'padded':
        monkeypatch.setenv('CLAUDE_CONFIG_DIR', '  ' + str(home / 'config') + '  ')
    else:
        monkeypatch.setenv('RECALL_HOOK_CONFIG_HOME', str(home / 'hooks'))
    project = tmp_path / 'project'
    project.mkdir()
    monkeypatch.chdir(project)
    expected = recall_hooks.claude_config_home()
    assert cred.token_cache_path(CONFIG).parent == expected
    assert cred.fallback_path().parent == expected
    cred.write_refresh_token(CONFIG, '<synthetic-refresh>')
    cred.store_access_token(CONFIG, cred.Credential('<synthetic-access>', 1000))
    assert not (project / '~').exists()
