"""CCA PR519 through PR510 behavioral regressions.

Red proof uses the unchanged tests against production bytes from commit 7b1ef2aa8.
BUG-001 must fail the escaped-error/invalid-warning assertions in run_checks;
BUG-002 must fail the registered-status assertion; BUG-003 must fail the call
count assertion. DAT-001 must fail visible-column and width assertions against
real PostgreSQL. NUM-001 rank-six cases must fail recovery assertions; rank-five
cases are unchanged-behavior controls. Exact node IDs and outcomes are retained
in the audit's red.xml and green.xml receipts.
"""

from __future__ import annotations

from dataclasses import replace
import json

import psycopg
from psycopg import sql
import pytest

from recall import claude_code, doctor
from recall.trust_policy import TrustPolicy
from recall_mcp import service
from scripts.summarize_query_construction_batch import summarize
from tests.conftest import TEST_DSN, requires_db
from tests.test_query_construction_service import _Store, _frame, _result


def _configure_client(monkeypatch, tmp_path, document):
    config = tmp_path / 'client.json'
    config.write_text(json.dumps(document), encoding='utf-8')
    monkeypatch.setattr(claude_code, 'claude_code_detected', lambda: True)
    monkeypatch.setattr(claude_code, 'client_config_path', lambda: config)


@pytest.mark.parametrize('layer', ['root', 'projects', 'project', 'servers'])
def test_doctor_reports_invalid_client_shape(monkeypatch, tmp_path, layer):
    """BUG-001: parsed JSON of the wrong shape must become a warning."""
    document = {
        'root': [],
        'projects': {'projects': ['invalid']},
        'project': {'projects': {str(tmp_path): ['invalid']}},
        'servers': {'projects': {str(tmp_path): {'mcpServers': ['invalid']}}},
    }[layer]
    _configure_client(monkeypatch, tmp_path, document)
    error = None
    report = None
    try:
        report = doctor.run_checks(
            dsn='sqlite:///' + (tmp_path / 'missing.sqlite').as_posix(),
            embedder='hashing', trust_mode='development', project_root=tmp_path,
        )
    except Exception as exc:
        error = type(exc).__name__
    assert error is None, f'client shape escaped the diagnostic as {error}'
    checks = [c for c in report.checks if c.name == 'claude code']
    assert any(c.status == 'warn' and 'invalid' in c.detail.lower() for c in checks)


def test_doctor_recognizes_resolving_project_alias(monkeypatch, tmp_path):
    """BUG-002: the writer's directory identity must survive doctor lookup."""
    (tmp_path / 'child').mkdir()
    alias = str(tmp_path / 'child' / '..')
    _configure_client(monkeypatch, tmp_path, {
        'projects': {alias: {'mcpServers': {'memory': {'command': 'recall-mcp'}}}},
    })
    report = doctor.run_checks(
        dsn='sqlite:///' + (tmp_path / 'missing.sqlite').as_posix(),
        embedder='hashing', trust_mode='development', project_root=tmp_path,
    )
    checks = [c for c in report.checks if c.name == 'claude code']
    assert any(c.status == 'ok' and 'memory' in c.detail for c in checks)


@pytest.mark.parametrize('failure', ['retrieve', 'generation'])
def test_construction_counts_failed_retrieval_attempt(monkeypatch, failure):
    """BUG-003: completed and failed candidate calls both consume an attempt."""
    calls = []
    store = _Store()
    store.generation_id = 'active-generation'

    def retrieve(*args, **kwargs):
        query = args[2]
        calls.append(query)
        if len(calls) == 1:
            result = replace(_result(query, ('c0',)), generation_id=store.generation_id)
        elif failure == 'retrieve':
            raise RuntimeError('candidate failed')
        else:
            result = replace(_result(query, ('c1',)), generation_id='other-generation')
        return type('Retrieved', (), {'result': result})()

    monkeypatch.setattr(service, '_retrieve_trusted', retrieve)
    response = service.query_construction_challenge(
        store, object(), 'Find the release procedure.', 'original query',
        arm='original_loop', frame=_frame('refined release procedure'),
        policy=TrustPolicy.development(),
    )
    assert len(calls) == 2
    assert len(response['diagnostics']['provider_failures']) == 1
    assert response['diagnostics']['retrieval_calls'] == 2


@requires_db
def test_doctor_does_not_merge_shadowed_table_columns():
    """DAT-001: metadata must describe the relation an unqualified read opens."""
    with psycopg.connect(TEST_DSN) as conn:
        conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        conn.execute('CREATE SCHEMA cca519_front')
        conn.execute('CREATE SCHEMA cca519_back')
        conn.execute('CREATE TABLE cca519_front.shadowed (tenant_id text)')
        conn.execute('CREATE TABLE cca519_back.shadowed (source text, embedding vector(7))')
        conn.execute('SET LOCAL search_path TO cca519_front, cca519_back, public')
        assert doctor._chunk_tables(conn)['shadowed'] == {'tenant_id'}
        checks = list(doctor._corpus_checks(conn, table='shadowed', tenant='acme'))
        assert any(c.name == 'schema' and c.status == 'fail' for c in checks)
        conn.rollback()


@requires_db
def test_doctor_embedding_width_follows_search_path():
    """DAT-001: reversing search_path must select the other visible vector width."""
    with psycopg.connect(TEST_DSN) as conn:
        conn.execute('CREATE EXTENSION IF NOT EXISTS vector')
        conn.execute('CREATE SCHEMA cca519_width_first')
        conn.execute('CREATE SCHEMA cca519_width_second')
        conn.execute('CREATE TABLE cca519_width_first.shadowed (embedding vector(3))')
        conn.execute('CREATE TABLE cca519_width_second.shadowed (embedding vector(7))')
        for first, second, width in (
            ('cca519_width_first', 'cca519_width_second', 3),
            ('cca519_width_second', 'cca519_width_first', 7),
        ):
            conn.execute(sql.SQL('SET LOCAL search_path TO {}, {}, public').format(
                sql.Identifier(first), sql.Identifier(second),
            ))
            assert doctor._embedding_width(conn, 'shadowed') == width
        conn.rollback()


@pytest.mark.parametrize('surface', ['retrieval', 'trusted_evidence'])
@pytest.mark.parametrize('rank', [5, 6])
def test_recovery_score_uses_top_five(surface, rank):
    """NUM-001: rank six is red pre-fix; rank five is the preservation control."""
    hits = [
        {'source': f'other-{i}.md', 'chunk_id': f'c{i}', 'verdict': 'ok'}
        for i in range(rank - 1)
    ] + [{'source': 'gold.md', 'chunk_id': 'gold', 'verdict': 'ok'}]
    final = {surface: {'hits' if surface == 'retrieval' else 'items': hits}}
    result = summarize({'rows': [{
        'task_id': 'rank-control', 'arm': 'original_loop',
        'gold': {'gold_sources': ['gold.md']}, 'final': final,
    }]})
    assert result['arms']['original_loop']['recovery_rate'] == (1 if rank == 5 else 0)
