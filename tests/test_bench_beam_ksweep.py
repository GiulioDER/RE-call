"""A coverage curve must identify its corpus and account for every eligible question.

Red proof: these refusal and provenance tests fail their intended assertions on
d03099027's benchmarks.beam.ksweep. Positive controls are additionally checked against
restrictive production mutations, recorded in the PR139 through130 audit proof receipt.
All provider and storage boundaries below are offline doubles.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest

from benchmarks.beam import ksweep

pytestmark = pytest.mark.benchharness


@pytest.fixture
def backend(monkeypatch):
    state = SimpleNamespace(counts={}, empty=set(), resolutions=[], tenants=[])

    class Store:
        def __init__(self, dsn, *, dim, tenant, table):
            self.tenant = tenant
            state.tenants.append(tenant)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def count(self):
            return state.counts.get(self.tenant, 2)

    def resolve(name):
        state.resolutions.append(name)
        return SimpleNamespace(
            dim=2, embed=lambda texts: [[1., 0.] if t == 'fact' else [0., 1.] for t in texts]
        )

    def search(store, embedder, question, **kwargs):
        texts = [] if store.tenant in state.empty else ['other', 'fact']
        return SimpleNamespace(hits=[SimpleNamespace(chunk=SimpleNamespace(text=t)) for t in texts])

    monkeypatch.setattr('benchmarks.systems.resolve_embedder', resolve)
    monkeypatch.setattr('benchmarks.systems.resolve_reranker', lambda name: None)
    monkeypatch.setattr('benchmarks._trust.bench_search', search)
    monkeypatch.setattr('recall.store.PgVectorStore', Store)
    return state


def _row(index=0):
    return {'conversation_idx': index, 'question': 'question', 'rubric': ['fact']}


def _sweep(rows, depths=None):
    return ksweep.sweep(
        rows, dsn='unused', table='bench_test', embedder_name='offline',
        depths=[1, 2] if depths is None else depths, reranker_name='none',
    )


@pytest.mark.parametrize('depths', [[], [0], [-1], [251], [1, 1], [2, 1], [1.5], [True]])
def test_invalid_depths_refuse_before_provider_resolution(backend, depths):
    """BUG-002: invalid slices and repeated depths must never become observations."""
    error = None
    try:
        _sweep([_row()], depths)
    except Exception as exc:
        error = exc
    assert isinstance(error, ValueError), f'expected depth validation, got {error!r}'
    assert 'depths' in str(error)
    assert backend.resolutions == []


@pytest.mark.parametrize('rows', [[], [{'question': 'question'}], [{'rubric': ['fact']}]])
def test_no_eligible_questions_refuses(backend, rows):
    """BUG-001: an empty selection must not publish a current artifact with n=0."""
    with pytest.raises(ValueError, match='eligible'):
        _sweep(rows)
    assert backend.resolutions == []


def test_missing_tenant_cannot_disappear_from_a_mixed_sample(backend):
    """BUG-001: the successful first conversation must not hide the absent second one."""
    backend.counts['beam-1m-1'] = 0
    backend.empty.add('beam-1m-1')
    with pytest.raises(SystemExit, match='ZERO chunks'):
        _sweep([_row(0), _row(1)])


def test_empty_retrieval_cannot_disappear_from_a_populated_tenant(backend):
    """BUG-001: a nonzero count does not license dropping a query with no hits."""
    backend.empty.add('beam-1m-1')
    with pytest.raises(RuntimeError, match='no memories'):
        _sweep([_row(0), _row(1)])


def test_complete_sample_preserves_prefix_scores_and_population(backend):
    """Control: replacing sweep's best scores with zeros fails the prefix-score assertion."""
    result = _sweep([_row(0), _row(1)])
    assert result['n'] == 2
    assert result['mean_nugget_sim'] == {'1': [0., 0.], '2': [1., 1.]}
    assert result['weakest_nugget_sim'] == result['mean_nugget_sim']
    assert backend.tenants == ['beam-1m-0', 'beam-1m-1']


def test_new_artifact_records_runtime_model_stack(backend, monkeypatch):
    """DAT-002: a newly generated artifact must not inherit the historical sentinel."""
    stack = {'torch': 'recorded-test-version'}
    monkeypatch.setattr('recall.eval.provenance.model_stack', lambda: stack)
    assert _sweep([_row()]).get('stack') == stack


def _cli(monkeypatch, tmp_path, payload, *, depths='1,2'):
    source = tmp_path / 'rows.json'
    source.write_text(json.dumps(payload), encoding='utf-8')
    output = tmp_path / 'output.json'
    monkeypatch.setattr(sys, 'argv', [
        'ksweep', '--rows', str(source), '--out', str(output), '--dsn', 'unused',
        '--table', 'bench_test', '--embedder', 'offline', '--reranker', 'none',
        '--depths', depths,
    ])
    return output


@pytest.mark.parametrize('chat_size', ['100K', '500K', '10M', None])
def test_cli_refuses_ambiguous_or_wrong_corpus_before_output(
    backend, monkeypatch, tmp_path, chat_size, capsys,
):
    """BUG-003: metadata must agree with the 1M tenant namespace before any model call."""
    output = _cli(monkeypatch, tmp_path, {'chat_size': chat_size, 'rows': [_row()]})
    with pytest.raises(SystemExit) as exc:
        ksweep.main()
    assert exc.value.code == 2
    assert 'chat_size' in capsys.readouterr().err
    assert not output.exists()
    assert backend.resolutions == []


def test_cli_accepts_1m_artifact_and_writes_its_population(backend, monkeypatch, tmp_path):
    """Control: mutating sweep's emitted n to zero fails the serialized population assertion."""
    output = _cli(monkeypatch, tmp_path, {'chat_size': '1M', 'rows': [_row()]})
    ksweep.main()
    assert json.loads(output.read_text())['n'] == 1


def test_cli_invalid_depth_cannot_create_output(backend, monkeypatch, tmp_path):
    """BUG-002: boundary validation protects the command's persisted result too."""
    output = _cli(monkeypatch, tmp_path, {'chat_size': '1M', 'rows': [_row()]}, depths='1,1')
    with pytest.raises(ValueError, match='depths'):
        ksweep.main()
    assert not output.exists()
