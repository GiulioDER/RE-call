"""Pinned aliases must reject local shadows before optional library code can run.

SEC-001/002 red proof: baseline 61b428432 invokes a recording loader from each
public construction path before refusing the shadow. Both refusal nodes must fail
the marker assertion, then pass unchanged. Pinned-load controls are also tested
against mutations dropping the revision argument; receipts retain the exact nodes.
"""

import sys
from types import SimpleNamespace

import pytest

from recall.embeddings import (
    SFR_CODE_EMBEDDER_MODEL,
    SFR_CODE_EMBEDDER_REVISION,
    resolve_embedder,
)
from recall.rerank import COREB_CODE_RERANKER_MODEL, COREB_CODE_RERANKER_REVISION
from recall_mcp import service


@pytest.fixture
def code_loaders(monkeypatch, tmp_path):
    marker = tmp_path / "loader-called"
    calls = []

    def load(model, **kwargs):
        marker.write_text("called", encoding="utf-8")
        calls.append((model, kwargs))
        return SimpleNamespace(
            get_sentence_embedding_dimension=lambda: 4,
            convert_tokens_to_ids=lambda token: {"yes": 1, "no": 0}[token],
            eval=lambda: None,
        )

    monkeypatch.setitem(sys.modules, "sentence_transformers", SimpleNamespace(SentenceTransformer=load))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=load),
        AutoModelForCausalLM=SimpleNamespace(from_pretrained=load),
    ))
    monkeypatch.chdir(tmp_path)
    return marker, calls


def _load_alias(alias):
    if alias == "sfr-code":
        return resolve_embedder(alias, {
            "RECALL_ACCEPT_RESEARCH_MODEL_LICENSE": "1",
            "RECALL_ACCEPT_REMOTE_MODEL_CODE": "1",
        })
    return service._new_reranker({
        "RECALL_RETRIEVAL_PROFILE": "code",
        "RECALL_ACCEPT_REMOTE_MODEL_CODE": "1",
    })


@pytest.mark.parametrize("alias,model", [
    ("sfr-code", SFR_CODE_EMBEDDER_MODEL),
    ("coreb-code", COREB_CODE_RERANKER_MODEL),
])
def test_pinned_alias_refuses_local_shadow_before_loading(code_loaders, tmp_path, alias, model):
    marker, calls = code_loaders
    (tmp_path / model).mkdir(parents=True)
    refusal = None
    try:
        _load_alias(alias)
    except ValueError as exc:
        refusal = str(exc)
    assert not marker.exists(), "pinned alias must refuse a local shadow before loading code"
    assert not calls
    assert refusal and "shadow" in refusal.lower()


@pytest.mark.parametrize("alias,model,revision", [
    ("sfr-code", SFR_CODE_EMBEDDER_MODEL, SFR_CODE_EMBEDDER_REVISION),
    ("coreb-code", COREB_CODE_RERANKER_MODEL, COREB_CODE_RERANKER_REVISION),
])
def test_unshadowed_alias_keeps_reviewed_revision(code_loaders, alias, model, revision):
    _, calls = code_loaders
    result = None
    refusal = None
    try:
        result = _load_alias(alias)
    except ValueError as exc:
        refusal = str(exc)
    assert result is not None, f"an unshadowed pinned alias must remain loadable: {refusal}"
    assert calls
    assert all(name == model and kw["revision"] == revision for name, kw in calls)
    assert all(kw["trust_remote_code"] is True for _, kw in calls)


def test_explicit_st_local_path_remains_loadable(code_loaders, tmp_path):
    """A blanket local-path refusal mutation must fail this compatibility assertion."""
    _, calls = code_loaders
    local = tmp_path / "custom-model"
    local.mkdir()
    (local / "config.json").write_text("{}", encoding="utf-8")
    result = None
    refusal = None
    try:
        result = resolve_embedder(f"st:{local}", {})
    except ValueError as exc:
        refusal = str(exc)
    assert result is not None, f"explicit st: local models must remain loadable: {refusal}"
    assert calls[0][0] == str(local)
    assert calls[0][1]["trust_remote_code"] is False
