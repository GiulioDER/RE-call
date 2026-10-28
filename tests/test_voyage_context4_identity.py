"""Contextual partition inputs must participate in the runtime embedding identity."""
from __future__ import annotations

import sys
import types

import pytest

from recall.embedding_providers import voyage
from recall.embedding_providers.voyage import _voyage_token_counter as load_token_counter
from recall.embedding_registry import registered_profile
from recall.generation_build import BuildRequest, pipeline_identity


class _Client:
    def __init__(self, **kwargs):
        self.documents = []

    def contextualized_embed(self, *, inputs, input_type, output_dimension, **kwargs):
        groups = [[inputs[0]]] if input_type == "query" else inputs
        if input_type == "document":
            self.documents.extend(groups)
        return types.SimpleNamespace(results=[
            types.SimpleNamespace(embeddings=[
                [float(len(group)), float(index)] + [0.0] * (output_dimension - 2)
                for index, _ in enumerate(group)
            ]) for group in groups
        ])


def _build(monkeypatch, *, artifact="vocab-a", available=True, version="1.0"):
    class Tokenizer:
        @staticmethod
        def from_pretrained(model):
            if not available:
                raise OSError("offline")
            return Tokenizer()

        def no_truncation(self):
            pass

        def to_str(self):
            return artifact

        def encode_batch(self, texts):
            return [types.SimpleNamespace(ids=[0] * max(1, len(text) // 4)) for text in texts]

    module = types.ModuleType("tokenizers")
    module.Tokenizer = Tokenizer
    monkeypatch.setitem(sys.modules, "tokenizers", module)
    monkeypatch.setattr(voyage, "_voyage_token_counter", load_token_counter)
    monkeypatch.setattr(voyage, "_package_version", lambda name: version, raising=False)
    monkeypatch.setattr(voyage, "_voyage_client_class", lambda owner: _Client)
    return registered_profile("voyage-context-4-v1").build(api_key="test")


@pytest.mark.parametrize("change", ["availability", "artifact", "runtime"])
def test_partition_counter_changes_runtime_and_pipeline_identity(monkeypatch, change):
    """Red proof: HEAD 4c89b241, unchanged test on the testbench, all three variants failed the
    profile-fingerprint inequality. VoyageContextualizedEmbedder retained the registry identity
    while tokenizer availability, serialized artifact, or runtime changed. Restored fix passes.
    """
    first = _build(monkeypatch)
    first_identity = first.profile.fingerprint()
    first_pipeline = pipeline_identity(first, BuildRequest()).fingerprint
    group = ["a" * 16_000, "b" * 16_000]
    first_vectors = first.embed_document_groups([group])
    second = _build(
        monkeypatch,
        available=change != "availability",
        artifact="vocab-b" if change == "artifact" else "vocab-a",
        version="2.0" if change == "runtime" else "1.0",
    )
    second_identity = second.profile.fingerprint()
    second_vectors = second.embed_document_groups([group])
    if change == "availability":
        assert first_vectors != second_vectors  # the provider sees different neighboring chunks
    assert first_identity != second_identity
    assert first_pipeline != pipeline_identity(second, BuildRequest()).fingerprint
    assert second_identity == second.profile.fingerprint()  # immutable after passage embedding
    same = _build(
        monkeypatch,
        available=change != "availability",
        artifact="vocab-b" if change == "artifact" else "vocab-a",
        version="2.0" if change == "runtime" else "1.0",
    )
    assert same.profile.fingerprint() == second_identity


def test_identified_custom_counter_requires_explicit_identity(monkeypatch):
    """Red proof: HEAD 4c89b241 accepted an unidentified custom counter with a registered
    profile, failing pytest.raises. The fix refuses to invent an identity for arbitrary code.
    """
    monkeypatch.setattr(voyage, "_voyage_client_class", lambda owner: _Client)
    identity = registered_profile("voyage-context-4-v1").identity()
    with pytest.raises(ValueError, match="token_counter_identity"):
        voyage.VoyageContextualizedEmbedder(
            api_key="test", identity=identity, token_counter=lambda texts: [1] * len(texts)
        )
    def build(counter_id):
        return voyage.VoyageContextualizedEmbedder(
            api_key="test", identity=identity, token_counter=lambda texts: [1] * len(texts),
            token_counter_identity=counter_id,
        )
    assert build("count-v1").profile.fingerprint() == build("count-v1").profile.fingerprint()
    assert build("count-v1").profile.fingerprint() != build("count-v2").profile.fingerprint()
