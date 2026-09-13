"""Production invariants for grouped Voyage Context 4 embedding.

These tests exercise the provider boundary with a shaped SDK stub. Mutation evidence for this
feature is recorded in the production handoff report. Flattening the nested request or dropping a
chunk makes the alignment and boundary tests fail before the implementation is restored.
"""
from __future__ import annotations

import hashlib
import sys
import types

import pytest

from recall.cache import embed_with_cache
from recall.embedding_registry import registered_profile
from recall.embeddings import EmbeddingProfile, VoyageContextualizedEmbedder
from recall.generation_build import BuildRequest, pipeline_identity
from recall.lineage import IndexManifestV1, ManifestObjectV1


class _ResultGroup:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embeddings = vectors


class _Result:
    def __init__(self, vectors: list[list[list[float]]]) -> None:
        self.results = [_ResultGroup(group) for group in vectors]


class _Client:
    calls: list[dict[str, object]] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def contextualized_embed(self, *, inputs, model, input_type, **kwargs):
        type(self).calls.append(
            {"inputs": inputs, "model": model, "input_type": input_type, "kwargs": kwargs}
        )
        if input_type == "query":
            return _Result([[[0.0, 0.0, 0.0]]])
        vectors = []
        for group in inputs:
            vectors.append([[float(index), float(len(text)), 1.0] for index, text in enumerate(group)])
        return _Result(vectors)


@pytest.fixture
def context_embedder(monkeypatch: pytest.MonkeyPatch) -> VoyageContextualizedEmbedder:
    _Client.calls = []
    module = types.ModuleType("voyageai")
    module.Client = _Client
    monkeypatch.setitem(sys.modules, "voyageai", module)
    return VoyageContextualizedEmbedder(
        api_key="test",
        output_dimension=3,
        max_request_inputs=3,
        max_request_chunks=4,
        max_request_chars=5,
        timeout=7,
    )


def test_context4_uses_nested_groups_and_preserves_order(context_embedder) -> None:
    vectors = context_embedder.embed_document_groups([["aa", "bbb"], ["c", "dddd"]])

    assert [len(group) for group in vectors] == [2, 2]
    assert [vector[1] for group in vectors for vector in group] == [2.0, 3.0, 1.0, 4.0]
    document_calls = [call for call in _Client.calls if call["input_type"] == "document"]
    assert document_calls
    assert all(all(isinstance(group, list) for group in call["inputs"]) for call in document_calls)
    assert [text for call in document_calls for group in call["inputs"] for text in group] == [
        "aa", "bbb", "c", "dddd"
    ]


def test_context4_splits_without_truncating_an_oversized_chunk(context_embedder) -> None:
    text = "x" * 17
    vectors = context_embedder.embed_document_groups([[text]])

    assert len(vectors) == 1
    assert len(vectors[0]) == 1
    document_calls = [call for call in _Client.calls if call["input_type"] == "document"]
    assert document_calls[-1]["inputs"] == [[text]]


def test_context4_passage_cache_is_refused() -> None:
    embedder = object.__new__(VoyageContextualizedEmbedder)
    with pytest.raises(ValueError, match="cannot use the per-text embedding cache"):
        embed_with_cache(embedder, ["chunk"], object(), purpose="passage")  # type: ignore[arg-type]


def test_context4_profile_and_pipeline_record_group_contract() -> None:
    entry = registered_profile("voyage-context-4-v1")
    identity = entry.identity()
    assert isinstance(identity, EmbeddingProfile)
    assert dict(identity.dependencies)["grouping_policy"] == "voyage-context-document-v1"
    fake = types.SimpleNamespace(name=entry.model_name, dim=entry.dimension, profile=identity)
    pipeline = pipeline_identity(fake, BuildRequest())
    assert pipeline.embedder.profile_id == entry.profile_id
    assert pipeline.embedder.profile_fingerprint == identity.fingerprint()
    assert pipeline.to_dict()["embedder"]["profile_fingerprint"] == identity.fingerprint()


def test_context_group_id_is_manifest_identity_and_round_trips(tmp_path) -> None:
    body = b"one"
    path = tmp_path / "one.md"
    path.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    grouped = ManifestObjectV1(path.as_uri(), digest, "text/markdown", len(body), digest, "chat-1")
    plain = ManifestObjectV1(path.as_uri(), digest, "text/markdown", len(body), digest)
    manifest = IndexManifestV1("tenant", "v1", (grouped,))
    assert IndexManifestV1.from_json(manifest.to_json()).objects[0].context_group_id == "chat-1"
    assert manifest.digest != IndexManifestV1("tenant", "v1", (plain,)).digest
