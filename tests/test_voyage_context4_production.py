"""Production invariants for grouped Voyage Context 4 embedding.

These tests exercise the provider boundary with a shaped SDK stub. Mutation evidence for this
feature is recorded in the production handoff report. Flattening the nested request or dropping a
chunk makes the alignment and boundary tests fail before the implementation is restored.
"""
from __future__ import annotations

import hashlib
import logging
import math
import sys
import types
from dataclasses import replace

import pytest

import recall.embedding_registry as embedding_registry
from recall.cache import embed_with_cache
from recall.embedding_registry import registered_profile
from recall.embeddings import (
    VOYAGE_CONTEXT_TOKEN_HEADROOM,
    VOYAGE_CONTEXT_WINDOW_TOKENS,
    EmbeddingProfile,
    VoyageContextualizedEmbedder,
    resolve_embedder,
)
from recall.generation_build import BuildRequest, pipeline_identity
from recall.lineage import IndexManifestV1, ManifestObjectV1, PipelineIdentity


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
    monkeypatch.setattr("recall._voyage_http.Client", module.Client)
    return VoyageContextualizedEmbedder(
        api_key="test",
        output_dimension=3,
        max_request_inputs=3,
        max_request_chunks=4,
        max_request_chars=5,
        token_counter=lambda texts: [1] * len(texts),
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


#: The token budget a part or a request may fill by the local count, at the provider's window.
_BUDGET = int(VOYAGE_CONTEXT_WINDOW_TOKENS * VOYAGE_CONTEXT_TOKEN_HEADROOM)


def _density_counter(chars_per_token: float):
    """A fake tokenizer: every text costs ``ceil(len / chars_per_token)`` tokens."""
    return lambda texts: [math.ceil(len(text) / chars_per_token) for text in texts]


def _production_limits_embedder(
    monkeypatch: pytest.MonkeyPatch, counter
) -> VoyageContextualizedEmbedder:
    """The embedder at the registered profile's real limits, with a fake token counter."""
    _Client.calls = []
    module = types.ModuleType("voyageai")
    module.Client = _Client
    monkeypatch.setitem(sys.modules, "voyageai", module)
    monkeypatch.setattr("recall._voyage_http.Client", module.Client)
    return VoyageContextualizedEmbedder(api_key="test", output_dimension=3, token_counter=counter)


def _document_requests() -> list[list[list[str]]]:
    return [call["inputs"] for call in _Client.calls if call["input_type"] == "document"]


def test_context4_bounds_every_part_by_tokens_not_characters(monkeypatch) -> None:
    """Invariant: no example sent to Voyage holds more tokens than the budget.

    The case is a real one, from a 2026-09-30 build of MMLongBench-Doc PDFs: about 59,600
    characters of one PDF at 1.66 characters per token, which the 60,000-character bound accepted
    as ONE part of about 36,000 tokens, over the 32,000-token window, so Voyage refused it and the
    generation build aborted.

    Red proof, run 2026-09-30: `or tokens + count > budget` deleted from
    `VoyageContextualizedEmbedder._split_group`, i.e. the character-only split. This test then
    failed on the per-example assertion (one part of 35,970 tokens against a budget of 28,800).
    Restored, it passes.
    """
    counter = _density_counter(1.66)
    embedder = _production_limits_embedder(monkeypatch, counter)
    group = [f"{index:04d}" + "9" * 1986 for index in range(30)]
    assert sum(len(text) for text in group) <= 60_000  # the character bound admits it whole
    assert sum(counter(group)) > VOYAGE_CONTEXT_WINDOW_TOKENS  # the provider would refuse it

    vectors = embedder.embed_document_groups([group])

    assert len(vectors[0]) == len(group)
    sent = [example for request in _document_requests() for example in request]
    assert [sum(counter(example)) for example in sent if sum(counter(example)) > _BUDGET] == []
    assert [text for example in sent for text in example] == group  # in order, none truncated
    assert len(sent) > 1


def test_context4_bounds_a_whole_request_by_tokens(monkeypatch) -> None:
    """Invariant: no REQUEST holds more tokens than the budget, even when each part fits.

    For pre-chunked input Voyage caps the whole request at the model's window, not only each
    example, so two documents that each fit may not share a request.

    Red proof, run 2026-09-30: `or tokens + part_tokens > budget` deleted from the request loop
    in `VoyageContextualizedEmbedder.embed_document_groups`. Both 20,000-token documents then
    went in one request (40,000 characters is under the character bound), and this test failed
    on the per-request assertion. Restored, it passes.
    """
    counter = _density_counter(1.0)
    embedder = _production_limits_embedder(monkeypatch, counter)
    groups = [[f"{doc}{index}" + "z" * 1998 for index in range(10)] for doc in "ab"]

    vectors = embedder.embed_document_groups(groups)

    assert [len(group) for group in vectors] == [10, 10]
    totals = [
        sum(sum(counter(example)) for example in request) for request in _document_requests()
    ]
    assert [total for total in totals if total > _BUDGET] == []
    assert len(totals) == 2


def test_context4_token_bound_leaves_character_bounded_splits_unchanged(monkeypatch) -> None:
    """Invariant: a group whose character-bounded parts all fit the token budget splits exactly
    as it did before tokens were counted, so the vectors a corpus already stores under the same
    profile identity stay reproducible.

    Red proof, run 2026-09-30: `VoyageContextualizedEmbedder._token_budget` mutated to take 0.4 of
    the window instead of `VOYAGE_CONTEXT_TOKEN_HEADROOM`, a margin wide enough to re-split
    ordinary prose. This test then failed on the part sizes ([10, 10, 10] against [12, 12, 6]).
    Restored, it passes.
    """
    embedder = _production_limits_embedder(monkeypatch, _density_counter(4.0))
    group = [f"{index:03d}" + "p" * 4997 for index in range(30)]

    embedder.embed_document_groups([group])

    sent = [example for request in _document_requests() for example in request]
    assert [len(example) for example in sent] == [12, 12, 6]  # 60,000 characters per part


def test_context4_sends_a_chunk_over_the_token_budget_alone(monkeypatch, caplog) -> None:
    """Invariant: a single chunk over the budget is never merged with a neighbour and never
    truncated; it is sent on its own with a warning, and the provider accepts or refuses it.

    Red proof, run 2026-09-30: the token condition in `VoyageContextualizedEmbedder._split_group`
    mutated to `(tokens + count > budget and count <= budget)`, i.e. "close a part only for a
    chunk that would fit alone". This test then failed on the part layout, the oversized chunk
    sharing a part with the chunk before it. Restored, it passes.
    """
    embedder = _production_limits_embedder(monkeypatch, _density_counter(1.0))
    huge = "h" * (_BUDGET + 500)
    group = ["before", huge, "after"]

    with caplog.at_level(logging.WARNING, logger="recall.embeddings"):
        vectors = embedder.embed_document_groups([group])

    assert len(vectors[0]) == 3
    sent = [example for request in _document_requests() for example in request]
    assert sent == [["before"], [huge], ["after"]]
    assert any(
        record.getMessage() == "voyage_context_chunk_over_token_budget"
        for record in caplog.records
    )


def test_the_registry_passes_the_declared_token_window(monkeypatch) -> None:
    """Invariant: the profile's declared `request_limit_tokens` reaches the embedder.

    It was declared, folded into the identity, and dropped on the way to the constructor, which
    is how a character budget came to stand in for it.

    Red proof, run 2026-09-30: the `max_request_tokens=` argument removed from the
    `voyage-context` branch of `RegisteredProfile._construct`. This test then failed on the
    recorded value (None against 12,345). Restored, it passes.
    """
    recorded: dict[str, object] = {}

    def build(**kwargs: object) -> object:
        recorded.update(kwargs)
        raise RuntimeError("stop after construction arguments")

    monkeypatch.setattr(embedding_registry, "VoyageContextualizedEmbedder", build)
    entry = replace(registered_profile("voyage-context-4-v1"), request_limit_tokens=12_345)
    with pytest.raises(RuntimeError, match="stop after construction arguments"):
        entry.build(api_key="k")

    assert recorded.get("max_request_tokens") == 12_345


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
    restored = PipelineIdentity.from_dict(pipeline.to_dict())
    assert restored.production_admissible


def test_context4_direct_resolver_uses_registered_profile(monkeypatch: pytest.MonkeyPatch) -> None:
    _Client.calls = []

    class _RegistryClient(_Client):
        def contextualized_embed(self, *, inputs, model, input_type, **kwargs):
            if input_type == "query":
                type(self).calls.append(
                    {"inputs": inputs, "model": model, "input_type": input_type, "kwargs": kwargs}
                )
                return _Result([[[0.0] * 1024]])
            return super().contextualized_embed(
                inputs=inputs, model=model, input_type=input_type, **kwargs
            )

    module = types.ModuleType("voyageai")
    module.Client = _RegistryClient
    monkeypatch.setitem(sys.modules, "voyageai", module)
    monkeypatch.setattr("recall._voyage_http.Client", module.Client)

    embedder = resolve_embedder(
        "voyage-context:voyage-context-4", {"VOYAGE_API_KEY": "test"}
    )

    assert embedder.name == "voyage-context:voyage-context-4"
    assert embedder.profile.profile_id == "voyage-context-4-v1"
    assert dict(embedder.profile.dependencies)["grouping_policy"] == "voyage-context-document-v1"


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
