"""A cached vector may only be served back to the identity that produced it.

`cache_key` used to hash `(profile_id, purpose, dim, text)`. Two of the fields that change the
stored vector were missing: `artifact_digest` and `context_version`. Both are independently
settable while `profile_id` stays fixed: `FastEmbedEmbedder.__init__` takes all three as separate
parameters, so two processes configured with the same `RECALL_EMBED_PROFILE` and a re-provisioned
artifact shared cache entries computed by different weights, and a context-mode change reused
vectors embedded from different text.

The failure is silent by construction: a cache hit returns a plausible vector of the right width.
Nothing downstream can tell it apart from a fresh one.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from recall.cache import EmbeddingCache, cache_key, embed_with_cache, embed_query_with_cache
from recall.embedding_registry import registered_profile
from recall.embeddings import EmbeddingProfile, legacy_embedding_profile

#: A profile whose every identity field is spelled out, so the fingerprint below pins all of them.
_PINNED = EmbeddingProfile(
    profile_id="bge-small-asymmetric-v1",
    model_name="BAAI/bge-small-en-v1.5",
    artifact_digest="9a443d711e063427f62cf559a38863122ee5ed107fdd7920de882fd66dbc919c",
    dimension=384,
    query_mode="query_embed",
    passage_mode="passage_embed",
    normalization="l2",
    instruction_version="none",
    chunker_version="chunk-text-v1",
    context_version="raw-v1",
    dependencies=(("fastembed", "0.8.0"),),
)

#: Computed from the encoding documented on `EmbeddingProfile.fingerprint` by an independent
#: transcription of that rule, NOT by calling the method. A test that reads its expectation out of
#: the object under test cannot fail when the object changes.
_PINNED_FINGERPRINT = "1bd9a91f09cc346c52cfa825f8a1dd3cac70ccf6c024f73de69a335106fa2bc7"


class _Recorder:
    """Embedder that reports which encoder it was asked for, under a settable identity."""

    def __init__(self, profile: EmbeddingProfile) -> None:
        self.profile = profile
        self.calls: list[tuple[str, tuple[str, ...]]] = []

    @property
    def dim(self) -> int:
        return self.profile.dimension

    @property
    def name(self) -> str:
        return self.profile.profile_id

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("legacy", tuple(texts)))
        return [[0.0] * self.dim for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls.append(("query", (text,)))
        return [1.0] + [0.0] * (self.dim - 1)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("passage", tuple(texts)))
        return [[0.0, 1.0] + [0.0] * (self.dim - 2) for _ in texts]


def _profile(**overrides: object) -> EmbeddingProfile:
    fields = {
        "profile_id": "p-v1",
        "model_name": "m",
        "artifact_digest": "a" * 64,
        "dimension": 4,
        "query_mode": "query_embed",
        "passage_mode": "passage_embed",
        "context_version": "raw-v1",
    }
    fields.update(overrides)
    return EmbeddingProfile(**fields)  # type: ignore[arg-type]


def test_profile_fingerprint_is_stable() -> None:
    """The fingerprint is durable key material: cached vectors outlive the process that wrote
    them, so a change to the encoding silently re-partitions every cache in existence."""
    assert _PINNED.fingerprint() == _PINNED_FINGERPRINT


def test_fingerprint_changes_when_any_identity_field_changes() -> None:
    """Every field, one at a time. A fingerprint that ignores one is a cache that aliases on it."""
    base = _profile()
    variants = {
        "profile_id": _profile(profile_id="p-v2"),
        "model_name": _profile(model_name="m2"),
        "artifact_digest": _profile(artifact_digest="b" * 64),
        "dimension": _profile(dimension=8),
        "query_mode": _profile(query_mode="embed"),
        "passage_mode": _profile(passage_mode="embed"),
        "normalization": _profile(normalization="none"),
        "instruction_version": _profile(instruction_version="retrieval-v1"),
        "chunker_version": _profile(chunker_version="chunk-text-v2"),
        "context_version": _profile(context_version="context-section-v1"),
        "dependencies": _profile(dependencies=(("fastembed", "0.8.1"),)),
    }
    assert set(variants) == {f.name for f in EmbeddingProfile.__dataclass_fields__.values()}
    for changed, variant in variants.items():
        assert variant.fingerprint() != base.fingerprint(), f"{changed} is not key material"


def test_query_passage_and_legacy_keys_are_three_distinct_spaces() -> None:
    profile = _profile()
    keys = {
        purpose: cache_key(profile, 4, "same text", purpose)  # type: ignore[arg-type]
        for purpose in ("query", "passage", "legacy")
    }
    assert len(set(keys.values())) == 3


def test_a_context_version_change_invalidates_the_cache(tmp_path: Path) -> None:
    """Same profile ID, different context version: the text handed to the encoder differs, so
    the vector must not be reused."""
    raw = _Recorder(_profile(context_version="raw-v1"))
    contextual = _Recorder(_profile(context_version="context-section-v1"))
    with EmbeddingCache(tmp_path / "vectors.sqlite") as cache:
        embed_with_cache(raw, ["chunk"], cache, purpose="passage")
        embed_with_cache(contextual, ["chunk"], cache, purpose="passage")
    assert contextual.calls == [("passage", ("chunk",))], (
        "the second profile read the first profile's vectors"
    )


def test_a_re_provisioned_artifact_invalidates_the_cache(tmp_path: Path) -> None:
    """Same profile ID, different weights. This is the shape a shadow generation and a
    re-provisioned model both take, and the one the old key could not see."""
    first = _Recorder(_profile(artifact_digest="a" * 64))
    second = _Recorder(_profile(artifact_digest="b" * 64))
    with EmbeddingCache(tmp_path / "vectors.sqlite") as cache:
        embed_with_cache(first, ["chunk"], cache, purpose="passage")
        embed_with_cache(second, ["chunk"], cache, purpose="passage")
    assert second.calls == [("passage", ("chunk",))]


def test_cross_profile_reuse_fails_closed(tmp_path: Path) -> None:
    """A miss costs one embed. A hit across identities costs a wrong vector, forever."""
    symmetric = _Recorder(
        registered_profile("bge-small-symmetric-v1").identity(artifact_digest="a" * 64)
    )
    asymmetric = _Recorder(
        registered_profile("bge-small-asymmetric-v1").identity(artifact_digest="a" * 64)
    )
    with EmbeddingCache(tmp_path / "vectors.sqlite") as cache:
        embed_with_cache(symmetric, ["chunk"], cache, purpose="passage")
        embed_with_cache(asymmetric, ["chunk"], cache, purpose="passage")
    assert asymmetric.calls == [("passage", ("chunk",))]


def test_a_query_never_reads_a_passage_vector(tmp_path: Path) -> None:
    embedder = _Recorder(_profile())
    with EmbeddingCache(tmp_path / "vectors.sqlite") as cache:
        passage = embed_with_cache(embedder, ["same"], cache, purpose="passage")[0]
        query = embed_query_with_cache(embedder, "same", cache)
    assert query != passage
    assert embedder.calls == [("passage", ("same",)), ("query", ("same",))]


def test_a_legacy_embedder_still_caches_under_its_own_identity(tmp_path: Path) -> None:
    """An embedder with no `profile` attribute keeps working, keyed by its legacy descriptor."""

    class LegacyOnly:
        dim = 2
        name = "legacy-2"

        def __init__(self) -> None:
            self.embedded: list[str] = []

        def embed(self, texts: list[str]) -> list[list[float]]:
            self.embedded.extend(texts)
            return [[float(len(t)), 1.0] for t in texts]

    embedder = LegacyOnly()
    with EmbeddingCache(tmp_path / "vectors.sqlite") as cache:
        first = embed_with_cache(embedder, ["alpha", "beta"], cache)
        second = embed_with_cache(embedder, ["alpha", "beta"], cache)
    assert embedder.embedded == ["alpha", "beta"]
    assert second == first
    assert legacy_embedding_profile(embedder).artifact_digest == "legacy-unverified"


def test_two_legacy_embedders_with_different_names_do_not_share_entries() -> None:
    class _Legacy:
        def __init__(self, name: str) -> None:
            self.name = name
            self.dim = 2

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0, 0.0] for _ in texts]

    a = legacy_embedding_profile(_Legacy("hashing-64"))
    b = legacy_embedding_profile(_Legacy("st:benchmarks/finetune/model"))
    assert cache_key(a, 2, "t") != cache_key(b, 2, "t")


def test_cache_key_refuses_a_bare_string_identity() -> None:
    """The old signature took the profile ID as a string, which is exactly the identity that is
    NOT sufficient. Passing one must not silently produce a key."""
    with pytest.raises((AttributeError, TypeError)):
        cache_key("bge-small-symmetric-v1", 384, "t")  # type: ignore[arg-type]
