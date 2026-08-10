"""Which encoder runs is decided by the profile, and the profile is the registry's.

An asymmetric model has two encoders, and using the wrong one is invisible: the vector has the
right width, the cosine is in range, and retrieval merely gets quietly worse. So the tests here
assert the METHOD that was called, not the shape of what came back.

`fastembed` is an optional extra and CI's `test` job installs without it, deliberately, so a test
that needed the real package would skip in exactly the environment that runs it. These use a stub
`fastembed` module: the subject is our dispatch, our offline preconditions and our identity
plumbing, all of which are ours to get wrong. What the stub cannot prove, that the real loader
honours `local_files_only`, is measured separately against the provisioned artifact on VPS2 and
recorded in `docs/archive/ENTERPRISE_PROGRAM_STATUS.md`.
"""
from __future__ import annotations

import sys
import types

import pytest

from recall.embedding_registry import registered_profile
from recall.embeddings import (
    Embedder,
    EmbeddingProfile,
    embed_passages,
    embed_query,
    embedding_profile,
    embedding_profile_id,
)
from recall.timing import TimedEmbedder

DIGEST = "9a443d711e063427f62cf559a38863122ee5ed107fdd7920de882fd66dbc919c"


class LegacyOnly:
    """The whole `Embedder` protocol and nothing else: no `embed_query`, no `profile`."""

    dim = 3
    name = "legacy-only"

    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(tuple(texts))
        return [[1.0, 0.0, 0.0] for _ in texts]


class Asymmetric:
    dim = 2
    name = "asymmetric"
    profile = EmbeddingProfile("fake-v1", "fake", "a" * 64, 2, "query_embed", "passage_embed")

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append("legacy")
        return [[0.0, 0.0] for _ in texts]

    def embed_query(self, text: str) -> list[float]:
        self.calls.append("query")
        return [1.0, 0.0]

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        self.calls.append("passage")
        return [[0.0, 1.0] for _ in texts]


# --------------------------------------------------------------------------------------------
# Legacy embedders keep working


def test_a_legacy_embed_only_embedder_still_satisfies_the_protocol() -> None:
    embedder = LegacyOnly()
    assert isinstance(embedder, Embedder)


def test_query_and_passage_helpers_fall_back_to_embed_for_a_legacy_embedder() -> None:
    embedder = LegacyOnly()

    assert embed_query(embedder, "q") == [1.0, 0.0, 0.0]
    assert embed_passages(embedder, ["p1", "p2"]) == [[1.0, 0.0, 0.0]] * 2

    assert embedder.calls == [("q",), ("p1", "p2")]


def test_a_legacy_embedder_gets_a_legacy_descriptor_not_a_registry_entry() -> None:
    profile = embedding_profile(LegacyOnly())
    assert profile.profile_id == "legacy-only"
    assert profile.artifact_digest == "legacy-unverified"
    assert profile.query_mode == profile.passage_mode == "legacy"


# --------------------------------------------------------------------------------------------
# Encoder selection


def test_query_and_passage_select_their_own_encoders() -> None:
    embedder = Asymmetric()

    assert embed_query(embedder, "q") == [1.0, 0.0]
    assert embed_passages(embedder, ["p"]) == [[0.0, 1.0]]

    assert embedder.calls == ["query", "passage"]
    assert embedding_profile_id(embedder) == "fake-v1"


def test_timed_embeddings_route_asymmetrically_and_are_recorded() -> None:
    inner = Asymmetric()
    timed = TimedEmbedder(inner)

    assert timed.embed_query("q") == [1.0, 0.0]
    assert timed.embed_passages(["p"]) == [[0.0, 1.0]]
    assert timed.embed(["l"]) == [[0.0, 0.0]]

    assert inner.calls == ["query", "passage", "legacy"]
    assert timed.stats.calls == 3
    assert timed.stats.total_ms >= 0.0


def test_a_timing_wrapper_does_not_erase_the_wrapped_identity() -> None:
    """A wrapper that dropped the profile would cache under `legacy-unverified` and alias
    every timed run of every profile into one key space."""
    timed = TimedEmbedder(Asymmetric())
    assert embedding_profile(timed).profile_id == "fake-v1"
    assert embedding_profile(timed).artifact_digest == "a" * 64


def test_a_timing_wrapper_over_a_legacy_embedder_still_falls_back() -> None:
    inner = LegacyOnly()
    timed = TimedEmbedder(inner)
    assert timed.embed_query("q") == [1.0, 0.0, 0.0]
    assert inner.calls == [("q",)]


# --------------------------------------------------------------------------------------------
# FastEmbed dispatches on the modes the registry declared


class _StubTextEmbedding:
    """Records construction kwargs and which encoder each call used."""

    instances: list["_StubTextEmbedding"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.calls: list[tuple[str, tuple[str, ...]]] = []
        _StubTextEmbedding.instances.append(self)

    def _vectors(self, tag: float, texts: list[str]) -> list[list[float]]:
        return [[tag] + [0.0] * 383 for _ in texts]

    def embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("embed", tuple(texts)))
        return self._vectors(1.0, texts)

    def query_embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("query_embed", tuple(texts)))
        return self._vectors(2.0, texts)

    def passage_embed(self, texts: list[str]) -> list[list[float]]:
        self.calls.append(("passage_embed", tuple(texts)))
        return self._vectors(3.0, texts)


@pytest.fixture
def stub_fastembed(monkeypatch: pytest.MonkeyPatch) -> type[_StubTextEmbedding]:
    _StubTextEmbedding.instances = []
    module = types.ModuleType("fastembed")
    module.TextEmbedding = _StubTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    return _StubTextEmbedding


@pytest.fixture
def provisioned(tmp_path):
    """A local artifact tree and its true digest, as an operator would provision one."""
    from recall.embeddings import artifact_tree_sha256

    root = tmp_path / "bge-cache"
    root.mkdir()
    (root / "model.onnx").write_bytes(b"weights")
    return root, artifact_tree_sha256(root)


def test_fastembed_uses_query_embed_and_passage_embed_for_an_asymmetric_profile(
    stub_fastembed, provisioned
) -> None:
    root, digest = provisioned
    entry = registered_profile("bge-small-asymmetric-v1")

    embedder = entry.build(artifact_path=root, artifact_digest=digest)
    model = stub_fastembed.instances[-1]

    embedder.embed_query("q")
    embedder.embed_passages(["p"])

    assert [call[0] for call in model.calls] == [
        "passage_embed",  # dimension discovery uses PASSAGE encoding
        "query_embed",
        "passage_embed",
    ]


def test_fastembed_uses_the_single_encoder_for_the_symmetric_profile(
    stub_fastembed, provisioned
) -> None:
    root, digest = provisioned
    entry = registered_profile("bge-small-symmetric-v1")

    embedder = entry.build(artifact_path=root, artifact_digest=digest)
    model = stub_fastembed.instances[-1]

    embedder.embed_query("q")
    embedder.embed_passages(["p"])

    assert [call[0] for call in model.calls] == ["embed", "embed", "embed"]


def test_a_built_embedder_carries_the_registry_identity_verbatim(
    stub_fastembed, provisioned
) -> None:
    root, digest = provisioned
    entry = registered_profile("bge-small-context-neighbor-v1")

    profile = embedding_profile(entry.build(artifact_path=root, artifact_digest=digest))

    assert profile.profile_id == "bge-small-context-neighbor-v1"
    assert profile.artifact_digest == digest
    assert profile.context_version == "context-neighbor-v1"
    assert profile.query_mode == "query_embed"
    assert profile.passage_mode == "passage_embed"

    # DELIBERATE CHANGE, not a relaxation. The registry declares WHAT the profile is; it cannot
    # declare the ONNX execution provider, which is resolved when the session is constructed and
    # is only knowable at build time. Since the provider moves vectors — measured: CPU vs CUDA
    # changed top-45 membership on 2 of 64 queries — it is identity, so `build` must add it. What
    # "verbatim" protects is every field the registry DOES declare, all asserted above and all
    # unchanged. This assertion is pinned to the exact key set rather than widened to a superset,
    # so a future entry sneaking into a registered profile's dependencies still fails here.
    deps = dict(profile.dependencies)
    assert deps.keys() == {"fastembed", "onnx-providers-source", "onnx-providers"}
    assert deps["fastembed"], "the registry's own dependency entry must survive the wrap"
    # Whichever way the stub resolves, the two provider fields must AGREE about whether the
    # session was readable — a source of "session" alongside a value of "unavailable" (or the
    # reverse) would mean the pair can report a provenance it did not observe.
    assert (deps["onnx-providers-source"] == "unavailable") == (
        deps["onnx-providers"] == "unavailable"
    )


def test_a_probed_dimension_that_contradicts_the_registry_refuses_to_start(
    stub_fastembed, provisioned, monkeypatch
) -> None:
    """The registry declares 384. An artifact that probes to anything else is not this profile,
    and starting anyway would write vectors no other process can interpret."""
    root, digest = provisioned

    def _short(self, texts):
        self.calls.append(("passage_embed", tuple(texts)))
        return [[0.0, 1.0] for _ in texts]

    monkeypatch.setattr(_StubTextEmbedding, "passage_embed", _short)
    entry = registered_profile("bge-small-asymmetric-v1")

    with pytest.raises(ValueError, match="dimension"):
        entry.build(artifact_path=root, artifact_digest=digest)


def test_a_backend_without_the_declared_query_encoder_refuses_at_startup(
    stub_fastembed, provisioned, monkeypatch
) -> None:
    """Both encoders are resolved when the embedder is built, not when they are first used.

    Resolving the query encoder lazily would let a deployment index a whole corpus and then fail
    on its first query, which is the worst possible time to discover it. Refusing at construction
    is also what `docs/ENTERPRISE_RETRIEVAL.md` claims, and a claim about when a guard fires is
    worth executing.
    """
    root, digest = provisioned
    monkeypatch.delattr(_StubTextEmbedding, "query_embed")
    entry = registered_profile("bge-small-asymmetric-v1")

    with pytest.raises(ValueError, match="no encoder named 'query_embed'"):
        entry.build(artifact_path=root, artifact_digest=digest)
