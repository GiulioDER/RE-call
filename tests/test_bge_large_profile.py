"""`bge-large-symmetric-v1` is registered, so a 1024-dim corpus can be served at all.

Before this profile existed the MCP server could not serve a bge-large corpus: `make_embedder`
accepts only `hashing` and `fastembed`, and `fastembed` alone builds bge-small. Measured against
a live 8,671-chunk bge-large corpus on VPS2, `RECALL_EMBEDDER=fastembed:BAAI/bge-large-en-v1.5`
raised `unknown embedder` before any search ran, so every one of the ten tools was unreachable.

⚠️ The obvious fix — letting `make_embedder` fall through to `resolve_embedder` — was tried and
REJECTED on audit. `resolve_embedder` builds fastembed embedders with no identity, and
`recall/embeddings.py:794-801` then hardcodes `profile_id="bge-small-symmetric-v1"` for EVERY
fastembed model. Measured: `fastembed`, `fastembed:BAAI/bge-large-en-v1.5` and
`fastembed:intfloat/e5-small-v2` all report that one registered id. Two different 384-dim models
would then be indistinguishable to every profile-id-keyed check in the serving path — readiness,
lineage and calibration all compare that string — so a server could search one model's vectors
with another model's queries and nothing would fire. The registry path avoids this by
construction: `RegisteredProfile.build` passes a real `identity`, so the id is the profile's own.

These tests use the `stub_fastembed` pattern rather than a real model, because CI installs only
the `dev` extra and a test that needs `fastembed` errors there rather than skipping.
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest

from recall.context import ContextPolicy, context_policy_for_profile
from recall.embedding_registry import (
    find_registered_profile,
    registered_profile,
    registered_profile_ids,
)
from recall.embeddings import artifact_tree_sha256

PROFILE_ID = "bge-large-symmetric-v1"
CONTEXT_PROFILE_ID = "bge-large-context-section-v1"


class _StubTextEmbedding:
    instances: list[dict] = []

    def __init__(self, **kwargs):
        type(self).instances.append(kwargs)

    def embed(self, texts, **_):
        return [[0.0] * 1024 for _ in texts]

    query_embed = embed
    passage_embed = embed


@pytest.fixture
def stub_fastembed(monkeypatch: pytest.MonkeyPatch):
    _StubTextEmbedding.instances = []
    module = types.ModuleType("fastembed")
    module.TextEmbedding = _StubTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    return _StubTextEmbedding


@pytest.fixture
def provisioned(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "models" / "bge-large"
    root.mkdir(parents=True)
    (root / "model.onnx").write_bytes(b"weights")
    (root / "tokenizer.json").write_bytes(b"{}")
    return root, artifact_tree_sha256(root)


def test_bge_large_is_registered_at_1024_dimensions() -> None:
    entry = registered_profile(PROFILE_ID)
    assert entry.model_name == "BAAI/bge-large-en-v1.5"
    assert entry.dimension == 1024
    assert entry.backend == "fastembed"
    assert PROFILE_ID in registered_profile_ids()


def test_bge_large_indexes_under_the_same_context_policy_as_a_default_index() -> None:
    """A corpus built with the default `ContextPolicy()` must not be re-contextualised at query
    time. The symmetric bge-small profile is `none` for the same reason; a mismatch here would
    mean query vectors and stored vectors were produced from differently-assembled text."""
    assert context_policy_for_profile(PROFILE_ID).mode == ContextPolicy().mode == "none"


def test_bge_large_context_profile_uses_section_policy() -> None:
    assert context_policy_for_profile(CONTEXT_PROFILE_ID).mode == "section"
    entry = registered_profile(CONTEXT_PROFILE_ID)
    assert entry.dimension == 1024
    assert entry.context_version == "context-section-v1"


def test_bge_large_has_its_own_identity_not_bge_small_s(stub_fastembed, provisioned) -> None:
    """The whole point of registering rather than widening the resolver.

    `embedding_profile_id` must return this profile's own id, not the hardcoded
    `bge-small-symmetric-v1` that every resolver-built fastembed embedder reports. Without a
    distinct id, readiness, lineage and calibration cannot tell the two models apart.
    """
    from recall.embeddings import embedding_profile_id

    root, digest = provisioned
    embedder = registered_profile(PROFILE_ID).build(artifact_path=root, artifact_digest=digest)

    assert embedding_profile_id(embedder) == PROFILE_ID
    assert embedding_profile_id(embedder) != "bge-small-symmetric-v1"
    assert embedder.dim == 1024
    # The model actually loaded is this profile's, taken from the identity rather than a default.
    assert _StubTextEmbedding.instances[-1]["model_name"] == "BAAI/bge-large-en-v1.5"


def test_every_registered_profile_has_a_distinct_fingerprint() -> None:
    """A fingerprint is the key a calibration binds to, and `load_for_profile` refuses on exactly
    this comparison. Built from the identities rather than by loading models: the fingerprint is a
    property of the declared identity, so no weights are needed to compare them.

    Asserted over the WHOLE registry, not just the pair this change adds — a collision anywhere
    would let one profile's calibration load for another."""
    digest = "a" * 64
    seen: dict[str, str] = {}
    for pid in registered_profile_ids():
        entry = registered_profile(pid)
        # A HOSTED profile refuses an operator digest outright (there is no artifact tree to have
        # hashed), so it completes its identity without one. It is included rather than skipped:
        # a hosted profile's fingerprint keys a calibration exactly as a local one's does, and
        # every hosted profile shares the same digest marker, so if anything were going to
        # collide here it would be these.
        fp = (
            entry.identity()
            if entry.hosted
            else entry.identity(artifact_digest=entry.artifact_digest or digest)
        ).fingerprint()
        assert fp not in seen, f"{pid} collides with {seen.get(fp)}"
        seen[fp] = pid
    assert len(seen) == len(registered_profile_ids())


def test_bge_large_weights_are_provisioned_not_pinned() -> None:
    """No `artifact_digest` is baked in: the weights are provisioned per deployment, so the
    identity cannot be completed without an operator-supplied digest. Asserting the refusal keeps
    a future pin from being added silently, which would make every existing index's fingerprint
    disagree with the registry's."""
    entry = find_registered_profile(PROFILE_ID)
    assert entry is not None and entry.artifact_digest is None
    with pytest.raises(ValueError, match="operator-supplied artifact digest"):
        entry.identity()
