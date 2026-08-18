"""One registry owns embedding profile identity, and nothing else may hold a second copy.

Before this, two independent dict literals described the same profile vocabulary: a
profile-ID -> context-version map inside `recall_mcp.service.make_embedder`, and a
profile-ID -> context-mode map inside `recall.context.context_policy_for_profile`. They already
disagreed in extent (six entries against three) and nothing asserted they agreed at all. A profile
added to one and not the other indexes under the wrong context mode instead of raising.

These tests pin the properties that make a second copy impossible to reintroduce quietly.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from recall.context import ContextPolicy, context_policy_for_profile
from recall.embedding_registry import (
    REGISTERED_PROFILES,
    RegisteredProfile,
    registered_profile,
    registered_profile_ids,
)
from recall.embeddings import EmbeddingProfile

#: The vocabulary this program deployed. Spelled out rather than derived from the registry: a test
#: that reads its expectation out of the object under test cannot report a missing entry.
EXPECTED_IDS = (
    "bge-small-symmetric-v1",
    "bge-small-asymmetric-v1",
    "bge-small-context-document-v1",
    "bge-small-context-section-v1",
    "bge-small-context-neighbor-v1",
    "bge-base-symmetric-v1",
    "bge-large-symmetric-v1",
    "bge-large-asymmetric-v1",
    "minilm-l6-symmetric-v1",
    "minilm-multilingual-symmetric-v1",
    "arctic-embed-xs-symmetric-v1",
    "qwen3-embedding-0.6b-384-v1",
)


def test_every_deployed_profile_is_registered_exactly_once() -> None:
    assert registered_profile_ids() == EXPECTED_IDS
    assert tuple(REGISTERED_PROFILES) == EXPECTED_IDS
    for profile_id in EXPECTED_IDS:
        assert registered_profile(profile_id).profile_id == profile_id


def test_an_unregistered_profile_id_is_refused_rather_than_defaulted() -> None:
    with pytest.raises(ValueError, match="unknown embedding profile"):
        registered_profile("bge-small-context-paragraph-v1")


def test_context_version_is_derived_from_context_mode_not_declared_beside_it() -> None:
    """The two maps cannot disagree because there is only one value and one derivation.

    `Indexer.__init__` refuses an embedder whose `context_version` is not exactly
    `raw-v1` / `context-<mode>-<policy version>`, so this derivation is the contract the index
    path enforces, restated at the only place a profile is defined.
    """
    for entry in REGISTERED_PROFILES.values():
        policy = context_policy_for_profile(entry.profile_id)
        assert policy.mode == entry.context_mode
        expected = (
            "raw-v1" if entry.context_mode == "none"
            else f"context-{entry.context_mode}-{policy.version}"
        )
        assert entry.context_version == expected


def test_an_unregistered_profile_still_gets_the_no_context_policy() -> None:
    """Legacy embedders (`hashing-64`, a fine-tuned `st:` model) have no registry entry.

    They must keep indexing raw chunk text rather than raising: the registry governs the
    enterprise profiles, and refusing here would break every offline and evaluation path.
    """
    assert context_policy_for_profile("hashing-64") == ContextPolicy(mode="none")


def test_the_registry_carries_the_whole_identity_of_a_profile() -> None:
    entry = registered_profile("bge-small-context-section-v1")
    assert entry.model_name == "BAAI/bge-small-en-v1.5"
    assert entry.dimension == 384
    assert entry.query_mode == "query_embed"
    assert entry.passage_mode == "passage_embed"
    assert entry.normalization == "l2"
    assert entry.instruction_version == "none"
    assert entry.chunker_version == "chunk-text-v1"
    assert entry.context_version == "context-section-v1"
    assert entry.backend == "fastembed"
    # Operator-provisioned: the digest is a property of the deployment's artifact tree, not of
    # the profile, so the registry declares its absence instead of inventing a value.
    assert entry.artifact_digest is None


def test_a_registered_profile_becomes_a_runtime_identity_without_restating_it() -> None:
    """`RegisteredProfile.identity()` is the only way a runtime `EmbeddingProfile` is built.

    Two constructors would be two identities, which is the defect this whole registry removes.
    """
    entry = registered_profile("bge-small-asymmetric-v1")
    identity = entry.identity(artifact_digest="a" * 64, dependencies=(("fastembed", "0.8.0"),))

    assert isinstance(identity, EmbeddingProfile)
    assert identity.profile_id == "bge-small-asymmetric-v1"
    assert identity.model_name == entry.model_name
    assert identity.artifact_digest == "a" * 64
    assert identity.dimension == 384
    assert identity.query_mode == "query_embed"
    assert identity.passage_mode == "passage_embed"
    assert identity.context_version == "raw-v1"
    assert identity.normalization == "l2"
    assert identity.instruction_version == "none"
    assert identity.chunker_version == "chunk-text-v1"
    assert identity.dependencies == (("fastembed", "0.8.0"),)


def test_an_identity_cannot_be_built_without_a_verified_digest() -> None:
    entry = registered_profile("bge-small-symmetric-v1")
    with pytest.raises(ValueError, match="artifact digest"):
        entry.identity(artifact_digest=None)


def test_a_pinned_digest_refuses_a_different_operator_supplied_one() -> None:
    """The Qwen entry pins the artifact it was measured on. A different tree is a different
    experiment, and must not inherit the recorded verdict."""
    entry = registered_profile("qwen3-embedding-0.6b-384-v1")
    assert entry.artifact_digest is not None
    with pytest.raises(ValueError, match="pinned artifact digest"):
        entry.identity(artifact_digest="b" * 64)
    assert entry.identity(artifact_digest=entry.artifact_digest).artifact_digest == (
        entry.artifact_digest
    )


def test_registered_profiles_are_immutable() -> None:
    entry = registered_profile("bge-small-symmetric-v1")
    with pytest.raises(Exception):
        entry.dimension = 512  # type: ignore[misc]


def test_only_the_qwen_profile_is_marked_rejected() -> None:
    rejected = [e.profile_id for e in REGISTERED_PROFILES.values() if e.rejection is not None]
    assert rejected == ["qwen3-embedding-0.6b-384-v1"]
    for entry in REGISTERED_PROFILES.values():
        assert entry.rejected is (entry.rejection is not None)


def test_the_qwen_rejection_record_carries_the_measurement_that_decided_it() -> None:
    """The negative result, preserved where the code can be asked for it.

    Measured on VPS2 at a four-thread budget against the provisioned artifact; the profile was
    rejected on CPU latency, not on quality, and the numbers are the record of that.
    """
    record = registered_profile("qwen3-embedding-0.6b-384-v1").rejection
    assert record is not None
    assert record.verdict == "rejected"
    assert record.reason == "cpu-latency"
    assert record.revision == "97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3"
    assert record.artifact_digest == (
        "0e9f06588b7e661b8d8e6d393b5936750e428ec422f9971c7f02838dbe70fc9f"
    )
    assert record.reference_cpu_threads == 4
    assert dict(record.measurements) == {
        "query_p50_ms": 4638.83,
        "query_p95_ms": 5816.34,
        "passage_batch_20_p50_ms": 41016.64,
        "load_ms": 24558.4,
        "max_rss_mb": 1739.47,
    }


def test_a_profile_added_to_the_registry_needs_no_second_edit_anywhere(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The property the two dict literals could not have.

    A seventh profile used to require editing `make_embedder`'s context-version map AND
    `context_policy_for_profile`'s context-mode map. Adding it to one and not the other did not
    raise; it indexed under the wrong context mode. Here the registry gains an entry and nothing
    else is touched: both the constructor and the context policy must already know about it.
    """
    from recall_mcp.service import make_embedder

    added = RegisteredProfile(
        profile_id="bge-small-context-paragraph-v1",
        model_name="BAAI/bge-small-en-v1.5",
        dimension=384,
        query_mode="query_embed",
        passage_mode="passage_embed",
        context_mode="section",
        backend="fastembed",
    )
    monkeypatch.setattr(
        "recall.embedding_registry.REGISTERED_PROFILES",
        {**REGISTERED_PROFILES, added.profile_id: added},
    )

    assert context_policy_for_profile(added.profile_id).mode == "section"
    # Reaches the artifact check, which means the vocabulary accepted it. A profile the
    # constructor did not know would have been refused before any path was resolved.
    with pytest.raises(FileNotFoundError):
        make_embedder(
            "fastembed",
            {
                "RECALL_EMBED_PROFILE": added.profile_id,
                "RECALL_MODEL_CACHE": "/definitely/not/provisioned",
                "RECALL_MODEL_SHA256": "0" * 64,
            },
        )


def test_a_rejection_record_must_name_what_was_measured() -> None:
    entry = registered_profile("qwen3-embedding-0.6b-384-v1")
    with pytest.raises(ValueError, match="measurement"):
        RegisteredProfile(
            profile_id="x-v1",
            model_name="x",
            dimension=8,
            query_mode="embed",
            passage_mode="embed",
            context_mode="none",
            backend="fastembed",
            rejection=type(entry.rejection)(  # type: ignore[misc]
                verdict="rejected",
                reason="cpu-latency",
                decided_on="2026-08-03",
                revision="0" * 40,
                artifact_digest="0" * 64,
                reference_cpu_threads=4,
                measurements=(),
            ),
        )


ENTERPRISE_DOC = Path(__file__).resolve().parent.parent / "docs" / "ENTERPRISE_RETRIEVAL.md"


def test_the_qwen_rejection_is_published_where_an_operator_will_find_it() -> None:
    """The record and the document must not be able to drift apart.

    A verdict that lives only in a Python constant is not published, and a verdict that lives
    only in prose is not checkable. This asserts the document carries every number the registry
    carries, so deleting one from either side fails here.
    """
    text = ENTERPRISE_DOC.read_text(encoding="utf-8")
    entry = registered_profile("qwen3-embedding-0.6b-384-v1")
    record = entry.rejection
    assert record is not None

    assert entry.profile_id in text
    assert record.revision in text
    assert record.artifact_digest in text
    assert str(record.reference_cpu_threads) in text
    for name, value in record.measurements:
        assert str(value) in text, f"{name} is recorded in the registry but not in the document"


def test_the_document_says_the_profile_is_rejected_not_pending() -> None:
    text = ENTERPRISE_DOC.read_text(encoding="utf-8").lower()
    assert "rejected" in text
    assert "promotion gated" not in text, (
        "'promotion gated' implies a gate that could still open; the verdict is final"
    )


def test_the_index_path_agrees_with_the_registry_for_every_profile() -> None:
    """A third copy of the context-version derivation lives in `Indexer.__init__`.

    It rejects an embedder whose profile does not spell its context exactly the way the index
    path expects, so a registry that derived the string differently would make every context
    profile unindexable. Constructing an Indexer for each registered identity is the only way to
    assert the two agree; reading the registry's own derivation twice would not.
    """
    from recall.index import Indexer

    class _Embedder:
        def __init__(self, profile: EmbeddingProfile) -> None:
            self.profile = profile
            self.dim = profile.dimension
            self.name = profile.profile_id

        def embed(self, texts: list[str]) -> list[list[float]]:
            return [[0.0] * self.dim for _ in texts]

    for entry in REGISTERED_PROFILES.values():
        # Three cases, not two. A pinned profile accepts only its own digest; a provisioned one
        # takes an operator-supplied digest; a HOSTED one refuses a digest outright, because
        # supplying one would assert a verification nobody performed.
        # A pinned profile accepts only its own digest; every other one is operator supplied.
        identity = entry.identity(artifact_digest=entry.artifact_digest or "a" * 64)
        Indexer(
            object(),  # type: ignore[arg-type]  # the context check runs before any store use
            _Embedder(identity),  # type: ignore[arg-type]
            context_policy=context_policy_for_profile(entry.profile_id),
        )
