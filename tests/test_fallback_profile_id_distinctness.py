"""An unregistered embedder must not claim a registered profile's identity.

`profile_id` is a CLAIM about which model wrote a vector. The no-identity path in
`FastEmbedEmbedder.__init__` used to mint one of two literals unconditionally, so ANY model built
without a registered identity reported `bge-small-symmetric-v1` (or its asymmetric twin).
Measured 2026-08-18: a `fastembed:BAAI/bge-large-en-v1.5` embedder reported ``dim=1024`` under
``profile_id='bge-small-symmetric-v1'``, whose registry entry is 384-dimensional, and a production
corpus of 8,716 chunks had stored that pairing in its chunk metadata.

Two consequences, and only the second was ever load-bearing:

* The embedding CACHE was never confused. `EmbeddingProfile.fingerprint` covers `model_name` and
  `dimension`, so the two models keyed apart despite sharing an id. `recall/cache.py` is sound.
* `recall.index._index_fingerprint` was. It hashes `embedding_profile_id` alone, with no dimension
  term, so a bge-small corpus and a bge-large corpus produced the SAME fingerprint for the same
  file and the incremental skip guard treated a model swap as a no-op. That is the failure these
  tests pin, which is why the fingerprint case is here and not only the id case.

The no-migration half is pinned too: the legacy literal must SURVIVE for the model it actually
names, because it is the key every shipped calibration file, every `results/` promotion decision
and every corpus indexed by a bare `FastEmbedEmbedder()` is already written under.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from recall.context import ContextPolicy
from recall.embeddings import (
    EmbeddingProfile,
    FastEmbedEmbedder,
    _fallback_profile_id,
    embedding_profile_id,
)
from recall.index import _index_fingerprint
from tests.conftest import requires_fastembed

BGE_SMALL = "BAAI/bge-small-en-v1.5"
BGE_LARGE = "BAAI/bge-large-en-v1.5"


class _Embedder:
    """An embedder that is nothing but its profile.

    Deliberately model-free. The subject is the identity comparison, and loading real weights
    would mean the assertion is skipped wherever the extra is absent, which is exactly the CI job
    where a regression would land unnoticed.
    """

    def __init__(self, profile: EmbeddingProfile) -> None:
        self.profile = profile

    @property
    def dim(self) -> int:
        return self.profile.dimension

    @property
    def name(self) -> str:
        return self.profile.model_name

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.0] * self.dim for _ in texts]


def _profile(model_name: str, dimension: int, asymmetric: bool = False) -> EmbeddingProfile:
    return EmbeddingProfile(
        profile_id=_fallback_profile_id(model_name, dimension, asymmetric),
        model_name=model_name,
        artifact_digest="legacy-unverified",
        dimension=dimension,
        query_mode="query_embed" if asymmetric else "embed",
        passage_mode="passage_embed" if asymmetric else "embed",
    )


def test_bge_large_does_not_report_a_bge_small_profile_id() -> None:
    """The exact defect, at the width it was measured at."""
    resolved = _fallback_profile_id(BGE_LARGE, 1024, asymmetric=False)
    assert resolved != "bge-small-symmetric-v1"
    assert "bge-small" not in resolved
    assert "bge-large-en-v1.5" in resolved and "1024" in resolved


def test_bge_large_does_not_report_a_bge_small_asymmetric_profile_id() -> None:
    """The asymmetric twin was minted by the same unconditional expression."""
    resolved = _fallback_profile_id(BGE_LARGE, 1024, asymmetric=True)
    assert resolved != "bge-small-asymmetric-v1"
    assert "bge-small" not in resolved


@pytest.mark.parametrize(
    ("first", "second"),
    [
        # Different model, different width: the measured case.
        ((BGE_LARGE, 1024, False), (BGE_SMALL, 384, False)),
        # Different model at the SAME width. No term outside the id separates these for
        # `_index_fingerprint`, which carries no dimension of its own.
        (("vendor/model-a", 768, False), ("vendor/model-b", 768, False)),
        # Same model and width, different encoder modes. This is the distinction the legacy
        # symmetric/asymmetric pair encoded, and dropping it would re-introduce a narrower
        # version of the same collision.
        ((BGE_LARGE, 1024, False), (BGE_LARGE, 1024, True)),
        # Same model, different width: a width change is a different embedder.
        (("vendor/model-a", 768, False), ("vendor/model-a", 1024, False)),
    ],
)
def test_different_unregistered_embedders_do_not_share_a_profile_id(
    first: tuple[str, int, bool], second: tuple[str, int, bool]
) -> None:
    assert _fallback_profile_id(*first) != _fallback_profile_id(*second)


def test_different_unregistered_models_do_not_share_an_index_fingerprint() -> None:
    """The consequence that made the id defect matter, asserted end to end.

    `_index_fingerprint` is what the incremental skip guard compares, so equal fingerprints mean
    a file is skipped rather than re-embedded after a model swap.
    """
    policy = ContextPolicy()
    embedders = {
        "small": _Embedder(_profile(BGE_SMALL, 384)),
        "large": _Embedder(_profile(BGE_LARGE, 1024)),
        "large-asym": _Embedder(_profile(BGE_LARGE, 1024, asymmetric=True)),
        "same-width-other-model": _Embedder(_profile("vendor/other-1024", 1024)),
    }
    fingerprints = {
        label: _index_fingerprint("deadbeef", emb, policy) for label, emb in embedders.items()
    }
    assert len(set(fingerprints.values())) == len(embedders), fingerprints


def test_the_registered_pairing_keeps_its_legacy_identifier() -> None:
    """No migration for the model the literal actually names.

    A bare `FastEmbedEmbedder()` is bge-small at 384 and reaches this path with no identity, so
    changing its id would re-partition every default corpus and orphan every calibration file
    keyed to it. That is a cost with no defect behind it.
    """
    assert _fallback_profile_id(BGE_SMALL, 384, asymmetric=False) == "bge-small-symmetric-v1"
    assert _fallback_profile_id(BGE_SMALL, 384, asymmetric=True) == "bge-small-asymmetric-v1"


def test_the_legacy_identifier_is_refused_at_the_wrong_width() -> None:
    """The registry entry is 384-dimensional; the model name alone does not earn the id.

    This is the branch that distinguishes the fix from a rename: the guard reads the registry's
    declared dimension rather than trusting the model string it was handed.
    """
    assert _fallback_profile_id(BGE_SMALL, 1024, asymmetric=False) != "bge-small-symmetric-v1"


def test_the_legacy_identifier_is_refused_for_another_model_at_the_same_width() -> None:
    """The mirror of the width case, and the one a mutant slipped through.

    Dropping the `model_name` term from the guard left every 384-dimensional model inheriting
    `bge-small-symmetric-v1`, and the whole suite stayed green: the width test above cannot see
    it, because it varies the width and holds the model fixed. Both halves of the comparison need
    their own assertion, or half the guard is decorative.
    """
    resolved = _fallback_profile_id("vendor/other-384", 384, asymmetric=False)
    assert resolved != "bge-small-symmetric-v1"
    assert "other-384" in resolved


@pytest.mark.parametrize(
    "model_name", [BGE_LARGE, "vendor/model-a", "org/sub/deep-name", "no-slash-model"]
)
def test_a_derived_identifier_is_safe_to_put_in_a_filename(model_name: str) -> None:
    """A profile id is interpolated into a path, so it may not carry path or reserved characters.

    `recall.eval.promotion.run.ArmConfig.key` builds a result FILENAME out of the profile id, and
    an unregistered embedder reaches it: `promotion/__main__` takes `embedding_profile(embedder)`
    for any embedder, not only a registered one. A raw HuggingFace name would put a `/` in that
    path and a `:` that Windows refuses, so a derived id that round-trips through a filename is
    part of the contract rather than a cosmetic choice.
    """
    resolved = _fallback_profile_id(model_name, 1024, asymmetric=False)
    assert not set(resolved) & set('/\\:*?"<>|')
    assert Path(resolved).name == resolved


@requires_fastembed
def test_the_constructor_is_wired_to_the_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """The real no-identity constructor path, exercised on the small model.

    Asserting `_fallback_profile_id` in isolation would leave the wiring untested, and the default
    embedder legitimately keeps the legacy literal, so a bare construction cannot tell a wired
    constructor from the unconditional expression it replaced. Removing the registry entry makes
    the DEFAULT model unregistered, which drives the derived branch through the real constructor
    without a 1.3 GB download.
    """
    import recall.embedding_registry as registry

    default = FastEmbedEmbedder()
    assert embedding_profile_id(default) == "bge-small-symmetric-v1"

    monkeypatch.setattr(registry, "find_registered_profile", lambda profile_id: None)
    derived = FastEmbedEmbedder()
    resolved = embedding_profile_id(derived)
    assert resolved != "bge-small-symmetric-v1"
    assert "bge-small-en-v1.5" in resolved and "384" in resolved


@requires_fastembed
def test_an_explicit_profile_id_still_wins() -> None:
    """`profile_id=` is a caller's deliberate claim, and the fallback must not override it.

    Through the real constructor: asserting this against a hand-built `EmbeddingProfile` would
    test the dataclass rather than the `profile_id or ...` precedence that carries it.
    """
    embedder = FastEmbedEmbedder(profile_id="caller-chosen-v1")
    assert embedding_profile_id(embedder) == "caller-chosen-v1"
