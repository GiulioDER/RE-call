"""The incremental skip guard must key on the whole embedding identity, not one field of it.

`_index_fingerprint` decides "this file is already indexed under this configuration". It used to
hash `embedding_profile_id(embedder)`, which is ONE field of an `EmbeddingProfile`, so any two
embedders sharing an id were the same embedder as far as the skip guard was concerned: the file
was skipped, the old vectors stayed, and the run reported success.

That was reachable, not theoretical. Until `92619999` the no-identity path minted
`bge-small-symmetric-v1` for every unregistered model, and a 384-dimension corpus and a
1024-dimension corpus produced EQUAL index fingerprints for the same file. Fixing the id closed
that route in; these tests close the hole itself, by pinning that every field of the identity
reaches the fingerprint.

`recall/cache.py` has always keyed on the whole profile and its docstring says why: "The ID alone
is not an identity". These tests assert the skip guard now agrees with the cache instead of
contradicting it.
"""
from __future__ import annotations

import pytest

from recall.context import ContextPolicy
from recall.embeddings import EmbeddingProfile
from recall.index import _index_fingerprint

CONTENT = "deadbeef"


class _Embedder:
    """An embedder that is nothing but its profile.

    Model-free on purpose: the subject is which identity fields reach the hash, and loading real
    weights would skip these wherever the fastembed extra is absent, which is exactly the CI job
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


def _profile(**overrides: object) -> EmbeddingProfile:
    base: dict[str, object] = {
        "profile_id": "shared-id-v1",
        "model_name": "vendor/model-a",
        "artifact_digest": "legacy-unverified",
        "dimension": 384,
        "query_mode": "embed",
        "passage_mode": "embed",
    }
    base.update(overrides)
    return EmbeddingProfile(**base)  # type: ignore[arg-type]


def _fingerprint(profile: EmbeddingProfile, policy: ContextPolicy | None = None) -> str:
    return _index_fingerprint(CONTENT, _Embedder(profile), policy or ContextPolicy())


#: Every field of `EmbeddingProfile` other than `profile_id`, each paired with a value that
#: differs from `_profile()`'s default. `profile_id` is excluded deliberately: it was the ONLY
#: field the old implementation covered, so including it would pass with or without the fix and
#: would be the one case that proves nothing.
IDENTITY_FIELDS = [
    ("model_name", "vendor/model-b"),
    ("artifact_digest", "9a443d711e063427"),
    ("dimension", 1024),
    ("query_mode", "query_embed"),
    ("passage_mode", "passage_embed"),
    ("normalization", "none"),
    ("instruction_version", "retrieval-v1"),
    ("chunker_version", "chunk-text-v2"),
    ("context_version", "ctx-document-v1"),
    ("dependencies", (("fastembed", "0.9.0"),)),
]


@pytest.mark.parametrize(("field", "value"), IDENTITY_FIELDS, ids=[f for f, _ in IDENTITY_FIELDS])
def test_every_identity_field_reaches_the_index_fingerprint(field: str, value: object) -> None:
    """Change one field of the identity, holding the profile id fixed, and the file must rebuild.

    Holding `profile_id` fixed across every case is the whole point: under the old implementation
    all of these hashed equal, so the guard would skip a file whose vectors had genuinely moved.
    """
    assert _fingerprint(_profile()) != _fingerprint(_profile(**{field: value}))


def test_all_identity_variants_are_mutually_distinct() -> None:
    """Not just different from the baseline, but from each other.

    A hash that collapsed two fields into one term would still pass every pairwise test above
    while losing the ability to tell those two configurations apart.
    """
    fingerprints = {"baseline": _fingerprint(_profile())}
    for field, value in IDENTITY_FIELDS:
        fingerprints[field] = _fingerprint(_profile(**{field: value}))
    assert len(set(fingerprints.values())) == len(fingerprints), fingerprints


def test_the_original_defect_two_widths_under_one_id() -> None:
    """The measured case, pinned at the layer it actually broke.

    A 384-dimension and a 1024-dimension embedder sharing `bge-small-symmetric-v1` produced equal
    index fingerprints, so swapping the model left every file skipped and every vector stale.
    """
    small = _profile(profile_id="bge-small-symmetric-v1", dimension=384)
    large = _profile(
        profile_id="bge-small-symmetric-v1", dimension=1024, model_name="BAAI/bge-large-en-v1.5"
    )
    assert _fingerprint(small) != _fingerprint(large)


def test_an_identical_identity_hashes_identically() -> None:
    """The other half, and the one that makes the guard usable rather than merely safe.

    A fingerprint that moved for an unchanged configuration would re-embed the whole corpus on
    every run, which is a worse failure than the one being fixed: it is silent, recurring, and
    costs an inference run each time rather than once.
    """
    assert _fingerprint(_profile()) == _fingerprint(_profile())


def test_the_content_hash_still_reaches_the_fingerprint() -> None:
    """A pre-existing term, re-pinned because this change rewrote the tuple around it."""
    embedder = _Embedder(_profile())
    policy = ContextPolicy()
    assert _index_fingerprint("aaaa", embedder, policy) != _index_fingerprint(
        "bbbb", embedder, policy
    )


@pytest.mark.parametrize(
    "policy",
    [ContextPolicy(mode="document"), ContextPolicy(version="ctx-v2")],
    ids=["mode", "version"],
)
def test_the_context_policy_terms_still_reach_the_fingerprint(policy: ContextPolicy) -> None:
    """Also pre-existing, also re-pinned.

    Widening the embedder term is exactly the kind of edit that quietly drops a neighbouring term,
    and `tests/test_context_modes_index.py` covers the mode only through the database.
    """
    assert _fingerprint(_profile()) != _fingerprint(_profile(), policy)


class _Tokenizer:
    """A real `recall.context.Tokenizer`, because `ContextPolicy` refuses `max_tokens` without one.

    An object with `count_tokens`, not a bare callable. `ContextPolicy.__post_init__` only checks
    that a tokenizer is PRESENT, so a plain function constructs happily and blows up later inside
    `contextual_passages`; using the real shape here keeps the fixture from encoding that mistake.
    """

    def count_tokens(self, text: str) -> int:
        return len(text.split())


_tokens = _Tokenizer()


def test_max_tokens_reaches_the_fingerprint() -> None:
    """The long-carried gap, closed in the same change that pays for it.

    `contextual_passages` selects a different rung of its degradation ladder when `max_tokens` is
    set, so two policies differing only in it build different passages from the same file. They
    used to hash equal, so the second was skipped and served passages built under the first.
    """
    unset = _fingerprint(_profile())
    capped = _fingerprint(_profile(), ContextPolicy(max_tokens=256, tokenizer=_tokens))
    assert unset != capped


def test_two_different_caps_do_not_share_a_fingerprint() -> None:
    """Set-versus-unset is the easy half; two set values are the one a sloppy term would miss.

    A term written as a truthy flag rather than the value would pass the test above and fail here.
    """
    a = _fingerprint(_profile(), ContextPolicy(max_tokens=256, tokenizer=_tokens))
    b = _fingerprint(_profile(), ContextPolicy(max_tokens=512, tokenizer=_tokens))
    assert a != b


def test_a_zero_cap_is_not_the_same_as_no_cap() -> None:
    """`None` and `0` are different policies, and a falsy-coercing term would merge them.

    A mutant writing `str(context_policy.max_tokens or 0)` survived every other test in this file.
    The distinction is real, not pedantic: measured through `contextual_passages`, `max_tokens=0`
    falls all the way down the degradation ladder to the bare chunk text, while `max_tokens=None`
    skips the ladder entirely and embeds the fully contextualised passage. Two genuinely different
    vectors, one fingerprint, and the second policy silently skipped.
    """
    no_cap = _fingerprint(_profile(), ContextPolicy())
    zero_cap = _fingerprint(_profile(), ContextPolicy(max_tokens=0, tokenizer=_tokens))
    assert no_cap != zero_cap


def test_an_unset_cap_is_stable_across_calls() -> None:
    """`None` is the shipped value everywhere, so its term must be constant, not merely present.

    `context_policy_for_profile` never sets `max_tokens`, so every shipped corpus hashes this
    branch on every run. A term that varied here would re-embed every corpus on every index.
    """
    assert _fingerprint(_profile()) == _fingerprint(_profile())
    assert _fingerprint(_profile(), ContextPolicy()) == _fingerprint(_profile(), ContextPolicy())


def test_the_tokenizer_is_knowingly_not_covered() -> None:
    """Pins the LIMIT, so it is a recorded decision rather than an untested assumption.

    A tokenizer changes the passage exactly as `max_tokens` does, but a callable has no identity
    that is stable across processes, and an unstable term would re-embed the whole corpus on every
    run: strictly worse than the gap. This test exists so that anyone closing the gap has to
    delete an assertion that states the reasoning, rather than discovering the omission from a
    fingerprint that quietly did not move.
    """
    def other_tokens(text: str) -> list[str]:
        return list(text)

    same_cap_other_tokenizer = (
        _fingerprint(_profile(), ContextPolicy(max_tokens=256, tokenizer=_tokens)),
        _fingerprint(_profile(), ContextPolicy(max_tokens=256, tokenizer=other_tokens)),
    )
    assert same_cap_other_tokenizer[0] == same_cap_other_tokenizer[1]
