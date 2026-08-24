"""Serving a corpus built with a hosted (API) embedding model, through the registry.

Pre-registration: `docs/preregistrations/2026-08-18-hosted-embedding-profiles.md`, predictions
P1 to P5. Each test here is driven from a CONSTRUCTED embedder rather than from a hand-made
`EmbeddingProfile`, and that is the point rather than a stylistic preference.

An earlier attempt at this feature shipped a security test that called the private
`_artifact_is_pinned` on a synthetic identity no runtime path produces. It passed. It would have
gone on passing with the entire hosted branch of `RegisteredProfile.build` deleted, because
nothing in it ever ran that branch. The defect that actually sank the attempt, namely `build()`
constructing an identity and then discarding it, because the hosted classes took no ``identity``
parameter, was invisible to its own test suite for exactly that reason.

So: stub the PROVIDER (the network boundary), never the registry, and assert on what
`build()` returns.

The stubs are shaped to the real SDKs' response objects, because a stub more cooperative than the
client it replaces proves nothing. `voyageai.Client.embed(...)` returns an object with an
``.embeddings`` list; `OpenAI().embeddings.create(...)` returns one with a ``.data`` list whose
items each carry ``.embedding``. Both are reproduced below.
"""
from __future__ import annotations

import sys
import types
from dataclasses import replace

import pytest

from recall.embedding_registry import (
    HOSTED_BACKENDS,
    REGISTERED_PROFILES,
    RegisteredProfile,
    registered_profile,
)
from recall.embeddings import (
    HOSTED_UNVERIFIED_DIGEST,
    LEGACY_UNVERIFIED_DIGEST,
    EmbeddingProfile,
    artifact_is_pinned,
    embedding_profile,
    embedding_profile_id,
)

VOYAGE_PROFILE = "voyage-code-3-v1"
OPENAI_PROFILE = "openai-text-embedding-3-small-v1"


# --------------------------------------------------------------------------------------------
# Provider stubs, installed as import-time modules because both embedder classes import their
# SDK inside __init__ and CI installs only the `dev` extra (no voyageai, no openai).
# --------------------------------------------------------------------------------------------
class _VoyageResult:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.embeddings = vectors


class _StubVoyageClient:
    """Mimics `voyageai.Client`, recording what the registry actually asked the provider for."""

    calls: list[dict] = []

    def __init__(self, api_key: str, max_retries: int = 0) -> None:
        self.api_key = api_key

    def embed(self, texts, model, **kwargs):
        type(self).calls.append({"model": model, "texts": list(texts)})
        return _VoyageResult([[0.5] * _StubVoyageClient.width for _ in texts])


class _OpenAIItem:
    def __init__(self, vector: list[float]) -> None:
        self.embedding = vector


class _OpenAIResponse:
    def __init__(self, vectors: list[list[float]]) -> None:
        self.data = [_OpenAIItem(v) for v in vectors]


class _StubEmbeddings:
    def __init__(self, outer: "_StubOpenAI") -> None:
        self._outer = outer

    def create(self, **kwargs):
        _StubOpenAI.calls.append(kwargs)
        n = len(kwargs["input"])
        return _OpenAIResponse([[0.5] * _StubOpenAI.width for _ in range(n)])


class _StubOpenAI:
    calls: list[dict] = []

    def __init__(self, api_key: str, base_url: str, max_retries: int = 0) -> None:
        self.api_key = api_key
        self.base_url = base_url
        type(self).calls.append({"__init__": True, "base_url": base_url, "api_key": api_key})
        self.embeddings = _StubEmbeddings(self)


@pytest.fixture
def stub_providers(monkeypatch):
    """Install both SDKs as stub modules and reset their recorded calls.

    Mirrors the `stub_fastembed` pattern in tests/test_embedding_offline_enforcement.py: the
    module is injected into `sys.modules` so the function-local `import voyageai` / `from openai
    import OpenAI` inside each constructor resolves to the stub. Without this the tests would
    ERROR rather than skip on a CI runner that installs only the `dev` extra.
    """
    _StubVoyageClient.calls = []
    _StubOpenAI.calls = []
    _StubVoyageClient.width = 1024
    _StubOpenAI.width = 1536

    voyage_mod = types.ModuleType("voyageai")
    voyage_mod.Client = _StubVoyageClient
    monkeypatch.setitem(sys.modules, "voyageai", voyage_mod)

    openai_mod = types.ModuleType("openai")
    openai_mod.OpenAI = _StubOpenAI
    monkeypatch.setitem(sys.modules, "openai", openai_mod)
    return types.SimpleNamespace(voyage=_StubVoyageClient, openai=_StubOpenAI)


# --------------------------------------------------------------------------------------------
# P1: the identity survives `build()`
# --------------------------------------------------------------------------------------------
def test_voyage_build_carries_the_registry_identity(stub_providers):
    """The exact defect that made the previous attempt decorative.

    Reproduced then: ``registered_profile('voyage-3-v1').build(api_key='k')`` returned an embedder
    with ``has .profile: False`` and ``profile_id: voyage:voyage-3``, because `VoyageEmbedder`
    took no ``identity``. Every downstream consumer fell back to `legacy_embedding_profile`, so a
    generation registered under the registry id could never match the runtime.
    """
    embedder = registered_profile(VOYAGE_PROFILE).build(api_key="k")

    assert isinstance(getattr(embedder, "profile", None), EmbeddingProfile)
    assert embedding_profile_id(embedder) == VOYAGE_PROFILE
    assert embedding_profile_id(embedder) != "voyage:voyage-code-3"
    profile = embedding_profile(embedder)
    assert profile.artifact_digest == HOSTED_UNVERIFIED_DIGEST
    assert profile.artifact_digest != LEGACY_UNVERIFIED_DIGEST
    assert profile.dimension == embedder.dim == 1024


def test_openai_build_carries_the_registry_identity(stub_providers):
    embedder = registered_profile(OPENAI_PROFILE).build(api_key="k")

    assert embedding_profile_id(embedder) == OPENAI_PROFILE
    assert embedding_profile(embedder).artifact_digest == HOSTED_UNVERIFIED_DIGEST
    assert embedder.dim == 1536


def test_build_sends_the_registry_model_name_not_the_class_default(stub_providers):
    """The registry must DECIDE the request, not merely describe it.

    `OpenAICompatEmbedder`'s own default model is `openai/text-embedding-3-small` and its default
    base_url is OpenRouter, so a build that silently used class defaults would look correct for
    that one profile and be wrong for every other. Asserting the large variant catches it.
    """
    _StubOpenAI.width = 3072  # this profile's declared width; the guard holds the stub to it
    registered_profile("openai-text-embedding-3-large-v1").build(api_key="k")

    creates = [c for c in _StubOpenAI.calls if "model" in c]
    assert creates, "the constructor must probe the endpoint for its width"
    assert creates[0]["model"] == "openai/text-embedding-3-large"
    constructed = [c for c in _StubOpenAI.calls if c.get("__init__")]
    assert constructed[0]["base_url"] == "https://openrouter.ai/api/v1"


def test_every_hosted_profile_builds_and_carries_its_own_identity(stub_providers):
    """Coverage asserted over the registry, not over a list repeated here.

    A test naming its own three profiles passes unchanged when a fourth is registered wrongly.
    """
    hosted = [e for e in REGISTERED_PROFILES.values() if e.hosted]
    assert hosted, "the registry must contain hosted profiles for this suite to mean anything"
    for entry in hosted:
        _StubVoyageClient.width = entry.dimension
        _StubOpenAI.width = entry.dimension
        embedder = entry.build(api_key="k")
        assert embedding_profile_id(embedder) == entry.profile_id
        assert embedder.dim == entry.dimension
        assert embedding_profile(embedder).context_version == entry.context_version


# --------------------------------------------------------------------------------------------
# P3: a declared/actual width disagreement raises at construction
# --------------------------------------------------------------------------------------------
def test_width_disagreement_raises_at_construction(stub_providers):
    """Measured before this guard existed: built cleanly, and the readiness gate passed.

    The gate could not catch it. `check_enterprise_readiness` compares
    ``profile.dimension != embedder.dim``, but with no identity the profile comes from
    `legacy_embedding_profile`, which sets `dimension` FROM `embedder.dim`, so the comparison was
    vacuously true for every hosted embedder ever built.
    """
    _StubVoyageClient.width = 512  # the provider changed the width behind the model name
    with pytest.raises(ValueError, match="declares dimension 1024"):
        registered_profile(VOYAGE_PROFILE).build(api_key="k")


def test_width_disagreement_raises_for_openai_too(stub_providers):
    _StubOpenAI.width = 999
    with pytest.raises(ValueError, match="declares dimension 1536"):
        registered_profile(OPENAI_PROFILE).build(api_key="k")


def test_the_width_guard_is_not_vacuous(stub_providers):
    """Pin the guard's ALLOW path as well as its refusal.

    A width check that refused everything would pass every test above and make the feature
    unusable. This asserts the matching width is accepted, so the guard is discriminating rather
    than merely present.
    """
    _StubVoyageClient.width = 1024
    embedder = registered_profile(VOYAGE_PROFILE).build(api_key="k")
    assert embedder.dim == 1024


# --------------------------------------------------------------------------------------------
# P2: the enterprise gate refuses a hosted profile
# --------------------------------------------------------------------------------------------
def test_hosted_profile_is_not_artifact_pinned(stub_providers):
    embedder = registered_profile(VOYAGE_PROFILE).build(api_key="k")
    assert artifact_is_pinned(embedding_profile(embedder)) is False


def test_enterprise_readiness_refuses_a_hosted_profile(stub_providers):
    """Driven end to end from a built embedder, and asserting the SPECIFIC failure.

    Asserting only ``ready is False`` would pass if the store stub had failed first, which is the
    confound the pre-registration names. The assertion below names the artifact failure, and the
    positive control underneath proves the harness can produce a passing gate at all.
    """
    from recall.readiness import check_enterprise_readiness

    embedder = registered_profile(VOYAGE_PROFILE).build(api_key="k")
    result = check_enterprise_readiness(_StubStore(), embedder)

    assert result.ready is False
    assert any("hosted API" in f for f in result.failures), result.failures
    # And the reason is attestation, not a width or identity mismatch.
    assert not any("dimension" in f for f in result.failures), result.failures


def test_enterprise_readiness_harness_can_pass(stub_providers):
    """Positive control for the test above: the same store, with a PINNED profile, is ready.

    Without this, `test_enterprise_readiness_refuses_a_hosted_profile` would be satisfied by a
    harness that fails every input, which is a guard that cannot fail.
    """
    from recall.readiness import check_enterprise_readiness

    pinned = EmbeddingProfile(
        profile_id="bge-small-symmetric-v1",
        model_name="BAAI/bge-small-en-v1.5",
        artifact_digest="a" * 64,
        dimension=384,
        query_mode="embed",
        passage_mode="embed",
    )
    result = check_enterprise_readiness(_StubStore(dim=384), _ProfiledEmbedder(pinned))
    assert not any("hosted API" in f for f in result.failures), result.failures
    assert not any("not pinned" in f for f in result.failures), result.failures


def test_operator_can_accept_the_unpinned_artifact_explicitly(stub_providers):
    """The escape hatch exists, and is the same one legacy profiles use."""
    from recall.readiness import check_enterprise_readiness

    embedder = registered_profile(VOYAGE_PROFILE).build(api_key="k")
    result = check_enterprise_readiness(_StubStore(), embedder, allow_legacy_profile=True)
    assert not any("hosted API" in f for f in result.failures), result.failures


# --------------------------------------------------------------------------------------------
# P4: the index fingerprint separates two widths of one hosted model
# --------------------------------------------------------------------------------------------
def test_index_fingerprint_separates_two_widths_of_one_hosted_model(stub_providers):
    """Two widths of one hosted model must never share an index fingerprint.

    🔁 **Superseded upstream while this was being written, and kept for the narrower claim it
    still makes.** Measured on 42bbe818 (2026-08-18): identical. Both widths of `voyage-code-3`
    minted ``profile_id='voyage:voyage-code-3'`` through `legacy_embedding_profile`, and
    `_index_fingerprint` hashed that id with no dimension term, so the incremental skip guard
    treated a width swap as a no-op. `voyage-code-3` really does serve 256/512/1024/2048, so it
    was reachable rather than theoretical.

    #381 then landed independently and re-keyed `_index_fingerprint` on the whole
    `EmbeddingProfile.fingerprint()`, which includes `dimension`. That fixes the collision at the
    root, for legacy hosted embedders as well as registered ones, and it is a better fix than
    giving hosted profiles distinct ids would have been. So this test no longer demonstrates a
    defect that the registry change repairs.

    It is kept because it still pins something neither change guarantees on its own: that two
    REGISTERED hosted profiles differing only in width stay distinguishable to the skip guard. A
    future fingerprint that dropped the dimension term, or a registry that let two widths share an
    id, would each reintroduce the original failure, and this is the assertion that would catch it.
    The two profiles differ ONLY in dimension, or the test would prove nothing about the width.
    """
    from recall.context import ContextPolicy
    from recall.index import _index_fingerprint

    base = registered_profile(VOYAGE_PROFILE)
    wide = replace(base, profile_id="voyage-code-3-2048-v1", dimension=2048)
    assert (base.model_name, base.context_mode) == (wide.model_name, wide.context_mode)

    _StubVoyageClient.width = 1024
    narrow_embedder = base.build(api_key="k")
    _StubVoyageClient.width = 2048
    wide_embedder = wide.build(api_key="k")

    policy = ContextPolicy()
    assert _index_fingerprint("hash", narrow_embedder, policy) != _index_fingerprint(
        "hash", wide_embedder, policy
    )


# --------------------------------------------------------------------------------------------
# P5: a hosted profile stays SUBJECT to the Indexer context check
# --------------------------------------------------------------------------------------------
def test_hosted_profile_is_not_exempt_from_the_index_context_check(stub_providers):
    """The over-application that sank the reverted attempt.

    It replaced `index.py`'s literal comparison with a shared `is_unverified_artifact()`
    predicate. Hosted profiles are unverified, so they became exempt from a context check they
    should pass, and a hosted identity declaring `raw-v1` was accepted under a `section` policy
    where a local profile raises.
    """
    from recall.context import ContextPolicy
    from recall.index import Indexer

    embedder = registered_profile(VOYAGE_PROFILE).build(api_key="k")
    assert embedding_profile(embedder).context_version == "raw-v1"

    with pytest.raises(ValueError, match="context"):
        Indexer(
            store=_StubStore(),
            embedder=embedder,
            context_policy=ContextPolicy(mode="section"),
        )


def test_a_legacy_profile_stays_exempt_from_that_check(stub_providers):
    """The other half: narrowing the exemption must not remove it from legacy embedders.

    A guard that blocks correct work gets deleted, which loses the coverage entirely.
    """
    from recall.context import ContextPolicy
    from recall.embeddings import HashingEmbedder
    from recall.index import Indexer

    Indexer(
        store=_StubStore(dim=64),
        embedder=HashingEmbedder(dim=64),
        context_policy=ContextPolicy(mode="section"),
    )


# --------------------------------------------------------------------------------------------
# Registry invariants
# --------------------------------------------------------------------------------------------
def test_a_hosted_profile_cannot_pin_an_artifact_digest():
    with pytest.raises(ValueError, match="cannot pin an artifact digest"):
        RegisteredProfile(
            profile_id="bad-v1",
            model_name="voyage-3",
            dimension=1024,
            query_mode="embed",
            passage_mode="embed",
            context_mode="none",
            backend="voyage",
            artifact_digest="b" * 64,
        )


def test_a_voyage_profile_cannot_declare_output_dimensions():
    """A request field the client never sends must be refused, not accepted and dropped."""
    with pytest.raises(ValueError, match="does not send"):
        RegisteredProfile(
            profile_id="bad-voyage-v1",
            model_name="voyage-code-3",
            dimension=512,
            query_mode="embed",
            passage_mode="embed",
            context_mode="none",
            backend="voyage",
            output_dimensions=512,
        )


def test_a_local_profile_cannot_declare_output_dimensions():
    with pytest.raises(ValueError, match="width comes from the artifact"):
        RegisteredProfile(
            profile_id="bad-local-v1",
            model_name="BAAI/bge-small-en-v1.5",
            dimension=384,
            query_mode="embed",
            passage_mode="embed",
            context_mode="none",
            backend="fastembed",
            output_dimensions=384,
        )


def test_build_refuses_artifact_arguments_for_a_hosted_profile(stub_providers):
    with pytest.raises(ValueError, match="takes an api_key"):
        registered_profile(VOYAGE_PROFILE).build(artifact_path="/x", artifact_digest="c" * 64)


def test_build_still_requires_artifact_arguments_for_a_local_profile():
    with pytest.raises(ValueError, match="needs both an artifact path"):
        registered_profile("bge-small-symmetric-v1").build(api_key="k")


def test_hosted_profiles_declare_only_measured_normalization():
    """Gemini is registered at its native width ONLY, and this asserts the reason.

    Measured 2026-08-18: `gemini-embedding-001` returns a unit vector at 3072 but norm 0.694 at
    1536 and 0.582 at 768. A profile declaring `normalization='l2'` at a truncated width would be
    stating something the provider does not do, and every cosine compared against a calibrated
    abstention threshold would be quietly wrong.
    """
    for entry in REGISTERED_PROFILES.values():
        if not entry.hosted:
            continue
        if entry.output_dimensions is not None:
            assert entry.output_dimensions == entry.dimension, entry.profile_id
        if "gemini" in entry.model_name:
            assert entry.dimension == 3072, (
                f"{entry.profile_id} registers a truncated Gemini width, whose vectors are not "
                "l2-normalised; see scripts/measure_hosted_embedding_widths.py"
            )


def test_hosted_backends_and_the_hosted_flag_agree():
    for entry in REGISTERED_PROFILES.values():
        assert entry.hosted == (entry.backend in HOSTED_BACKENDS)
        if entry.hosted and entry.backend == "openai-compat":
            assert entry.base_url, f"{entry.profile_id} has no endpoint"


# --------------------------------------------------------------------------------------------
# The production path
# --------------------------------------------------------------------------------------------
def test_make_embedder_builds_a_hosted_profile(stub_providers):
    """`RegisteredProfile.build` had exactly one non-test caller, and it refused every hosted
    route: `RECALL_EMBED_PROFILE` required `RECALL_EMBEDDER=fastembed`, and the profile branch
    then demanded `RECALL_MODEL_CACHE` + `RECALL_MODEL_SHA256`, which a hosted profile has no
    business supplying. Without this test the feature is unreachable in production.
    """
    from recall_mcp.service import make_embedder

    embedder = make_embedder(
        "voyage",
        {"RECALL_EMBED_PROFILE": VOYAGE_PROFILE, "VOYAGE_API_KEY": "k"},
    )
    assert embedding_profile_id(embedder) == VOYAGE_PROFILE


def test_make_embedder_needs_no_artifact_variables_for_a_hosted_profile(stub_providers):
    from recall_mcp.service import make_embedder

    _StubOpenAI.width = 3072
    embedder = make_embedder(
        "openrouter",
        {"RECALL_EMBED_PROFILE": "gemini-embedding-001-v1", "OPENROUTER_API_KEY": "k"},
    )
    assert embedding_profile_id(embedder) == "gemini-embedding-001-v1"


def test_make_profile_embedder_serves_a_hosted_generation(stub_providers):
    """The SECOND production path, and the one that decides whether a tenant can be served.

    `recall.enterprise_cli` calls `make_profile_embedder(route.active.embedding_profile)`, i.e.
    with whatever profile the tenant's ACTIVE GENERATION was built under. It hard-coded
    `RECALL_EMBEDDER=fastembed`, so a tenant whose generation is hosted raised instead of serving,
    which would have left this feature reachable from `make_embedder` and unreachable from the
    path that actually serves a corpus. Fixing only the first caller is fixing half of it.
    """
    from recall_mcp.service import make_profile_embedder

    embedder = make_profile_embedder(VOYAGE_PROFILE, env={"VOYAGE_API_KEY": "k"})
    assert embedding_profile_id(embedder) == VOYAGE_PROFILE


def test_make_profile_embedder_local_route_is_unchanged():
    """The local branch must keep asking for its artifact variables."""
    from recall_mcp.service import make_profile_embedder

    with pytest.raises(ValueError, match="RECALL_MODEL_CACHE and RECALL_MODEL_SHA256"):
        make_profile_embedder("bge-small-symmetric-v1", env={})


def test_make_embedder_refuses_a_mismatched_embedder_spelling(stub_providers):
    from recall_mcp.service import make_embedder

    with pytest.raises(ValueError, match="needs RECALL_EMBEDDER=voyage"):
        make_embedder(
            "fastembed", {"RECALL_EMBED_PROFILE": VOYAGE_PROFILE, "VOYAGE_API_KEY": "k"}
        )


def test_make_embedder_still_refuses_an_unknown_profile():
    from recall_mcp.service import make_embedder

    with pytest.raises(ValueError, match="unknown RECALL_EMBED_PROFILE"):
        make_embedder("voyage", {"RECALL_EMBED_PROFILE": "nope-v1"})


def test_make_embedder_local_profile_route_is_unchanged():
    """The pairing every existing deployment already has written down."""
    from recall_mcp.service import make_embedder

    with pytest.raises(ValueError, match="RECALL_MODEL_CACHE and RECALL_MODEL_SHA256"):
        make_embedder("fastembed", {"RECALL_EMBED_PROFILE": "bge-small-symmetric-v1"})


# --------------------------------------------------------------------------------------------
# Minimal stubs for the gate's collaborators
# --------------------------------------------------------------------------------------------
class _StubStore:
    """Enough `PgVectorStore` surface for `check_enterprise_readiness` and `Indexer.__init__`."""

    def __init__(self, dim: int = 1024) -> None:
        self.dim = dim
        self.tenant = "t"
        self.generation_id = "g"
        self.table = "chunks"


class _ProfiledEmbedder:
    def __init__(self, profile: EmbeddingProfile) -> None:
        self.profile = profile
        self.dim = profile.dimension
        self.name = profile.profile_id

    def embed(self, texts):
        return [[0.0] * self.dim for _ in texts]
