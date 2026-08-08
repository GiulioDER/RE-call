"""The execution provider must be fingerprint key material on EVERY path, not just the legacy one.

The first attempt at this fix built the provider pair inside the right-hand side of
``self._profile = identity or EmbeddingProfile(...)``. Because a supplied ``identity`` is always
truthy, the right-hand side never evaluated for a REGISTERED profile — the path that carries a
pinned artifact and a certified calibration, and therefore the path where serving CPU vectors to a
GPU-configured embedder does the most damage. The fix read as protection and could not fire.

Read the split below carefully, because the first version of this file got it wrong. The tests
above the WIRING divider exercise the HELPERS: against the broken version they fail only by
ImportError, since none of those symbols existed there. That is a missing-symbol failure, not a
behavioural RED, and it would NOT catch the call site regressing to ``identity or …`` tomorrow.

The two tests below the divider are the actual regression guard. They go through
``registered_profile(...).build(...)``, which passes a registry ``identity`` — the exact argument
that made the original fix short-circuit — and they assert on the built embedder's profile. Revert
the call site and they go red on the assertion.

They also cover the KNOWN-provider branch, which nothing in the repo previously reached: the
suite's shared fastembed stub exposes no session, so every other test takes ``unavailable``.
"""

from __future__ import annotations

import pytest

from recall.embeddings import (
    PROVIDERS_UNKNOWN,
    EmbeddingProfile,
    _provider_dependencies,
    _with_provider_dependency,
)


def _profile(**kw: object) -> EmbeddingProfile:
    base = dict(
        profile_id="bge-small-symmetric-v1",
        model_name="BAAI/bge-small-en-v1.5",
        artifact_digest="sha256:deadbeef",
        dimension=384,
        query_mode="embed",
        passage_mode="embed",
        context_version="v1",
        dependencies=(("fastembed", "0.8.0"),),
    )
    base.update(kw)
    return EmbeddingProfile(**base)  # type: ignore[arg-type]


def test_a_registered_profile_separates_cpu_from_gpu():
    """The defect, stated as an assertion: two provenances, one fingerprint.

    RED on the old code — `identity` short-circuited, so both sides returned the registry's
    dependencies verbatim and the two fingerprints were equal.
    """
    registered = _profile()
    cpu = _with_provider_dependency(registered, ["CPUExecutionProvider"])
    gpu = _with_provider_dependency(registered, ["CUDAExecutionProvider"])

    assert cpu.fingerprint() != gpu.fingerprint(), (
        "a registered profile still shares one cache key and one calibration binding across "
        "CPU and CUDA"
    )


def test_the_provider_actually_lands_in_a_registered_profiles_dependencies():
    """Not merely 'the fingerprint moved' — the reason it moved has to be the provider."""
    wrapped = _with_provider_dependency(_profile(), ["CUDAExecutionProvider"])

    assert ("onnx-providers", "CUDAExecutionProvider") in wrapped.dependencies
    assert ("fastembed", "0.8.0") in wrapped.dependencies, "registry entries must survive"


def test_wrapping_twice_does_not_move_the_fingerprint_again():
    """Idempotence is load-bearing: a double wrap would invalidate every cache a second time."""
    once = _with_provider_dependency(_profile(), ["CPUExecutionProvider"])
    twice = _with_provider_dependency(once, ["CUDAExecutionProvider"])

    assert twice is once or twice.fingerprint() == once.fingerprint()
    assert [k for k, _ in twice.dependencies].count("onnx-providers") == 1


@pytest.mark.parametrize("providers", [[PROVIDERS_UNKNOWN], []])
def test_an_unreadable_session_is_a_third_state_not_a_cpu_run(providers):
    """"I could not tell" must not be recorded as a provider, and must not read as CPU.

    `_session_providers` is documented as observational and must never fail a run, so its sentinel
    reaches this code on healthy deployments. Collapsing it into the provider value would repeat
    the negative-guard mistake: an unrecorded session comparing equal to a known one.
    """
    deps = dict(_provider_dependencies(providers))

    assert deps["onnx-providers-source"] == "unavailable"
    assert deps["onnx-providers"] == "unavailable"
    assert PROVIDERS_UNKNOWN not in deps["onnx-providers"], (
        "the sentinel must not travel as if it were a provider name"
    )


def test_unknown_is_distinguishable_from_a_real_cpu_run():
    known = _with_provider_dependency(_profile(), ["CPUExecutionProvider"])
    unknown = _with_provider_dependency(_profile(), [PROVIDERS_UNKNOWN])

    assert known.fingerprint() != unknown.fingerprint()
    assert dict(known.dependencies)["onnx-providers-source"] == "session"
    assert dict(unknown.dependencies)["onnx-providers-source"] == "unavailable"


# ---------------------------------------------------------------------------
# The WIRING. Everything above tests the helpers; none of it would notice the
# call site reverting to `identity or EmbeddingProfile(...)`. These do.
# ---------------------------------------------------------------------------


class _StubModel:
    """A fastembed stand-in whose ONNX session IS introspectable.

    The suite's existing stub exposes no session, so every other test in the repo takes the
    `unavailable` branch. Nothing exercised the KNOWN branch — the one that carries a real
    provider into the fingerprint — which is the branch the whole change is for.
    """

    instances: list["_StubModel"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        self.model = _StubSession()
        _StubModel.instances.append(self)

    def embed(self, texts):
        return [[1.0] + [0.0] * 383 for _ in texts]

    query_embed = embed
    passage_embed = embed


class _StubSession:
    def get_providers(self) -> list[str]:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]


@pytest.fixture
def stub_with_session(monkeypatch):
    import sys
    import types

    _StubModel.instances = []
    module = types.ModuleType("fastembed")
    module.TextEmbedding = _StubModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    return _StubModel


@pytest.fixture
def provisioned(tmp_path):
    from recall.embeddings import artifact_tree_sha256

    root = tmp_path / "bge-cache"
    root.mkdir()
    (root / "model.onnx").write_bytes(b"weights")
    return root, artifact_tree_sha256(root)


def test_a_registered_profile_built_through_the_registry_carries_the_provider(
    stub_with_session, provisioned
):
    """THE regression test: revert the call site and this goes red.

    `entry.build(...)` is the enterprise path — it passes a registry `identity`, which is exactly
    the argument that made the original fix short-circuit.
    """
    from recall.embedding_registry import registered_profile

    root, digest = provisioned
    embedder = registered_profile("bge-small-symmetric-v1").build(
        artifact_path=root, artifact_digest=digest
    )
    deps = dict(embedder.profile.dependencies)

    assert "onnx-providers" in deps, "the registered path lost the provider again"
    assert deps["onnx-providers-source"] == "session"
    assert deps["onnx-providers"] == "CUDAExecutionProvider,CPUExecutionProvider"


def test_two_registry_builds_on_different_providers_do_not_share_a_fingerprint(
    stub_with_session, provisioned, monkeypatch
):
    """The consequence, asserted end to end rather than on the helper."""
    from recall.embedding_registry import registered_profile

    root, digest = provisioned
    gpu = registered_profile("bge-small-symmetric-v1").build(
        artifact_path=root, artifact_digest=digest
    )

    monkeypatch.setattr(_StubSession, "get_providers", lambda self: ["CPUExecutionProvider"])
    cpu = registered_profile("bge-small-symmetric-v1").build(
        artifact_path=root, artifact_digest=digest
    )

    assert gpu.profile.fingerprint() != cpu.profile.fingerprint()
