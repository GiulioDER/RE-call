"""The onnxruntime and tokenizers versions must be fingerprint key material for FastEmbed.

Invariant: two FastEmbed embedders that differ ONLY in the installed onnxruntime version have
different `EmbeddingProfile.fingerprint()` values, on the registered path
(`RegisteredProfile.build`) and on the unregistered one (`FastEmbedEmbedder(...)` directly).

Failure mode caught: an onnxruntime-only upgrade (PR 943 moved 1.30.0 to 1.31.0 with fastembed
unchanged) left the fingerprint byte-identical, so the embedding cache key (`recall/cache.py`),
the index skip-guard fingerprint (`recall/index.py`) and the v1 calibration binding all reused
vectors produced under the old runtime, while `EmbeddingProfile.fingerprint`'s docstring said the
dependencies existed to prevent exactly that.

Red proof, recorded per the repository rule. Baseline: `recall/embedding_providers/local.py` at
726a65ff4, which records `fastembed` and the ONNX providers but no onnxruntime version. The
pre-fix code has no `_onnxruntime_dependency`, so these tests go through the real consumer
boundary instead and fake the version where every version lookup in that module goes,
`recall.embedding_providers.local._package_version`, which exists on both sides of the fix.

- `test_a_registered_profile_moves_when_only_onnxruntime_moves`: RED at the baseline on
  `assert old.profile.fingerprint() != new.profile.fingerprint()` (the two were equal).
- `test_an_unregistered_embedder_moves_when_only_onnxruntime_moves`: RED at the baseline on the
  same assertion for the directly constructed embedder.
- `test_a_gpu_build_is_found_under_its_own_distribution_name`: RED at the baseline on
  `assert dict(gpu.profile.dependencies).get("onnxruntime") == "onnxruntime-gpu==1.31.0"`
  (`None == ...`), because no entry for the runtime existed in the dependencies at all.

- `test_a_tokenizers_only_upgrade_moves_the_fingerprint`: RED at the baseline on
  `assert old.profile.fingerprint() != new.profile.fingerprint()` (equal), and RED again against
  the fix with only the `("tokenizers", ...)` entry deleted from `_provider_dependencies`, which is
  the mutation that isolates it from the onnxruntime entry.

All four GREEN after the fix (`_onnxruntime_dependency` and the `tokenizers` entry, both folded
into `_provider_dependencies`).
"""

from __future__ import annotations

import sys
import types

import pytest

import recall.embedding_providers.local as local


class _StubModel:
    """A fastembed stand-in; everything about it stays fixed between the two builds."""

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs

    def embed(self, texts):
        return [[1.0] + [0.0] * 383 for _ in texts]

    query_embed = embed
    passage_embed = embed


@pytest.fixture
def stub_fastembed(monkeypatch):
    module = types.ModuleType("fastembed")
    module.TextEmbedding = _StubModel  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)


@pytest.fixture
def provisioned(tmp_path):
    from recall.embeddings import artifact_tree_sha256

    root = tmp_path / "bge-cache"
    root.mkdir()
    (root / "model.onnx").write_bytes(b"weights")
    return root, artifact_tree_sha256(root)


def _installed(monkeypatch, **versions: str) -> None:
    """Pretend exactly these distributions are installed, fastembed fixed at one version.

    Keyword names use underscores for the hyphens of a distribution name.
    """
    table = {
        "fastembed": "0.8.0",
        "tokenizers": "0.22.1",
        **{k.replace("_", "-"): v for k, v in versions.items()},
    }
    monkeypatch.setattr(local, "_package_version", lambda name: table.get(name, "not-installed"))


def test_a_registered_profile_moves_when_only_onnxruntime_moves(
    stub_fastembed, provisioned, monkeypatch
):
    from recall.embedding_registry import registered_profile

    root, digest = provisioned
    entry = registered_profile("bge-small-symmetric-v1")

    _installed(monkeypatch, onnxruntime="1.30.0")
    old = entry.build(artifact_path=root, artifact_digest=digest)
    _installed(monkeypatch, onnxruntime="1.31.0")
    new = entry.build(artifact_path=root, artifact_digest=digest)

    assert old.profile.fingerprint() != new.profile.fingerprint(), (
        "an onnxruntime-only upgrade left a registered FastEmbed fingerprint unchanged"
    )
    assert dict(old.profile.dependencies)["onnxruntime"] == "1.30.0"
    assert dict(new.profile.dependencies)["onnxruntime"] == "1.31.0"


def test_an_unregistered_embedder_moves_when_only_onnxruntime_moves(stub_fastembed, monkeypatch):
    _installed(monkeypatch, onnxruntime="1.30.0")
    old = local.FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
    _installed(monkeypatch, onnxruntime="1.31.0")
    new = local.FastEmbedEmbedder("BAAI/bge-small-en-v1.5")

    assert old.profile.fingerprint() != new.profile.fingerprint(), (
        "an onnxruntime-only upgrade left an unregistered FastEmbed fingerprint unchanged"
    )
    assert dict(old.profile.dependencies)["fastembed"] == dict(new.profile.dependencies)[
        "fastembed"
    ], "the control: fastembed itself must not have moved"


def test_a_gpu_build_is_found_under_its_own_distribution_name(stub_fastembed, monkeypatch):
    """onnxruntime-gpu installs the same module under another distribution name."""
    _installed(monkeypatch, onnxruntime_gpu="1.31.0")
    gpu = local.FastEmbedEmbedder("BAAI/bge-small-en-v1.5")

    assert dict(gpu.profile.dependencies).get("onnxruntime") == "onnxruntime-gpu==1.31.0"

    _installed(monkeypatch, onnxruntime="1.31.0")
    cpu = local.FastEmbedEmbedder("BAAI/bge-small-en-v1.5")

    assert gpu.profile.fingerprint() != cpu.profile.fingerprint()


def test_a_tokenizers_only_upgrade_moves_the_fingerprint(stub_fastembed, monkeypatch):
    """fastembed tokenizes before inference, so the tokenizer is identity like the runtime."""
    _installed(monkeypatch, onnxruntime="1.31.0", tokenizers="0.22.1")
    old = local.FastEmbedEmbedder("BAAI/bge-small-en-v1.5")
    _installed(monkeypatch, onnxruntime="1.31.0", tokenizers="0.23.0")
    new = local.FastEmbedEmbedder("BAAI/bge-small-en-v1.5")

    assert old.profile.fingerprint() != new.profile.fingerprint(), (
        "a tokenizers-only upgrade left a FastEmbed fingerprint unchanged"
    )
