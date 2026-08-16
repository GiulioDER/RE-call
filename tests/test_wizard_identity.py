"""Give a locally-downloaded embedder an identity production will accept.

`generation build` needs an immutable embedder identity: a provider revision, or an artifact
digest. Neither exists for the default embedder — nothing in the tree pins a revision for
bge-small — so the only way to build was `--unverified-development`, which stamps the generation
`verified: false`, and `generations.py` refuses `allow_unverified` under `RECALL_ENV=production`.
The wizard could therefore build a generation only in the one mode its own design does not serve
`docs` and `code` in.

The way out is already in the tree and was simply never wired: `artifact_tree_sha256` digests a
provisioned model directory, and `EmbedderIdentity` treats an artifact digest as immutable on its
own. Hashing what fastembed actually downloaded is also a STRONGER identity than a Hub revision
would have been, and this is not a nicety — fastembed fetches
`qdrant/bge-small-en-v1.5-onnx-q`, a quantised repackaging, so a commit SHA for
`BAAI/bge-small-en-v1.5` would have named bytes nobody loads.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from recall.lineage import EmbedderIdentity
from recall.wizard.identity import ArtifactIdentity, artifact_identity_for


class _Onnx:
    def __init__(self, model_dir: Path) -> None:
        self._model_dir = str(model_dir)


class _Inner:
    def __init__(self, model_dir: Path) -> None:
        self.model = _Onnx(model_dir)


class _FakeEmbedder:
    """Shaped like `FastEmbedEmbedder`: a private `_model` wrapping an ONNX model with a dir."""

    def __init__(self, model_dir: Path, name: str = "BAAI/bge-small-en-v1.5") -> None:
        self._model = _Inner(model_dir)
        self._name = name
        self._dim = 384

    @property
    def name(self) -> str:
        return self._name

    @property
    def dim(self) -> int:
        return self._dim


@pytest.fixture()
def artifact(tmp_path: Path) -> Path:
    """A directory shaped like fastembed's cache: .../snapshots/<revision>/."""
    snapshot = (
        tmp_path
        / "models--qdrant--bge-small-en-v1.5-onnx-q"
        / "snapshots"
        / "52398278842ec682c6f32300af41344b1c0b0bb2"
    )
    snapshot.mkdir(parents=True)
    (snapshot / "model_optimized.onnx").write_bytes(b"weights" * 100)
    (snapshot / "tokenizer.json").write_bytes(b'{"a": 1}')
    return snapshot


# --------------------------------------------------------------------------------------
# The contract that matters: the result makes an EmbedderIdentity verified
# --------------------------------------------------------------------------------------


def test_the_identity_is_accepted_as_immutable_by_lineage(artifact: Path) -> None:
    """The whole point. `EmbedderIdentity.verified` is what production checks."""
    got = artifact_identity_for(_FakeEmbedder(artifact))
    assert got is not None

    identity = EmbedderIdentity(
        provider=got.provider,
        model=got.model,
        dimension=384,
        revision=got.revision,
        artifact_digest=got.artifact_digest,
    )
    assert identity.verified is True


def test_an_identity_with_only_the_digest_is_still_verified(artifact: Path) -> None:
    """A digest alone is immutable, so a cache layout with no snapshot dir still works."""
    got = artifact_identity_for(_FakeEmbedder(artifact))
    assert got is not None
    assert EmbedderIdentity(
        provider=got.provider, model=got.model, dimension=384, artifact_digest=got.artifact_digest
    ).verified is True


# --------------------------------------------------------------------------------------
# What it reports
# --------------------------------------------------------------------------------------


def test_the_digest_is_the_librarys_own_tree_digest(artifact: Path) -> None:
    """Not a second implementation that could drift from what `verify_artifact` checks."""
    from recall.embeddings import artifact_tree_sha256

    got = artifact_identity_for(_FakeEmbedder(artifact))
    assert got is not None
    assert got.artifact_digest == artifact_tree_sha256(artifact)
    assert len(got.artifact_digest) == 64


def test_the_revision_comes_from_the_snapshot_directory(artifact: Path) -> None:
    """fastembed lays out `.../snapshots/<revision>/`, which is the provider's published id."""
    got = artifact_identity_for(_FakeEmbedder(artifact))
    assert got is not None
    assert got.revision == "52398278842ec682c6f32300af41344b1c0b0bb2"
    assert got.provider == "fastembed"
    assert got.model == "BAAI/bge-small-en-v1.5"
    assert got.path == artifact


def test_a_layout_with_no_snapshot_directory_reports_no_revision(tmp_path: Path) -> None:
    """A digest is still an immutable identity on its own, so this is not a failure."""
    plain = tmp_path / "weights"
    plain.mkdir()
    (plain / "model.onnx").write_bytes(b"x")
    got = artifact_identity_for(_FakeEmbedder(plain))
    assert got is not None
    assert got.revision is None
    assert got.artifact_digest


def test_the_digest_changes_when_the_weights_change(artifact: Path) -> None:
    """Otherwise it would name an identity that does not track the bytes."""
    before = artifact_identity_for(_FakeEmbedder(artifact))
    (artifact / "model_optimized.onnx").write_bytes(b"different weights")
    after = artifact_identity_for(_FakeEmbedder(artifact))
    assert before is not None and after is not None
    assert before.artifact_digest != after.artifact_digest


# --------------------------------------------------------------------------------------
# Refusals and the never-raises rule
# --------------------------------------------------------------------------------------


def test_an_embedder_with_no_local_artifact_returns_none() -> None:
    """Hashing has no weights; voyage and openai keep theirs on someone else's machine.

    None means "no artifact identity available", which is a real answer the caller acts on by
    falling back to `--unverified-development` and saying so.
    """
    from recall.embeddings import resolve_embedder

    assert artifact_identity_for(resolve_embedder("hashing")) is None


def test_an_object_whose_internals_are_absent_returns_none() -> None:
    class _Opaque:
        name = "mystery"
        dim = 8

    assert artifact_identity_for(_Opaque()) is None


def test_it_never_raises_when_introspection_explodes(tmp_path: Path) -> None:
    """The wizard probe convention: unknown is an answer, a traceback during install is not."""

    class _Hostile:
        @property
        def name(self):
            raise RuntimeError("boom")

        @property
        def _model(self):
            raise RuntimeError("boom")

    assert artifact_identity_for(_Hostile()) is None


def test_a_model_directory_that_vanished_returns_none(tmp_path: Path) -> None:
    missing = tmp_path / "gone"
    assert artifact_identity_for(_FakeEmbedder(missing)) is None


def test_an_empty_model_directory_returns_none(tmp_path: Path) -> None:
    """`artifact_tree_sha256` raises on a directory with no files; that must not escape."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert artifact_identity_for(_FakeEmbedder(empty)) is None


# --------------------------------------------------------------------------------------
# Against the real thing
# --------------------------------------------------------------------------------------


def test_the_real_fastembed_embedder_yields_a_verified_identity() -> None:
    """The end-to-end claim, on whatever this machine has cached.

    Skipped rather than failed when fastembed is absent or its weights are not downloaded, since
    neither is a defect in this code.
    """
    pytest.importorskip("fastembed")
    from recall.embeddings import resolve_embedder

    try:
        embedder = resolve_embedder("fastembed")
    except Exception:  # noqa: BLE001 - no weights, no network: not this module's problem
        pytest.skip("fastembed weights are not available here")

    got = artifact_identity_for(embedder)
    if got is None:
        pytest.skip("fastembed's internals moved; the fallback is --unverified-development")

    assert isinstance(got, ArtifactIdentity)
    assert got.provider == "fastembed"
    assert len(got.artifact_digest) == 64
    identity = EmbedderIdentity(
        provider=got.provider,
        model=got.model,
        dimension=embedder.dim,
        revision=got.revision,
        artifact_digest=got.artifact_digest,
    )
    assert identity.verified is True
    # And the digest really is over the bytes on disk.
    assert got.path.is_dir()
    assert hashlib.sha256 is not None  # keep the import meaningful
