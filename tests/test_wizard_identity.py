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
from recall.wizard.identity import ArtifactIdentity, _artifact_digest, artifact_identity_for


class _Onnx:
    def __init__(self, model_dir: Path) -> None:
        self._model_dir = str(model_dir)


class _Inner:
    def __init__(self, model_dir: Path) -> None:
        self.model = _Onnx(model_dir)


class FastEmbedEmbedder:  # noqa: N801 - the guard matches on this exact class name
    """Shaped like the real `FastEmbedEmbedder`, and NAMED like it.

    `artifact_identity_for` guards on the class name before writing `provider="fastembed"`
    into a lineage record, so a fake that is merely shaped like one is correctly refused.
    """

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
    got = artifact_identity_for(FastEmbedEmbedder(artifact))
    assert got is not None

    identity = EmbedderIdentity(
        provider=got.provider,
        model=got.model,
        dimension=384,
        revision=got.lineage_revision,
        artifact_digest=got.artifact_digest,
    )
    assert identity.verified is True


def test_an_identity_with_only_the_digest_is_still_verified(artifact: Path) -> None:
    """A digest alone is immutable, so a cache layout with no snapshot dir still works."""
    got = artifact_identity_for(FastEmbedEmbedder(artifact))
    assert got is not None
    assert EmbedderIdentity(
        provider=got.provider, model=got.model, dimension=384, artifact_digest=got.artifact_digest
    ).verified is True


# --------------------------------------------------------------------------------------
# What it reports
# --------------------------------------------------------------------------------------


def test_the_digest_is_this_modules_own(artifact: Path) -> None:
    """Deliberately not `artifact_tree_sha256`: that one refuses the symlinks an HF cache is
    made of, and sorts Path objects, which is case-folded on Windows and not on POSIX."""
    from recall.wizard.identity import _artifact_digest

    got = artifact_identity_for(FastEmbedEmbedder(artifact))
    assert got is not None
    assert got.artifact_digest == _artifact_digest(artifact)
    assert len(got.artifact_digest) == 64


def test_the_revision_comes_from_the_snapshot_directory(artifact: Path) -> None:
    """fastembed lays out `.../snapshots/<revision>/`, which is the provider's published id."""
    got = artifact_identity_for(FastEmbedEmbedder(artifact))
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
    got = artifact_identity_for(FastEmbedEmbedder(plain))
    assert got is not None
    assert got.revision is None
    assert got.artifact_digest


def test_the_digest_changes_when_the_weights_change(artifact: Path) -> None:
    """Otherwise it would name an identity that does not track the bytes."""
    before = artifact_identity_for(FastEmbedEmbedder(artifact))
    (artifact / "model_optimized.onnx").write_bytes(b"different weights")
    after = artifact_identity_for(FastEmbedEmbedder(artifact))
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
    """Named FastEmbedEmbedder ON PURPOSE, so it gets past the class guard.

    This test is about "internals absent", not "wrong class". When the guard was added as the
    first statement it started returning at the guard, and the test passed without ever reaching
    the introspection it exists to exercise.
    """

    class FastEmbedEmbedder:  # noqa: N801 - must match the guard to reach the probing
        name = "mystery"
        dim = 8

    assert artifact_identity_for(FastEmbedEmbedder()) is None


def test_it_never_raises_when_introspection_explodes(tmp_path: Path) -> None:
    """The wizard probe convention: unknown is an answer, a traceback during install is not.

    Named FastEmbedEmbedder so the class guard lets it through to the exploding properties. With
    the plain name it returned at the guard, and this — the only test of the module's stated
    "a probe returns, it does not raise" contract — proved nothing.
    """

    class FastEmbedEmbedder:  # noqa: N801 - must match the guard to reach the probing
        @property
        def name(self):
            raise RuntimeError("boom")

        @property
        def _model(self):
            raise RuntimeError("boom")

    assert artifact_identity_for(FastEmbedEmbedder()) is None


def test_a_model_directory_that_vanished_returns_none(tmp_path: Path) -> None:
    missing = tmp_path / "gone"
    assert artifact_identity_for(FastEmbedEmbedder(missing)) is None


def test_an_empty_model_directory_returns_none(tmp_path: Path) -> None:
    """The digest raises on a directory with no files; that must not escape."""
    empty = tmp_path / "empty"
    empty.mkdir()
    assert artifact_identity_for(FastEmbedEmbedder(empty)) is None


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
    # NOT a skip. An earlier version skipped here "because fastembed's internals may have moved",
    # which silently absorbed the real failure: on Linux the HF cache is symlinks into ../../blobs
    # and the digest refused them, so this returned None on every deployment platform while the
    # suite stayed green. If a resolvable fastembed embedder yields no identity, that is the
    # defect this test exists to catch.
    assert got is not None, (
        "a working fastembed embedder must yield an artifact identity; None here means the "
        "cache layout or fastembed's internals changed and the module needs updating"
    )

    assert isinstance(got, ArtifactIdentity)
    assert got.provider == "fastembed"
    assert len(got.artifact_digest) == 64
    identity = EmbedderIdentity(
        provider=got.provider,
        model=got.model,
        dimension=embedder.dim,
        revision=got.lineage_revision,
        artifact_digest=got.artifact_digest,
    )
    assert identity.verified is True
    # And the digest really is over the bytes on disk.
    assert got.path.is_dir()
    assert hashlib.sha256 is not None  # keep the import meaningful


# --------------------------------------------------------------------------------------
# The three structural findings, each measured before it was fixed
# --------------------------------------------------------------------------------------


def test_a_snapshot_of_symlinks_into_blobs_still_digests(tmp_path: Path) -> None:
    """The HF cache layout on Linux and macOS, which is where recall is deployed and CI'd.

    `huggingface_hub` fills `snapshots/<rev>/` with symlinks into `../../blobs/<sha>`; Windows
    falls back to real copies, which is why the first version of this module measured fine locally
    and returned None on every Linux install. `artifact_tree_sha256` refuses a symlink leaving its
    root, and `artifact_identity_for` swallowed that into a None.
    """
    import os

    repo = tmp_path / "models--qdrant--bge-small-en-v1.5-onnx-q"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    (blobs / "deadbeef").write_bytes(b"weights")
    snap = repo / "snapshots" / ("5" * 40)
    snap.mkdir(parents=True)
    (snap / "config.json").write_bytes(b"{}")
    try:
        os.symlink(blobs / "deadbeef", snap / "model.onnx")
    except OSError:
        pytest.skip("creating a symlink needs privilege on this machine")

    got = artifact_identity_for(FastEmbedEmbedder(snap))
    assert got is not None, "the HF cache layout must be digestible"
    assert len(got.artifact_digest) == 64


def test_the_digest_ordering_is_the_same_on_every_platform(tmp_path: Path) -> None:
    """Sorting Path objects is case-folded on Windows and case-sensitive on POSIX, so identical
    weights hashed to two different digests depending on the machine."""
    from pathlib import PurePosixPath, PureWindowsPath

    import hashlib as _h

    names = ["README.md", "config.json", "Tokenizer.json", "model.onnx"]
    win = [p.name for p in sorted(PureWindowsPath("/r") / n for n in names)]
    pos = [p.name for p in sorted(PurePosixPath("/r") / n for n in names)]
    assert win != pos, "precondition: these orders really do differ by platform"

    d = tmp_path / "snap"
    d.mkdir()
    for n in names:
        (d / n).write_bytes(n.encode())

    # An INDEPENDENT oracle, not a re-implementation of the module's sort. An earlier version of
    # this test sorted the files itself and then only asserted the digest was 64 characters long,
    # so changing the module's ordering changed nothing the test looked at and the guard could not
    # fail. Comparing against a digest computed here from a fixed, platform-independent order is
    # what makes a different ordering in the module show up.
    expected = _h.sha256()
    for n in sorted(names):  # plain str sort == the POSIX-relative-path order the module uses
        expected.update(n.encode("utf-8"))
        expected.update(b"\x00")
        expected.update((d / n).read_bytes())

    assert _artifact_digest(d) == expected.hexdigest()


def test_a_revision_from_another_repo_is_not_offered_to_lineage(artifact: Path) -> None:
    """fastembed serves BAAI/... out of qdrant/..., so the snapshot SHA is a qdrant commit.

    Recording it under the BAAI name yields a revision nobody can resolve. The raw value is still
    reported, attributed to the repo it belongs to; only `lineage_revision` withholds it.
    """
    got = artifact_identity_for(FastEmbedEmbedder(artifact))
    assert got is not None
    assert got.source_repo == "qdrant/bge-small-en-v1.5-onnx-q"
    assert got.model == "BAAI/bge-small-en-v1.5"
    assert got.revision == "5239827884" + "2ec682c6f32300af41344b1c0b0bb2"
    assert got.lineage_revision is None, "a cross-repo revision must not reach a lineage record"


def test_a_matching_repo_does_offer_its_revision(artifact: Path) -> None:
    """The other half: when the model IS the source repo, the revision is usable."""
    got = artifact_identity_for(
        FastEmbedEmbedder(artifact, name="qdrant/bge-small-en-v1.5-onnx-q")
    )
    assert got is not None
    assert got.lineage_revision == got.revision


def test_a_non_hex_snapshot_name_is_not_a_revision(tmp_path: Path) -> None:
    """A user-provisioned `.../snapshots/latest/` is not a published commit, and
    `EmbedderIdentity` applies no format check of its own."""
    snap = tmp_path / "models--org--repo" / "snapshots" / "latest"
    snap.mkdir(parents=True)
    (snap / "model.onnx").write_bytes(b"x")
    got = artifact_identity_for(FastEmbedEmbedder(snap))
    assert got is not None
    assert got.revision is None


def test_a_non_fastembed_embedder_is_not_stamped_with_fastembed_provenance(
    artifact: Path,
) -> None:
    """`provider` goes verbatim into a lineage record treated as immutable evidence, and the
    attribute chains probed are generic (`_model_dir` is not a fastembed-only name)."""

    class SentenceTransformerEmbedder:  # noqa: N801 - mirrors the real class name
        def __init__(self, model_dir: Path) -> None:
            self._model = _Inner(model_dir)

        name = "sentence-transformers/all-MiniLM-L6-v2"
        dim = 384

    assert artifact_identity_for(SentenceTransformerEmbedder(artifact)) is None


def test_a_dangling_symlink_refuses_rather_than_hashing_a_partial_tree(tmp_path: Path) -> None:
    """`is_file()` follows a link and returns False for a broken one, so it would be dropped.

    A snapshot whose blobs were cleaned up, or copied without `-a`, would then hash only the
    surviving files and record that as immutable provenance for bytes that are not there. On
    Linux every file in the snapshot IS a link into ../../blobs, so this is the platform this
    module targets.
    """
    import os

    repo = tmp_path / "models--org--repo"
    blobs = repo / "blobs"
    blobs.mkdir(parents=True)
    blob = blobs / "cafe"
    blob.write_bytes(b"weights")
    snap = repo / "snapshots" / ("7" * 40)
    snap.mkdir(parents=True)
    (snap / "config.json").write_bytes(b"{}")
    try:
        os.symlink(blob, snap / "model.onnx")
    except OSError:
        pytest.skip("creating a symlink needs privilege on this machine")

    assert artifact_identity_for(FastEmbedEmbedder(snap)) is not None
    blob.unlink()  # the cache is now incomplete
    assert artifact_identity_for(FastEmbedEmbedder(snap)) is None


def test_a_directory_symlink_refuses_rather_than_hiding_its_contents(tmp_path: Path) -> None:
    """The dangling-link fix one level up, and the same silent-omission failure.

    A live directory symlink passes the dangling check, fails `is_file()`, and `rglob` does not
    descend into it — so everything underneath is omitted from the digest with no error, and the
    result is recorded as immutable provenance for a tree it does not cover.
    """
    import os

    repo = tmp_path / "models--org--repo"
    outside = repo / "extra"
    outside.mkdir(parents=True)
    (outside / "hidden.bin").write_bytes(b"never hashed")
    snap = repo / "snapshots" / ("9" * 40)
    snap.mkdir(parents=True)
    (snap / "config.json").write_bytes(b"{}")
    try:
        os.symlink(outside, snap / "linked", target_is_directory=True)
    except OSError:
        pytest.skip("creating a symlink needs privilege on this machine")

    assert artifact_identity_for(FastEmbedEmbedder(snap)) is None


def test_an_actionable_refusal_reaches_the_log(tmp_path: Path, caplog) -> None:
    """"Your cache is incomplete" is the one failure a reader can fix, so it must not be logged
    as a bare class name indistinguishable from a fastembed internals change."""
    import logging

    empty = tmp_path / "models--org--repo" / "snapshots" / ("3" * 40)
    empty.mkdir(parents=True)
    with caplog.at_level(logging.WARNING):
        assert artifact_identity_for(FastEmbedEmbedder(empty)) is None
    assert "has no files" in caplog.text
    assert str(empty) in caplog.text
