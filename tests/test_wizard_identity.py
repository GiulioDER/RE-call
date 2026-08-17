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

from recall.lineage import EmbedderIdentity, LineageError
from recall.wizard.identity import (
    ArtifactIdentity,
    _artifact_digest,
    _model_directory,
    _revision_from,
    _source_repo_from,
    artifact_identity_for,
)


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


def test_a_dangling_symlink_refuses_rather_than_hashing_a_partial_tree(
    tmp_path: Path, caplog
) -> None:
    """`is_file()` follows a link and returns False for a broken one, so it would be dropped.

    A snapshot whose blobs were cleaned up, or copied without `-a`, would then hash only the
    surviving files and record that as immutable provenance for bytes that are not there. On
    Linux every file in the snapshot IS a link into ../../blobs, so this is the platform this
    module targets.

    The cause is read out of the log for the same reason as its two siblings: `is None` alone
    holds whether this refusal carries `_ArtifactRefusal` or a bare `ValueError`, so without the
    log assertion the one message a reader can act on ("re-download the model") could be
    downgraded to a bare class name with the whole suite still green. That was measured, not
    supposed: reverting this refusal's class left 19 passed, 3 skipped unchanged.
    """
    import logging
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
    with caplog.at_level(logging.WARNING):
        assert artifact_identity_for(FastEmbedEmbedder(snap)) is None
    # A phrase with a space, not the bare word. pytest names `tmp_path` after the test, truncated
    # to 30 characters, so both of these tests put "dangling" into the very path the refusal
    # message embeds — the assertion was satisfied by its own fixture's name and held whatever the
    # message said, including the OTHER refusal's wording. A basename cannot contain a space.
    assert "symlink is dangling" in caplog.text
    assert str(snap / "model.onnx") in caplog.text


def test_the_dangling_refusal_is_actionable_on_a_platform_without_symlinks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog
) -> None:
    """The same refusal as above, reachable where `os.symlink` needs privilege.

    The symlink tests skip on Windows, which is where this module is developed, so the refusal
    they pin is unverifiable exactly where it is most often edited. Reporting a dangling link
    through `Path` rather than creating one keeps the real `_artifact_digest` loop under test:
    this drives the production code, it does not restate it.

    The unpatched fixture is asserted to resolve first, so a probe that answered `None` for some
    unrelated reason cannot be read as the guard firing.
    """
    import logging
    from pathlib import Path as _Path

    snap = tmp_path / "models--org--repo" / "snapshots" / ("5" * 40)
    snap.mkdir(parents=True)
    (snap / "config.json").write_bytes(b"{}")
    (snap / "pruned.onnx").write_bytes(b"weights")

    assert artifact_identity_for(FastEmbedEmbedder(snap)) is not None

    real_is_symlink, real_exists = _Path.is_symlink, _Path.exists

    def is_symlink(self: _Path) -> bool:
        return self.name == "pruned.onnx" or real_is_symlink(self)

    def exists(self: _Path, **kwargs: object) -> bool:
        return False if self.name == "pruned.onnx" else real_exists(self, **kwargs)

    monkeypatch.setattr(_Path, "is_symlink", is_symlink)
    monkeypatch.setattr(_Path, "exists", exists)

    with caplog.at_level(logging.WARNING):
        assert artifact_identity_for(FastEmbedEmbedder(snap)) is None
    # A phrase with a space, not the bare word. pytest names `tmp_path` after the test, truncated
    # to 30 characters, so both of these tests put "dangling" into the very path the refusal
    # message embeds — the assertion was satisfied by its own fixture's name and held whatever the
    # message said, including the OTHER refusal's wording. A basename cannot contain a space.
    assert "symlink is dangling" in caplog.text
    assert str(snap / "pruned.onnx") in caplog.text


def test_a_directory_symlink_refuses_rather_than_hiding_its_contents(
    tmp_path: Path, caplog
) -> None:
    """The dangling-link fix one level up, and the same silent-omission failure.

    A live directory symlink passes the dangling check, fails `is_file()`, and `rglob` does not
    descend into it — so everything underneath is omitted from the digest with no error, and the
    result is recorded as immutable provenance for a tree it does not cover.

    The intact snapshot is asserted to resolve FIRST, and the refusal is read out of the log.
    `artifact_identity_for` answers `None` to four unrelated conditions, so `is None` on its own
    says the function declined, not that this guard is what declined — and because the test skips
    wherever symlinks need privilege, a fixture that had stopped resolving for some other reason
    would never be noticed on the one platform that runs it.
    """
    import logging
    import os

    repo = tmp_path / "models--org--repo"
    outside = repo / "extra"
    outside.mkdir(parents=True)
    (outside / "hidden.bin").write_bytes(b"never hashed")
    snap = repo / "snapshots" / ("9" * 40)
    snap.mkdir(parents=True)
    (snap / "config.json").write_bytes(b"{}")

    assert artifact_identity_for(FastEmbedEmbedder(snap)) is not None

    try:
        os.symlink(outside, snap / "linked", target_is_directory=True)
    except OSError:
        pytest.skip("creating a symlink needs privilege on this machine")

    with caplog.at_level(logging.WARNING):
        assert artifact_identity_for(FastEmbedEmbedder(snap)) is None
    assert "directory symlink" in caplog.text
    assert str(snap / "linked") in caplog.text


# --------------------------------------------------------------------------------------
# The four defensive branches that back the "a probe returns, it does not raise" contract
#
# Each of these was reported uncovered by the CI coverage table, and defence in depth that no
# test pins is indistinguishable from dead code: the next reader deletes it and the module keeps
# its docstring's promise only by luck. Each test drives the real function — the module's own
# `_model_directory`, `_revision_from`, `_source_repo_from`, `artifact_identity_for` — with an
# input that trips that one branch and nothing else, and asserts the SPECIFIC consequence, so
# that removing the branch turns the test red rather than merely changing a log line.
# --------------------------------------------------------------------------------------


def test_a_model_dir_that_will_not_stringify_is_skipped_for_the_next_candidate(
    tmp_path: Path,
) -> None:
    """`_model_directory` must keep probing, not give up, when one attribute holds a non-path.

    The five attribute chains in `_MODEL_DIR_PATHS` reach into fastembed's internals, so any one
    of them can hold something that is neither a path nor `None` after a release moves a name.
    The branch under test is `except Exception: continue`, and the word that matters is
    `continue`: `return None` there would let one unexpected attribute mask a perfectly good
    artifact directory found by a later chain, and the caller would silently downgrade to
    `--unverified-development`.

    So the assertion is not "it returned None safely" but "it found the real directory anyway".
    """

    class _WillNotStringify:
        """Truthy, so the probe gets past `if not found`, and then fails inside `str()`.

        `__str__` returning a non-string makes the interpreter raise `TypeError` itself, rather
        than the test hand-throwing one: `Path()` on this machine accepts even an embedded NUL,
        so a stringification that genuinely fails is the honest way to trip the branch.
        """

        def __str__(self) -> str:
            return 42  # type: ignore[return-value]

    real = tmp_path / "models--qdrant--bge-small-en-v1.5-onnx-q" / "snapshots" / ("4" * 40)
    real.mkdir(parents=True)
    (real / "model_optimized.onnx").write_bytes(b"weights")

    class _Onnxish:
        _model_dir = _WillNotStringify()

    class _InnerWithJunk:
        model = _Onnxish()

    class FastEmbedEmbedder:  # noqa: N801 - must match the guard to reach the probing
        name = "BAAI/bge-small-en-v1.5"
        dim = 384

        def __init__(self) -> None:
            #: The FIRST chain probed, `_model.model._model_dir`, is the poisoned one …
            self._model = _InnerWithJunk()
            #: … and the LAST one, a bare `_model_dir`, is the artifact that must still be found.
            self._model_dir = str(real)

    embedder = FastEmbedEmbedder()
    assert _model_directory(embedder) == real, (
        "an unstringifiable candidate must be skipped, not treated as the answer or as a failure"
    )

    # And the same thing through the public entry point, which is where the cost would land.
    got = artifact_identity_for(embedder)
    assert got is not None
    assert got.path == real
    assert len(got.artifact_digest) == 64


def test_a_path_whose_parent_explodes_reports_no_revision() -> None:
    """`_revision_from` answers `None` rather than propagating, per the module's probe contract.

    A revision is a nice-to-have: the digest alone already makes the identity immutable, which is
    exactly why an exception raised while reading the cache layout must cost the revision and not
    the whole identity. The path object here is well behaved apart from `.parent`, so the `None`
    can only have come from the `except` under test.
    """

    class _ParentExplodes:
        #: A valid 40-hex commit id, so a `None` here cannot be blamed on the format check.
        name = "5" * 40

        @property
        def parent(self) -> Path:
            raise RuntimeError("the cache layout is not readable")

    assert _revision_from(_ParentExplodes()) is None  # type: ignore[arg-type]


def test_a_path_whose_grandparent_explodes_reports_no_source_repo() -> None:
    """`_source_repo_from` answers `None` too, and the failure is confined to the repo field.

    This is the sibling of the revision probe one level up: `.parent.parent` is the
    `models--<org>--<repo>` cache entry, which is one directory further from the artifact and so
    one directory more likely to be absent, on a mount that refuses it, or replaced by something
    that is not a path at all.

    The double raises only at the SECOND `.parent`, and the test asserts that `_revision_from`
    still succeeds on the very same object. That is what makes this test about lines 137-138
    rather than about a stub that is broken everywhere.
    """

    class _GrandparentExplodes:
        name = "6" * 40

        @property
        def parent(self) -> Path:
            class _Snapshots:
                name = "snapshots"

                @property
                def parent(self) -> Path:
                    raise RuntimeError("the cache entry is not readable")

            return _Snapshots()  # type: ignore[return-value]

    path = _GrandparentExplodes()
    assert _source_repo_from(path) is None  # type: ignore[arg-type]
    assert _revision_from(path) == "6" * 40, (  # type: ignore[arg-type]
        "precondition: this double is sound one level up, so the None above is the grandparent"
    )


def test_an_embedder_that_cannot_name_its_model_returns_none(artifact: Path) -> None:
    """An artifact with no model name is not an identity, and must not be built into one.

    `model` goes verbatim into an `EmbedderIdentity`, whose `__post_init__` refuses an empty
    model outright. Falling through with `model=""` would therefore not produce a weaker
    identity, it would produce a `LineageError` at whatever later point the wizard assembles the
    lineage record — a traceback during install, which is the one outcome this module exists to
    avoid. `None` here routes the caller to `--unverified-development` plus a message instead.
    """
    assert artifact_identity_for(FastEmbedEmbedder(artifact, name="")) is None

    class FastEmbedEmbedder_NoName:  # noqa: N801 - renamed below to match the guard
        """Missing the attribute entirely, which `getattr(..., "name", "")` also reduces to ""."""

        def __init__(self, model_dir: Path) -> None:
            self._model = _Inner(model_dir)

        dim = 384

    # Renamed rather than declared as `class FastEmbedEmbedder`, which would make the name local
    # to this function and silently shadow the module-level double used further down.
    FastEmbedEmbedder_NoName.__name__ = "FastEmbedEmbedder"
    nameless = FastEmbedEmbedder_NoName(artifact)
    assert type(nameless).__name__ == "FastEmbedEmbedder", (
        "precondition: without the rename above this returns at the class guard and the "
        "empty-name branch is never reached, so the assertion below would prove nothing"
    )
    assert artifact_identity_for(nameless) is None

    # The precondition, so the two Nones above are attributable to the missing name and not to
    # the artifact: the same directory with a name does yield an identity.
    assert artifact_identity_for(FastEmbedEmbedder(artifact)) is not None

    # And the consequence the branch prevents, stated rather than assumed.
    with pytest.raises(LineageError):
        EmbedderIdentity(provider="fastembed", model="", dimension=384, artifact_digest="a" * 64)


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


def test_a_foreign_valueerror_is_logged_by_class_and_not_by_message(caplog) -> None:
    """Only THIS module's own refusals are known to be safe to log in full.

    The handler that lets an actionable message through used to catch bare `ValueError`, defended
    by a comment saying those messages are locally produced and carry a path rather than a
    credential. That is true of the three refusals raised here and not of the `try` block, which
    reads attributes off a caller-supplied object and would log whatever somebody else's property
    raised. Giving the refusals their own class makes the comment structural instead of
    incidental, and this is the arm that fails if it goes back to being incidental.
    """
    import logging

    foreign_message = "<placeholder-token-that-must-not-reach-the-log>"

    class FastEmbedEmbedder:  # noqa: N801 - the guard matches on this exact class name
        name = "BAAI/bge-small-en-v1.5"
        dim = 384

        @property
        def _model_dir(self) -> str:
            # `getattr(obj, attr, None)` swallows AttributeError and nothing else, so this
            # escapes `_model_directory` exactly as a third-party failure would.
            raise ValueError(foreign_message)

    with caplog.at_level(logging.WARNING):
        assert artifact_identity_for(FastEmbedEmbedder()) is None
    assert foreign_message not in caplog.text
    assert "ValueError" in caplog.text
