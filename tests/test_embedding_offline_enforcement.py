"""Model artifacts are local, verified, and never fetched at runtime.

Runtime model downloads are on this program's standing out-of-scope list, and the reason is not
bandwidth: a weight file fetched at startup is an unpinned dependency on a third party, reachable
from a production workload that is supposed to have no outbound network at all. The control has to
be that startup FAILS without a provisioned, checksum-matching tree; a warning would be a
detector, not a guard.

What the socket test proves and what it does not: sockets are blocked at the standard entry
points and our whole startup path runs to completion, so nothing WE do reaches the network. The
model loader itself is stubbed here, because `fastembed` is an optional extra that CI's `test` job
deliberately does not install, and a test that skipped in CI would prove nothing at all. The real
loader was exercised against the provisioned artifact on VPS2; see
`docs/archive/ENTERPRISE_PROGRAM_STATUS.md` for that measurement.
"""
from __future__ import annotations

import socket
import sys
import types
from pathlib import Path

import pytest

from recall.embedding_registry import registered_profile
from recall.embeddings import artifact_tree_sha256, verify_artifact
from recall_mcp.service import make_embedder


class _StubTextEmbedding:
    instances: list["_StubTextEmbedding"] = []

    def __init__(self, **kwargs: object) -> None:
        self.kwargs = kwargs
        _StubTextEmbedding.instances.append(self)

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [[0.1] + [0.0] * 383 for _ in texts]

    query_embed = embed
    passage_embed = embed


@pytest.fixture
def stub_fastembed(monkeypatch: pytest.MonkeyPatch) -> type[_StubTextEmbedding]:
    _StubTextEmbedding.instances = []
    module = types.ModuleType("fastembed")
    module.TextEmbedding = _StubTextEmbedding  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "fastembed", module)
    return _StubTextEmbedding


@pytest.fixture
def provisioned(tmp_path: Path) -> tuple[Path, str]:
    root = tmp_path / "models" / "bge-small"
    root.mkdir(parents=True)
    (root / "model.onnx").write_bytes(b"weights")
    (root / "tokenizer.json").write_bytes(b"{}")
    return root, artifact_tree_sha256(root)


def _refuse(*args: object, **kwargs: object):
    raise AssertionError("startup attempted a network connection")


class _NoNetworkSocket(socket.socket):
    """A socket that cannot connect.

    Deliberately still a CLASS, and still a `socket.socket` subclass. Replacing
    `socket.socket` with a function looks stricter and is not usable: `ssl` builds
    `class SSLSocket(socket)` at import time, so the swap turns any later `import ssl` into a
    `TypeError` and the test then proves that imports break, not that nothing dialled out.
    That is not hypothetical, it is what happened when this check was first run against the
    real fastembed on VPS2, whose import chain reaches `requests` and therefore `ssl`.
    """

    def connect(self, *args: object, **kwargs: object):  # type: ignore[override]
        _refuse()

    def connect_ex(self, *args: object, **kwargs: object):  # type: ignore[override]
        _refuse()


@pytest.fixture
def no_network(monkeypatch: pytest.MonkeyPatch) -> None:
    """Refuse every standard way a Python process opens a connection."""
    monkeypatch.setattr(socket, "socket", _NoNetworkSocket)
    monkeypatch.setattr(socket, "create_connection", _refuse)
    monkeypatch.setattr(socket, "getaddrinfo", _refuse)


def test_the_socket_block_itself_fires(no_network) -> None:
    """The guard's own positive control. Without it, a green offline test could mean the block
    was never installed rather than that nothing reached for the network."""
    with pytest.raises(AssertionError, match="attempted a network connection"):
        socket.create_connection(("example.invalid", 443))
    with pytest.raises(AssertionError, match="attempted a network connection"):
        socket.getaddrinfo("example.invalid", 443)
    with socket.socket() as opened:
        with pytest.raises(AssertionError, match="attempted a network connection"):
            opened.connect(("127.0.0.1", 9))


def test_offline_startup_succeeds_with_every_socket_blocked(
    stub_fastembed, provisioned, no_network
) -> None:
    root, digest = provisioned
    entry = registered_profile("bge-small-asymmetric-v1")

    embedder = entry.build(artifact_path=root, artifact_digest=digest)

    assert embedder.dim == 384
    kwargs = stub_fastembed.instances[-1].kwargs
    assert kwargs["local_files_only"] is True
    assert kwargs["cache_dir"] == str(root.resolve())
    assert kwargs["model_name"] == "BAAI/bge-small-en-v1.5"


def test_offline_startup_through_make_embedder_needs_no_network(
    stub_fastembed, provisioned, no_network
) -> None:
    root, digest = provisioned
    embedder = make_embedder(
        "fastembed",
        {
            "RECALL_EMBED_PROFILE": "bge-small-context-document-v1",
            "RECALL_MODEL_CACHE": str(root),
            "RECALL_MODEL_SHA256": digest,
        },
    )
    assert embedder.profile.profile_id == "bge-small-context-document-v1"
    assert embedder.profile.context_version == "context-document-v1"


def test_a_missing_artifact_refuses_to_start(stub_fastembed, tmp_path: Path) -> None:
    entry = registered_profile("bge-small-symmetric-v1")
    with pytest.raises(FileNotFoundError):
        entry.build(artifact_path=tmp_path / "not-provisioned", artifact_digest="0" * 64)


def test_a_checksum_mismatch_refuses_to_start(stub_fastembed, provisioned) -> None:
    root, digest = provisioned
    entry = registered_profile("bge-small-symmetric-v1")

    assert entry.build(artifact_path=root, artifact_digest=digest) is not None

    (root / "model.onnx").write_bytes(b"tampered")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        entry.build(artifact_path=root, artifact_digest=digest)


def test_an_added_file_changes_the_digest(provisioned) -> None:
    """A tree hash that ignored new files would pass a tampered artifact that only ADDS."""
    root, digest = provisioned
    (root / "extra.bin").write_bytes(b"")
    assert artifact_tree_sha256(root) != digest


def test_a_malformed_digest_is_refused_before_any_io(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="64 hexadecimal"):
        verify_artifact(tmp_path / "does-not-exist", "not-a-digest")


def test_an_empty_artifact_tree_is_refused(tmp_path: Path) -> None:
    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="no files"):
        artifact_tree_sha256(empty)


@pytest.mark.skipif(sys.platform == "win32", reason="directory symlinks need privilege on Windows")
def test_a_symlink_escaping_the_artifact_root_is_refused(tmp_path: Path) -> None:
    """A security control that has never been shown to fire is a hypothesis.

    Without this, a provisioned tree could hash a file outside itself, so the digest would
    describe an artifact the operator did not provision.
    """
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"someone else's weights")
    root = tmp_path / "artifact"
    root.mkdir()
    (root / "model.onnx").write_bytes(b"weights")
    (root / "escape.bin").symlink_to(outside)

    with pytest.raises(ValueError, match="symlink escapes"):
        artifact_tree_sha256(root)


def test_the_offline_path_refuses_without_a_provisioned_cache(stub_fastembed) -> None:
    from recall.embeddings import FastEmbedEmbedder

    with pytest.raises(ValueError, match="provisioned cache_dir"):
        FastEmbedEmbedder(require_local=True)


def test_the_offline_path_refuses_without_a_digest(stub_fastembed, provisioned) -> None:
    from recall.embeddings import FastEmbedEmbedder

    root, _ = provisioned
    with pytest.raises(ValueError, match="artifact_sha256"):
        FastEmbedEmbedder(require_local=True, cache_dir=root)


def test_make_embedder_refuses_a_registered_profile_without_its_artifact_env(
    stub_fastembed, provisioned
) -> None:
    root, digest = provisioned
    for env in (
        {"RECALL_EMBED_PROFILE": "bge-small-symmetric-v1", "RECALL_MODEL_SHA256": digest},
        {"RECALL_EMBED_PROFILE": "bge-small-symmetric-v1", "RECALL_MODEL_CACHE": str(root)},
    ):
        with pytest.raises(ValueError, match="RECALL_MODEL_CACHE and RECALL_MODEL_SHA256"):
            make_embedder("fastembed", env)


def test_make_embedder_refuses_an_unregistered_profile() -> None:
    with pytest.raises(ValueError, match="unknown RECALL_EMBED_PROFILE"):
        make_embedder("fastembed", {"RECALL_EMBED_PROFILE": "bge-small-context-paragraph-v1"})


def test_make_embedder_refuses_a_profile_on_the_wrong_backend() -> None:
    with pytest.raises(ValueError, match="RECALL_EMBEDDER=fastembed"):
        make_embedder("hashing", {"RECALL_EMBED_PROFILE": "bge-small-symmetric-v1"})


def test_the_qwen_profile_reads_its_own_artifact_path_variable(provisioned) -> None:
    """`RECALL_MODEL_CACHE` is the fastembed tree. The Qwen artifact has its own variable.

    The digest is supplied and `RECALL_MODEL_CACHE` points at a REAL provisioned tree, so the
    only thing still missing is `RECALL_QWEN_MODEL_PATH`. A `make_embedder` that read the
    fastembed variable for this profile would find a usable path and get further, which is what
    makes this assertion discriminate: an earlier version of this test left the digest unset, so
    it refused for the wrong reason and passed under a mutation that hardcoded the variable.
    """
    root, digest = provisioned
    entry = registered_profile("qwen3-embedding-0.6b-384-v1")
    assert entry.artifact_path_env == "RECALL_QWEN_MODEL_PATH"
    with pytest.raises(ValueError, match="requires RECALL_QWEN_MODEL_PATH"):
        make_embedder(
            "fastembed",
            {
                "RECALL_EMBED_PROFILE": "qwen3-embedding-0.6b-384-v1",
                "RECALL_MODEL_CACHE": str(root),
                "RECALL_MODEL_SHA256": digest,
            },
        )


def test_the_shadow_path_reads_the_shadow_artifact_variables(
    stub_fastembed, provisioned, tmp_path: Path
) -> None:
    """A shadow generation may be built on a DIFFERENT artifact at the same profile ID.

    That is the exact aliasing shape the cache key now covers, and it is reachable only through
    `make_profile_embedder(shadow=True)`, which had no test at all. If the remapping silently
    fell through to the active artifact, the shadow would be built from the active weights and
    the cutover would compare a generation with itself.
    """
    from recall.embeddings import artifact_tree_sha256
    from recall_mcp.service import make_profile_embedder

    active, active_digest = provisioned
    shadow = tmp_path / "models" / "bge-small-next"
    shadow.mkdir(parents=True)
    (shadow / "model.onnx").write_bytes(b"different weights")
    shadow_digest = artifact_tree_sha256(shadow)
    assert shadow_digest != active_digest

    env = {
        "RECALL_MODEL_CACHE": str(active),
        "RECALL_MODEL_SHA256": active_digest,
        "RECALL_SHADOW_MODEL_CACHE": str(shadow),
        "RECALL_SHADOW_MODEL_SHA256": shadow_digest,
    }

    built = make_profile_embedder("bge-small-asymmetric-v1", shadow=True, env=env)
    assert built.profile.artifact_digest == shadow_digest

    active_built = make_profile_embedder("bge-small-asymmetric-v1", env=env)
    assert active_built.profile.artifact_digest == active_digest
    assert active_built.profile.fingerprint() != built.profile.fingerprint()


# ----------------------------------------------------------------------------------------------
# Artifact digests: the provenance a production ingest now rests on
# ----------------------------------------------------------------------------------------------


def test_an_embedder_with_no_weights_has_no_digest_and_says_so() -> None:
    """⛔ **None is a real answer and must stay one.**

    `HashingEmbedder` is defined by code, not weights: there is nothing on disk to hash.
    Manufacturing a digest for it — over the model name, say — would turn an honest "unverified"
    into a claim of provenance that no bytes back, which is worse than the refusal it bypasses. A
    production ingest with this embedder is SUPPOSED to be refused.
    """
    from recall.embeddings import HashingEmbedder, embedder_artifact_digest, embedder_artifact_path

    embedder = HashingEmbedder(dim=64)

    assert embedder_artifact_path(embedder) is None
    assert embedder_artifact_digest(embedder) is None


def test_a_digest_is_over_the_model_directory_not_the_shared_cache(tmp_path: Path) -> None:
    """⚠️ The scope is what makes the digest mean anything.

    Measured on this machine: the fastembed `cache_dir` held 45 files and 1.5 GB across SEVERAL
    models, so a digest over it would change whenever an unrelated model was downloaded, and would
    identify nothing. `_model_dir` is one model's own snapshot — 5 files, 67 MB.

    Stubbed rather than downloaded, so this runs offline and pins the SHAPE reached for rather than
    one vendor's directory layout.
    """
    from recall.embeddings import artifact_tree_sha256, embedder_artifact_digest

    model_dir = tmp_path / "snapshots" / "abc123"
    model_dir.mkdir(parents=True)
    (model_dir / "model.onnx").write_bytes(b"weights")
    (model_dir / "tokenizer.json").write_text("{}", encoding="utf-8")
    sibling = tmp_path / "snapshots" / "other-model"
    sibling.mkdir()
    (sibling / "model.onnx").write_bytes(b"a different model entirely")

    # ⚠️ **The stub carries BOTH attributes, because the real object does.** A first version
    # defined only `_model_dir`, so a mutation that preferred `cache_dir` was invisible: the stub
    # had no `cache_dir` to prefer, `getattr` fell through, and the test passed against code that
    # would hash 1.5 GB of unrelated models on a real machine. Caught by mutation, not by reading.
    class _Inner:
        cache_dir = str(tmp_path / "snapshots")
        _model_dir = model_dir

    class _Model:
        model = _Inner()

    class _Embedder:
        _model = _Model()

    digest = embedder_artifact_digest(_Embedder())

    assert digest == artifact_tree_sha256(model_dir)
    assert digest != artifact_tree_sha256(tmp_path / "snapshots"), (
        "hashing the whole cache would change when an unrelated model appears"
    )


def test_an_unreachable_artifact_path_yields_none_rather_than_raising() -> None:
    """This reaches into another library's internals, which change between versions.

    A wrong answer is worse than no answer, because it feeds an identity claiming to be verified.
    So every failure mode resolves to None and the caller falls back to an honest unverified
    identity.
    """
    from recall.embeddings import embedder_artifact_digest

    class _Model:
        class model:  # noqa: N801 - mimicking an attribute chain, not naming a class
            _model_dir = "/a/path/that/does/not/exist/anywhere"

    class _Gone:
        _model = _Model()

    assert embedder_artifact_digest(_Gone()) is None
    assert embedder_artifact_digest(object()) is None, "an embedder of another shape must not raise"
