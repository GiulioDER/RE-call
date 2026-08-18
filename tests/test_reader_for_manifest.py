"""Choosing the reader from the manifest, and refusing a manifest that needs two.

`recall/cli.py` built `S3ObjectReader.from_environment()` unconditionally, before it knew whether
the manifest named S3 objects at all. With `file://` objects now valid, the reader has to be chosen
from what the manifest actually contains.

The mixed case is the one worth being strict about. A manifest holding both `s3://` and `file://`
objects carries **two different immutability guarantees at once**: the S3 half is pinned to a
version, the local half only detectable after the fact. A generation built from it would be
described by a single lineage record that is true of one half and overstated for the other, and
nothing downstream could tell which hit came from which. Refusing is the only honest option, and it
is refused at the point the reader is chosen rather than discovered later as a read failure.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest

from recall.lineage import IndexManifestV1, ManifestObjectV1
from recall.manifest import LocalObjectReader, reader_for_manifest


def _s3_entry(key: str = "k") -> ManifestObjectV1:
    return ManifestObjectV1(f"s3://b/{key}", "v1", "text/markdown", 1, "a" * 64)


def _file_entry(tmp_path: pathlib.Path, name: str = "a.md") -> ManifestObjectV1:
    body = b"x"
    p = tmp_path / name
    p.write_bytes(body)
    digest = hashlib.sha256(body).hexdigest()
    return ManifestObjectV1(p.as_uri(), digest, "text/markdown", len(body), digest)


def _manifest(*objects: ManifestObjectV1) -> IndexManifestV1:
    return IndexManifestV1(
        schema_version=1, tenant_id="t", corpus_version="cv", objects=tuple(objects)
    )


def test_all_file_objects_choose_the_local_reader(tmp_path: pathlib.Path) -> None:
    reader = reader_for_manifest(_manifest(_file_entry(tmp_path)), local_roots=(tmp_path,))
    assert isinstance(reader, LocalObjectReader)


def test_all_s3_objects_choose_the_s3_reader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("RECALL_S3_ALLOWLIST", "b")
    sentinel = object()
    reader = reader_for_manifest(_manifest(_s3_entry()), s3_factory=lambda: sentinel)
    assert reader is sentinel


def test_a_mixed_manifest_is_refused(tmp_path: pathlib.Path) -> None:
    """Two immutability guarantees in one lineage record is a record that lies about half itself."""
    with pytest.raises(ValueError, match="mixes"):
        reader_for_manifest(
            _manifest(_s3_entry(), _file_entry(tmp_path)),
            local_roots=(tmp_path,),
            s3_factory=lambda: object(),
        )


def test_an_empty_manifest_is_refused(tmp_path: pathlib.Path) -> None:
    """Defaulting an empty manifest to either backend would pick one for no reason."""
    with pytest.raises(ValueError, match="no objects"):
        reader_for_manifest(_manifest(), local_roots=(tmp_path,))


def test_local_roots_default_to_the_environment(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("RECALL_LOCAL_ALLOWLIST", str(tmp_path))
    reader = reader_for_manifest(_manifest(_file_entry(tmp_path)))
    assert isinstance(reader, LocalObjectReader)


def test_a_local_manifest_without_an_allowlist_says_what_to_set(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RECALL_LOCAL_ALLOWLIST", raising=False)
    with pytest.raises(ValueError, match="RECALL_LOCAL_ALLOWLIST"):
        reader_for_manifest(_manifest(_file_entry(tmp_path)))


def test_the_s3_reader_is_not_built_for_a_local_manifest(
    tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building it needs boto3 and an allowlist, neither of which a local user has.

    This is the actual bug: the CLI built the S3 reader before knowing the manifest was local, so a
    local-only user hit an S3 configuration error while doing nothing involving S3.
    """
    monkeypatch.delenv("RECALL_S3_ALLOWLIST", raising=False)

    def _boom():
        raise AssertionError("the S3 reader must not be constructed for a file:// manifest")

    reader = reader_for_manifest(
        _manifest(_file_entry(tmp_path)), local_roots=(tmp_path,), s3_factory=_boom
    )
    assert isinstance(reader, LocalObjectReader)
