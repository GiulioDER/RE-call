"""A corpus that lives in a directory must be able to become a generation.

Calibration requires a generation (`recall calibration calibrate --generation`), generation build
requires a manifest, and `ManifestObjectV1` refused any URI that was not `s3://`, while
`S3ObjectReader` was the only reader in the codebase. So a single-machine user, which is every user
who has not stood up object storage, **could not calibrate at all**. That is the same shape as the
`RECALL_TRUST_MODE` gap fixed earlier today: a documented capability, unreachable from the
configuration a new user actually has.

The guarantees are NOT equivalent, and this file pins that rather than blurring it.

S3 object versioning makes a fetched object immutable: `version_id` names one specific set of bytes
forever, so a manifest entry and its object cannot drift apart. A local file has no such thing. It
can be rewritten in place after the manifest is written. The local reader therefore **detects**
divergence by hashing what it read, and cannot **prevent** it. Detection is a weaker promise and is
stated as one everywhere it appears.
"""

from __future__ import annotations

import hashlib
import pathlib

import pytest

from recall.lineage import LineageError, ManifestObjectV1


def _write(tmp_path: pathlib.Path, name: str, body: bytes) -> tuple[str, str, int]:
    p = tmp_path / name
    p.write_bytes(body)
    return p.as_uri(), hashlib.sha256(body).hexdigest(), len(body)


class TestS3IsUnchanged:
    """Widening the schema must not loosen anything S3 already guaranteed."""

    def test_s3_still_requires_a_version_id(self) -> None:
        with pytest.raises(LineageError, match="version_id"):
            ManifestObjectV1(
                uri="s3://b/k", version_id="", media_type="text/markdown", size=1, sha256="a" * 64
            )

    def test_s3_still_refuses_the_literal_null_version(self) -> None:
        """`null` is what an unversioned bucket returns. Accepting it would name nothing."""
        with pytest.raises(LineageError, match="version_id"):
            ManifestObjectV1(
                uri="s3://b/k", version_id="null", media_type="text/markdown", size=1, sha256="a" * 64
            )

    def test_an_unknown_scheme_is_still_refused(self) -> None:
        for uri in ("http://example.com/x", "ftp://h/x", "/plain/path", "gs://b/k"):
            with pytest.raises(LineageError):
                ManifestObjectV1(
                    uri=uri, version_id="v", media_type="text/markdown", size=1, sha256="a" * 64
                )

    def test_a_query_or_fragment_is_still_refused(self) -> None:
        for uri in ("s3://b/k?x=1", "s3://b/k#frag"):
            with pytest.raises(LineageError):
                ManifestObjectV1(
                    uri=uri, version_id="v", media_type="text/markdown", size=1, sha256="a" * 64
                )


class TestFileUris:
    def test_a_file_uri_is_accepted(self, tmp_path: pathlib.Path) -> None:
        uri, digest, size = _write(tmp_path, "a.md", b"hello")
        entry = ManifestObjectV1(
            uri=uri, version_id=digest, media_type="text/markdown", size=size, sha256=digest
        )
        assert entry.uri == uri

    def test_a_file_version_id_must_be_its_content_digest(self, tmp_path: pathlib.Path) -> None:
        """The only honest version a local file has is what is inside it.

        An arbitrary version string would look like an S3 version id and promise what it cannot
        deliver: that these bytes are pinned. Requiring it to equal the digest makes the version
        mean exactly what it can mean, which is 'the content that hashed to this'.
        """
        uri, digest, size = _write(tmp_path, "b.md", b"hello")
        with pytest.raises(LineageError, match="content digest"):
            ManifestObjectV1(
                uri=uri, version_id="v1", media_type="text/markdown", size=size, sha256=digest
            )


class TestLocalObjectReader:
    def test_it_reads_and_verifies(self, tmp_path: pathlib.Path) -> None:
        from recall.manifest import LocalObjectReader

        uri, digest, size = _write(tmp_path, "c.md", b"contents")
        entry = ManifestObjectV1(
            uri=uri, version_id=digest, media_type="text/markdown", size=size, sha256=digest
        )
        got = LocalObjectReader(roots=(tmp_path,)).fetch(entry)
        assert got.data == b"contents"

    def test_it_detects_a_file_rewritten_after_the_manifest(self, tmp_path: pathlib.Path) -> None:
        """The property that replaces S3 immutability: detection, not prevention.

        This is the whole reason the local path is a weaker guarantee, so it is the test that must
        exist. The manifest is written, the file is then changed underneath it, and the read fails
        loudly rather than serving different bytes under the same identity.
        """
        from recall.manifest import LocalObjectReader, ManifestVerificationError

        uri, digest, size = _write(tmp_path, "d.md", b"original")
        entry = ManifestObjectV1(
            uri=uri, version_id=digest, media_type="text/markdown", size=size, sha256=digest
        )
        (tmp_path / "d.md").write_bytes(b"tampered")
        with pytest.raises(ManifestVerificationError, match="checksum|size"):
            LocalObjectReader(roots=(tmp_path,)).fetch(entry)

    def test_it_confines_reads_to_the_allowed_roots(self, tmp_path: pathlib.Path) -> None:
        """The local analogue of the S3 allowlist. Without it a manifest names any file on disk."""
        from recall.manifest import LocalObjectReader, ObjectNotAllowed

        outside = tmp_path / "outside"
        outside.mkdir()
        allowed = tmp_path / "allowed"
        allowed.mkdir()
        uri, digest, size = _write(outside, "secret.md", b"not yours")
        entry = ManifestObjectV1(
            uri=uri, version_id=digest, media_type="text/markdown", size=size, sha256=digest
        )
        with pytest.raises(ObjectNotAllowed):
            LocalObjectReader(roots=(allowed,)).fetch(entry)

    def test_a_missing_file_fails_as_verification_not_as_oserror(
        self, tmp_path: pathlib.Path
    ) -> None:
        from recall.manifest import LocalObjectReader, ManifestVerificationError

        uri, digest, size = _write(tmp_path, "e.md", b"x")
        entry = ManifestObjectV1(
            uri=uri, version_id=digest, media_type="text/markdown", size=size, sha256=digest
        )
        (tmp_path / "e.md").unlink()
        with pytest.raises(ManifestVerificationError, match="unavailable"):
            LocalObjectReader(roots=(tmp_path,)).fetch(entry)
