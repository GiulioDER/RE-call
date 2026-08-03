from __future__ import annotations

import hashlib
from io import BytesIO

import pytest

from recall.lineage import IndexManifestV1, ManifestObjectV1
from recall.manifest import (
    ManifestVerificationError,
    ObjectNotAllowed,
    S3Allowlist,
    S3ObjectReader,
)


class _S3:
    def __init__(self, data: bytes, *, version: str = "version-1") -> None:
        self.data = data
        self.version = version
        self.calls: list[dict] = []

    def get_object(self, **kwargs):
        self.calls.append(kwargs)
        return {
            "Body": BytesIO(self.data),
            "ContentLength": len(self.data),
            "VersionId": self.version,
        }


def _entry(data: bytes = b"immutable corpus") -> ManifestObjectV1:
    return ManifestObjectV1(
        uri="s3://approved/corpora/tenant-a/memo.md",
        version_id="version-1",
        media_type="text/markdown",
        size=len(data),
        sha256=hashlib.sha256(data).hexdigest(),
    )


def test_reader_requests_exact_version_and_verifies_bytes() -> None:
    client = _S3(b"immutable corpus")
    reader = S3ObjectReader(client, S3Allowlist.parse("s3://approved/corpora/"))

    result = reader.fetch(_entry())

    assert result.data == b"immutable corpus"
    assert client.calls == [
        {
            "Bucket": "approved",
            "Key": "corpora/tenant-a/memo.md",
            "VersionId": "version-1",
        }
    ]


@pytest.mark.parametrize("mutation", ["version", "size", "checksum"])
def test_object_mutation_or_wrong_version_is_detected(mutation: str) -> None:
    original = b"immutable corpus"
    client = _S3(original)
    entry = _entry(original)
    if mutation == "version":
        client.version = "version-2"
    elif mutation == "size":
        entry = ManifestObjectV1(
            entry.uri, entry.version_id, entry.media_type, entry.size + 1, entry.sha256
        )
    else:
        client.data = b"mutable!! corpus"
    reader = S3ObjectReader(client, S3Allowlist.parse("approved/corpora/"))

    with pytest.raises(ManifestVerificationError, match=mutation):
        reader.fetch(entry)


def test_allowlist_is_checked_before_any_s3_request() -> None:
    client = _S3(b"immutable corpus")
    reader = S3ObjectReader(client, S3Allowlist.parse("approved/other-prefix/"))

    with pytest.raises(ObjectNotAllowed):
        reader.fetch(_entry())
    assert client.calls == []


def test_manifest_verification_stops_on_a_missing_object() -> None:
    class Missing(_S3):
        def get_object(self, **kwargs):
            raise FileNotFoundError(kwargs["Key"])

    entry = _entry()
    manifest = IndexManifestV1("tenant-a", "v1", (entry,))
    reader = S3ObjectReader(Missing(b""), S3Allowlist.parse("approved/corpora/"))

    with pytest.raises(ManifestVerificationError, match="unavailable"):
        reader.verify(manifest)
