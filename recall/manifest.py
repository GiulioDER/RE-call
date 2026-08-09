"""Canonical manifests and immutable, allowlisted S3 object retrieval."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import unquote, urlsplit

from recall.lineage import IndexManifestV1, LineageError, ManifestObjectV1


class ManifestVerificationError(RuntimeError):
    """An object disappeared or no longer matches its immutable manifest entry."""


class ObjectNotAllowed(ManifestVerificationError):
    """An S3 URI lies outside the deployment-owned bucket and prefix allowlist."""


class _Body(Protocol):
    def read(self) -> bytes: ...


class S3Client(Protocol):
    def get_object(self, **kwargs: Any) -> Mapping[str, Any]: ...


@dataclass(frozen=True)
class S3Location:
    bucket: str
    key: str

    @classmethod
    def parse(cls, uri: str) -> "S3Location":
        parsed = urlsplit(uri)
        key = unquote(parsed.path.lstrip("/"))
        if parsed.scheme != "s3" or not parsed.netloc or not key:
            raise ManifestVerificationError(f"invalid S3 URI {uri!r}")
        if parsed.query or parsed.fragment:
            raise ManifestVerificationError("S3 object URI cannot contain query or fragment")
        return cls(parsed.netloc, key)


@dataclass(frozen=True)
class S3Prefix:
    bucket: str
    prefix: str

    def contains(self, location: S3Location) -> bool:
        return location.bucket == self.bucket and location.key.startswith(self.prefix)


@dataclass(frozen=True)
class S3Allowlist:
    prefixes: tuple[S3Prefix, ...]

    def __post_init__(self) -> None:
        if not self.prefixes:
            raise ValueError("at least one S3 bucket/prefix must be allowlisted")

    @classmethod
    def parse(cls, value: str) -> "S3Allowlist":
        prefixes: list[S3Prefix] = []
        for raw in value.split(","):
            item = raw.strip()
            if not item:
                continue
            parsed = urlsplit(item if "://" in item else f"s3://{item}")
            if parsed.scheme != "s3" or not parsed.netloc or parsed.query or parsed.fragment:
                raise ValueError(f"invalid S3 allowlist entry {item!r}")
            prefixes.append(S3Prefix(parsed.netloc, unquote(parsed.path.lstrip("/"))))
        return cls(tuple(prefixes))

    @classmethod
    def from_environment(cls) -> "S3Allowlist":
        raw = os.environ.get("RECALL_S3_ALLOWLIST", "")
        if not raw:
            raise ValueError("RECALL_S3_ALLOWLIST is required for S3 manifest access")
        return cls.parse(raw)

    def require(self, uri: str) -> S3Location:
        location = S3Location.parse(uri)
        if not any(prefix.contains(location) for prefix in self.prefixes):
            raise ObjectNotAllowed(f"S3 object {uri!r} is outside RECALL_S3_ALLOWLIST")
        return location


@dataclass(frozen=True)
class VerifiedObject:
    entry: ManifestObjectV1
    data: bytes


class S3ObjectReader:
    """Read only exact object versions selected by deployment configuration."""

    def __init__(self, client: S3Client, allowlist: S3Allowlist) -> None:
        self._client = client
        self._allowlist = allowlist

    @classmethod
    def from_environment(cls) -> "S3ObjectReader":
        """Build from the process credential chain and deployment-owned endpoint only.

        No request object is accepted here, so a tenant cannot supply credentials or redirect the
        service to an attacker-controlled S3 endpoint.
        """
        try:
            import boto3
        except ImportError as exc:  # pragma: no cover, exercised without the optional extra
            raise ImportError("S3 access requires: pip install recall-rag[s3]") from exc
        endpoint = os.environ.get("RECALL_S3_ENDPOINT_URL")
        client = boto3.client("s3", endpoint_url=endpoint) if endpoint else boto3.client("s3")
        return cls(client, S3Allowlist.from_environment())

    def fetch(self, entry: ManifestObjectV1) -> VerifiedObject:
        location = self._allowlist.require(entry.uri)
        try:
            response = self._client.get_object(
                Bucket=location.bucket,
                Key=location.key,
                VersionId=entry.version_id,
            )
            body = response.get("Body")
            if body is None or not hasattr(body, "read"):
                raise ManifestVerificationError(f"S3 returned no body for {entry.uri}")
            data = body.read()
        except ManifestVerificationError:
            raise
        except Exception as exc:
            raise ManifestVerificationError(
                f"immutable object {entry.uri}@{entry.version_id} is unavailable: "
                f"{type(exc).__name__}"
            ) from exc

        returned_version = response.get("VersionId")
        if returned_version != entry.version_id:
            raise ManifestVerificationError(
                f"object version mismatch for {entry.uri}: expected {entry.version_id!r}, "
                f"received {returned_version!r}"
            )
        content_length = response.get("ContentLength")
        if content_length != entry.size or len(data) != entry.size:
            raise ManifestVerificationError(
                f"object size mismatch for {entry.uri}: expected {entry.size}, "
                f"received header={content_length!r}, body={len(data)}"
            )
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry.sha256:
            raise ManifestVerificationError(
                f"object checksum mismatch for {entry.uri}: expected {entry.sha256}, "
                f"received {digest}"
            )
        return VerifiedObject(entry, data)

    def verify(self, manifest: IndexManifestV1) -> tuple[VerifiedObject, ...]:
        return tuple(self.fetch(entry) for entry in manifest.objects)


def load_manifest(path: str | Path) -> IndexManifestV1:
    try:
        return IndexManifestV1.from_json(Path(path).read_bytes())
    except OSError as exc:
        raise LineageError(f"cannot read manifest {path}: {exc}") from exc


def load_inventory(path: str | Path) -> tuple[ManifestObjectV1, ...]:
    """Load a JSON array used by ``recall manifest create``."""
    import json

    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise LineageError(f"cannot read object inventory {path}: {exc}") from exc
    if not isinstance(value, list):
        raise LineageError("object inventory must be a JSON array")
    wrapper = {
        "schema_version": 1,
        "tenant_id": "inventory-validation",
        "corpus_version": "inventory-validation",
        "objects": value,
    }
    return IndexManifestV1.from_dict(wrapper).objects
