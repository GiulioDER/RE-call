"""Canonical manifests and immutable, allowlisted S3 object retrieval."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable
from collections.abc import Callable
from urllib.request import url2pathname
from urllib.parse import unquote, urlsplit

from recall.extraction import (
    DocumentExtractionError,
    ExtractedBlock,
    extract_document,
    extraction_path_for,
)
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
    """A manifest checked object, optionally carrying its extracted generation view.

    ``data`` is raw immutable source bytes for base readers and UTF 8 extracted bytes for
    extracting readers. ``metadata`` and ``blocks`` are populated only for generation building;
    ``verify()`` deliberately returns the raw form so checksum verification never invokes an
    optional parser or LibreOffice.
    """

    entry: ManifestObjectV1
    data: bytes
    metadata: dict[str, Any] = field(default_factory=dict)
    blocks: tuple[ExtractedBlock, ...] = ()


@runtime_checkable
class ObjectReader(Protocol):
    """What a manifest reader must do, independent of where the bytes live.

    Introduced so callers can be typed against the capability rather than against S3 specifically.
    `recall/cli.py` and `GenerationManager.build` both named `S3ObjectReader` concretely, which is
    what made the local backend untypeable even once it existed.
    """

    def fetch(self, entry: ManifestObjectV1) -> VerifiedObject: ...

    def verify(self, manifest: IndexManifestV1) -> tuple[VerifiedObject, ...]: ...


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
            raise ImportError('S3 access requires: pip install "recall-rag[s3]"') from exc
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


def _unc_supported() -> bool:
    """Whether this platform can address a `\\\\server\\share` path.

    A named seam rather than an inline `os.name` test, so a test can exercise the non-Windows
    refusal branch without patching `os.name` itself — which `pathlib` reads to choose its Path
    class, so patching it breaks every path operation in the process, including the reader's own.
    """
    return os.name == "nt"


class LocalObjectReader:
    """Read manifest objects from the filesystem, confined to allowed roots.

    Exists so a corpus that lives in a directory can become a generation. Calibration requires a
    generation and generation build required S3, so before this a single-machine user could not
    calibrate at all: a documented capability unreachable from the configuration they actually had.

    ⚠️ **This is a weaker guarantee than `S3ObjectReader`, deliberately and visibly.** S3 object
    versioning pins one specific set of bytes forever, so a manifest entry and its object cannot
    drift apart. A local file can be rewritten in place after the manifest is written. What this
    reader provides is **detection**: it hashes what it actually read and refuses when that differs
    from the entry. It cannot provide prevention, and nothing here should be read as though it does.
    """

    def __init__(self, roots: "tuple[Path, ...] | list[Path]") -> None:
        if not roots:
            raise ValueError("at least one local root must be allowlisted")
        # Resolved once, so a symlink cannot be swapped underneath the containment check between
        # construction and read.
        self._roots = tuple(Path(r).resolve() for r in roots)

    @classmethod
    def from_environment(cls) -> "LocalObjectReader":
        raw = os.environ.get("RECALL_LOCAL_ALLOWLIST", "")
        if not raw:
            raise ValueError(
                "RECALL_LOCAL_ALLOWLIST is required for file:// manifest access. Without it a "
                "manifest could name any file on the machine."
            )
        return cls(tuple(Path(p.strip()) for p in raw.split(os.pathsep) if p.strip()))

    def _resolve(self, entry: ManifestObjectV1) -> Path:
        parsed = urlsplit(entry.uri)
        if parsed.scheme != "file":
            raise ObjectNotAllowed(f"{entry.uri!r} is not a file:// object")
        # `url2pathname` ALONE. It already percent-decodes, so wrapping it in `unquote` decoded
        # twice and resolved a different file than the manifest named. Measured on 3.14/win32
        # against `Path.as_uri()` output: `hash#tag.md` resolved to `...\hash` and
        # `quest?ion.md` to `...\quest` (truncated at the decoded delimiter), while a file
        # genuinely named `percent%20literal.md` resolved to `percent literal.md` — a DIFFERENT
        # existing file. The containment check below still held, so nothing escaped the allowlist;
        # what broke is that legitimate corpus files became unreadable, reported as a checksum or
        # availability failure that named neither the file nor the cause.
        # The authority is part of the path for a UNC share, and dropping it silently rebased the
        # file onto the current local drive: `file://nas1/share/docs/a.md` resolved to
        # `\share\docs\a.md`, i.e. `C:\share\docs\a.md`. `Path("//nas1/share/...").as_uri()`
        # produces exactly that URI, so a network-share corpus, which is an ordinary thing to have
        # on the platform this targets, was unreadable. `localhost` and the empty authority both
        # mean "this machine" per RFC 8089 and must NOT be re-prefixed.
        # The UNC anchor is built here rather than handed back to `url2pathname` as a
        # `//authority/path` string. On 3.13+ that function re-splits what it is given and takes
        # the LOCAL branch whenever the authority is this machine's own hostname, so
        # `file://MYHOST/share/docs/a.md` — which `Path("//MYHOST/share/docs/a.md").as_uri()`
        # produces for a share on the local machine — collapsed back to `C:\share\docs\a.md`, the
        # exact defect the authority was carried to avoid. Decoding the path alone and prefixing
        # the anchor directly is version-independent.
        authority = unquote(parsed.netloc)
        try:
            local = url2pathname(parsed.path)
        except (OSError, ValueError) as exc:
            # The decode runs BEFORE the authority guard below, so guarding only the authority
            # left this hole open: a URI whose PATH begins with `//` (`file:////share/x.md`,
            # `file://localhost//share/x.md`) reads as an authority to 3.14's POSIX
            # `url2pathname` and raises `URLError`, which is an OSError and would propagate
            # untyped through `verify()` and `generation build`.
            #
            # `recall`'s own inventory cannot emit this form, because it calls `as_uri()` on
            # paths `candidate_files` has already resolved and `resolve()` collapses a POSIX `//`
            # root to `/`. Bare `Path("//share/x.md").as_uri()` DOES produce it on POSIX, so the
            # form is reachable from a hand-written or third-party manifest, and from any caller
            # that skips the resolve.
            raise ObjectNotAllowed(
                f"local object {entry.uri!r} does not name a readable local path "
                f"({type(exc).__name__})."
            ) from exc
        if authority and authority.lower() != "localhost":
            if not _unc_supported():
                # 3.14's POSIX `url2pathname` raises `URLError` here, which is neither
                # `ObjectNotAllowed` nor `ManifestVerificationError` and would propagate untyped
                # through `verify()` and `generation build`. A UNC share is a Windows concept, so
                # this is refused in the reader's own vocabulary instead.
                raise ObjectNotAllowed(
                    f"local object {entry.uri!r} names the remote authority {authority!r}. "
                    "A UNC share cannot be read on this platform; rewrite the manifest with a "
                    "local path."
                )
            path = Path(f"//{authority}{local}").resolve()
        else:
            path = Path(local).resolve()
        # `is_relative_to` on the RESOLVED path, so `..` and symlinks cannot escape a root. This is
        # the local analogue of the S3 allowlist: without it a manifest names any file on disk.
        if not any(path.is_relative_to(root) for root in self._roots):
            raise ObjectNotAllowed(
                f"local object {entry.uri!r} is outside RECALL_LOCAL_ALLOWLIST"
            )
        return path

    def fetch(self, entry: ManifestObjectV1) -> VerifiedObject:
        path = self._resolve(entry)
        try:
            data = path.read_bytes()
        except OSError as exc:
            # Surfaced as a verification failure, not an OSError, so a caller handling manifest
            # problems does not have to know which storage backend produced them.
            raise ManifestVerificationError(
                f"immutable object {entry.uri} is unavailable: {type(exc).__name__}"
            ) from exc

        if len(data) != entry.size:
            raise ManifestVerificationError(
                f"object size mismatch for {entry.uri}: expected {entry.size}, "
                f"received {len(data)}"
            )
        digest = hashlib.sha256(data).hexdigest()
        if digest != entry.sha256:
            raise ManifestVerificationError(
                f"object checksum mismatch for {entry.uri}: expected {entry.sha256}, "
                f"received {digest}. A local file changed after its manifest was written; this "
                f"reader detects that and cannot prevent it."
            )
        return VerifiedObject(entry, data)

    def verify(self, manifest: IndexManifestV1) -> tuple[VerifiedObject, ...]:
        return tuple(self.fetch(entry) for entry in manifest.objects)


class ExtractingLocalObjectReader(LocalObjectReader):
    """Verify original bytes, then expose a UTF 8 extracted view to generation building."""

    def fetch(self, entry: ManifestObjectV1) -> VerifiedObject:
        verified = super().fetch(entry)
        path = extraction_path_for(self._resolve(entry), entry.media_type)
        try:
            extracted = extract_document(path, verified.data)
        except DocumentExtractionError:
            raise
        except Exception as exc:
            raise ManifestVerificationError(
                f"could not extract {entry.uri}: {type(exc).__name__}"
            ) from exc
        return VerifiedObject(
            entry,
            extracted.text.encode("utf-8"),
            {**verified.metadata, **extracted.metadata},
            extracted.blocks,
        )

    def verify(self, manifest: IndexManifestV1) -> tuple[VerifiedObject, ...]:
        return tuple(LocalObjectReader.fetch(self, entry) for entry in manifest.objects)


class ExtractingS3ObjectReader:
    """Apply the same format extraction to verified S3 bytes."""

    def __init__(self, base: S3ObjectReader) -> None:
        self._base = base

    def fetch(self, entry: ManifestObjectV1) -> VerifiedObject:
        verified = self._base.fetch(entry)
        name = extraction_path_for(Path(unquote(urlsplit(entry.uri).path)).name, entry.media_type)
        try:
            extracted = extract_document(Path(name), verified.data)
        except Exception as exc:
            if isinstance(exc, ManifestVerificationError):
                raise
            raise ManifestVerificationError(
                f"could not extract {entry.uri}: {type(exc).__name__}"
            ) from exc
        return VerifiedObject(
            entry,
            extracted.text.encode("utf-8"),
            {**verified.metadata, **extracted.metadata},
            extracted.blocks,
        )

    def verify(self, manifest: IndexManifestV1) -> tuple[VerifiedObject, ...]:
        return tuple(self._base.fetch(entry) for entry in manifest.objects)


def reader_for_manifest(
    manifest: IndexManifestV1,
    *,
    local_roots: "tuple[Path, ...] | list[Path] | None" = None,
    s3_factory: "Callable[[], ObjectReader] | None" = None,
) -> ObjectReader:
    """Pick the reader the manifest's objects actually need.

    The CLI used to build `S3ObjectReader.from_environment()` unconditionally, before it knew what
    the manifest contained. That needs boto3 and `RECALL_S3_ALLOWLIST`, so a local-only user hit an
    S3 configuration error while doing nothing that involved S3.

    A manifest mixing `s3://` and `file://` objects is REFUSED. It would carry two different
    immutability guarantees at once: the S3 half pinned to a version, the local half only
    detectable after the fact. One lineage record would then describe both, true of one half and
    overstated for the other, with nothing downstream able to tell which hit came from which.

    `s3_factory` is injectable so the choice can be tested without boto3 or an allowlist.
    """
    if not manifest.objects:
        raise ValueError("manifest has no objects, so no reader can be chosen for it")

    schemes = {urlsplit(entry.uri).scheme for entry in manifest.objects}
    if len(schemes) > 1:
        raise ValueError(
            f"manifest mixes object schemes {sorted(schemes)}. A generation built from it would "
            "carry one lineage record describing two different immutability guarantees: S3 objects "
            "are pinned to a version, local files are only checked after the fact. Split it."
        )

    scheme = schemes.pop()
    if scheme == "file":
        if local_roots is not None:
            return ExtractingLocalObjectReader(local_roots)
        return ExtractingLocalObjectReader.from_environment()
    if scheme == "s3":
        reader = (s3_factory or S3ObjectReader.from_environment)()
        if isinstance(reader, S3ObjectReader):
            return ExtractingS3ObjectReader(reader)
        return reader
    raise ValueError(f"manifest objects use unsupported scheme {scheme!r}")
