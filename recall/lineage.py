"""Immutable identities for retrieval pipelines, corpora, and index generations."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType
from typing import Any, TypeAlias
from urllib.parse import urlsplit
from recall.errors import RecallError

JsonPrimitive: TypeAlias = str | int | float | bool | None
FrozenJson: TypeAlias = JsonPrimitive | tuple["FrozenJson", ...] | Mapping[str, "FrozenJson"]

PIPELINE_SCHEMA_VERSION = 1
MANIFEST_SCHEMA_VERSION = 1
DEFAULT_FTS_CONFIGURATION: Mapping[str, FrozenJson] = MappingProxyType(
    {"language": "english", "schema_version": 1}
)


class LineageError(ValueError, RecallError):
    """An identity is mutable, malformed, or cannot be canonicalised safely."""


class UnverifiedPipelineError(LineageError):
    """Production was asked to use an embedder without immutable provenance."""


def _freeze_json(value: Any, *, path: str = "$") -> FrozenJson:
    """Return a deeply immutable JSON value, rejecting ambiguous representations."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise LineageError(f"{path} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        frozen: dict[str, FrozenJson] = {}
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise LineageError(f"{path} has a non-string or empty object key")
            frozen[key] = _freeze_json(item, path=f"{path}.{key}")
        return MappingProxyType(dict(sorted(frozen.items())))
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return tuple(_freeze_json(item, path=f"{path}[]") for item in value)
    raise LineageError(f"{path} contains non-JSON value {type(value).__name__}")


def _thaw_json(value: FrozenJson) -> Any:
    if isinstance(value, Mapping):
        return {key: _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


def canonical_json(value: Any) -> bytes:
    """Canonical UTF-8 JSON used by every lineage and manifest digest."""
    frozen = _freeze_json(value)
    return json.dumps(
        _thaw_json(frozen),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json(value)).hexdigest()


def _sha256(value: str, *, field_name: str) -> str:
    normalised = value.lower()
    if len(normalised) != 64 or any(ch not in "0123456789abcdef" for ch in normalised):
        raise LineageError(f"{field_name} must be a 64-character SHA-256 digest")
    return normalised


@dataclass(frozen=True)
class EmbedderIdentity:
    """One provider artifact at one immutable revision and vector width."""

    provider: str
    model: str
    dimension: int
    revision: str | None = None
    artifact_digest: str | None = None
    unverified_reason: str | None = None

    def __post_init__(self) -> None:
        if not self.provider.strip() or not self.model.strip():
            raise LineageError("embedder provider and model must be non-empty")
        if self.dimension < 1:
            raise LineageError("embedder dimension must be positive")
        if self.artifact_digest is not None:
            object.__setattr__(
                self,
                "artifact_digest",
                _sha256(self.artifact_digest, field_name="artifact_digest"),
            )
        immutable = bool(self.revision or self.artifact_digest)
        if immutable and self.unverified_reason is not None:
            raise LineageError("an immutable embedder identity cannot also be marked unverified")
        if not immutable and not self.unverified_reason:
            raise LineageError(
                "embedder identity needs an immutable revision or artifact digest; "
                "development-only identities must state unverified_reason explicitly"
            )

    @property
    def verified(self) -> bool:
        return bool(self.revision or self.artifact_digest)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provider": self.provider,
            "model": self.model,
            "revision": self.revision,
            "artifact_digest": self.artifact_digest,
            "dimension": self.dimension,
            "verified": self.verified,
            "unverified_reason": self.unverified_reason,
        }


@dataclass(frozen=True)
class ChunkerIdentity:
    """One chunking algorithm with a versioned, deeply immutable configuration."""

    algorithm: str
    schema_version: int
    configuration: Mapping[str, FrozenJson] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.algorithm.strip():
            raise LineageError("chunker algorithm must be non-empty")
        if self.schema_version < 1:
            raise LineageError("chunker schema_version must be positive")
        frozen = _freeze_json(self.configuration, path="$.chunker.configuration")
        if not isinstance(frozen, Mapping):  # pragma: no cover, Mapping input guarantees this
            raise LineageError("chunker configuration must be an object")
        object.__setattr__(self, "configuration", frozen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "algorithm": self.algorithm,
            "schema_version": self.schema_version,
            "configuration": _thaw_json(self.configuration),
        }


@dataclass(frozen=True)
class PipelineIdentity:
    """Every input that can change chunks, embeddings, or sparse retrieval semantics."""

    embedder: EmbedderIdentity
    chunker: ChunkerIdentity
    fts_configuration: Mapping[str, FrozenJson] = field(
        default_factory=lambda: DEFAULT_FTS_CONFIGURATION
    )
    schema_version: int = PIPELINE_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version < 1:
            raise LineageError("pipeline schema_version must be positive")
        frozen = _freeze_json(self.fts_configuration, path="$.fts_configuration")
        if not isinstance(frozen, Mapping):  # pragma: no cover, Mapping input guarantees this
            raise LineageError("FTS configuration must be an object")
        object.__setattr__(self, "fts_configuration", frozen)

    @property
    def verified(self) -> bool:
        return self.embedder.verified

    @property
    def fingerprint(self) -> str:
        return canonical_sha256(self.to_dict())

    def require_production_identity(self) -> None:
        if not self.verified:
            raise UnverifiedPipelineError(
                "production generation builds require an immutable embedder revision or "
                "artifact digest"
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "embedder": self.embedder.to_dict(),
            "chunker": self.chunker.to_dict(),
            "fts_configuration": _thaw_json(self.fts_configuration),
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "PipelineIdentity":
        embedder = value.get("embedder")
        chunker = value.get("chunker")
        if not isinstance(embedder, Mapping) or not isinstance(chunker, Mapping):
            raise LineageError("pipeline embedder and chunker must be objects")
        chunker_configuration = chunker.get("configuration")
        if not isinstance(chunker_configuration, Mapping):
            chunker_configuration = {}
        fts_configuration = value.get("fts_configuration")
        if not isinstance(fts_configuration, Mapping):
            fts_configuration = DEFAULT_FTS_CONFIGURATION
        return cls(
            schema_version=int(value.get("schema_version", PIPELINE_SCHEMA_VERSION)),
            embedder=EmbedderIdentity(
                provider=str(embedder.get("provider", "")),
                model=str(embedder.get("model", "")),
                revision=(str(embedder["revision"]) if embedder.get("revision") else None),
                artifact_digest=(
                    str(embedder["artifact_digest"])
                    if embedder.get("artifact_digest")
                    else None
                ),
                dimension=int(embedder.get("dimension", 0)),
                unverified_reason=(
                    str(embedder["unverified_reason"])
                    if embedder.get("unverified_reason")
                    else None
                ),
            ),
            chunker=ChunkerIdentity(
                algorithm=str(chunker.get("algorithm", "")),
                schema_version=int(chunker.get("schema_version", 0)),
                configuration=chunker_configuration,
            ),
            fts_configuration=fts_configuration,
        )


@dataclass(frozen=True)
class ManifestObjectV1:
    uri: str
    version_id: str
    media_type: str
    size: int
    sha256: str

    def __post_init__(self) -> None:
        parsed = urlsplit(self.uri)
        if parsed.query or parsed.fragment:
            raise LineageError("manifest object URI cannot contain a query or fragment")

        # `file://` is accepted alongside `s3://` so a corpus that lives in a directory can become
        # a generation. Without it, calibration was unreachable for anyone who had not stood up
        # object storage, because calibration requires a generation and a generation required S3.
        #
        # ⚠️ The two are NOT equivalent guarantees, and the difference is enforced below rather
        # than left to a reader's goodwill. S3 `version_id` names one specific set of bytes
        # forever, so an entry and its object cannot drift apart. A local file has no version and
        # can be rewritten in place after the manifest is written. The only honest version a local
        # file has is what is inside it, so `version_id` is REQUIRED to equal the content digest.
        # An arbitrary string there would look like an S3 version id while promising something it
        # cannot deliver. What the local path buys is DETECTION of divergence, never prevention.
        if parsed.scheme == "file":
            if not parsed.path.lstrip("/"):
                raise LineageError("manifest object URI must be file:///absolute/path")
            # Compared case-insensitively because `sha256` is normalised to lowercase at the END of
            # this method, after this point. Comparing the raw values would reject a correct entry
            # written with an uppercase digest, which is a validator failing on presentation rather
            # than on substance.
            if self.version_id.lower() != self.sha256.lower():
                raise LineageError(
                    "a file:// manifest object's version_id must be its content digest. A local "
                    "file has no version other than its contents, and any other value would name "
                    "an immutability guarantee the filesystem does not provide."
                )
        elif parsed.scheme != "s3" or not parsed.netloc or not parsed.path.lstrip("/"):
            raise LineageError("manifest object URI must be s3://bucket/key or file:///path")

        if not self.version_id or self.version_id.lower() == "null":
            raise LineageError("manifest object version_id must name an immutable object version")
        if not self.media_type:
            raise LineageError("manifest object media_type must be non-empty")
        if self.size < 0:
            raise LineageError("manifest object size cannot be negative")
        object.__setattr__(self, "sha256", _sha256(self.sha256, field_name="sha256"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "uri": self.uri,
            "version_id": self.version_id,
            "media_type": self.media_type,
            "size": self.size,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class IndexManifestV1:
    """Canonical inventory of immutable S3 object versions for one tenant corpus."""

    tenant_id: str
    corpus_version: str
    objects: tuple[ManifestObjectV1, ...]
    schema_version: int = MANIFEST_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != MANIFEST_SCHEMA_VERSION:
            raise LineageError(f"unsupported manifest schema_version {self.schema_version}")
        if not self.tenant_id.strip() or not self.corpus_version.strip():
            raise LineageError("manifest tenant_id and corpus_version must be non-empty")
        ordered = tuple(sorted(self.objects, key=lambda item: (item.uri, item.version_id)))
        uris = [item.uri for item in ordered]
        if len(uris) != len(set(uris)):
            raise LineageError("manifest contains a duplicate object URI")
        object.__setattr__(self, "objects", ordered)

    @property
    def digest(self) -> str:
        return canonical_sha256(self.to_dict())

    @property
    def corpus_fingerprint(self) -> str:
        return self.digest

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "tenant_id": self.tenant_id,
            "corpus_version": self.corpus_version,
            "objects": [item.to_dict() for item in self.objects],
        }

    def to_json(self) -> str:
        return canonical_json(self.to_dict()).decode("utf-8") + "\n"

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IndexManifestV1":
        raw_objects = value.get("objects")
        if not isinstance(raw_objects, list):
            raise LineageError("manifest objects must be an array")
        objects: list[ManifestObjectV1] = []
        for raw in raw_objects:
            if not isinstance(raw, Mapping):
                raise LineageError("every manifest object must be an object")
            objects.append(
                ManifestObjectV1(
                    uri=str(raw.get("uri", "")),
                    version_id=str(raw.get("version_id", "")),
                    media_type=str(raw.get("media_type", "")),
                    size=int(raw.get("size", -1)),
                    sha256=str(raw.get("sha256", "")),
                )
            )
        return cls(
            schema_version=int(value.get("schema_version", 0)),
            tenant_id=str(value.get("tenant_id", "")),
            corpus_version=str(value.get("corpus_version", "")),
            objects=tuple(objects),
        )

    @classmethod
    def from_json(cls, data: str | bytes) -> "IndexManifestV1":
        try:
            value = json.loads(data)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise LineageError(f"invalid manifest JSON: {exc}") from exc
        if not isinstance(value, Mapping):
            raise LineageError("manifest root must be an object")
        return cls.from_dict(value)


class GenerationState(StrEnum):
    BUILDING = "building"
    VALIDATING = "validating"
    READY = "ready"
    ACTIVE = "active"
    RETIRED = "retired"
    FAILED = "failed"
    LEGACY_UNVERIFIED = "legacy_unverified"
