"""Bounded provenance and the isolated Voyage multimodal embedding tenant.

This module deliberately separates three things that are easy to conflate: the original media
object, the bounded text sidecar stored with a retrieval record, and the provider request. The
first is addressed by a controlled URI and digest, the second never contains binary data, and the
third may contain a transient base64 representation only while a request is in flight.
"""

from __future__ import annotations

import base64
import hashlib
import os
import posixpath
from urllib.parse import unquote, urlsplit
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from recall.embeddings import (
    EmbeddingProfile,
    _check_declared_width,
    retry_with_backoff,
)
from recall.errors import RecallError


MULTIMODAL_TENANT = "re-call-multimodal"
MULTIMODAL_EMBEDDING_MODEL = "voyage-multimodal-3.5"
MULTIMODAL_EMBEDDING_PROFILE = "voyage-multimodal-3.5-v1"
MULTIMODAL_DIMENSION = 1024
DEFAULT_MAX_MEDIA_BYTES = 30 * 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 30 * 1024 * 1024
DEFAULT_MAX_ITEMS = 20
DEFAULT_MAX_SIDECAR_CHARS = 64_000

SUPPORTED_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
PROTECTED_TENANTS = frozenset({"default", "memory", "re-call-code-gen", "re-call-docs"})
AccessPolicy = Literal["private", "tenant", "public"]


class MultimodalError(ValueError, RecallError):
    """Base class for fail closed multimodal validation and access errors."""


class MediaValidationError(MultimodalError):
    """The media or its provenance is not safe to admit."""


class MediaAccessDenied(MultimodalError):
    """The caller cannot access the original media reference."""


class MediaBudgetExceeded(MultimodalError):
    """The response would exceed its explicit media byte budget."""


def media_digest(payload: bytes) -> str:
    """Return the content address used for media provenance and linkage."""
    return hashlib.sha256(payload).hexdigest()


def _validate_digest(value: str) -> str:
    normalized = value.strip().lower()
    if len(normalized) != 64 or any(char not in "0123456789abcdef" for char in normalized):
        raise MediaValidationError("media content_digest must be a SHA256 hex digest")
    return normalized


def _validate_uri(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized or not normalized.startswith(("s3://", "file://")):
        raise MediaValidationError(f"{field} must use a controlled s3:// or file:// URI")
    return normalized


def _validate_object_root(object_uri: str, object_root: str) -> str:
    uri = urlsplit(_validate_uri(object_uri, "object_uri"))
    root = urlsplit(_validate_uri(object_root, "object_root"))
    if (uri.scheme, uri.netloc) != (root.scheme, root.netloc):
        raise MediaValidationError("object_uri is outside the configured multimodal object root")
    uri_path = posixpath.normpath(unquote(uri.path))
    root_path = posixpath.normpath(unquote(root.path))
    if uri_path == root_path or not uri_path.startswith(root_path.rstrip("/") + "/"):
        raise MediaValidationError("object_uri is outside the configured multimodal object root")
    return uri.geturl()


@dataclass(frozen=True)
class MediaObjectRef:
    """Bounded metadata for an original object held outside vector storage."""

    content_digest: str
    media_type: str
    byte_size: int
    object_uri: str
    tenant_id: str
    source_uri: str | None = None
    authoritative_timestamp: datetime | None = None
    access_policy: AccessPolicy = "tenant"
    ocr_text: str = ""
    caption: str = ""
    entities: tuple[str, ...] = ()
    location: str | None = None
    regions: tuple[str, ...] = ()
    page_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "content_digest", _validate_digest(self.content_digest))
        if self.media_type not in SUPPORTED_MEDIA_TYPES:
            raise MediaValidationError(f"unsupported media type: {self.media_type!r}")
        if self.byte_size < 1:
            raise MediaValidationError("media byte_size must be positive")
        if self.byte_size > DEFAULT_MAX_MEDIA_BYTES:
            raise MediaBudgetExceeded("media byte_size exceeds the default admission budget")
        _validate_uri(self.object_uri, "object_uri")
        if self.source_uri is not None:
            _validate_uri(self.source_uri, "source_uri")
        if not self.tenant_id.strip():
            raise MediaValidationError("media tenant_id must be non-empty")
        if self.access_policy not in {"private", "tenant", "public"}:
            raise MediaValidationError("access_policy must be private, tenant, or public")
        for field_name, value in (
            ("ocr_text", self.ocr_text),
            ("caption", self.caption),
            ("location", self.location or ""),
        ):
            if len(value) > DEFAULT_MAX_SIDECAR_CHARS:
                raise MediaValidationError(f"{field_name} exceeds the bounded sidecar limit")
        if any(not item.strip() for item in (*self.entities, *self.regions, *self.page_refs)):
            raise MediaValidationError(
                "media entities, regions, and page_refs must be non-empty strings"
            )

    @property
    def media_id(self) -> str:
        return f"{self.tenant_id}:{self.content_digest}"

    def metadata(self) -> dict[str, object]:
        """Return storage-safe metadata. It intentionally has no payload or data URL."""
        return {
            "media_id": self.media_id,
            "content_digest": self.content_digest,
            "media_type": self.media_type,
            "media_byte_size": self.byte_size,
            "object_uri": self.object_uri,
            "source_uri": self.source_uri,
            "tenant_id": self.tenant_id,
            "access_policy": self.access_policy,
            "authoritative_timestamp": (
                self.authoritative_timestamp.isoformat()
                if self.authoritative_timestamp is not None
                else None
            ),
            "ocr_text": self.ocr_text,
            "caption": self.caption,
            "entities": list(self.entities),
            "location": self.location,
            "regions": list(self.regions),
            "page_refs": list(self.page_refs),
        }


def build_media_ref(
    payload: bytes,
    *,
    media_type: str,
    object_uri: str,
    object_root: str | None = None,
    tenant_id: str = MULTIMODAL_TENANT,
    source_uri: str | None = None,
    authoritative_timestamp: datetime | None = None,
    access_policy: AccessPolicy = "tenant",
    ocr_text: str = "",
    caption: str = "",
    entities: Sequence[str] = (),
    location: str | None = None,
    regions: Sequence[str] = (),
    page_refs: Sequence[str] = (),
    max_bytes: int = DEFAULT_MAX_MEDIA_BYTES,
) -> MediaObjectRef:
    """Validate an upload and return only its controlled, digest-linked reference."""
    if tenant_id != MULTIMODAL_TENANT:
        raise MediaValidationError(
            f"media tenant is fixed at {MULTIMODAL_TENANT!r}; refusing tenant aliasing"
        )
    if len(payload) < 1:
        raise MediaValidationError("media payload must not be empty")
    if len(payload) > max_bytes:
        raise MediaBudgetExceeded(
            f"media payload is {len(payload)} bytes, over the {max_bytes} byte admission budget"
        )
    root = object_root or os.environ.get("RECALL_MULTIMODAL_OBJECT_ROOT", "").strip()
    if not root:
        raise MediaValidationError(
            "RECALL_MULTIMODAL_OBJECT_ROOT or an explicit object_root is required"
        )
    normalized_uri = _validate_object_root(object_uri, root)
    return MediaObjectRef(
        content_digest=media_digest(payload),
        media_type=media_type,
        byte_size=len(payload),
        object_uri=normalized_uri,
        tenant_id=tenant_id,
        source_uri=source_uri,
        authoritative_timestamp=authoritative_timestamp,
        access_policy=access_policy,
        ocr_text=ocr_text,
        caption=caption,
        entities=tuple(entities),
        location=location,
        regions=tuple(regions),
        page_refs=tuple(page_refs),
    )


@dataclass(frozen=True)
class MultimodalSidecar:
    """A retrieval record linking text to one original media object."""

    media: MediaObjectRef
    text: str = ""

    def __post_init__(self) -> None:
        if len(self.text) > DEFAULT_MAX_SIDECAR_CHARS:
            raise MediaValidationError("multimodal sidecar text exceeds the bounded limit")

    def metadata(self) -> dict[str, object]:
        result = self.media.metadata()
        result["sidecar_text"] = self.text
        return result


@dataclass(frozen=True)
class MultimodalQuery:
    """Text or transient image input accepted by the multimodal provider."""

    text: str | None = None
    image_bytes: bytes | None = None
    media_type: str | None = None
    max_bytes: int = DEFAULT_MAX_MEDIA_BYTES

    def __post_init__(self) -> None:
        if self.text is None and self.image_bytes is None:
            raise MediaValidationError("a multimodal query needs text or image bytes")
        if self.text is not None and len(self.text) > DEFAULT_MAX_SIDECAR_CHARS:
            raise MediaValidationError("multimodal query text exceeds the bounded limit")
        if self.image_bytes is not None:
            if self.max_bytes < 1 or self.max_bytes > DEFAULT_MAX_MEDIA_BYTES:
                raise MediaValidationError(
                    f"max_bytes must be between 1 and {DEFAULT_MAX_MEDIA_BYTES}"
                )
            if self.media_type not in SUPPORTED_MEDIA_TYPES:
                raise MediaValidationError("image queries need a supported media type")
            if len(self.image_bytes) > self.max_bytes:
                raise MediaBudgetExceeded("image query exceeds the media byte budget")

    def provider_input(self) -> dict[str, object]:
        content: list[dict[str, str]] = []
        if self.text is not None:
            content.append({"type": "text", "text": self.text})
        if self.image_bytes is not None:
            encoded = base64.b64encode(self.image_bytes).decode("ascii")
            content.append(
                {
                    "type": "image_base64",
                    "image_base64": f"data:{self.media_type};base64,{encoded}",
                }
            )
        return {"content": content}


def project_media_evidence(
    media: MediaObjectRef,
    *,
    requester_tenant: str,
    principal: str,
    include_original: bool = True,
    used_bytes: int = 0,
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES,
) -> dict[str, object]:
    """Project provenance and, only when authorized, the controlled original reference."""
    if requester_tenant != media.tenant_id:
        raise MediaAccessDenied("media may only be accessed from its owning tenant")
    if media.access_policy == "private" and not principal.strip():
        raise MediaAccessDenied("private media requires an authenticated principal")
    if used_bytes < 0 or used_bytes + media.byte_size > max_response_bytes:
        raise MediaBudgetExceeded("original media would exceed the response byte budget")
    result = media.metadata()
    result["principal"] = principal
    result["original_media"] = (
        {"object_uri": media.object_uri, "content_digest": media.content_digest}
        if include_original
        else None
    )
    return result


@dataclass(frozen=True)
class MultimodalTenantConfig:
    """Feature flagged physical tenant configuration."""

    enabled: bool = False
    tenant_id: str = MULTIMODAL_TENANT
    embedding_profile: str = MULTIMODAL_EMBEDDING_PROFILE
    object_root: str | None = None
    max_media_bytes: int = DEFAULT_MAX_MEDIA_BYTES
    max_response_bytes: int = DEFAULT_MAX_RESPONSE_BYTES
    max_items: int = DEFAULT_MAX_ITEMS

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> "MultimodalTenantConfig":
        source = os.environ if env is None else env
        return cls(
            enabled=source.get("RECALL_MULTIMODAL_ENABLED", "0").strip().lower()
            in {"1", "true", "yes", "on"},
            tenant_id=source.get("RECALL_MULTIMODAL_TENANT", MULTIMODAL_TENANT).strip(),
            embedding_profile=source.get(
                "RECALL_MULTIMODAL_EMBED_PROFILE", MULTIMODAL_EMBEDDING_PROFILE
            ).strip(),
            object_root=source.get("RECALL_MULTIMODAL_OBJECT_ROOT", "").strip() or None,
            max_media_bytes=int(source.get("RECALL_MULTIMODAL_MAX_MEDIA_BYTES", DEFAULT_MAX_MEDIA_BYTES)),
            max_response_bytes=int(
                source.get("RECALL_MULTIMODAL_MAX_RESPONSE_BYTES", DEFAULT_MAX_RESPONSE_BYTES)
            ),
            max_items=int(source.get("RECALL_MULTIMODAL_MAX_ITEMS", DEFAULT_MAX_ITEMS)),
        )

    def __post_init__(self) -> None:
        if self.tenant_id != MULTIMODAL_TENANT:
            raise MediaValidationError(
                f"multimodal tenant is fixed at {MULTIMODAL_TENANT!r}; refusing tenant aliasing"
            )
        if self.tenant_id in PROTECTED_TENANTS:
            raise MediaValidationError("multimodal tenant cannot overlap a text specialist tenant")
        if self.embedding_profile != MULTIMODAL_EMBEDDING_PROFILE:
            raise MediaValidationError("multimodal embedding profile is fixed and registered")
        if not 1 <= self.max_media_bytes <= DEFAULT_MAX_MEDIA_BYTES:
            raise MediaValidationError("multimodal max_media_bytes is outside the bounded limit")
        if not 1 <= self.max_response_bytes <= DEFAULT_MAX_RESPONSE_BYTES:
            raise MediaValidationError(
                "multimodal max_response_bytes is outside the bounded limit"
            )
        if not 1 <= self.max_items <= DEFAULT_MAX_ITEMS:
            raise MediaValidationError("multimodal max_items is outside the bounded limit")
        if self.enabled and not self.object_root:
            raise MediaValidationError(
                "RECALL_MULTIMODAL_OBJECT_ROOT is required when multimodal retrieval is enabled"
            )

    def physical_identity(self) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "embedding_profile": self.embedding_profile,
            "object_root": self.object_root,
            "max_media_bytes": self.max_media_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_items": self.max_items,
        }


class _MultimodalClient(Protocol):
    def multimodal_embed(self, **kwargs: object) -> object: ...

    def embed(self, texts: Sequence[str], model: str, **kwargs: object) -> object: ...


class VoyageMultimodalEmbedder:
    """Voyage multimodal provider adapter carrying the registered profile identity."""

    def __init__(
        self,
        model: str = MULTIMODAL_EMBEDDING_MODEL,
        api_key: str | None = None,
        batch_size: int = 32,
        max_retries: int = 3,
        identity: EmbeddingProfile | None = None,
        client: _MultimodalClient | None = None,
    ) -> None:
        self._model = identity.model_name if identity is not None else model
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._profile = identity
        if client is None:
            key = api_key or os.environ.get("VOYAGE_API_KEY")
            if not key:
                raise RuntimeError(
                    "VoyageMultimodalEmbedder needs VOYAGE_API_KEY or an explicit api_key"
                )
            try:
                import voyageai
            except ImportError as exc:  # pragma: no cover
                raise ImportError(
                    'VoyageMultimodalEmbedder requires: pip install "recall-rag[voyage]"'
                ) from exc
            client = voyageai.Client(api_key=key, max_retries=0)
        self._client = client
        probe = self._embed_provider_inputs([{"content": [{"type": "text", "text": "probe"}]}])
        self._dim = len(probe[0])
        _check_declared_width(identity, self._dim, "the Voyage multimodal endpoint")

    @property
    def dim(self) -> int:
        return self._dim

    @property
    def name(self) -> str:
        return f"voyage-multimodal:{self._model}"

    @property
    def profile(self) -> EmbeddingProfile | None:
        return self._profile

    def _embed_provider_inputs(self, inputs: list[dict[str, object]]) -> list[list[float]]:
        multimodal_embed = getattr(self._client, "multimodal_embed", None)
        if callable(multimodal_embed):
            result = retry_with_backoff(
                lambda: multimodal_embed(
                    inputs=inputs,
                    model=self._model,
                    input_type="document",
                ),
                attempts=self._max_retries,
            )
        else:
            texts: list[str] = []
            for item in inputs:
                content = item.get("content")
                if (
                    not isinstance(content, list)
                    or len(content) != 1
                    or not isinstance(content[0], Mapping)
                    or content[0].get("type") != "text"
                    or not isinstance(content[0].get("text"), str)
                ):
                    raise RuntimeError(
                        "the Voyage client lacks multimodal_embed for non-text input"
                    )
                texts.append(content[0]["text"])
            result = retry_with_backoff(
                lambda: self._client.embed(
                    texts,
                    model=self._model,
                    input_type="document",
                ),
                attempts=self._max_retries,
            )
        vectors = getattr(result, "embeddings", None)
        if not isinstance(vectors, list) or len(vectors) != len(inputs):
            raise RuntimeError("Voyage multimodal response did not preserve input alignment")
        return [[float(value) for value in vector] for vector in vectors]

    def embed_multimodal(self, inputs: Sequence[Mapping[str, object]]) -> list[list[float]]:
        normalized = [dict(item) for item in inputs]
        vectors: list[list[float]] = []
        for start in range(0, len(normalized), self._batch_size):
            vectors.extend(self._embed_provider_inputs(normalized[start : start + self._batch_size]))
        return vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self.embed_multimodal(
            [{"content": [{"type": "text", "text": text}]} for text in texts]
        )

    def embed_query(self, query: str | MultimodalQuery) -> list[float]:
        provider_input = (
            {"content": [{"type": "text", "text": query}]}
            if isinstance(query, str)
            else query.provider_input()
        )
        result = retry_with_backoff(
            lambda: self._client.multimodal_embed(
                inputs=[provider_input], model=self._model, input_type="query"
            ),
            attempts=self._max_retries,
        )
        vectors = getattr(result, "embeddings", None)
        if not isinstance(vectors, list) or len(vectors) != 1:
            raise RuntimeError("Voyage multimodal query response did not contain one vector")
        vector = [float(value) for value in vectors[0]]
        _check_declared_width(self._profile, len(vector), "the Voyage multimodal endpoint")
        return vector


__all__ = [
    "DEFAULT_MAX_MEDIA_BYTES",
    "DEFAULT_MAX_RESPONSE_BYTES",
    "MediaAccessDenied",
    "MediaBudgetExceeded",
    "MediaObjectRef",
    "MediaValidationError",
    "MultimodalQuery",
    "MultimodalSidecar",
    "MultimodalTenantConfig",
    "MULTIMODAL_DIMENSION",
    "MULTIMODAL_EMBEDDING_MODEL",
    "MULTIMODAL_EMBEDDING_PROFILE",
    "MULTIMODAL_TENANT",
    "SUPPORTED_MEDIA_TYPES",
    "VoyageMultimodalEmbedder",
    "build_media_ref",
    "media_digest",
    "project_media_evidence",
]
