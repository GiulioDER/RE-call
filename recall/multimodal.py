"""Bounded provenance and the isolated Voyage multimodal embedding tenant.

This module deliberately separates three things that are easy to conflate: the original media
object, the bounded text sidecar stored with a retrieval record, and the provider request. The
first is addressed by a controlled URI and digest, the second never contains binary data, and the
third may contain a transient base64 representation only while a request is in flight.
"""

from __future__ import annotations

import base64
import hashlib
from io import BytesIO
import logging
import math
import os
import posixpath
from urllib.parse import unquote, urlsplit
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from recall import embeddings as _embeddings
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

#: Voyage's multimodal limits (docs.voyageai.com/reference/multimodal-embeddings-api, read
#: 2026-09-30). An image counts one token per 560 pixels. Over either request cap, or over an
#: image's pixel or byte limit, Voyage refuses the whole request with HTTP 400. Over the per-input
#: context it truncates instead (`truncation` defaults to true), discarding an image cut mid-way.
VOYAGE_MAX_REQUEST_INPUTS = 1_000
VOYAGE_MAX_REQUEST_TOKENS = 320_000
VOYAGE_MAX_INPUT_TOKENS = 32_000
VOYAGE_PIXELS_PER_TOKEN = 560
VOYAGE_MAX_IMAGE_PIXELS = 16_000_000
#: "20 MB" in the documentation, read as decimal megabytes, the stricter of the two readings.
VOYAGE_MAX_IMAGE_BYTES = 20_000_000
#: The share of the request token cap a planned request may fill by the local estimate. The
#: estimate counts text with Voyage's published tokenizer and images from their pixels; the
#: provider decides, so the remaining tenth absorbs any disagreement.
VOYAGE_REQUEST_TOKEN_HEADROOM = 0.9

_IMAGE_FORMATS = {"image/jpeg": "JPEG", "image/png": "PNG", "image/webp": "WEBP"}

log = logging.getLogger("recall.multimodal")


def _split_data_url(data_url: str) -> tuple[str, bytes] | None:
    """The media type and decoded bytes of a base64 data URL, or None if it is not one."""
    header, separator, payload = data_url.partition(",")
    if not separator or not header.startswith("data:") or not header.endswith(";base64"):
        return None
    try:
        return header[len("data:") : -len(";base64")], base64.b64decode(payload, validate=True)
    except ValueError:
        return None


def _image_pixels(payload: bytes) -> int | None:
    """Width times height of an encoded image, or None when it cannot be read."""
    try:
        from PIL import Image

        with Image.open(BytesIO(payload)) as image:
            return int(image.width * image.height)
    except Exception:  # BROAD-CATCH: fail-open; ImportError or any decoder error means unknown
        return None


def fit_voyage_image(data_url: str, *, strict: bool = False) -> tuple[str, bool]:
    """Fit an image data URL within Voyage's pixel and byte limits, and say whether it changed.

    An image already within both limits is returned exactly as given, so nothing Voyage accepted
    before is re-encoded. One over either limit is scaled down, keeping its aspect ratio and
    format, until it fits: first to the pixel limit, then by a further fifth of each side while
    the encoding is still over the byte limit. Only the transient provider input changes; nothing
    that stores or returns the original ever sees the fitted copy. Without Pillow, or for bytes
    that do not decode, the input is returned unchanged and the provider decides.

    With ``strict``, an image Pillow cannot read raises (as does a missing Pillow) instead of
    passing through. The AML adapter uses it: it has always refused such an image rather than
    sending it, and its admission checks only the MIME type, size and magic bytes.
    """
    decoded = _split_data_url(data_url)
    if decoded is None:
        return data_url, False
    media_type, payload = decoded
    image_format = _IMAGE_FORMATS.get(media_type)
    if strict:
        from PIL import Image

        with Image.open(BytesIO(payload)) as probe:
            pixels: int | None = int(probe.width * probe.height)
    else:
        pixels = _image_pixels(payload)
    if image_format is None or pixels is None:
        return data_url, False
    if pixels <= VOYAGE_MAX_IMAGE_PIXELS and len(payload) <= VOYAGE_MAX_IMAGE_BYTES:
        return data_url, False
    from PIL import Image

    with Image.open(BytesIO(payload)) as image:
        scale = min(1.0, math.sqrt(VOYAGE_MAX_IMAGE_PIXELS / (image.width * image.height)))
        while True:
            width = max(1, int(image.width * scale))
            height = max(1, int(image.height * scale))
            while width * height > VOYAGE_MAX_IMAGE_PIXELS:
                if width >= height:
                    width -= 1
                else:
                    height -= 1
            resized = image.resize((width, height), Image.Resampling.LANCZOS)
            if image_format == "JPEG" and resized.mode not in {"L", "RGB"}:
                resized = resized.convert("RGB")
            output = BytesIO()
            resized.save(output, format=image_format)
            fitted = output.getvalue()
            if len(fitted) <= VOYAGE_MAX_IMAGE_BYTES or width * height <= 1:
                break
            scale *= 0.8
    encoded = base64.b64encode(fitted).decode("ascii")
    return f"data:{media_type};base64,{encoded}", True


def fit_voyage_input(item: Mapping[str, object]) -> tuple[dict[str, object], int]:
    """A copy of one provider input with every image fitted, and how many images changed."""
    fitted = dict(item)
    content = item.get("content")
    if not isinstance(content, list):
        return fitted, 0
    parts: list[object] = []
    changed = 0
    for part in content:
        if (
            isinstance(part, Mapping)
            and part.get("type") == "image_base64"
            and isinstance(part.get("image_base64"), str)
        ):
            url, transformed = fit_voyage_image(str(part["image_base64"]))
            if transformed:
                changed += 1
                part = {**part, "image_base64": url}
        parts.append(part)
    fitted["content"] = parts
    return fitted, changed


def estimate_voyage_tokens(
    item: Mapping[str, object], count_text: Callable[[list[str]], list[int]]
) -> tuple[int, int]:
    """Estimated tokens of one provider input, and its image count.

    Text is counted with ``count_text``, an image as its pixels over `VOYAGE_PIXELS_PER_TOKEN`.
    A part this cannot measure (an image that will not decode, or a content type RE-call never
    builds) counts as a whole context window, which at worst sends that input in a request of
    its own.
    """
    content = item.get("content")
    parts = content if isinstance(content, list) else []
    texts = [
        str(part["text"])
        for part in parts
        if isinstance(part, Mapping)
        and part.get("type") == "text"
        and isinstance(part.get("text"), str)
    ]
    tokens = sum(count_text(texts)) if texts else 0
    images = 0
    for part in parts:
        if isinstance(part, Mapping) and part.get("type") == "text":
            continue
        if not isinstance(part, Mapping) or part.get("type") != "image_base64":
            tokens += VOYAGE_MAX_INPUT_TOKENS
            continue
        images += 1
        decoded = _split_data_url(str(part.get("image_base64")))
        pixels = _image_pixels(decoded[1]) if decoded is not None else None
        tokens += (
            VOYAGE_MAX_INPUT_TOKENS
            if pixels is None
            else math.ceil(pixels / VOYAGE_PIXELS_PER_TOKEN)
        )
    return tokens, images


def plan_voyage_requests(
    estimates: Sequence[int], max_inputs: int = VOYAGE_MAX_REQUEST_INPUTS
) -> list[tuple[int, int]]:
    """Consecutive ``[start, stop)`` ranges under the input and token caps, in order.

    An input whose estimate alone passes the budget still goes, in a request of its own.
    """
    budget = int(VOYAGE_MAX_REQUEST_TOKENS * VOYAGE_REQUEST_TOKEN_HEADROOM)
    limit = max(1, min(max_inputs, VOYAGE_MAX_REQUEST_INPUTS))
    ranges: list[tuple[int, int]] = []
    start = 0
    used = 0
    for index, tokens in enumerate(estimates):
        if index > start and (index - start >= limit or used + tokens > budget):
            ranges.append((start, index))
            start = index
            used = 0
        used += tokens
    if start < len(estimates):
        ranges.append((start, len(estimates)))
    return ranges


TokenCounter = Callable[[list[str]], list[int]]


def lazy_voyage_text_counter(model: str) -> TokenCounter:
    """A text token counter for ``model`` that loads Voyage's tokenizer on its first call.

    Falls back, with a warning, to the UTF-8 byte bound when the tokenizer cannot be loaded. The
    loader is looked up through `recall.embeddings` at call time, so a test's patch applies.
    """
    loaded: list[TokenCounter] = []

    def count(texts: list[str]) -> list[int]:
        if not loaded:
            counter = _embeddings._voyage_token_counter(model)
            if counter is None:
                log.warning(
                    "voyage_multimodal_token_bound_fallback",
                    extra={"model": model, "bound": "utf8-bytes"},
                )
                counter = _embeddings._utf8_token_bound
            loaded.append(counter)
        return loaded[0](texts)

    return count


def estimate_voyage_inputs(
    items: Sequence[Mapping[str, object]], count_text: TokenCounter
) -> list[tuple[int, int]]:
    """Token and image estimates for provider inputs, calling ``count_text`` only if it matters.

    UTF-8 bytes bound the text tokens from above. When that bound already puts every input under
    the context and the whole call under the request budget, an exact count could not change the
    request plan or any warning, so ``count_text`` is never called and its tokenizer never loads.
    That keeps an ordinary query from paying the first load, about 4 s, or waiting on Hugging
    Face.
    """
    bound = [estimate_voyage_tokens(item, _embeddings._utf8_token_bound) for item in items]
    budget = int(VOYAGE_MAX_REQUEST_TOKENS * VOYAGE_REQUEST_TOKEN_HEADROOM)
    if (
        all(tokens <= VOYAGE_MAX_INPUT_TOKENS for tokens, _ in bound)
        and sum(tokens for tokens, _ in bound) <= budget
    ):
        return bound
    return [estimate_voyage_tokens(item, count_text) for item in items]



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
    def decode_path(path: str) -> str:
        decoded = path
        for _ in range(3):
            if any(part == ".." for part in decoded.split("/")):
                raise MediaValidationError(
                    "object_uri is outside the configured multimodal object root"
                )
            next_decoded = unquote(decoded)
            if next_decoded == decoded:
                break
            decoded = next_decoded
        normalized = posixpath.normpath(decoded)
        if (
            any(part == ".." for part in decoded.split("/"))
            or normalized == ".."
            or normalized.startswith("../")
        ):
            raise MediaValidationError(
                "object_uri is outside the configured multimodal object root"
            )
        return normalized

    uri_path = decode_path(uri.path)
    root_path = decode_path(root.path)
    if (
        not root_path.startswith("/")
    ):
        raise MediaValidationError("object_uri is outside the configured multimodal object root")
    normalized_uri_path = posixpath.normpath(uri_path)
    normalized_root_path = posixpath.normpath(root_path)
    try:
        common_path = posixpath.commonpath([normalized_uri_path, normalized_root_path])
    except ValueError:
        common_path = ""
    if (
        normalized_uri_path == normalized_root_path
        or common_path != normalized_root_path
        or not normalized_uri_path.startswith(normalized_root_path.rstrip("/") + "/")
    ):
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
    if object_root is not None:
        root = object_root.strip()
    else:
        configured_root = os.environ.get("RECALL_MULTIMODAL_OBJECT_ROOT")
        if configured_root is None or not configured_root.strip():
            raise MediaValidationError(
                "RECALL_MULTIMODAL_OBJECT_ROOT or an explicit object_root is required"
            )
        root = configured_root.strip()
    if not root:
        raise MediaValidationError("object_root must not be empty")
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
        token_counter: Callable[[list[str]], list[int]] | None = None,
    ) -> None:
        self._model = identity.model_name if identity is not None else model
        self._batch_size = batch_size
        self._max_retries = max_retries
        self._text_counter = token_counter or lazy_voyage_text_counter(self._model)
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
            from recall.embedding_registry import _voyage_timeout

            # The SDK's default timeout is none, so a hung socket would block the caller forever.
            client = voyageai.Client(
                api_key=key, max_retries=0, timeout=_voyage_timeout(os.environ)
            )
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

    def _estimate(self, items: list[dict[str, object]]) -> list[tuple[int, int]]:
        return estimate_voyage_inputs(items, self._text_counter)

    def _log_input(
        self, index: int, input_type: str, tokens: int, images: int, fitted: int
    ) -> None:
        if fitted:
            log.info(
                "voyage_multimodal_image_fitted",
                extra={"input_index": index, "input_type": input_type, "fitted_images": fitted},
            )
        if tokens > VOYAGE_MAX_INPUT_TOKENS:
            # Logged only, by the owner's decision of 2026-09-30: the input is sent unchanged and
            # the provider truncates it, since splitting it would change what its vector means.
            log.warning(
                "voyage_multimodal_input_over_context",
                extra={
                    "input_index": index,
                    "input_type": input_type,
                    "estimated_tokens": tokens,
                    "image_count": images,
                    "context_tokens": VOYAGE_MAX_INPUT_TOKENS,
                },
            )

    def embed_multimodal(self, inputs: Sequence[Mapping[str, object]]) -> list[list[float]]:
        """Embed every input, in order, in requests Voyage's caps accept.

        Requests were cut by count alone, so large images or long texts could pass the
        320K-token request cap and be refused whole. They are now packed under that cap as well
        as under ``batch_size``, and every image is fitted to Voyage's per-image pixel and byte
        limits first. The provider embeds each input on its own, so packing changes no vector,
        and a call already under every cap is sent exactly as before.
        """
        prepared = [fit_voyage_input(item) for item in inputs]
        normalized = [item for item, _ in prepared]
        estimates = self._estimate(normalized)
        for index, ((tokens, images), (_, fitted)) in enumerate(zip(estimates, prepared)):
            self._log_input(index, "document", tokens, images, fitted)
        vectors: list[list[float]] = []
        for start, stop in plan_voyage_requests(
            [tokens for tokens, _ in estimates], self._batch_size
        ):
            vectors.extend(self._embed_provider_inputs(normalized[start:stop]))
        return vectors

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self.embed_multimodal(
            [{"content": [{"type": "text", "text": text}]} for text in texts]
        )

    def embed_query(self, query: str | MultimodalQuery) -> list[float]:
        provider_input, fitted = fit_voyage_input(
            {"content": [{"type": "text", "text": query}]}
            if isinstance(query, str)
            else query.provider_input()
        )
        [(tokens, images)] = self._estimate([provider_input])
        self._log_input(0, "query", tokens, images, fitted)
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
