"""Bounded provenance and the isolated Voyage multimodal embedding tenant.

This module deliberately separates three things that are easy to conflate: the original media
object, the bounded text sidecar stored with a retrieval record, and the provider request. The
first is addressed by a controlled URI and digest, the second never contains binary data, and the
third may contain a transient base64 representation only while a request is in flight.
"""

from __future__ import annotations

import base64
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
from io import BytesIO
from itertools import islice
import logging
import math
import os
import posixpath
import warnings
from urllib.parse import SplitResult, unquote, urlsplit
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Protocol

from recall import embeddings as _embeddings
from recall._env import strict_bool
from recall.embeddings import (
    MULTIMODAL_EMBEDDING_MODEL as MULTIMODAL_EMBEDDING_MODEL,
    MULTIMODAL_EMBEDDING_PROFILE as MULTIMODAL_EMBEDDING_PROFILE,
    EmbeddingProfile,
    _check_declared_width,
    retry_with_backoff,
)
from recall.errors import RecallError
from recall.lineage import is_sha256_hex


MULTIMODAL_TENANT = "re-call-multimodal"
MULTIMODAL_DIMENSION = 1024
DEFAULT_MAX_MEDIA_BYTES = 30 * 1024 * 1024
DEFAULT_MAX_RESPONSE_BYTES = 30 * 1024 * 1024
DEFAULT_MAX_ITEMS = 20
DEFAULT_MAX_SIDECAR_CHARS = 64_000

SUPPORTED_MEDIA_TYPES = frozenset({"image/jpeg", "image/png", "image/webp"})
PROTECTED_TENANTS = frozenset({"default", "memory", "re-call-code-gen", "re-call-docs"})
# The fixed multimodal tenant must never alias a text specialist; checked once, at import.
if MULTIMODAL_TENANT in PROTECTED_TENANTS:  # pragma: no cover - a constant invariant
    raise RuntimeError("MULTIMODAL_TENANT overlaps a protected text specialist tenant")
#: Every policy requires the owning tenant. "private" also requires a non-empty principal;
#: "public" is recorded but enforced exactly as "tenant" is today.
AccessPolicy = Literal["private", "tenant", "public"]


class MultimodalError(ValueError, RecallError):
    """Base class for fail closed multimodal validation and access errors."""


class MediaValidationError(MultimodalError):
    """The media or its provenance is not safe to admit."""


class MediaAccessDenied(MultimodalError):
    """The caller cannot access the original media reference."""


class MediaBudgetExceeded(MultimodalError):
    """An upload, an image query, or a projected response would exceed its media byte budget."""

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
_INSTALL_HINT = (
    "VoyageMultimodalEmbedder requires voyageai>=0.3 and Pillow>=10: "
    'pip install "recall-rag[voyage]" "voyageai>=0.3" "pillow>=10"'
)

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


def _image_pixels(payload: bytes, image_format: str | None) -> int | None:
    """Width times height of an encoded image, or None when it cannot be read.

    Only the decoder of the declared format is tried, so a PNG label cannot hand Pillow an EPS
    file and with it Ghostscript; bytes of a type outside `_IMAGE_FORMATS` are not opened at all.
    """
    if image_format is None:
        return None
    try:
        from PIL import Image

        with Image.open(BytesIO(payload), formats=[image_format]) as image:
            return int(image.width * image.height)
    except Exception:  # BROAD-CATCH: fail-open; ImportError or any decoder error means unknown
        return None


def fit_voyage_image(data_url: str, *, strict: bool = False) -> tuple[str, bool]:
    """Fit an image data URL within Voyage's pixel and byte limits, and say whether it changed.

    An image already within both limits is returned exactly as given, so nothing Voyage accepted
    before is re-encoded. One over either limit is scaled down, keeping its aspect ratio and
    format, until it fits: first to the pixel limit, then by a further fifth of each side while
    the encoding is still over the byte limit. Only the transient provider input changes; nothing
    that stores or returns the original ever sees the fitted copy. Without Pillow, for bytes
    that do not decode, or for a type other than JPEG, PNG or WebP (never opened, in either
    mode), the input is returned unchanged and the provider decides.

    With ``strict``, an image of a supported type whose header Pillow cannot open raises (as
    does a missing Pillow) instead of passing through: usually ``UnidentifiedImageError``, though a truncated file can
    raise a plain ``OSError``, and so can a resize of an image over the pixel limit. Only the
    header is read, so an image within both limits whose pixel data is corrupt behind a readable
    header is still returned unchanged and sent. The AML adapter uses it, because its admission
    checks only the MIME type, size and magic bytes.
    """
    fitted, changed, _ = _fit_voyage_image(data_url, strict=strict)
    return fitted, changed


def _fit_voyage_image(
    data_url: str, *, strict: bool = False
) -> tuple[str, bool, int | None]:
    """Fit an image and retain the resulting pixel count for this request's estimator."""
    decoded = _split_data_url(data_url)
    if decoded is None:
        return data_url, False, None
    media_type, payload = decoded
    image_format = _IMAGE_FORMATS.get(media_type)
    if image_format is None:
        # A type RE-call cannot fit is not opened at all, in either mode; the provider decides.
        return data_url, False, None
    if strict:
        from PIL import Image

        with Image.open(BytesIO(payload), formats=[image_format]) as probe:
            pixels: int | None = int(probe.width * probe.height)
    else:
        pixels = _image_pixels(payload, image_format)
    if pixels is None:
        return data_url, False, pixels
    if pixels <= VOYAGE_MAX_IMAGE_PIXELS and len(payload) <= VOYAGE_MAX_IMAGE_BYTES:
        return data_url, False, pixels
    from PIL import Image, ImageOps

    # Fitting decodes the whole image. Pillow refuses one over twice its `MAX_IMAGE_PIXELS` but
    # only warns below that, so a small compressed file declaring ~150M pixels is decoded in full
    # here. Strict mode refuses it at Pillow's own limit. The default mode keeps fitting it: a
    # legitimate large scan within the caller's byte cap still embeds, where passing it through
    # unfitted would make Voyage refuse the whole request it was batched into.
    ceiling = Image.MAX_IMAGE_PIXELS
    if strict and ceiling is not None and pixels > ceiling:
        raise Image.DecompressionBombError(
            f"image declares {pixels} pixels, over the decode limit of {ceiling}"
        )

    try:
        with Image.open(BytesIO(payload), formats=[image_format]) as original:
            image = ImageOps.exif_transpose(original)
            scale = min(1.0, math.sqrt(VOYAGE_MAX_IMAGE_PIXELS / (image.width * image.height)))
            while True:
                width = max(1, int(image.width * scale))
                height = max(1, int(image.height * scale))
                # The jumps reproduce the old one-pixel-per-step loop's result in at most a few
                # steps; one pixel at a time took millions of steps for a 1 x N strip.
                while width * height > VOYAGE_MAX_IMAGE_PIXELS:
                    if width >= height:
                        width = max(VOYAGE_MAX_IMAGE_PIXELS // height, height - 1)
                    else:
                        height = max(VOYAGE_MAX_IMAGE_PIXELS // width, width)
                resized = image.resize((width, height), Image.Resampling.LANCZOS)
                if image_format == "JPEG" and resized.mode not in {"L", "RGB"}:
                    resized = resized.convert("RGB")
                output = BytesIO()
                resized.save(output, format=image_format)
                fitted = output.getvalue()
                if len(fitted) <= VOYAGE_MAX_IMAGE_BYTES or width * height <= 1:
                    break
                scale *= 0.8
    except (OSError, ValueError, SyntaxError):
        if strict:
            raise
        return data_url, False, pixels
    encoded = base64.b64encode(fitted).decode("ascii")
    return f"data:{media_type};base64,{encoded}", True, width * height


def fit_voyage_input(item: Mapping[str, object]) -> tuple[dict[str, object], int]:
    """A copy of one provider input with every image fitted, and how many images changed.

    Deprecated: nothing calls it; `VoyageMultimodalEmbedder.embed_multimodal` fits its inputs.
    """
    warnings.warn(
        "recall.multimodal.fit_voyage_input is deprecated and will be removed",
        DeprecationWarning,
        stacklevel=2,
    )
    return _fit_voyage_input(item, {})


def _fit_voyage_input(
    item: Mapping[str, object], image_pixels: dict[str, int | None]
) -> tuple[dict[str, object], int]:
    """Prepare one input, carrying scalar image measurements to the next stage."""
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
            url, transformed, pixels = _fit_voyage_image(str(part["image_base64"]))
            image_pixels[url] = pixels
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

    Deprecated: use `estimate_voyage_inputs`, which also skips the tokenizer when it cannot
    matter.
    """
    warnings.warn(
        "recall.multimodal.estimate_voyage_tokens is deprecated; use estimate_voyage_inputs",
        DeprecationWarning,
        stacklevel=2,
    )
    return _estimate_voyage_tokens(item, count_text, {})


def _estimate_voyage_tokens(
    item: Mapping[str, object],
    count_text: Callable[[list[str]], list[int]],
    image_pixels: dict[str, int | None],
) -> tuple[int, int]:
    """Estimate one input, reusing pixel counts keyed by the exact provider data URL."""
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
        url = str(part.get("image_base64"))
        if url not in image_pixels:
            decoded = _split_data_url(url)
            image_pixels[url] = (
                _image_pixels(decoded[1], _IMAGE_FORMATS.get(decoded[0]))
                if decoded is not None
                else None
            )
        pixels = image_pixels[url]
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
    return _estimate_voyage_inputs(items, count_text, {})


def _estimate_voyage_inputs(
    items: Sequence[Mapping[str, object]],
    count_text: TokenCounter,
    image_pixels: dict[str, int | None],
) -> list[tuple[int, int]]:
    """Reuse request-local scalar measurements, never decoded image buffers, across text passes."""
    bound = [
        _estimate_voyage_tokens(item, _embeddings._utf8_token_bound, image_pixels)
        for item in items
    ]
    budget = int(VOYAGE_MAX_REQUEST_TOKENS * VOYAGE_REQUEST_TOKEN_HEADROOM)
    if (
        all(tokens <= VOYAGE_MAX_INPUT_TOKENS for tokens, _ in bound)
        and sum(tokens for tokens, _ in bound) <= budget
    ):
        return bound
    return [_estimate_voyage_tokens(item, count_text, image_pixels) for item in items]



def media_digest(payload: bytes) -> str:
    """Return the content address used for media provenance and linkage."""
    return hashlib.sha256(payload).hexdigest()


def _validate_digest(value: str) -> str:
    normalized = value.strip().lower()
    if not is_sha256_hex(normalized):
        raise MediaValidationError("media content_digest must be a SHA256 hex digest")
    return normalized


def _validate_uri(value: str, field: str) -> str:
    normalized = value.strip()
    if not normalized or not normalized.startswith(("s3://", "file://")):
        raise MediaValidationError(f"{field} must use a controlled s3:// or file:// URI")
    return normalized


_MAX_URI_DECODE_PASSES = 3
_OUTSIDE_ROOT = "object_uri is outside the configured multimodal object root"


def _split_controlled_uri(value: str, field: str) -> SplitResult:
    """A controlled URI's parts, refused with this module's own error when it does not parse."""
    validated = _validate_uri(value, field)
    try:
        return urlsplit(validated)
    except ValueError as exc:
        raise MediaValidationError(f"{field} is not a valid URI") from exc


def _decoded_path(path: str, error: str) -> tuple[str, str]:
    """The path decoded to its fixpoint, literally and normalised, refusing what a consumer could escape by."""
    decoded = path
    for _ in range(_MAX_URI_DECODE_PASSES):
        if any(part == ".." for part in decoded.split("/")):
            raise MediaValidationError(error)
        next_decoded = unquote(decoded)
        if next_decoded == decoded:
            break
        decoded = next_decoded
    # A path still changing after the last pass would hide its remaining encodings from the
    # checks below, and a backslash is a separator to a Windows consumer but not to posixpath.
    # A leading empty segment is refused because normpath folds it away and the '//' left in the
    # stored URI is a UNC host to a Windows consumer. One later in the path is a legitimate key
    # (S3 allows them); `_validate_object_root` admits it only after the root's own prefix.
    if (
        unquote(decoded) != decoded
        or "\\" in decoded
        or "\x00" in decoded
        or decoded.startswith("//")
    ):
        raise MediaValidationError(error)
    normalized = posixpath.normpath(decoded)
    if (
        any(part == ".." for part in decoded.split("/"))
        or normalized == ".."
        or normalized.startswith("../")
    ):
        raise MediaValidationError(error)
    return decoded, normalized


def _object_root_path(object_root: str, field: str) -> tuple[SplitResult, str, str]:
    """A root's parts, its normalised and its literal decoded paths, refused naming the root."""
    root = _split_controlled_uri(object_root, field)
    literal_root, root_path = _decoded_path(root.path, f"{field} is not a usable object root")
    if not root_path.startswith("/"):
        raise MediaValidationError(
            f"{field} must name an absolute path prefix, such as s3://bucket/prefix/"
        )
    return root, root_path, literal_root


def _validate_object_root(object_uri: str, object_root: str) -> str:
    uri = _split_controlled_uri(object_uri, "object_uri")
    root, root_path, literal_root = _object_root_path(object_root, "object_root")
    if (uri.scheme, uri.netloc) != (root.scheme, root.netloc):
        raise MediaValidationError(_OUTSIDE_ROOT)
    literal_path, uri_path = _decoded_path(uri.path, _OUTSIDE_ROOT)
    try:
        common_path = posixpath.commonpath([uri_path, root_path])
    except ValueError:
        common_path = ""
    if (
        uri_path == root_path
        or common_path != root_path
        or not uri_path.startswith(root_path.rstrip("/") + "/")
        # An S3 key is literal: an empty or '.' segment in either path names another key than
        # normpath suggests, so the literal paths must agree on the root's own prefix too.
        or not literal_path.startswith(
            literal_root if literal_root.endswith("/") else literal_root + "/"
        )
    ):
        raise MediaValidationError(_OUTSIDE_ROOT)
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
        # Stored as validated, as content_digest is: a padded tenant_id would make
        # project_media_evidence deny the media its own tenant.
        object.__setattr__(self, "object_uri", _validate_uri(self.object_uri, "object_uri"))
        if self.source_uri is not None:
            object.__setattr__(self, "source_uri", _validate_uri(self.source_uri, "source_uri"))
        object.__setattr__(self, "tenant_id", self.tenant_id.strip())
        if not self.tenant_id:
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
            if not self.image_bytes:
                raise MediaValidationError("image query bytes must not be empty")
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
    object_root: str | None = None,
) -> dict[str, object]:
    """Project a media reference's provenance for an authorized requester, or refuse.

    Raises `MediaAccessDenied` for a requester outside the owning tenant, or for private media
    without a principal, and `MediaBudgetExceeded` when ``used_bytes`` plus the media's
    ``byte_size`` passes ``max_response_bytes``; nothing is projected in either case. The budget
    is charged whatever ``include_original`` says. ``include_original`` controls only the
    ``original_media`` entry: the provenance itself always carries ``object_uri`` and
    ``content_digest``.

    A reference need not come from `build_media_ref`, so its ``object_uri`` is checked again
    under ``object_root``, or under ``RECALL_MULTIMODAL_OBJECT_ROOT`` when that is set, and
    refused with `MediaValidationError` if it lies outside. With neither, it is not checked.
    """
    if requester_tenant != media.tenant_id:
        raise MediaAccessDenied("media may only be accessed from its owning tenant")
    if media.access_policy == "private" and not principal.strip():
        raise MediaAccessDenied("private media requires an authenticated principal")
    root = object_root
    if root is None:
        root = os.environ.get("RECALL_MULTIMODAL_OBJECT_ROOT", "").strip() or None
    if root is not None:
        _validate_object_root(media.object_uri, root)
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
            # Strict, as the server's own settings read it: a typo must not quietly disable the tenant.
            enabled=strict_bool(
                source.get("RECALL_MULTIMODAL_ENABLED"), name="RECALL_MULTIMODAL_ENABLED"
            ),
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
        if self.embedding_profile != MULTIMODAL_EMBEDDING_PROFILE:
            raise MediaValidationError("multimodal embedding profile is fixed and registered")
        for variable, value, ceiling in (
            ("RECALL_MULTIMODAL_MAX_MEDIA_BYTES", self.max_media_bytes, DEFAULT_MAX_MEDIA_BYTES),
            (
                "RECALL_MULTIMODAL_MAX_RESPONSE_BYTES",
                self.max_response_bytes,
                DEFAULT_MAX_RESPONSE_BYTES,
            ),
            ("RECALL_MULTIMODAL_MAX_ITEMS", self.max_items, DEFAULT_MAX_ITEMS),
        ):
            if not 1 <= value <= ceiling:
                raise MediaValidationError(f"{variable} is {value}; it must be between 1 and {ceiling}")
        if self.enabled and not self.object_root:
            raise MediaValidationError(
                "RECALL_MULTIMODAL_OBJECT_ROOT is required when multimodal retrieval is enabled"
            )
        # build_media_ref refuses every upload under a root it cannot contain, so refuse that
        # root here, at startup, rather than at the first admission.
        if self.enabled and self.object_root:
            _object_root_path(self.object_root, "RECALL_MULTIMODAL_OBJECT_ROOT")

    def physical_identity(self) -> dict[str, object]:
        return {
            "tenant_id": self.tenant_id,
            "embedding_profile": self.embedding_profile,
            "object_root": self.object_root,
            "max_media_bytes": self.max_media_bytes,
            "max_response_bytes": self.max_response_bytes,
            "max_items": self.max_items,
        }


class _TextClient(Protocol):
    """The least a client needs. A client with ``multimodal_embed`` too (the Voyage SDK's) is
    used through that, found at call time; one without it serves text inputs through ``embed``."""

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
        client: _TextClient | None = None,
        token_counter: Callable[[list[str]], list[int]] | None = None,
        timeout: float | None = None,
        max_parallel_requests: int = 1,
    ) -> None:
        """Build the client and probe the endpoint's width.

        Construction sends one embedding request (a short text) to learn the vector width and
        check it against ``identity``, so it needs a reachable provider and spends credit.
        ``identity.model_name``, when given, overrides ``model``. ``timeout`` (seconds) applies
        only to the client built here, not to an injected ``client``; without it the
        ``RECALL_VOYAGE_TIMEOUT_SECONDS`` value of the process environment is used. A client
        without ``multimodal_embed`` is accepted and serves text inputs through ``embed``.
        ``max_parallel_requests`` bounds the planned requests one call keeps in flight.
        """
        if max_parallel_requests < 1:
            raise ValueError("Voyage multimodal parallel requests must be positive")
        self._max_parallel_requests = max_parallel_requests
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
                raise ImportError(_INSTALL_HINT) from exc
            if not callable(getattr(voyageai.Client, "multimodal_embed", None)):
                # An older SDK has no multimodal endpoint; its text fallback would serve the
                # probe and then fail on the first image.
                raise ImportError(_INSTALL_HINT)
            from recall.embedding_registry import _voyage_timeout

            # The SDK's default timeout is none, so a hung socket would block the caller forever.
            client = voyageai.Client(
                api_key=key,
                max_retries=0,
                timeout=_voyage_timeout(os.environ) if timeout is None else timeout,
            )
        try:
            import PIL  # noqa: F401
        except ImportError:
            # Without Pillow no image can be measured or fitted, so an image over Voyage's
            # limits is sent as given and the provider refuses the whole request holding it.
            log.warning(
                "voyage_multimodal_images_unfitted",
                extra={"reason": 'Pillow is not installed; pip install "pillow>=10"'},
            )
        self._client = client
        self._dim = 0  # set from the probe below, which every later response must match
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

    def _embed_provider_inputs(
        self, inputs: list[dict[str, object]], input_type: str = "document"
    ) -> list[list[float]]:
        multimodal_embed = getattr(self._client, "multimodal_embed", None)
        if callable(multimodal_embed):
            result = retry_with_backoff(
                lambda: multimodal_embed(
                    inputs=inputs,
                    model=self._model,
                    input_type=input_type,
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
                    input_type=input_type,
                ),
                attempts=self._max_retries,
            )
        vectors = getattr(result, "embeddings", None)
        if not isinstance(vectors, list) or len(vectors) != len(inputs):
            raise RuntimeError("Voyage multimodal response did not preserve input alignment")
        floats = [[float(value) for value in vector] for vector in vectors]
        # Every vector must have the width the construction probe measured (the probe itself
        # sets it), so a model change or a truncated response cannot mix widths in one store.
        if any(not vector for vector in floats):
            raise RuntimeError("Voyage multimodal response holds an empty vector")
        expected = self._dim or (len(floats[0]) if floats else 0)
        if any(len(vector) != expected for vector in floats):
            widths = sorted({len(vector) for vector in floats})
            raise RuntimeError(
                f"Voyage multimodal response vectors are {widths} wide, not all {expected}"
            )
        return floats

    def _estimate(
        self, items: list[dict[str, object]], image_pixels: dict[str, int | None]
    ) -> list[tuple[int, int]]:
        return _estimate_voyage_inputs(items, self._text_counter, image_pixels)

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
        """Embed every input, returning the vectors in input order.

        Every image is first fitted, non-strictly, to Voyage's per-image pixel and byte limits;
        one that cannot be read is sent as given. The inputs are then packed into consecutive
        requests of at most ``batch_size`` inputs and, by the local estimate, 90% of Voyage's
        320K-token request cap; an input over that budget alone is sent in a request of its own.
        An input over the 32K-token context is sent unchanged and logged, and the provider
        truncates it. The provider embeds each input on its own, so packing changes no vector.
        Up to ``max_parallel_requests`` requests are in flight at once; when one fails, the
        requests not yet sent are cancelled and its error is raised once those in flight end.
        """
        image_pixels: dict[str, int | None] = {}
        prepared = [_fit_voyage_input(item, image_pixels) for item in inputs]
        normalized = [item for item, _ in prepared]
        estimates = self._estimate(normalized, image_pixels)
        for index, ((tokens, images), (_, fitted)) in enumerate(zip(estimates, prepared)):
            self._log_input(index, "document", tokens, images, fitted)
        requests = [
            normalized[start:stop]
            for start, stop in plan_voyage_requests(
                [tokens for tokens, _ in estimates], self._batch_size
            )
        ]
        if self._max_parallel_requests > 1 and len(requests) > 1:
            workers = min(self._max_parallel_requests, len(requests))
            # A request is submitted only when a worker is free, so nothing waits in the pool's
            # queue: a worker that has just failed would otherwise take the next request before
            # the failure is seen. After a failure no further request is sent; one already in
            # flight cannot be recalled and is waited for by the shutdown below.
            pending = iter(enumerate(requests))
            by_index: dict[int, list[list[float]]] = {}
            pool = ThreadPoolExecutor(max_workers=workers)
            try:
                running = {
                    pool.submit(self._embed_provider_inputs, request): index
                    for index, request in islice(pending, workers)
                }
                while running:
                    done, _ = wait(running, return_when=FIRST_COMPLETED)
                    for future in done:
                        by_index[running.pop(future)] = future.result()
                    for index, request in islice(pending, len(done)):
                        running[pool.submit(self._embed_provider_inputs, request)] = index
            finally:
                pool.shutdown(wait=True)
            responses = [by_index[index] for index in range(len(requests))]
        else:
            responses = [self._embed_provider_inputs(request) for request in requests]
        return [vector for response in responses for vector in response]

    def embed(self, texts: list[str]) -> list[list[float]]:
        return self.embed_passages(texts)

    def embed_passages(self, texts: list[str]) -> list[list[float]]:
        return self.embed_multimodal(
            [{"content": [{"type": "text", "text": text}]} for text in texts]
        )

    def embed_query(self, query: str | MultimodalQuery) -> list[float]:
        image_pixels: dict[str, int | None] = {}
        provider_input, fitted = _fit_voyage_input(
            {"content": [{"type": "text", "text": query}]}
            if isinstance(query, str)
            else query.provider_input(),
            image_pixels,
        )
        [(tokens, images)] = self._estimate([provider_input], image_pixels)
        self._log_input(0, "query", tokens, images, fitted)
        [vector] = self._embed_provider_inputs([provider_input], input_type="query")
        _check_declared_width(self._profile, len(vector), "the Voyage multimodal endpoint")
        return vector


__all__ = [
    "DEFAULT_MAX_ITEMS",
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
    "VOYAGE_MAX_INPUT_TOKENS",
    "VoyageMultimodalEmbedder",
    "build_media_ref",
    "estimate_voyage_inputs",
    "fit_voyage_image",
    "lazy_voyage_text_counter",
    "media_digest",
    "plan_voyage_requests",
    "project_media_evidence",
]
