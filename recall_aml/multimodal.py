"""Ordered multimodal preservation and Voyage retrieval helpers for AML Hosted."""

from __future__ import annotations

import base64
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import math
from typing import Any, Protocol

from recall.types import Chunk, ScoredChunk
from recall_aml.config import EMBEDDING_PROFILE, RETRIEVAL_PROFILE
from recall_aml.identity import canonical_digest, session_digest
from recall_aml.models import (
    ContentValue,
    ImageContentPart,
    ImageURLValue,
    Message,
    SearchItem,
    TextContentPart,
    content_media_bytes,
    decode_image_data_url,
)


MULTIMODAL_EMBEDDING_MODEL = "voyage-multimodal-3.5"
MULTIMODAL_EMBEDDING_PROFILE = "voyage-multimodal-3.5-v2"
MULTIMODAL_DIMENSION = 1024
MULTIMODAL_RRF_CONSTANT = 60
MAX_VOYAGE_IMAGE_PIXELS = 16_000_000
RAW_SEGMENT_CHARS = 4_500
MAX_RESPONSE_MEDIA_BYTES = 30 * 1024 * 1024


def media_tenant(tenant: str) -> str:
    return tenant + "_media"


def multimodal_tenant(tenant: str) -> str:
    return tenant + "_mm"


def is_multimodal(value: ContentValue) -> bool:
    return not isinstance(value, str)


def content_text(value: ContentValue) -> str:
    """Return only supplied text, never generated image claims or encoded media."""
    if isinstance(value, str):
        return value
    text = "\n".join(part.text for part in value if isinstance(part, TextContentPart))
    return text if text.strip() else "[image evidence]"


def _iso(value: datetime | None) -> str | None:
    if value is None:
        return None
    return value.astimezone(timezone.utc).isoformat()


def _voyage_image_data_url(data_url: str) -> tuple[str, bool]:
    """Fit a derived embedding image to Voyage limits while preserving source bytes elsewhere."""
    media_type, payload = decode_image_data_url(data_url)
    from PIL import Image

    with Image.open(BytesIO(payload)) as image:
        pixels = image.width * image.height
        if pixels <= MAX_VOYAGE_IMAGE_PIXELS:
            return data_url, False
        scale = math.sqrt(MAX_VOYAGE_IMAGE_PIXELS / pixels)
        width = max(1, int(image.width * scale))
        height = max(1, int(image.height * scale))
        while width * height > MAX_VOYAGE_IMAGE_PIXELS:
            if width >= height:
                width -= 1
            else:
                height -= 1
        resized = image.resize((width, height), Image.Resampling.LANCZOS)
        image_format = {
            "image/jpeg": "JPEG",
            "image/png": "PNG",
            "image/webp": "WEBP",
        }[media_type]
        if image_format == "JPEG" and resized.mode not in {"L", "RGB"}:
            resized = resized.convert("RGB")
        output = BytesIO()
        resized.save(output, format=image_format)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:{media_type};base64,{encoded}", True


@dataclass(frozen=True)
class PreparedMultimodal:
    primary_chunks: list[Chunk]
    media_chunks: list[Chunk]
    vector_chunks: list[Chunk]
    voyage_inputs: list[dict[str, Any]]


def prepare_messages(
    messages: Sequence[Message],
    *,
    request_id: str,
    session_id: str,
    source: str,
    source_nul_replacements: int = 0,
) -> PreparedMultimodal:
    """Build text chunks, content addressed media, and ordered visual vector records."""
    primary: list[Chunk] = []
    media: dict[str, Chunk] = {}
    vectors: list[Chunk] = []
    voyage_inputs: list[dict[str, Any]] = []
    session_hash = session_digest(session_id)

    for ordinal, message in enumerate(messages):
        value = message.content
        if isinstance(value, str):
            parts: list[TextContentPart | ImageContentPart] = [
                TextContentPart(type="text", text=value)
            ]
        else:
            parts = list(value)

        manifest: list[dict[str, str]] = []
        voyage_content: list[dict[str, str]] = []
        image_transform_count = 0
        for part in parts:
            if isinstance(part, TextContentPart):
                manifest.append({"type": "text", "text": part.text})
                voyage_content.append({"type": "text", "text": part.text})
                continue
            media_type, payload = decode_image_data_url(part.image_url.url)
            digest = hashlib.sha256(payload).hexdigest()
            media_id = "media_" + digest
            manifest.append({"type": "image_ref", "media_id": media_id})
            voyage_image, transformed = _voyage_image_data_url(part.image_url.url)
            image_transform_count += int(transformed)
            voyage_content.append(
                {"type": "image_base64", "image_base64": voyage_image}
            )
            media.setdefault(
                media_id,
                Chunk(
                    id=media_id,
                    source="aml://media/" + digest,
                    text=f"{media_type} image sha256:{digest}",
                    metadata={
                        "record_type": "media",
                        "media_type": media_type,
                        "decoded_bytes": len(payload),
                        "sha256": digest,
                        "data_url": part.image_url.url,
                    },
                ),
            )

        supplied_text = content_text(value)
        segments = [
            supplied_text[index : index + RAW_SEGMENT_CHARS]
            for index in range(0, len(supplied_text), RAW_SEGMENT_CHARS)
        ] or ["[image evidence]"]
        timestamp = _iso(message.timestamp)
        parent_payload = {
            "request_id": request_id,
            "ordinal": ordinal,
            "role": message.role,
            "manifest": manifest,
            "timestamp": timestamp,
        }
        parent_id = "raw_" + canonical_digest(parent_payload)
        for segment_index, segment in enumerate(segments):
            chunk_id = (
                parent_id
                if segment_index == 0
                else parent_id + f"_segment_{segment_index:04d}"
            )
            prefix = f"timestamp: {timestamp}\n" if timestamp else ""
            metadata: dict[str, Any] = {
                "record_type": "raw",
                "kind": "multimodal",
                "source_session_id": session_id,
                "session_digest": session_hash,
                "event_time": timestamp,
                "embedding_profile": EMBEDDING_PROFILE,
                "retrieval_profile": RETRIEVAL_PROFILE,
                "ordinal": ordinal,
                "segment": segment_index,
                "segment_count": len(segments),
                "multimodal_parent_id": parent_id,
                "source_nul_replacements": source_nul_replacements,
                "file": f"{chunk_id}.md",
            }
            if segment_index == 0:
                metadata["multimodal_manifest"] = manifest
                metadata["multimodal_scalar"] = isinstance(value, str)
            primary.append(
                Chunk(
                    id=chunk_id,
                    source=source,
                    text=f"{prefix}role: {message.role}\ncontent: {segment}",
                    metadata=metadata,
                )
            )

        vectors.append(
            Chunk(
                id=parent_id,
                source=source,
                text=supplied_text,
                metadata={
                    "record_type": "multimodal_vector",
                    "primary_id": parent_id,
                    "source_session_id": session_id,
                    "session_digest": session_hash,
                    "event_time": timestamp,
                    "embedding_profile": MULTIMODAL_EMBEDDING_PROFILE,
                    "image_transform_count": image_transform_count,
                },
            )
        )
        voyage_inputs.append({"content": voyage_content})

    return PreparedMultimodal(primary, list(media.values()), vectors, voyage_inputs)


def to_voyage_input(value: ContentValue) -> dict[str, Any]:
    if isinstance(value, str):
        return {"content": [{"type": "text", "text": value}]}
    content: list[dict[str, str]] = []
    for part in value:
        if isinstance(part, TextContentPart):
            content.append({"type": "text", "text": part.text})
        else:
            voyage_image, _ = _voyage_image_data_url(part.image_url.url)
            content.append({"type": "image_base64", "image_base64": voyage_image})
    return {"content": content}


class MultimodalEmbedder(Protocol):
    dim: int
    profile: str

    def embed_documents(self, inputs: Sequence[dict[str, Any]]) -> list[list[float]]: ...
    def embed_query(self, value: ContentValue) -> list[float]: ...


class VoyageMultimodalEmbedder:
    """Single owner of Voyage multimodal embedding calls, with provider retries disabled."""

    dim = MULTIMODAL_DIMENSION
    profile = MULTIMODAL_EMBEDDING_PROFILE

    def __init__(self, api_key: str, *, client: Any = None, timeout: float = 180.0) -> None:
        if client is None:
            import voyageai

            client = voyageai.Client(api_key=api_key, max_retries=0, timeout=timeout)
        self._client = client

    def embed_documents(self, inputs: Sequence[dict[str, Any]]) -> list[list[float]]:
        materialized = list(inputs)
        if not materialized:
            return []
        response = self._client.multimodal_embed(
            materialized,
            model=MULTIMODAL_EMBEDDING_MODEL,
            input_type="document",
            truncation=True,
        )
        vectors = [list(vector) for vector in response.embeddings]
        self._validate(vectors, len(materialized))
        return vectors

    def embed_query(self, value: ContentValue) -> list[float]:
        response = self._client.multimodal_embed(
            [to_voyage_input(value)],
            model=MULTIMODAL_EMBEDDING_MODEL,
            input_type="query",
            truncation=True,
        )
        vectors = [list(vector) for vector in response.embeddings]
        self._validate(vectors, 1)
        return vectors[0]

    def _validate(self, vectors: Sequence[Sequence[float]], expected: int) -> None:
        if len(vectors) != expected or any(len(vector) != self.dim for vector in vectors):
            raise RuntimeError("Voyage multimodal embedder returned invalid dimensions")


def fuse_hits(
    text_hits: Sequence[ScoredChunk],
    visual_hits: Sequence[ScoredChunk],
) -> list[ScoredChunk]:
    """Fuse text and visual rankings by primary chunk id with deterministic RRF."""
    by_id = {hit.chunk.id: hit for hit in text_hits}
    visual_by_primary: dict[str, ScoredChunk] = {}
    for hit in visual_hits:
        primary_id = str(hit.chunk.metadata.get("primary_id", hit.chunk.id))
        visual_by_primary.setdefault(primary_id, hit)
    scores: dict[str, float] = {}
    for ranking in (
        [hit.chunk.id for hit in text_hits],
        list(visual_by_primary),
    ):
        for rank, chunk_id in enumerate(ranking, start=1):
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (
                MULTIMODAL_RRF_CONSTANT + rank
            )
    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))
    output: list[ScoredChunk] = []
    for chunk_id in ordered:
        if chunk_id in by_id:
            chunk = by_id[chunk_id].chunk
        else:
            chunk = visual_by_primary[chunk_id].chunk
        output.append(ScoredChunk(chunk=chunk, score=scores[chunk_id]))
    return output


def render_preserved(
    hits: Sequence[ScoredChunk],
    *,
    primary_by_id: dict[str, Chunk],
    media_by_id: dict[str, Chunk],
    top_k: int,
) -> list[SearchItem]:
    """Reconstruct exact ordered AML parts while enforcing the response media limit."""
    output: list[SearchItem] = []
    seen: set[str] = set()
    used_media_bytes = 0
    for hit in hits:
        parent_id = str(hit.chunk.metadata.get("multimodal_parent_id", hit.chunk.id))
        if parent_id in seen:
            continue
        parent = primary_by_id.get(parent_id)
        if parent is None:
            continue
        manifest = parent.metadata.get("multimodal_manifest")
        if not isinstance(manifest, list):
            continue
        parts: list[TextContentPart | ImageContentPart] = []
        candidate_bytes = 0
        valid = True
        for entry in manifest:
            if not isinstance(entry, dict):
                valid = False
                break
            if entry.get("type") == "text" and isinstance(entry.get("text"), str):
                parts.append(TextContentPart(type="text", text=str(entry["text"])))
            elif entry.get("type") == "image_ref" and isinstance(entry.get("media_id"), str):
                media_chunk = media_by_id.get(str(entry["media_id"]))
                if media_chunk is None:
                    valid = False
                    break
                data_url = media_chunk.metadata.get("data_url")
                if not isinstance(data_url, str):
                    valid = False
                    break
                part = ImageContentPart(
                    type="image_url", image_url=ImageURLValue(url=data_url)
                )
                candidate_bytes += content_media_bytes([part])
                parts.append(part)
            else:
                valid = False
                break
        if not valid or not parts or used_media_bytes + candidate_bytes > MAX_RESPONSE_MEDIA_BYTES:
            continue
        used_media_bytes += candidate_bytes
        seen.add(parent_id)
        rendered_content: ContentValue = parts
        if (
            parent.metadata.get("multimodal_scalar") is True
            and len(parts) == 1
            and isinstance(parts[0], TextContentPart)
        ):
            rendered_content = parts[0].text
        output.append(
            SearchItem(
                id=parent_id,
                content=rendered_content,
                created_at=None,
                source=parent.source,
                session_id=str(parent.metadata.get("source_session_id", "")),
                kind=str(parent.metadata.get("kind", "multimodal")),
                score=float(hit.score),
            )
        )
        if len(output) >= top_k:
            break
    return output
