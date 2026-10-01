"""Ordered multimodal preservation and Voyage retrieval helpers for AML Hosted."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import logging
from typing import Any, Protocol

from recall.multimodal import (
    VOYAGE_MAX_INPUT_TOKENS,
    estimate_voyage_inputs,
    fit_voyage_image,
    lazy_voyage_text_counter,
    plan_voyage_requests,
)
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
from recall_aml.window_compose import render_facts


MULTIMODAL_EMBEDDING_MODEL = "voyage-multimodal-3.5"
MULTIMODAL_EMBEDDING_PROFILE = "voyage-multimodal-3.5-v2"
MULTIMODAL_DIMENSION = 1024
MULTIMODAL_RRF_CONSTANT = 60
RAW_SEGMENT_CHARS = 4_500
MAX_RESPONSE_MEDIA_BYTES = 30 * 1024 * 1024
#: Voyage's multimodal request limits, and the estimating and packing against them, are
#: shared with the core tenant and live in `recall.multimodal`.

log = logging.getLogger("recall_aml")


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
    """Fit a derived embedding image to Voyage limits while preserving source bytes elsewhere.

    The data URL is validated exactly as admission validates it, and a malformed one raises. The
    fitting is `recall.multimodal.fit_voyage_image`, shared with the core tenant, in strict mode
    so an image Pillow cannot read still raises rather than being sent. It was measured byte
    identical to this module's former copy on 2026-09-30 (six cases across PNG, JPEG, WebP, RGBA
    and greyscale, fitted and untouched), so no stored vector changes.
    """
    decode_image_data_url(data_url)
    return fit_voyage_image(data_url, strict=True)


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

    def __init__(
        self,
        api_key: str,
        *,
        client: Any = None,
        timeout: float = 180.0,
        token_counter: Callable[[list[str]], list[int]] | None = None,
    ) -> None:
        """Build the adapter.

        ``token_counter`` returns one text token count per string. It defaults to Voyage's
        published tokenizer for the model, loaded on first use, and falls back to a UTF-8 byte
        bound when that cannot be loaded.
        """
        if client is None:
            import voyageai

            client = voyageai.Client(api_key=api_key, max_retries=0, timeout=timeout)
        self._client = client
        self._text_counter = token_counter or lazy_voyage_text_counter(
            MULTIMODAL_EMBEDDING_MODEL
        )

    def _warn_if_over_context(
        self, index: int, tokens: int, images: int, input_type: str
    ) -> None:
        """Log an input the provider will truncate. It is still sent, unchanged."""
        if tokens > VOYAGE_MAX_INPUT_TOKENS:
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

    def embed_documents(self, inputs: Sequence[dict[str, Any]]) -> list[list[float]]:
        """Embed every input, in as few requests as Voyage's request caps allow.

        An Add used to go out as ONE request, so enough large images (about twelve at the 16M
        pixel fit) passed the 320K-token request cap and the whole Add was refused on every
        resend. Inputs are now packed, in order, into requests under the caps. The provider embeds
        each input on its own, so which request carries it changes nothing about its vector, and
        an Add that fits is sent exactly as before, as one request. An input over the 32K-token
        context is sent unchanged and logged, since the provider truncates it.
        """
        materialized = list(inputs)
        if not materialized:
            return []
        estimates: list[int] = []
        for index, (tokens, images) in enumerate(
            estimate_voyage_inputs(materialized, self._text_counter)
        ):
            self._warn_if_over_context(index, tokens, images, "document")
            estimates.append(tokens)
        vectors: list[list[float]] = []
        for start, stop in plan_voyage_requests(estimates):
            response = self._client.multimodal_embed(
                materialized[start:stop],
                model=MULTIMODAL_EMBEDDING_MODEL,
                input_type="document",
                truncation=True,
            )
            vectors.extend(list(vector) for vector in response.embeddings)
        self._validate(vectors, len(materialized))
        return vectors

    def embed_query(self, value: ContentValue) -> list[float]:
        provider_input = to_voyage_input(value)
        [(tokens, images)] = estimate_voyage_inputs([provider_input], self._text_counter)
        self._warn_if_over_context(0, tokens, images, "query")
        response = self._client.multimodal_embed(
            [provider_input],
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


def _created_at(metadata: dict[str, Any]) -> datetime | None:
    event_time = metadata.get("event_time")
    if isinstance(event_time, datetime):
        return event_time
    if isinstance(event_time, str):
        try:
            return datetime.fromisoformat(event_time)
        except ValueError:
            return None
    return None


def _text_item(hit: ScoredChunk) -> SearchItem:
    """Render a text record exactly as ``retrieval.render_full_evidence`` does."""
    metadata = hit.chunk.metadata
    return SearchItem(
        id=hit.chunk.id,
        content=hit.chunk.text,
        created_at=_created_at(metadata),
        source=hit.chunk.source,
        session_id=str(metadata.get("source_session_id", "")),
        kind=str(metadata.get("kind", metadata.get("record_type", "raw"))),
        score=float(hit.score),
        render_facts=render_facts(metadata),
    )


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
        manifest = parent.metadata.get("multimodal_manifest") if parent is not None else None
        if parent is None or not isinstance(manifest, list):
            # A routed specialist variant stores text-only Adds as ordinary text windows, which
            # carry no multimodal manifest. Dropping them here made every text memory vanish
            # whenever the router sent a query to this route (a query mentioning an "image",
            # "chart" or "UI"), so render them exactly as the text-only path does. A visual
            # vector chunk has a parent id and cannot be reconstructed without it.
            if "multimodal_parent_id" in hit.chunk.metadata:
                continue
            seen.add(parent_id)
            output.append(_text_item(hit))
            if len(output) >= top_k:
                break
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
                # The manifest holds no timestamp, so without this the Answer model sees no
                # time for any item on the multimodal route (official run teval_dcc1109c4331c3e3).
                created_at=_created_at(parent.metadata),
                source=parent.source,
                session_id=str(parent.metadata.get("source_session_id", "")),
                kind=str(parent.metadata.get("kind", "multimodal")),
                score=float(hit.score),
            )
        )
        if len(output) >= top_k:
            break
    return output
