"""MM-4: machine-read text from each image, kept as a sidecar beside the image it came from.

Pre-registration: docs/preregistrations/2026-09-25-aml-c9-image-text-sidecar.md

C9 stores an image message's supplied text and the image, and nothing read FROM the image, so a
question about what a receipt says or how many chairs a room holds can only be answered if the
reader is handed the image. At Add, one vision call per image extracts what is visibly there
(legible text verbatim, objects with counts, people by appearance, setting, visible dates, screen
state) as short labelled lines. The result lives in its own namespace (``image_text_tenant``), so
raw windows keep ``content_text``'s rule of carrying no generated claims. At Search it is used two
ways, each behind its own default-off flag: as an extra retrieval leg whose hits stand for their
parent image message (``image_text_leg``), and as text appended to the returned image message,
labelled as machine-read (``image_text_shown``).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import logging
from typing import Any

from recall.types import Chunk
from recall_aml.models import SearchItem, TextContentPart

log = logging.getLogger("recall_aml")

#: Label on every appended sidecar, so the reader knows the text is a model's reading of an image.
SHOWN_LABEL = "[machine-read from the image]"
MAX_OUTPUT_TOKENS = 300
MAX_IMAGES_PER_MESSAGE = 16
IMAGE_TEXT_PROMPT = (
    "List only what is visibly present in this image, as short labelled lines. Do not guess or "
    "infer anything that is not visible.\n"
    "Text: every legible piece of text, verbatim, including numbers, prices, dates and names on "
    "signs, labels, receipts or screens.\n"
    "Objects: each distinct kind of object with its count, colour and position.\n"
    "People: by appearance, clothing and action only; never guess who they are.\n"
    "Setting: the place or scene.\n"
    "Time: any date or time visible in the image.\n"
    "Screen: if this is a screenshot, the app and what the screen shows.\n"
    "Omit a line when there is nothing for it."
)


def image_text_tenant(tenant: str) -> str:
    return tenant + "_imgtext"


def sidecar_id(parent_id: str, index: int) -> str:
    return f"{parent_id}_imgtext_{index}"


class ImageTextExtractor:
    """One bounded vision call per image through an OpenAI-compatible client (OpenRouter)."""

    def __init__(self, client: Any, model: str, *, provider: str | None = None) -> None:
        self._client = client
        self.model = model
        self._provider = provider

    def extract(self, data_url: str) -> str | None:
        """The image's visible content as text, or None when the provider fails (no sidecar)."""
        extra: dict[str, Any] = {"reasoning": {"enabled": False}}
        if self._provider:
            extra["provider"] = {"order": [self._provider], "allow_fallbacks": False}
        try:
            response = self._client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": IMAGE_TEXT_PROMPT},
                            {"type": "image_url", "image_url": {"url": data_url, "detail": "low"}},
                        ],
                    }
                ],
                temperature=0,
                max_tokens=MAX_OUTPUT_TOKENS,
                extra_body=extra,
            )
            text = str(response.choices[0].message.content or "").strip()
        except Exception as exc:  # BROAD-CATCH: a failed reading means no sidecar, never a failed Add
            log.info("image_text_extract_failed", extra={"error_class": type(exc).__name__})
            return None
        return text or None


def sidecar_chunks(
    primary_chunks: Sequence[Chunk],
    media_chunks: Sequence[Chunk],
    extract: Callable[[str], str | None],
) -> list[Chunk]:
    """One sidecar per image of each multimodal parent, in manifest order."""
    media_by_id = {chunk.id: chunk for chunk in media_chunks}
    sidecars: list[Chunk] = []
    for parent in primary_chunks:
        manifest = parent.metadata.get("multimodal_manifest")
        if not isinstance(manifest, list):
            continue
        refs = [entry for entry in manifest if isinstance(entry, dict) and entry.get("type") == "image_ref"]
        for index, entry in enumerate(refs[:MAX_IMAGES_PER_MESSAGE]):
            media = media_by_id.get(str(entry.get("media_id")))
            data_url = media.metadata.get("data_url") if media is not None else None
            if not isinstance(data_url, str):
                continue
            text = extract(data_url)
            if not text:
                continue
            sidecars.append(
                Chunk(
                    id=sidecar_id(parent.id, index),
                    source=parent.source,
                    text=text,
                    metadata={
                        "record_type": "image_text",
                        "kind": "image_text",
                        "primary_id": parent.id,
                        "media_id": str(entry.get("media_id")),
                        "source_session_id": parent.metadata.get("source_session_id"),
                        "event_time": parent.metadata.get("event_time"),
                    },
                )
            )
    return sidecars


def shown_items(items: Sequence[SearchItem], sidecars: Mapping[str, Sequence[str]]) -> list[SearchItem]:
    """Append each returned image message's sidecar text once, labelled; leave the rest alone."""
    output: list[SearchItem] = []
    for item in items:
        texts = [text for text in sidecars.get(item.id, ()) if text.strip()]
        if not texts:
            output.append(item)
            continue
        addition = f"{SHOWN_LABEL}\n" + "\n".join(texts)
        if isinstance(item.content, str):
            output.append(item.model_copy(update={"content": f"{item.content}\n{addition}"}))
        else:
            part = TextContentPart(type="text", text=addition)
            output.append(item.model_copy(update={"content": [*item.content, part]}))
    return output
