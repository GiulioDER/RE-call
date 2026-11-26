"""Deterministic anchors over one source body, and the payload a compile call carries."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

#: Window and overlap of the AML anchored compiler (``recall_aml/compiler.py``), unchanged.
ANCHOR_CHARS = 1_600
ANCHOR_OVERLAP_CHARS = 160


@dataclass(frozen=True)
class SourceAnchor:
    """One byte-for-byte excerpt of a source body, created before any model call."""

    id: str
    start: int
    end: int
    quote: str


def build_source_anchors(source: str, body: str) -> list[SourceAnchor]:
    """Segment ``body`` into overlapping windows with ids bound to ``source`` and the text."""
    anchors: list[SourceAnchor] = []
    step = ANCHOR_CHARS - ANCHOR_OVERLAP_CHARS
    for start in range(0, len(body), step):
        end = min(start + ANCHOR_CHARS, len(body))
        quote = body[start:end]
        if quote.strip():
            payload = {"source": source, "start": start, "end": end, "quote": quote}
            digest = hashlib.sha256(
                json.dumps(
                    payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
                ).encode("utf-8")
            ).hexdigest()
            anchors.append(
                SourceAnchor(id=f"a{len(anchors):03d}_{digest[:16]}", start=start, end=end, quote=quote)
            )
        if end == len(body):
            break
    return anchors


def compile_payload(source: str, anchors: Sequence[SourceAnchor]) -> dict[str, Any]:
    return {
        "source": source,
        "anchors": [{"id": anchor.id, "excerpt": anchor.quote} for anchor in anchors],
    }


def encode_stored_data(payload: Mapping[str, Any]) -> str:
    """The exact text a compile prompt carries inside ``<stored_data>``."""
    return (
        json.dumps(payload, ensure_ascii=True, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def fit_payload(payload: Mapping[str, Any], budget: int) -> dict[str, Any] | None:
    """Keep the first and last anchors that fit ``budget`` encoded characters.

    Returns None when the payload already fits, or when not even one anchor would. Each end gets
    half of the room and the kept anchors keep their order and ids, which is all the evidence
    checks after the call rely on.
    """
    if len(encode_stored_data(payload)) <= budget:
        return None
    anchors = list(payload["anchors"])
    room = budget - len(encode_stored_data({**payload, "anchors": []}))
    sizes = [len(encode_stored_data(anchor)) + 1 for anchor in anchors]
    head_end, used = 0, 0
    while head_end < len(anchors) and used + sizes[head_end] <= room // 2:
        used += sizes[head_end]
        head_end += 1
    tail_start = len(anchors)
    while tail_start > head_end and used + sizes[tail_start - 1] <= room:
        used += sizes[tail_start - 1]
        tail_start -= 1
    kept = anchors[:head_end] + anchors[tail_start:]
    if not kept:
        return None
    return {**payload, "anchors": kept}
