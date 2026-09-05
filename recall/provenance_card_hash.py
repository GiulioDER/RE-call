"""Canonical hashing helpers for immutable evidence cards."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections.abc import Mapping
from datetime import datetime
from typing import Any


def _canonical(value: object) -> object:
    if isinstance(value, str):
        return unicodedata.normalize("NFC", value)
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {
            unicodedata.normalize("NFC", str(key)): _canonical(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_canonical(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"unsupported value in evidence-card content: {type(value).__name__}")


def card_payload(card: Any) -> dict[str, object]:
    return {
        "chunk_id": card.chunk_id,
        "source": card.source,
        "source_digest": card.source_digest,
        "valid_from": card.valid_from,
        "valid_until": card.valid_until,
        "first_indexed_at": card.first_indexed_at,
        "indexed_at": card.indexed_at,
        "tenant_id": card.tenant_id,
        "generation_id": card.generation_id,
        "pipeline_fingerprint": card.pipeline_fingerprint,
        "corpus_fingerprint": card.corpus_fingerprint,
        "calibration_id": card.calibration_id,
        "calibration_status": card.calibration_status,
        "trust_state": card.trust_state,
        "verdict": card.verdict,
        "confidence": card.confidence,
        "rank": card.rank,
        "supersession_links": card.supersession_links,
        "contradiction_links": card.contradiction_links,
        "support_refs": card.support_refs,
        "structured_facts": [fact.to_payload() for fact in card.structured_facts],
        "schema_version": card.schema_version,
    }


def card_id_for_payload(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(_canonical(payload), ensure_ascii=False, separators=(",", ":"), sort_keys=True)
    return "card_" + hashlib.sha256(encoded.encode("utf-8")).hexdigest()[:32]
