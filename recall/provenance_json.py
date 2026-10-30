"""Preserve immutable provenance identities across PostgreSQL numeric normalization.

JSONB expands exponent numbers and drops negative zero. A storage-only companion
keeps the original JSON spelling when that would change a Python float's identity.
Readers validate the companion against the entire stored object before using it.
Existing plain payloads and public payload shapes remain unchanged.
"""
from __future__ import annotations

import json
from collections.abc import Mapping
from decimal import Decimal
from typing import Any

_ORIGINAL = "__recall_json"


def _needs_original(value: Any) -> bool:
    if isinstance(value, float):
        return "e+" in repr(value) or repr(value) == "-0.0"
    if isinstance(value, Mapping):
        return any(_needs_original(item) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_needs_original(item) for item in value)
    return False


def encode_jsonb(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Add the original numeric spelling only to payloads JSONB would alter."""
    if _ORIGINAL in payload:
        raise ValueError("reserved provenance storage field")
    result = dict(payload)
    if _needs_original(payload):
        # Apply JSON's key coercion before sorting: Python facts can contain mixed
        # integer/string keys, while JSONB and the identity canonicalizers use strings.
        normalized = json.loads(json.dumps(payload, ensure_ascii=False, separators=(",", ":")))
        result[_ORIGINAL] = json.dumps(normalized, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return result


def _same(left: Any, right: Any) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return type(left) is type(right) and left == right
    if isinstance(left, (int, float)) and isinstance(right, (int, float)):
        return Decimal(str(left)) == Decimal(str(right))
    if isinstance(left, Mapping) and isinstance(right, Mapping):
        return left.keys() == right.keys() and all(_same(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(_same(a, b) for a, b in zip(left, right))
    return type(left) is type(right) and left == right


def decode_jsonb(payload: Mapping[str, Any]) -> Mapping[str, Any]:
    """Restore the original spelling, refusing an inconsistent storage companion."""
    if _ORIGINAL not in payload:
        return payload
    encoded = payload[_ORIGINAL]
    if not isinstance(encoded, str):
        raise ValueError("invalid provenance storage companion")
    original = json.loads(encoded)
    stored = {key: value for key, value in payload.items() if key != _ORIGINAL}
    if not isinstance(original, dict) or not _same(original, stored):
        raise ValueError("provenance storage companion mismatch")
    return original
