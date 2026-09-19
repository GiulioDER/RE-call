"""Opaque and stable identities for hosted evaluation data."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tenant_for(user_id: str) -> str:
    return "aml_" + digest(user_id)


def session_digest(session_id: str) -> str:
    return digest(session_id)


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
