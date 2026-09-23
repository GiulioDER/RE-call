"""Opaque and stable identities for hosted evaluation data."""

from __future__ import annotations

import hashlib
import json
from typing import Any


def digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def tenant_for(user_id: str) -> str:
    return "aml_" + digest(user_id)


def graph_tenant(tenant: str) -> str:
    """Return an opaque tenant for derived records without weakening user isolation."""
    return "aml_graph_" + digest(tenant)


def specialist_tenant(tenant: str, embedding_profile: str) -> str:
    """Return an isolated physical namespace for one semantic embedding space."""
    return "aml_specialist_" + digest(tenant + "\0" + embedding_profile)


def atomic_view_tenant(scope_tenant: str) -> str:
    """Return the isolated namespace for the atomic views of one retrieval scope.

    ``scope_tenant`` is the tenant a Search reads (the Code4 tenant, or its Context specialist),
    so each scope's views live beside, and are embedded like, the windows they rescue.
    """
    return "aml_atomic_" + digest(scope_tenant)


def session_digest(session_id: str) -> str:
    return digest(session_id)


def canonical_digest(payload: Any) -> str:
    encoded = json.dumps(
        payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
