"""Schema constants and small helpers for the prior art evidence corpus."""

from __future__ import annotations

from datetime import date
from typing import Any, Mapping

SOURCE_TYPES = frozenset(
    {
        "paper",
        "repository",
        "official_docs",
        "product_docs",
        "benchmark",
        "protocol",
        "standard",
        "survey",
    }
)
SOURCE_TIERS = frozenset({"primary", "secondary", "discovery_only"})
SOURCE_STATUSES = frozenset({"candidate", "accepted", "superseded", "inaccessible"})
SYSTEM_TYPES = frozenset(
    {
        "open_source_library",
        "agent_runtime",
        "commercial_service",
        "research_system",
        "benchmark",
        "protocol",
    }
)
REVIEW_STATUSES = frozenset({"draft", "accepted", "rejected", "disputed"})
CLAIM_VALUES = frozenset(
    {"verified", "partial", "not_evidenced", "contradicted", "unknown"}
)
REVIEW_DECISIONS = frozenset({"accepted", "rejected", "disputed"})
EVIDENCE_TYPES = frozenset({"explicit", "demonstrated", "vendor_claim", "inferred"})
LOCATOR_KINDS = frozenset({"section", "page", "anchor", "line", "repository_path", "api"})


def as_date(value: Any) -> date | None:
    """Parse an ISO date, returning None for an explicit null value."""

    if value is None:
        return None
    if not isinstance(value, str):
        raise ValueError("date must be an ISO string or null")
    return date.fromisoformat(value)


def is_mapping(value: Any) -> bool:
    return isinstance(value, Mapping)


def nonempty_string(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def record_id(record: Mapping[str, Any], field: str) -> str | None:
    value = record.get(field)
    return value if isinstance(value, str) and value else None
