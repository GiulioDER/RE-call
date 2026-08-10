"""Validation helpers for benchmark artifact claims."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

from recall.provider_metadata import ProviderMetadata, provider_metadata_from_any, validate_cost_claim


def provider_metadata_from_payload(payload: Mapping[str, object]) -> tuple[ProviderMetadata, ...]:
    raw = payload.get("provider_metadata", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("provider_metadata must be an array")
    return tuple(provider_metadata_from_any(item) for item in raw)


def reject_unauditable_cost_claims(payload: Mapping[str, object]) -> None:
    """Reject monetary benchmark claims without provider revision and cost fields."""

    cost_claims = payload.get("cost_claims", ())
    if not cost_claims:
        return
    metadata = provider_metadata_from_payload(payload)
    if not metadata:
        raise ValueError("benchmark cost claims require provider_metadata")
    for item in metadata:
        validate_cost_claim(item)


__all__ = ["provider_metadata_from_payload", "reject_unauditable_cost_claims"]
