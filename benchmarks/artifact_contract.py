"""Validation helpers for benchmark artifact claims."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

from recall.provider_metadata import ProviderMetadata, provider_metadata_from_any, validate_cost_claim

#: A monetary FIGURE in prose: `$7.29`, `0.75 USD`, `USD 3`, `5 dollars`.
#: Figures are what auditability needs; a wordy "costs about five dollars" is not caught, and
#: that limit is deliberate rather than an oversight.
_MONETARY_PROSE = re.compile(r"\$\s*\d|\bUSD\s*\d|\d\s*(?:USD\b|dollars?\b)", re.IGNORECASE)

#: Keys whose subtrees hold benchmark SOURCE text copied in verbatim, not claims the artifact
#: makes. A LOCOMO conversation that mentions a price is not a cost claim, and rejecting one
#: would fail a completed multi-minute run at the write site for a number nobody published.
VERBATIM_SOURCE_KEYS = frozenset({"outcomes"})


def provider_metadata_from_payload(payload: Mapping[str, object]) -> tuple[ProviderMetadata, ...]:
    raw = payload.get("provider_metadata", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("provider_metadata must be an array")
    return tuple(provider_metadata_from_any(item) for item in raw)


def _publishes_monetary_prose(value: object) -> bool:
    """Walk the artifact's own claim surface looking for a published monetary figure."""

    if isinstance(value, str):
        return _MONETARY_PROSE.search(value) is not None
    if isinstance(value, Mapping):
        return any(
            _publishes_monetary_prose(item)
            for key, item in value.items()
            if key not in VERBATIM_SOURCE_KEYS
        )
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return any(_publishes_monetary_prose(item) for item in value)
    return False


def reject_unauditable_cost_claims(payload: Mapping[str, object]) -> None:
    """Reject monetary benchmark claims without provider revision and cost fields.

    Two ways an artifact makes a monetary claim, and both are checked. It can populate
    ``cost_claims``, or it can print a dollar figure in prose. The prose route used to pass
    unexamined: this returned early whenever ``cost_claims`` was falsy, so an artifact that
    published "$7.29 per run" while OMITTING the key entirely was never audited at all.

    ``cost_claims`` is therefore mandatory even when empty. An absent key is an undeclared
    posture; ``[]`` is a positive declaration that the artifact claims no money, and the prose
    scan is what holds that declaration honest.
    """

    if "cost_claims" not in payload:
        raise ValueError("benchmark artifacts must declare cost_claims, even as an empty array")
    cost_claims = payload["cost_claims"]
    if not isinstance(cost_claims, Sequence) or isinstance(cost_claims, (str, bytes, bytearray)):
        raise ValueError("cost_claims must be an array")

    if not cost_claims and not _publishes_monetary_prose(payload):
        return
    metadata = provider_metadata_from_payload(payload)
    if not metadata:
        raise ValueError("benchmark cost claims require provider_metadata")
    for item in metadata:
        validate_cost_claim(item)


__all__ = [
    "VERBATIM_SOURCE_KEYS",
    "provider_metadata_from_payload",
    "reject_unauditable_cost_claims",
]
