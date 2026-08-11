"""Validation helpers for benchmark artifact claims."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet
import re

from recall.provider_metadata import ProviderMetadata, provider_metadata_from_any, validate_cost_claim

#: A monetary FIGURE in prose: `$7.29`, `0.75 USD`, `USD 3`, `5 dollars`, `EUR 6.60`, `£5.00`.
#: Figures are what auditability needs; a wordy "costs about five dollars" is not caught, and
#: that limit is deliberate rather than an oversight. A bare number under a cost named key is
#: NOT caught either, which is a real gap, but closing it would make every run that reports a
#: provider cost raise at the write site and is therefore not a change to make in passing.
#:
#: `pounds` is deliberately absent as a BARE word while `£`, `GBP` and `pounds sterling` are
#: present: "weighs 5 pounds" is not a cost claim, and a false positive here aborts a completed
#: paid run at the write site. The disambiguated form reclaims the money case at a cost of one
#: remaining weight phrasing, "N pounds sterling silver", which this project will not publish.
#: `dollars` and `euros` carry no unit ambiguity at all.
_MONETARY_PROSE = re.compile(
    r"[$€£]\s*\d"
    r"|\b(?:USD|EUR|GBP)\s*\d"
    r"|\d\s*pounds?\s+sterling\b"
    r"|\d\s*(?:USD\b|EUR\b|GBP\b|dollars?\b|euros?\b)",
    re.IGNORECASE,
)

#: TOP LEVEL keys whose subtrees hold benchmark SOURCE text copied in verbatim, not claims the
#: artifact makes. A LOCOMO conversation that mentions a price is not a cost claim, and rejecting
#: one would fail a completed multi-minute run at the write site for a number nobody published.
#:
#: Root only, deliberately. Applied at every depth this becomes an audit BYPASS: `config["system"]`
#: is `describe()` output from a duck typed adapter, so a nested key that happens to be named
#: `outcomes` would silently exempt its whole subtree, and that key namespace is not ours to trust.
VERBATIM_SOURCE_KEYS = frozenset({"outcomes"})


def provider_metadata_from_payload(payload: Mapping[str, object]) -> tuple[ProviderMetadata, ...]:
    raw = payload.get("provider_metadata", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("provider_metadata must be an array")
    return tuple(provider_metadata_from_any(item) for item in raw)


def _publishes_monetary_prose(value: object, seen: set[int] | None = None) -> bool:
    """Walk the artifact's own claim surface looking for a published monetary figure.

    `seen` guards against a self referential payload. This walk runs BEFORE `json.dumps`, so
    without it a cycle surfaces as a RecursionError from the contract instead of the clean
    `ValueError: Circular reference detected` the serializer would have raised.
    """

    if isinstance(value, str):
        return _MONETARY_PROSE.search(value) is not None
    if isinstance(value, (bytes, bytearray)):
        # Defensive only: bytes are not JSON serialisable, so a payload carrying them cannot be
        # published either way. Non-UTF-8 encodings are a known miss, not covered coverage.
        return _MONETARY_PROSE.search(value.decode("utf-8", "replace")) is not None

    seen = set() if seen is None else seen
    if id(value) in seen:
        return False
    if isinstance(value, (Mapping, Sequence, AbstractSet)):
        seen = seen | {id(value)}
    if isinstance(value, Mapping):
        return any(_publishes_monetary_prose(item, seen) for item in value.values())
    if isinstance(value, (Sequence, AbstractSet)):
        return any(_publishes_monetary_prose(item, seen) for item in value)
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

    claim_surface = {
        key: item for key, item in payload.items() if key not in VERBATIM_SOURCE_KEYS
    }
    if not cost_claims and not _publishes_monetary_prose(claim_surface):
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
