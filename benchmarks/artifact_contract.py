"""Validation helpers for benchmark artifact claims."""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set as AbstractSet
import json
from pathlib import Path
import re
from typing import Any

from recall.provider_metadata import ProviderMetadata, provider_metadata_from_any, validate_cost_claim

#: A monetary FIGURE in prose: `$7.29`, `0.75 USD`, `USD 3`, `5 dollars`, `EUR 6.60`, `£5.00`.
#: Figures are what auditability needs; a wordy "costs about five dollars" is not caught, and
#: that limit is deliberate rather than an oversight. A bare number under a cost named key is
#: NOT caught either, which is a real gap, but closing it would make every run that reports a
#: provider cost trip the contract and is therefore not a change to make in passing.
#:
#: `pounds` is deliberately absent as a BARE word while `£`, `GBP` and `pounds sterling` are
#: present: "weighs 5 pounds" is not a cost claim, and a false positive costs the operator a
#: republish (`benchmarks.run` quarantines a refused artifact rather than destroying it, but it
#: still does not publish it). The disambiguated form reclaims the money case at a cost of one
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
#: one would refuse to publish a completed multi-minute run over a number nobody claimed.
#:
#: Root only, deliberately. Applied at every depth this becomes an audit BYPASS: `config["system"]`
#: is `describe()` output from a duck typed adapter, so a nested key that happens to be named
#: `outcomes` would silently exempt its whole subtree, and that key namespace is not ours to trust.
VERBATIM_SOURCE_KEYS = frozenset({"outcomes"})



def load_published_artifact(path: Path) -> dict[str, Any]:
    """Load a benchmark artifact, refusing one `benchmarks.run` marked as never published.

    The other half of the quarantine. `benchmarks.run` keeps a refused artifact out of the
    `results/*.json` glob AND marks it in band, and this is what makes the mark mean something:
    without a reader that honours it, a quarantined file handed over directly — which is how
    every one of these tools is invoked — is byte identical to a real measurement and gets
    tabulated as one.

    `SystemExit` rather than `ValueError` because every caller is a CLI entry point, and it
    matches what `analyze._expand` already raises for an unusable input.
    """

    doc: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise SystemExit(f"{path} is not a JSON object, so it is not a results artifact")
    if doc.get("unpublished"):
        raise SystemExit(
            f"{path} was REFUSED publication by benchmarks.run and is not a measurement: "
            f"{doc.get('unpublished_reason', 'no reason recorded')}"
        )
    validate_evidence_cost_contract(doc)
    validate_operational_claim_separation(doc)
    validate_routing_experiment(doc)
    return doc


def validate_evidence_cost_contract(payload: Mapping[str, object]) -> None:
    """Validate the additive exact evidence cost fields when an artifact declares them."""
    evidence_cost = payload.get("evidence_cost")
    if evidence_cost is not None:
        if not isinstance(evidence_cost, Mapping):
            raise ValueError("evidence_cost must be an object or null")
        if evidence_cost.get("claim_family") != "evidence_cost":
            raise ValueError("evidence_cost.claim_family must be 'evidence_cost'")
        curve = evidence_cost.get("curve")
        if curve is not None:
            if not isinstance(curve, Sequence) or isinstance(curve, (str, bytes, bytearray)):
                raise ValueError("evidence_cost.curve must be an array or null")
            from benchmarks.evidence_curve import EVIDENCE_BUDGETS

            budgets = tuple(point.get("budget_tokens") for point in curve if isinstance(point, Mapping))
            if budgets != EVIDENCE_BUDGETS:
                raise ValueError("evidence_cost.curve must use the preregistered budget ladder")
            for index, point in enumerate(curve):
                if not isinstance(point, Mapping):
                    raise ValueError(f"evidence_cost.curve[{index}] must be an object")
                records = point.get("records")
                if not isinstance(records, Sequence) or isinstance(records, (str, bytes, bytearray)):
                    raise ValueError(f"evidence_cost.curve[{index}].records must be an array")
                if point.get("measured_budget") is True:
                    budget = point["budget_tokens"]
                    for record in records:
                        if not isinstance(record, Mapping) or record.get("evidence_budget") != budget:
                            raise ValueError(
                                f"evidence_cost.curve[{index}] contains a mismatched budget record"
                            )
    metadata = payload.get("tokenizer_metadata")
    if metadata is None:
        outcomes = payload.get("outcomes")
        if isinstance(outcomes, Sequence) and any(
            isinstance(outcome, Mapping) and outcome.get("evidence_budget") is not None
            for outcome in outcomes
        ):
            raise ValueError("budgeted evidence artifacts require tokenizer_metadata")
        return
    if not isinstance(metadata, Mapping):
        raise ValueError("tokenizer_metadata must be an object or null")
    if metadata.get("tokenizer_id") != "cl100k_base":
        raise ValueError("exact evidence artifacts must use cl100k_base")
    for key in ("tokenizer_revision", "tokenizer_hash"):
        if not isinstance(metadata.get(key), str) or not metadata[key]:
            raise ValueError(f"tokenizer_metadata.{key} is required")
    outcomes = payload.get("outcomes")
    if not isinstance(outcomes, Sequence) or isinstance(outcomes, (str, bytes, bytearray)):
        raise ValueError("exact evidence artifacts must contain an outcomes array")
    for index, outcome in enumerate(outcomes):
        if not isinstance(outcome, Mapping):
            raise ValueError(f"outcomes[{index}] must be an object")
        for key in ("evidence_tokens_exact", "input_tokens_exact"):
            value = outcome.get(key)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"outcomes[{index}].{key} must be a nonnegative integer")


def validate_operational_claim_separation(payload: Mapping[str, object]) -> None:
    """Reject an operational block that presents itself as retrieval quality."""
    operational = payload.get("operational_metrics")
    if operational is None:
        return
    if not isinstance(operational, Mapping):
        raise ValueError("operational_metrics must be an object")
    if operational.get("claim_family") != "operational":
        raise ValueError("operational_metrics.claim_family must be 'operational'")
    if operational.get("retrieval_quality_claim") is not False:
        raise ValueError("operational metrics cannot make a retrieval quality claim")


def validate_routing_experiment(payload: Mapping[str, object]) -> None:
    """Validate additive routing provenance without interpreting quality results."""
    experiment = payload.get("routing_experiment")
    if experiment is None:
        return
    if not isinstance(experiment, Mapping):
        raise ValueError("routing_experiment must be an object or null")
    if experiment.get("mode") not in {"shadow", "active"}:
        raise ValueError("routing_experiment.mode must be shadow or active")
    from recall.query_class import QUERY_CLASS_VERSION, ROUTING_POLICY_VERSION

    if experiment.get("classifier_version") != QUERY_CLASS_VERSION:
        raise ValueError("routing_experiment.classifier_version is unsupported")
    if experiment.get("policy_version") != ROUTING_POLICY_VERSION:
        raise ValueError("routing_experiment.policy_version is unsupported")


def provider_metadata_from_payload(payload: Mapping[str, object]) -> tuple[ProviderMetadata, ...]:
    raw = payload.get("provider_metadata", ())
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, bytearray)):
        raise ValueError("provider_metadata must be an array")
    return tuple(provider_metadata_from_any(item) for item in raw)


def _publishes_monetary_prose(value: object, seen: set[int] | None = None) -> bool:
    """Walk the artifact's own claim surface looking for a published monetary figure.

    `seen` guards against a self referential payload, so a cycle cannot surface as a
    RecursionError from the contract. `benchmarks.run` now serialises before calling the
    contract, so at that call site `json.dumps` reports the cycle first and this guard never
    fires; it remains for direct callers, which have no such ordering.
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
    "load_published_artifact",
    "provider_metadata_from_payload",
    "reject_unauditable_cost_claims",
    "validate_evidence_cost_contract",
    "validate_operational_claim_separation",
    "validate_routing_experiment",
]
