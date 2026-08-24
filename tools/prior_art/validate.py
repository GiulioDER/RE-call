"""Fail closed validation for the prior art evidence corpus."""

from __future__ import annotations

from collections import defaultdict
from datetime import date
from typing import Any
from urllib.parse import urlparse

from .schema import (
    CLAIM_VALUES,
    EVIDENCE_TYPES,
    LOCATOR_KINDS,
    REVIEW_DECISIONS,
    REVIEW_STATUSES,
    SOURCE_STATUSES,
    SOURCE_TIERS,
    SOURCE_TYPES,
    SYSTEM_TYPES,
    as_date,
    is_mapping,
    nonempty_string,
    record_id,
)

MAX_EVIDENCE_WORDS = 25


def _ids(records: list[dict[str, Any]], field: str, label: str, errors: list[str]) -> set[str]:
    values: set[str] = set()
    for index, record in enumerate(records):
        value = record_id(record, field)
        if value is None:
            errors.append(f"{label}[{index}] missing nonempty {field}")
            continue
        if value in values:
            errors.append(f"duplicate {label} identifier: {value}")
        values.add(value)
    return values


def _check_date(
    value: Any,
    location: str,
    today: date,
    errors: list[str],
    *,
    required: bool = False,
) -> None:
    if value is None:
        if required:
            errors.append(f"{location} is required")
        return
    try:
        parsed = as_date(value)
    except ValueError as exc:
        errors.append(f"{location} is invalid: {exc}")
        return
    if parsed is not None and parsed > today:
        errors.append(f"{location} is in the future: {value}")


def _check_url(value: Any, location: str, errors: list[str]) -> None:
    if not nonempty_string(value):
        errors.append(f"{location} must be a nonempty URL")
        return
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        errors.append(f"{location} must use http or https: {value}")


def _taxonomy_capabilities(taxonomy: dict[str, Any], errors: list[str]) -> dict[str, dict[str, Any]]:
    groups = taxonomy.get("groups")
    if not isinstance(groups, list) or not groups:
        errors.append("taxonomy.groups must be a nonempty list")
        return {}
    capabilities: dict[str, dict[str, Any]] = {}
    for group_index, group in enumerate(groups):
        if not is_mapping(group):
            errors.append(f"taxonomy.groups[{group_index}] must be an object")
            continue
        group_id = group.get("group_id")
        entries = group.get("capabilities")
        if not nonempty_string(group_id):
            errors.append(f"taxonomy.groups[{group_index}] missing group_id")
        if not isinstance(entries, list) or not entries:
            errors.append(f"taxonomy group {group_id!r} must have capabilities")
            continue
        for capability_index, capability in enumerate(entries):
            if not is_mapping(capability):
                errors.append(f"taxonomy capability {group_id}[{capability_index}] must be an object")
                continue
            capability_id = capability.get("capability_id")
            if not nonempty_string(capability_id):
                errors.append(f"taxonomy capability {group_id}[{capability_index}] missing capability_id")
                continue
            if not capability_id.startswith(f"{group_id}."):
                errors.append(f"capability {capability_id} must be namespaced under {group_id}")
            if capability_id in capabilities:
                errors.append(f"duplicate capability identifier: {capability_id}")
            allowed_values = capability.get("allowed_values")
            if set(allowed_values or []) != set(CLAIM_VALUES):
                errors.append(f"capability {capability_id} must declare all claim values")
            for field in ("name", "definition", "minimum_evidence"):
                if not nonempty_string(capability.get(field)):
                    errors.append(f"capability {capability_id} missing {field}")
            capabilities[capability_id] = dict(capability)
    return capabilities


def validate_dataset(dataset: dict[str, Any], today: date | None = None) -> list[str]:
    """Return all validation errors. An empty list means the corpus is valid."""

    today = today or date.today()
    errors: list[str] = []
    taxonomy = dataset.get("taxonomy")
    if not is_mapping(taxonomy):
        errors.append("taxonomy must be an object")
        taxonomy = {}
    capabilities = _taxonomy_capabilities(dict(taxonomy), errors)

    report_config = dataset.get("report_config")
    if not is_mapping(report_config):
        errors.append("report_config must be an object")
        report_config = {}
    target_hypothesis = report_config.get("target_hypothesis", [])
    if not isinstance(target_hypothesis, list) or not target_hypothesis:
        errors.append("report_config.target_hypothesis must be a nonempty list")
    else:
        for capability_id in target_hypothesis:
            if capability_id not in capabilities:
                errors.append(
                    f"report_config.target_hypothesis references a missing capability {capability_id}"
                )

    sources = dataset.get("sources", [])
    systems = dataset.get("systems", [])
    claims = dataset.get("claims", [])
    reviews = dataset.get("reviews", [])
    for label, records in (("sources", sources), ("systems", systems), ("claims", claims), ("reviews", reviews)):
        if not isinstance(records, list):
            errors.append(f"{label} must be a list")

    if not all(isinstance(records, list) for records in (sources, systems, claims, reviews)):
        return errors

    source_ids = _ids(sources, "source_id", "source", errors)
    system_ids = _ids(systems, "system_id", "system", errors)
    claim_ids = _ids(claims, "claim_id", "claim", errors)
    review_ids = _ids(reviews, "review_id", "review", errors)

    source_by_id = {record.get("source_id"): record for record in sources}
    review_by_claim: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for index, source in enumerate(sources):
        prefix = f"sources[{index}]"
        if source.get("source_type") not in SOURCE_TYPES:
            errors.append(f"{prefix}.source_type is invalid")
        if source.get("source_tier") not in SOURCE_TIERS:
            errors.append(f"{prefix}.source_tier is invalid")
        if source.get("status") not in SOURCE_STATUSES:
            errors.append(f"{prefix}.status is invalid")
        for field in ("title", "canonical_url", "accessed_at"):
            if field != "canonical_url" and not nonempty_string(source.get(field)):
                errors.append(f"{prefix}.{field} must be nonempty")
        _check_url(source.get("canonical_url"), f"{prefix}.canonical_url", errors)
        _check_date(source.get("published_at"), f"{prefix}.published_at", today, errors)
        _check_date(source.get("updated_at"), f"{prefix}.updated_at", today, errors)
        _check_date(source.get("accessed_at"), f"{prefix}.accessed_at", today, errors, required=True)
        if source.get("supersedes_source_id") is not None and source.get("supersedes_source_id") not in source_ids:
            errors.append(f"{prefix}.supersedes_source_id does not exist")

    for index, system in enumerate(systems):
        prefix = f"systems[{index}]"
        if system.get("system_type") not in SYSTEM_TYPES:
            errors.append(f"{prefix}.system_type is invalid")
        if system.get("review_status") not in REVIEW_STATUSES:
            errors.append(f"{prefix}.review_status is invalid")
        for field in ("display_name", "organization", "homepage_url", "as_of_date"):
            if not nonempty_string(system.get(field)):
                errors.append(f"{prefix}.{field} must be nonempty")
        _check_url(system.get("homepage_url"), f"{prefix}.homepage_url", errors)
        _check_date(system.get("as_of_date"), f"{prefix}.as_of_date", today, errors, required=True)
        source_refs = system.get("source_ids")
        if not isinstance(source_refs, list) or not source_refs:
            errors.append(f"{prefix}.source_ids must be a nonempty list")
        else:
            for source_id in source_refs:
                if source_id not in source_ids:
                    errors.append(f"{prefix}.source_ids references missing source {source_id}")

    claims_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for index, claim in enumerate(claims):
        prefix = f"claims[{index}]"
        claim_id = claim.get("claim_id")
        if claim_id not in claim_ids:
            errors.append(f"{prefix}.claim_id is invalid")
        if claim.get("system_id") not in system_ids:
            errors.append(f"{prefix}.system_id references a missing system")
        capability_id = claim.get("capability_id")
        if capability_id not in capabilities:
            errors.append(f"{prefix}.capability_id references a missing capability")
        if claim.get("value") not in CLAIM_VALUES:
            errors.append(f"{prefix}.value is invalid")
        if claim.get("review_status") not in REVIEW_STATUSES:
            errors.append(f"{prefix}.review_status is invalid")
        if claim.get("evidence_type") not in EVIDENCE_TYPES:
            errors.append(f"{prefix}.evidence_type is invalid")
        if not nonempty_string(claim.get("claim_text")):
            errors.append(f"{prefix}.claim_text must be nonempty")
        source_id = claim.get("source_id")
        if source_id not in source_ids:
            errors.append(f"{prefix}.source_id references a missing source")
        elif claim.get("value") == "verified" and source_by_id[source_id].get("source_tier") != "primary":
            errors.append(f"{prefix}.verified claims require a primary source")
        locator = claim.get("evidence_locator")
        if not is_mapping(locator):
            errors.append(f"{prefix}.evidence_locator must be an object")
        else:
            if locator.get("kind") not in LOCATOR_KINDS:
                errors.append(f"{prefix}.evidence_locator.kind is invalid")
            if not nonempty_string(locator.get("value")):
                errors.append(f"{prefix}.evidence_locator.value must be nonempty")
        excerpt = claim.get("evidence_excerpt")
        if excerpt is not None:
            if not isinstance(excerpt, str):
                errors.append(f"{prefix}.evidence_excerpt must be a string or null")
            elif len(excerpt.split()) > MAX_EVIDENCE_WORDS:
                errors.append(f"{prefix}.evidence_excerpt exceeds {MAX_EVIDENCE_WORDS} words")
        review_id = claim.get("review_id")
        if review_id not in review_ids:
            errors.append(f"{prefix}.review_id references a missing review")
        claims_by_cell[(claim.get("system_id"), capability_id)].append(claim)

    for index, review in enumerate(reviews):
        prefix = f"reviews[{index}]"
        claim_id = review.get("claim_id")
        if claim_id not in claim_ids:
            errors.append(f"{prefix}.claim_id references a missing claim")
        if review.get("decision") not in REVIEW_DECISIONS:
            errors.append(f"{prefix}.decision is invalid")
        if not nonempty_string(review.get("reviewer")):
            errors.append(f"{prefix}.reviewer must be nonempty")
        if not nonempty_string(review.get("reason")):
            errors.append(f"{prefix}.reason must be nonempty")
        _check_date(review.get("reviewed_at"), f"{prefix}.reviewed_at", today, errors, required=True)
        conflict_ids = review.get("conflict_claim_ids", [])
        if not isinstance(conflict_ids, list):
            errors.append(f"{prefix}.conflict_claim_ids must be a list")
        else:
            for conflict_id in conflict_ids:
                if conflict_id not in claim_ids:
                    errors.append(f"{prefix}.conflict_claim_ids references a missing claim")
        review_by_claim[claim_id].append(review)

    for claim_id, claim_reviews in review_by_claim.items():
        if len(claim_reviews) > 1:
            errors.append(f"claim {claim_id} has multiple review records")

    for (system_id, capability_id), cell_claims in claims_by_cell.items():
        if len(cell_claims) <= 1:
            continue
        claim_ids_in_cell = {claim["claim_id"] for claim in cell_claims}
        linked = set()
        for claim in cell_claims:
            for review in review_by_claim.get(claim["claim_id"], []):
                linked.update(review.get("conflict_claim_ids", []))
        if not claim_ids_in_cell.issubset(linked | {next(iter(claim_ids_in_cell))}):
            errors.append(
                f"duplicate claims for {system_id}/{capability_id} require explicit conflict_claim_ids"
            )

    return errors
