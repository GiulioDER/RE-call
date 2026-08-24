"""Deterministic Markdown rendering for the prior art evidence corpus."""

from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
from typing import Any

from .loader import DATA_ROOT
from .validate import validate_dataset


def _capabilities(dataset: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    values: list[tuple[str, dict[str, Any]]] = []
    for group in dataset["taxonomy"]["groups"]:
        for capability in group["capabilities"]:
            values.append((group["group_id"], capability))
    return values


def _cell(claims: list[dict[str, Any]]) -> str:
    if not claims:
        return "unknown"
    values = {claim["value"] for claim in claims}
    if "contradicted" in values and ("verified" in values or "partial" in values):
        return "contested"
    for value in ("verified", "partial", "not_evidenced", "unknown", "contradicted"):
        if value in values:
            return value
    return "unknown"


def _source_link(source: dict[str, Any]) -> str:
    return f"[{source['source_id']}]({source['canonical_url']})"


def _target_hypothesis(dataset: dict[str, Any]) -> list[str]:
    target = dataset["report_config"].get("target_hypothesis", [])
    return [capability_id for capability_id in target if isinstance(capability_id, str)]


def _combination_status(values: list[str]) -> str:
    if any(value in {"contested", "contradicted"} for value in values):
        return "contested_combination"
    if values and all(value == "verified" for value in values):
        return "verified_combination"
    if any(value in {"verified", "partial"} for value in values):
        return "partial_combination"
    return "unverified_combination"


def _combination_rows(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    target = _target_hypothesis(dataset)
    claims_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for claim in dataset["claims"]:
        claims_by_cell[(claim["system_id"], claim["capability_id"])].append(claim)

    rows: list[dict[str, Any]] = []
    for system in sorted(dataset["systems"], key=lambda item: item["system_id"]):
        values = {
            capability_id: _cell(claims_by_cell.get((system["system_id"], capability_id), []))
            for capability_id in target
        }
        rows.append(
            {
                "system_id": system["system_id"],
                "status": _combination_status(list(values.values())),
                "verified": [capability_id for capability_id, value in values.items() if value == "verified"],
                "partial": [capability_id for capability_id, value in values.items() if value == "partial"],
                "missing": [
                    capability_id
                    for capability_id, value in values.items()
                    if value in {"unknown", "not_evidenced"}
                ],
                "conflicts": [
                    capability_id
                    for capability_id, value in values.items()
                    if value in {"contested", "contradicted"}
                ],
            }
        )
    return rows


def _incomplete_claims(dataset: dict[str, Any]) -> list[dict[str, Any]]:
    return sorted(
        [
            claim
            for claim in dataset["claims"]
            if claim["value"] in {"unknown", "not_evidenced"}
            or claim.get("review_status") != "accepted"
        ],
        key=lambda claim: claim["claim_id"],
    )


def _conflicting_cells(dataset: dict[str, Any]) -> list[tuple[str, str, list[str]]]:
    claims_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for claim in dataset["claims"]:
        claims_by_cell[(claim["system_id"], claim["capability_id"])].append(claim)
    conflicts: list[tuple[str, str, list[str]]] = []
    for (system_id, capability_id), claims in sorted(claims_by_cell.items()):
        linked_conflict = any(
            conflict_id
            for review in dataset["reviews"]
            if review.get("claim_id") in {claim["claim_id"] for claim in claims}
            for conflict_id in review.get("conflict_claim_ids", [])
        )
        if _cell(claims) == "contested" or linked_conflict:
            conflicts.append((system_id, capability_id, sorted(claim["claim_id"] for claim in claims)))
    return conflicts


def render_matrix(dataset: dict[str, Any]) -> str:
    systems = sorted(dataset["systems"], key=lambda item: item["system_id"])
    sources = {source["source_id"]: source for source in dataset["sources"]}
    claims_by_cell: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for claim in dataset["claims"]:
        claims_by_cell[(claim["system_id"], claim["capability_id"])].append(claim)

    lines = [
        "# RE-call prior art capability matrix",
        "",
        "Generated from the canonical evidence records in `docs/prior_art/`.",
        "",
        "This matrix records evidence, not absolute absence. `unknown` means the investigation is incomplete; `not_evidenced` means the reviewed sources did not establish the capability.",
        "",
    ]
    for group_id, capability in _capabilities(dataset):
        capability_id = capability["capability_id"]
        lines.extend(
            [
                f"## {group_id}: {capability['name']}",
                "",
                capability["definition"],
                "",
                "| System | Value | Evidence | Claim |",
                "| --- | --- | --- | --- |",
            ]
        )
        for system in systems:
            claims = claims_by_cell.get((system["system_id"], capability_id), [])
            value = _cell(claims)
            if claims:
                evidence = "; ".join(
                    _source_link(sources[claim["source_id"]]) for claim in sorted(claims, key=lambda item: item["claim_id"])
                )
                claim_text = "<br>".join(claim["claim_text"] for claim in sorted(claims, key=lambda item: item["claim_id"]))
            else:
                evidence = ""
                claim_text = "No accepted claim in the corpus."
            lines.append(f"| {system['display_name']} | `{value}` | {evidence} | {claim_text} |")
        lines.append("")

    lines.extend(
        [
            "## Value legend",
            "",
            "`verified` means primary evidence directly supports the claim. `partial` means the evidence covers a narrower case. `not_evidenced` means the reviewed sources did not establish the capability. `unknown` means research is incomplete. `contradicted` means available evidence conflicts with the claim. `contested` is rendered when accepted claims for one cell conflict.",
            "",
            "## Incomplete or unresolved claims",
            "",
        ]
    )
    incomplete_claims = _incomplete_claims(dataset)
    if incomplete_claims:
        for claim in incomplete_claims:
            lines.append(
                f"* `{claim['claim_id']}` for `{claim['system_id']}` and `{claim['capability_id']}` is `{claim['value']}`."
            )
    else:
        lines.append("No incomplete or unresolved claims are recorded.")
    lines.extend(["", "## Conflicting evidence", ""])
    conflicts = _conflicting_cells(dataset)
    if conflicts:
        for system_id, capability_id, claim_ids in conflicts:
            lines.append(
                f"* `{system_id}` and `{capability_id}` has conflicting evidence across "
                + ", ".join(f"`{claim_id}`" for claim_id in claim_ids)
                + "."
            )
    else:
        lines.append("No conflicting evidence is recorded.")
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _gap_classification(values: list[str]) -> str:
    if "contradicted" in values and ("verified" in values or "partial" in values):
        return "contested"
    if values.count("verified") >= 2:
        return "established"
    if "verified" in values or "partial" in values:
        return "emerging"
    return "unverified_gap"


def render_gap_report(dataset: dict[str, Any]) -> str:
    capabilities = _capabilities(dataset)
    claims_by_capability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in dataset["claims"]:
        claims_by_capability[claim["capability_id"]].append(claim)
    lines = [
        "# RE-call prior art gap report",
        "",
        "This report identifies research candidates. It does not assert that a capability has never been implemented.",
        "",
        "## Capability status",
        "",
    ]
    for group_id, capability in capabilities:
        claims = claims_by_capability.get(capability["capability_id"], [])
        values = [claim["value"] for claim in claims]
        classification = _gap_classification(values)
        systems = sorted({claim["system_id"] for claim in claims})
        lines.append(f"### {capability['capability_id']}")
        lines.append("")
        lines.append(f"Group: `{group_id}`. Status: **{classification}**.")
        lines.append("")
        lines.append(f"Definition: {capability['definition']}")
        lines.append("")
        if systems:
            lines.append("Systems with reviewed claims: " + ", ".join(f"`{system}`" for system in systems) + ".")
        else:
            lines.append("No reviewed system claim exists for this capability.")
        lines.append("")

    lines.extend(
        [
            "## RE-call research hypothesis",
            "",
            "The current hypothesis is a combination of evidence backed claims, explicit validity and supersession, reversible provenance lineage, authority and scope enforcement, deletion propagation through derived artifacts, abstention based on support and conflict, and action outcome feedback into future belief state.",
            "",
            "The matrix must establish the evidence boundary before this combination is described as novel.",
            "",
            "## Target combination analysis",
            "",
            "This section reports coverage of the configured RE-call hypothesis. It is not a novelty claim. A missing cell means that this corpus has not accepted evidence for that capability in that system.",
            "",
            "Target capabilities: " + ", ".join(f"`{capability_id}`" for capability_id in _target_hypothesis(dataset)) + ".",
            "",
            "| System | Combination status | Verified support | Partial support | Missing evidence | Conflicting evidence |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
    )
    for row in _combination_rows(dataset):
        lines.append(
            "| "
            + " | ".join(
                [
                    f"`{row['system_id']}`",
                    f"`{row['status']}`",
                    ", ".join(f"`{value}`" for value in row["verified"]) or "none",
                    ", ".join(f"`{value}`" for value in row["partial"]) or "none",
                    ", ".join(f"`{value}`" for value in row["missing"]) or "none",
                    ", ".join(f"`{value}`" for value in row["conflicts"]) or "none",
                ]
            )
            + "|"
        )
    lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_summary(dataset: dict[str, Any]) -> str:
    claims_by_capability: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for claim in dataset["claims"]:
        claims_by_capability[claim["capability_id"]].append(claim)
    capability_status: dict[str, str] = {}
    for _, capability in _capabilities(dataset):
        capability_id = capability["capability_id"]
        values = [claim["value"] for claim in claims_by_capability.get(capability_id, [])]
        capability_status[capability_id] = _gap_classification(values)
    unresolved_claim_ids = sorted(
        claim["claim_id"]
        for claim in dataset["claims"]
        if claim["value"] in {"unknown", "not_evidenced"}
    )
    combination = {
        row["system_id"]: {
            "status": row["status"],
            "verified": row["verified"],
            "partial": row["partial"],
            "missing": row["missing"],
            "conflicts": row["conflicts"],
        }
        for row in _combination_rows(dataset)
    }
    return json.dumps(
        {
            "schema_version": "1.0",
            "search_cutoff_date": dataset["report_config"]["search_cutoff_date"],
            "system_count": len(dataset["systems"]),
            "source_count": len(dataset["sources"]),
            "claim_count": len(dataset["claims"]),
            "review_count": len(dataset["reviews"]),
            "unresolved_claim_ids": unresolved_claim_ids,
            "capability_status": capability_status,
            "target_combination": {
                "capability_ids": _target_hypothesis(dataset),
                "systems": combination,
            },
        },
        indent=2,
        sort_keys=True,
    ) + "\n"


def render_files(dataset: dict[str, Any], *, check: bool = False) -> list[Path]:
    errors = validate_dataset(dataset)
    if errors:
        raise ValueError("prior art corpus is invalid:\n" + "\n".join(f"  {error}" for error in errors))
    root = Path(dataset.get("root", DATA_ROOT))
    outputs = {
        root / "generated_matrix.md": render_matrix(dataset),
        root / "generated_gap_report.md": render_gap_report(dataset),
        root / "generated_summary.json": render_summary(dataset),
    }
    changed: list[Path] = []
    for path, content in outputs.items():
        if check:
            if not path.exists() or path.read_text(encoding="utf-8") != content:
                changed.append(path)
        else:
            path.write_text(content, encoding="utf-8", newline="\n")
    return changed
