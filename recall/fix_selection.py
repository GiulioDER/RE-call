"""Select one current audit fix from retrieved IDs and externally observed failures."""

from __future__ import annotations

from collections import Counter
from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from recall.eval.bm25 import BM25Index, tokenize
from recall.frontmatter import validity_bounds
from recall.records import MAX_RECORDS

SHORTLIST_SIZE = 3
Revision = Annotated[str, StringConstraints(pattern=r"\A[0-9a-f]{40}\z")]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=4096)]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=512)]
Alternatives = Annotated[list[Identifier], Field(min_length=1, max_length=8)]


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class FixCandidate(_StrictModel):
    id: Identifier
    fix_commit: Revision
    pre_fix_revision: Revision
    test_node: Identifier
    assertion_text: Annotated[str, Field(max_length=8192)] = ""
    symptom_groups: Annotated[list[Alternatives], Field(min_length=2, max_length=8)]
    failure_marker: Text
    repair: Text


class FixObservation(_StrictModel):
    revision: Revision
    test_node: Identifier
    outcome: Literal["fail", "pass", "error", "skipped"]
    diagnostic: Annotated[str, StringConstraints(strip_whitespace=True, max_length=4096)] = ""


class FixRequest(_StrictModel):
    report: Text
    revision: Revision
    file_hints: Annotated[list[Identifier], Field(max_length=32)] = Field(default_factory=list)
    candidates: Annotated[list[FixCandidate], Field(min_length=1, max_length=MAX_RECORDS)]
    observations: Annotated[list[FixObservation], Field(max_length=64)] = Field(default_factory=list)

    @model_validator(mode="after")
    def unique_candidates(self) -> FixRequest:
        if len({candidate.id for candidate in self.candidates}) != len(self.candidates):
            raise ValueError("candidate IDs must be unique")
        return self


def _valid_bug(record: dict[str, Any], now: datetime) -> bool:
    if record["kind"] != "bug" or record["expired"]:
        return False
    try:
        start, end = validity_bounds(record)
    except ValueError:
        return False
    return (start is None or start <= now) and (end is None or now <= end)


def _matches_symptom(report: str, candidate: FixCandidate) -> bool:
    return all(
        any((phrase := " ".join(tokenize(alternative))) and f" {phrase} " in report for alternative in group)
        for group in candidate.symptom_groups
    )


def select_fix(records: list[dict[str, Any]], request: FixRequest, *, now: datetime) -> dict[str, Any]:
    """Accept entries from one policy-filtered lookup, using the same time as that lookup.

    The caller supplies trusted candidate manifests and observations from a runner at the given
    clean revision. This function does not execute a test or attest to the caller's inputs.
    """
    declared = {candidate.id: candidate for candidate in request.candidates}
    counts = Counter(record["id"] for record in records if record["id"] in declared)
    if any(count > 1 for count in counts.values()):
        return {"decision": "abstain", "reason": "ambiguous_records", "shortlist": []}
    # The same snapshot reports supersession even when the successor is outside the request
    # or hidden by source policy. Any such verdict blocks only the exact requested record.
    superseded = {entry["id"] for entry in records if entry["id"] in declared and entry["superseded_by"]}
    eligible = {
        entry["id"]: entry for entry in records
        if entry["id"] in declared
        and entry["id"] not in superseded
        and _valid_bug(entry, now)
        and entry["recall_record"]["tier"] == "proved"
        and entry["recall_record"]["fix_commit"] == declared[entry["id"]].fix_commit
        and declared[entry["id"]].pre_fix_revision == request.revision
    }
    index = BM25Index(
        (key, " ".join((entry["recall_record"]["summary"], declared[key].test_node, declared[key].assertion_text)))
        for key, entry in eligible.items()
    )
    files = set(request.file_hints)
    ranked = [key for key, score in index.rank(request.report) if score > 0]
    ranked.sort(key=lambda key: files.isdisjoint(eligible[key]["recall_record"]["files"]))
    shortlist = ranked[:SHORTLIST_SIZE]
    result: dict[str, Any] = {
        "decision": "abstain",
        "shortlist": [{"id": key, "test_node": declared[key].test_node} for key in shortlist],
    }
    if not shortlist:
        return result | {"reason": "superseded_fix" if superseded else "no_relevant_candidates"}

    report = " " + " ".join(tokenize(request.report)) + " "
    matches = [key for key in eligible if _matches_symptom(report, declared[key])]
    if len(matches) > 1:
        return result | {"reason": "ambiguous_symptom"}
    if not matches or matches[0] not in shortlist:
        return result | {"reason": "symptom_not_correlated"}

    key = matches[0]
    candidate = declared[key]
    # A single test node used for another reachable repair cannot identify this symptom.
    if sum(declared[other].test_node == candidate.test_node for other in eligible) > 1:
        return result | {"reason": "ambiguous_test_node"}
    observations = [observation for observation in request.observations
                    if observation.revision == request.revision and observation.test_node == candidate.test_node]
    if len(observations) != 1:
        return result | {"reason": "missing_observation" if not observations else "ambiguous_observation"}
    observation = observations[0]
    if observation.outcome != "fail" or observation.diagnostic != candidate.failure_marker:
        return result | {"reason": "failure_not_matched"}
    return result | {
        "decision": "serve",
        "reason": "observed_symptom",
        "fix": {
            "id": key,
            "source": eligible[key]["source"],
            "source_commit": candidate.fix_commit,
            "verification": candidate.test_node,
            "repair": candidate.repair,
        },
    }
