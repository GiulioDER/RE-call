"""Select one current audit fix from retrieved IDs and externally observed failures."""

from __future__ import annotations

import posixpath
from collections import Counter
from datetime import datetime
from typing import Annotated, Any, Literal, NotRequired, TypedDict

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator, model_validator

from recall.eval.bm25 import BM25Index, tokenize
from recall.frontmatter import validity_bounds
from recall.records import MAX_RECORDS

SHORTLIST_SIZE = 3
MAX_ID_CHARS = 512
# A failure marker and the diagnostic compared with it for equality share one bound.
MAX_TEXT_CHARS = 4096
MAX_ASSERTION_CHARS = 8192
MIN_SYMPTOM_GROUPS = 2
MAX_SYMPTOM_GROUPS = 8
MAX_ALTERNATIVES = 8
MAX_FILE_HINTS = 32
MAX_OBSERVATIONS = 64
Revision = Annotated[str, StringConstraints(pattern=r"\A[0-9a-f]{40}\z")]
Text = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TEXT_CHARS)]
Identifier = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_ID_CHARS)]
Alternatives = Annotated[list[Identifier], Field(min_length=1, max_length=MAX_ALTERNATIVES)]

Decision = Literal["serve", "abstain", "unavailable"]
Reason = Literal[
    "observed_symptom", "ambiguous_records", "superseded_fix", "no_relevant_candidates",
    "ambiguous_symptom", "symptom_not_correlated", "ambiguous_test_node", "missing_observation",
    "ambiguous_observation", "failure_not_matched", "store_missing", "snapshot_unavailable",
]


class ShortlistEntry(TypedDict):
    id: str
    test_node: str


class ServedFix(TypedDict):
    id: str
    source: str
    source_commit: str
    verification: str
    repair: str


class FixDecision(TypedDict):
    decision: Decision
    reason: Reason
    shortlist: list[ShortlistEntry]
    fix: NotRequired[ServedFix]


class _StrictModel(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")


class FixCandidate(_StrictModel):
    """One retrieved bug ID with the caller's manifest for it.

    Every symptom group must match the report; a group matches when any of its alternatives does,
    compared as lowercased alphanumeric tokens, so case and punctuation are ignored.
    """

    id: Identifier
    fix_commit: Revision
    pre_fix_revision: Revision
    test_node: Identifier
    assertion_text: Annotated[str, Field(max_length=MAX_ASSERTION_CHARS)] = ""
    symptom_groups: Annotated[list[Alternatives], Field(min_length=MIN_SYMPTOM_GROUPS, max_length=MAX_SYMPTOM_GROUPS)]
    failure_marker: Text
    repair: Text


class FixObservation(_StrictModel):
    """One test outcome a trusted runner measured at a revision, with its extracted diagnostic."""

    revision: Revision
    test_node: Identifier
    outcome: Literal["fail", "pass", "error", "skipped"]
    diagnostic: Annotated[str, StringConstraints(strip_whitespace=True, max_length=MAX_TEXT_CHARS)] = ""


class FixRequest(_StrictModel):
    """A strict request: unknown keys are refused and candidate IDs must be unique."""

    report: Text
    revision: Revision
    file_hints: Annotated[list[Identifier], Field(max_length=MAX_FILE_HINTS)] = Field(default_factory=list)
    candidates: Annotated[list[FixCandidate], Field(min_length=1, max_length=MAX_RECORDS)]
    observations: Annotated[list[FixObservation], Field(max_length=MAX_OBSERVATIONS)] = Field(default_factory=list)

    @field_validator("file_hints")
    @classmethod
    def repository_paths(cls, hints: list[str]) -> list[str]:
        # Records name repository-relative POSIX paths; `./x` and a Windows `x\y` name the same file.
        return [posixpath.normpath(hint.replace("\\", "/")) for hint in hints]

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


def unavailable(reason: Literal["store_missing", "snapshot_unavailable"]) -> FixDecision:
    return {"decision": "unavailable", "reason": reason, "shortlist": []}


def _abstain(reason: Reason, shortlist: list[ShortlistEntry]) -> FixDecision:
    return {"decision": "abstain", "reason": reason, "shortlist": shortlist}


def select_fix(records: list[dict[str, Any]], request: FixRequest, *, now: datetime) -> FixDecision:
    """Accept entries from one policy-filtered lookup, using the same time as that lookup.

    The caller supplies trusted candidate manifests and observations from a runner at the given
    clean revision. This function does not execute a test or attest to the caller's inputs.
    """
    declared = {candidate.id: candidate for candidate in request.candidates}
    counts = Counter(record["id"] for record in records if record["id"] in declared)
    if any(count > 1 for count in counts.values()):
        return _abstain("ambiguous_records", [])
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
    entries: list[ShortlistEntry] = [{"id": key, "test_node": declared[key].test_node} for key in shortlist]
    if not shortlist:
        return _abstain("superseded_fix" if superseded else "no_relevant_candidates", entries)

    report = " " + " ".join(tokenize(request.report)) + " "
    matches = [key for key in eligible if _matches_symptom(report, declared[key])]
    if len(matches) > 1:
        return _abstain("ambiguous_symptom", entries)
    if not matches or matches[0] not in shortlist:
        return _abstain("symptom_not_correlated", entries)

    key = matches[0]
    candidate = declared[key]
    # A single test node used for another reachable repair cannot identify this symptom.
    if sum(declared[other].test_node == candidate.test_node for other in eligible) > 1:
        return _abstain("ambiguous_test_node", entries)
    observations = [observation for observation in request.observations
                    if observation.revision == request.revision and observation.test_node == candidate.test_node]
    if len(observations) != 1:
        return _abstain("missing_observation" if not observations else "ambiguous_observation", entries)
    observation = observations[0]
    if observation.outcome != "fail" or observation.diagnostic != candidate.failure_marker:
        return _abstain("failure_not_matched", entries)
    return {
        "decision": "serve",
        "reason": "observed_symptom",
        "shortlist": entries,
        "fix": {
            "id": key,
            "source": eligible[key]["source"],
            "source_commit": candidate.fix_commit,
            "verification": candidate.test_node,
            "repair": candidate.repair,
        },
    }
