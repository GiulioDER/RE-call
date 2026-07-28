"""Exclude LOCOMO's known-corrupt questions from a results artifact — after VERIFYING the mapping.

::

    python -m benchmarks.locomo_audit --results benchmarks/results/recall_...json

Why
---
An independent audit of LOCOMO (github.com/dial481/locomo-audit) found **156 defective questions,
of which 99 carry a wrong golden answer** — 99 of the 1,540 answerable questions, 6.4%, which caps
any judged score at 93.57%. The remaining 57 are ``WRONG_CITATION``: the golden answer is right
and only the cited evidence turn is wrong, so they do not corrupt a score that grades the answer.

A benchmark number published against a competitor has to be reported BOTH ways — on the full set
(comparable to every other published LOCOMO result) and on the clean subset (what the systems
would score against a correct answer key). This module produces the second one.

The safety requirement, and why it is most of this file
------------------------------------------------------
The audit keys its findings ``locomo_{c}_qa{n}``; this harness keys its outcomes
``{sample_id}:{index}``. Joining those two is an index translation across two independently
authored files, and the failure mode is silent: an off-by-one still excludes exactly 99
questions — just the WRONG 99 — and the resulting "corrected" score is fabricated, in a direction
nobody can see from the output. That is a worse outcome than not excluding anything at all.

So the mapping is never trusted. For EVERY audit entry, the question text (and the golden answer,
where the entry carries one) is compared against what actually sits at the computed position in
``locomo10.json``, and any mismatch raises with the offending ids named. A convention error, a
re-ordered dataset, or a dataset revision upstream all surface as a loud failure rather than as a
quietly different number.

The numbering convention was ESTABLISHED, not assumed. Verified against
``locomo10.json`` (SHA256-pinned by the audit repo as byte-identical to snap-research/locomo):
``n`` is **0-based**, the same convention as this harness's own ``index``. All 156 entries match
their question text at offset 0 and all 156 match their golden answer; at offset 1, 0 of 156 match
the question. ``locomo_0_qa1`` is "When did Melanie paint a sunrise?", which is ``qa[1]`` of
``conv-26``, not ``qa[0]`` ("When did Caroline go to the LGBTQ support group?"). The evidence is
re-derived on every call by the verification pass above — this docstring records the conclusion,
the code re-earns it.
"""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from benchmarks.pipeline import aggregate
from benchmarks.rejudge import to_outcome
from benchmarks.systems import sample_id_of

#: The vendored audit findings. See ``audit_data/README.md`` for source, revision and licence.
DEFAULT_AUDIT_PATH = Path(__file__).parent / "audit_data" / "locomo_errors.json"

#: The error class that does NOT corrupt a judged score: the golden answer is correct and only the
#: cited evidence turn is wrong. 57 of the audit's 156 entries. Excluding these would throw away
#: 57 perfectly gradeable questions and overstate the correction.
CITATION_ONLY_ERROR = "WRONG_CITATION"

#: ``locomo_{conversation}_qa{question}``. Anchored: a partial match on a malformed id would
#: produce a plausible integer and a silently wrong mapping.
_AUDIT_ID = re.compile(r"^locomo_(\d+)_qa(\d+)$")


def parse_audit_id(audit_id: str) -> tuple[int, int]:
    """``locomo_3_qa68`` -> ``(3, 68)``: the 0-based conversation index and 0-based qa index."""
    match = _AUDIT_ID.match(audit_id.strip())
    if match is None:
        raise ValueError(
            f"audit id {audit_id!r} does not have the form 'locomo_<conversation>_qa<n>'"
        )
    return int(match.group(1)), int(match.group(2))


def _normalise(text: object) -> str:
    """Whitespace- and case-insensitive comparison key for the verification pass.

    Deliberately forgiving about spacing and case and about NOTHING else. The pass exists to catch
    an index that points at a DIFFERENT QUESTION, which no amount of normalisation can disguise;
    making it stricter would turn a trailing space in either file into a false alarm that a reader
    would eventually learn to suppress — and a suppressed integrity check is not one.
    """
    return " ".join(str(text).split()).casefold()


def load_audit(audit_path: Path | str = DEFAULT_AUDIT_PATH) -> list[dict[str, Any]]:
    """The audit entries, as a list of dicts."""
    entries: list[dict[str, Any]] = json.loads(Path(audit_path).read_text(encoding="utf-8"))
    return entries


def _locomo_index(locomo_path: Path | str) -> list[tuple[str, list[dict[str, Any]]]]:
    """``locomo10.json`` as ``[(sample_id, qa rows), ...]`` in file order — the mapping's domain.

    Conversation index is POSITIONAL in the audit's ids, so it has to be positional here too;
    `sample_id_of` supplies the other half of our id and raises on a conversation without one
    (the same identity rule the run script uses, so a translated id is the id the artifact holds).
    """
    conversations: list[dict[str, Any]] = json.loads(
        Path(locomo_path).read_text(encoding="utf-8")
    )
    return [(sample_id_of(c), list(c.get("qa") or [])) for c in conversations]


def verified_mapping(
    locomo_path: Path | str, audit_path: Path | str = DEFAULT_AUDIT_PATH
) -> dict[str, dict[str, Any]]:
    """``{our question_id: audit entry}`` for every audit entry, or raise naming the mismatches.

    This is the function that makes the exclusion safe to publish. Every entry is checked, not a
    sample: the candidate id is computed, the row at that position is read out of the LOCOMO file,
    and the entry is accepted only if the question text matches — plus the golden answer, when the
    entry carries one.

    Every failure mode lands in the same raise, with ids: an out-of-range conversation or qa index,
    a question that does not match, a golden answer that does not match, and two audit entries
    mapping onto the same question. The message says how many failed and shows the first few,
    because a whole-file mismatch (wrong convention, wrong dataset revision) and a single bad row
    need different responses and the count is what tells them apart.
    """
    conversations = _locomo_index(locomo_path)
    mapped: dict[str, dict[str, Any]] = {}
    problems: list[str] = []

    for entry in load_audit(audit_path):
        audit_id = str(entry.get("question_id", ""))
        conversation_index, qa_index = parse_audit_id(audit_id)
        if not 0 <= conversation_index < len(conversations):
            problems.append(f"{audit_id}: conversation {conversation_index} is not in the dataset")
            continue
        sample_id, qa_rows = conversations[conversation_index]
        if not 0 <= qa_index < len(qa_rows):
            problems.append(
                f"{audit_id}: qa index {qa_index} is out of range for {sample_id} "
                f"({len(qa_rows)} rows)"
            )
            continue

        row = qa_rows[qa_index]
        if _normalise(row.get("question")) != _normalise(entry.get("question")):
            problems.append(
                f"{audit_id}: question text does not match {sample_id}:{qa_index} "
                f"(audit={entry.get('question')!r}, dataset={row.get('question')!r})"
            )
            continue
        golden = entry.get("golden_answer")
        if golden is not None and _normalise(row.get("answer")) != _normalise(golden):
            problems.append(
                f"{audit_id}: golden answer does not match {sample_id}:{qa_index} "
                f"(audit={golden!r}, dataset={row.get('answer')!r})"
            )
            continue

        question_id = f"{sample_id}:{qa_index}"
        if question_id in mapped:
            problems.append(
                f"{audit_id}: maps onto {question_id}, already claimed by another entry"
            )
            continue
        mapped[question_id] = entry

    if problems:
        shown = "\n  ".join(problems[:10])
        raise ValueError(
            f"the audit ids do not line up with {locomo_path}: {len(problems)} of "
            f"{len(mapped) + len(problems)} entries failed verification. Excluding questions on an "
            f"unverified mapping would drop the WRONG questions and silently fabricate the "
            f"corrected score, so nothing is excluded.\n  {shown}"
        )
    return mapped


def bad_question_ids(
    locomo_path: Path | str,
    audit_path: Path | str = DEFAULT_AUDIT_PATH,
    *,
    include_citation_only: bool = False,
) -> set[str]:
    """The ids, in THIS harness's ``{sample_id}:{index}`` format, of audited-defective questions.

    Defaults to the 99 SCORE-CORRUPTING entries — the ones with a wrong golden answer, which is
    the population behind the audit's headline "99 of 1,540 (6.4%), ceiling 93.57%".
    ``include_citation_only=True`` widens it to all 156 by adding the ``WRONG_CITATION`` entries,
    whose golden answers are correct; that set is the right one for an evidence/retrieval metric
    (which reads the citations) and the wrong one for a judged answer score.
    """
    mapped = verified_mapping(locomo_path, audit_path)
    return {
        question_id
        for question_id, entry in mapped.items()
        if include_citation_only or entry.get("error_type") != CITATION_ONLY_ERROR
    }


def split_by_audit(
    outcomes: Sequence[Mapping[str, Any]], bad_ids: set[str]
) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    """Partition `outcomes` into (clean, excluded) by `bad_ids`. Order preserved, nothing lost.

    A partition, not a filter: the excluded questions are returned rather than dropped, so the
    count that was removed is reportable and the removed questions themselves are inspectable. A
    correction that only ever shows the number it produced is one nobody can check.
    """
    clean: list[Mapping[str, Any]] = []
    excluded: list[Mapping[str, Any]] = []
    for outcome in outcomes:
        target = excluded if str(outcome["question_id"]) in bad_ids else clean
        target.append(outcome)
    return clean, excluded


def audited_report(
    doc: Mapping[str, Any], locomo_path: Path | str, audit_path: Path | str = DEFAULT_AUDIT_PATH
) -> dict[str, Any]:
    """Both aggregates for one results artifact: as published, and with the bad questions removed.

    Reported side by side on purpose. The full-set number is the one comparable to every published
    LOCOMO result (all of which are scored against the same defective key); the clean number is the
    one that says what the system actually does. Neither replaces the other, and quoting only the
    clean one would be exactly the selective-denominator move this whole exercise is auditing.
    """
    bad_ids = bad_question_ids(locomo_path, audit_path)
    clean, excluded = split_by_audit(doc["outcomes"], bad_ids)
    return {
        "arm": doc.get("arm"),
        "audit_bad_ids": len(bad_ids),
        "n_all": len(doc["outcomes"]),
        "n_clean": len(clean),
        "n_excluded": len(excluded),
        "excluded_ids": sorted(str(o["question_id"]) for o in excluded),
        "aggregate_all": doc.get("aggregate"),
        "aggregate_clean": aggregate([to_outcome(o) for o in clean]),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m benchmarks.locomo_audit",
        description="Report a results artifact with and without LOCOMO's audited-bad questions.",
    )
    parser.add_argument("--results", type=Path, required=True, help="a results artifact")
    parser.add_argument("--data", type=Path, default=Path("locomo10.json"))
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT_PATH)
    parser.add_argument("--out", type=Path, help="write the full report here as JSON")
    args = parser.parse_args(argv)

    for path in (args.results, args.data, args.audit):
        if not path.exists():
            parser.error(f"{path} not found")

    doc: dict[str, Any] = json.loads(args.results.read_text(encoding="utf-8"))
    report = audited_report(doc, args.data, args.audit)

    all_rate = (report["aggregate_all"] or {}).get("answerable_accuracy") or {}
    clean_rate = report["aggregate_clean"]["answerable_accuracy"]
    print(f"{report['arm']}: {report['n_excluded']} of {report['n_all']} questions excluded")
    print(f"  as published: n={all_rate.get('n')} accuracy={all_rate.get('rate')}")
    print(f"  audit-clean : n={clean_rate['n']} accuracy={clean_rate['rate']}")
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"report -> {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
