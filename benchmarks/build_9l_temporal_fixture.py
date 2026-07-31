"""Rebuild `results/beam_9l_temporal.json`: the cases §9l's conclusion was drawn from.

Issue #167 step 1 — *"Reproduce the 7 questions in §9l as a fixture, so any fix is checked against
the cases that produced the claim."* Until this existed, §9l was a paragraph about seven questions
nobody could enumerate, and any future temporal selector would have been scored on an aggregate
rather than on the cases that motivated it.

PRIOR WORK — searched, but NOT with `docs_search`, which was unavailable. The memory corpus is
indexed on VPS2 only and VPS2 was down; `qwen-vps3`'s copy of the tool answers `relation
"docs_chunks" does not exist`. Stated rather than skipped, because "I searched" and "the search
tool was dead" are different claims and only one of them is true here. Fallback was a grep over
the memory directory plus the in-repo temporal work, which surfaced:

  - [[project-recall-entailment-supersession-phase0-done-2026-07-18]] — supersession is already
    shipped and studied; this file does not propose adding it.
  - [[project-recall-beam-benchmark-2026-07-28]] — the BEAM run this reads was ~$45 of OpenRouter
    on VPS2, and it carries §9l's overclaim verbatim ("a known limit, not an open task").
  - `benchmarks/check_temporal_inert.py`, `benchmarks/check_p1_supersession_density.py`,
    `docs/REFERENCE_TIME_DESIGN.md` — the temporal work already on master.

None of them enumerates §9l's failures; that gap is what this file closes.
`benchmarks/audit_data/locomo_errors.json` is the shape precedent — a vendored error table with a
hand `error_type` and a stated `reasoning` per row.

WHY THE ARTIFACT IS COMMITTED AND THE RUNS ARE NOT
--------------------------------------------------
`benchmarks/results/` is gitignored, so the two run files this reads cannot live in the repo. A
figure whose only backing sits outside the repo is the defect this project has already paid for
once, so the *derived* fixture is committed under `results/` — where `benchmarks/claim_gate.py`
can resolve `<!--@ beam_9l_temporal.json#... -->` markers against it and machine-check every
number §9l publishes about these cases.

The run files are identified by **sha256, not by path**. A path says where a file was; a hash says
which file it was. `--recall-run` / `--mem0-run` therefore accept any location, and a mismatch is
refused rather than silently rebuilt from a different run.

WHAT IS MEASURED AND WHAT IS JUDGED
-----------------------------------
Two different kinds of content, kept separable on purpose:

  - **Measured** — ids, scores, gold rubric, our verbatim answer, and every count in `summary`.
    All of it is re-derived from the runs on each rebuild.
  - **Judged** — `mechanism` and `reasoning` in `CLASSIFICATION` below. These are a hand reading
    of five answers, made from the run artifact ONLY: the BEAM corpus is not cached locally, so
    the turns themselves were never consulted. Where a classification claims something about the
    corpus (e.g. that 25 March was the original deadline), the evidence is our own answer's
    wording, quoted in the reason. Five items is not a sample, and nothing here is a rate.

SELF-CHECKS (pre-registered — these are §9l's own published counts, and the rebuild fails if the
runs stop reproducing them):

  S1  the paired temporal_reasoning means are 0.408 (RE-call) and 0.567 (Mem0), n=30
  S2  exactly 7 questions lose by a full point
  S3  exactly 1 of those 7 had empty retrieval
  S4  exactly 5 of those 7 were answered without abstaining  ("confidently and wrongly")
  S5  in every confident-wrong case our stated day-count is CORRECT arithmetic on the operands we
      chose, and our answer text contains that count
  S6  a case carries transcribed operands if and only if it is confident-wrong
  S7  each transcribed gold interval reproduces the day count the rubric itself states, and its
      endpoints match the "from X till Y" the rubric spells out

Added later, and listed here because an inventory that omits a check is how a stale label starts:

  S2b every case really is a full-point loss, checked where CI can see it
  S6b the mechanism label agrees with the `abstained` / `retrieval_empty` flags it describes
  S8  intervals, mechanism and reasoning all match `CLASSIFICATION`, the authoritative copy in
      CODE, which is the backstop for what the prose checks cannot constrain

S5 is not in §9l. It is the check that decides whether a retrieval-side fix is even the right
shape: if the arithmetic were wrong, date selection would not be the bottleneck.

S6 and S7 were added after a mutation pass: four deliberate corruptions were run against the
build, and the first three were caught while "give the ABSTAINED case a pair of operands" was
not. A fixture that accepts invented operands for a question where we answered nothing would
report a date-selection failure that never happened, which is the one error this file exists to
prevent. S7 closes the matching hole on the other side — before it, a mistyped gold endpoint was
caught only when it happened to collide with our own answer.

Every check above that does not need the runs lives in `validate()`, which runs against the
COMMITTED fixture. That split is the guard, not a tidiness choice: `benchmarks/results/` is
gitignored, so anything left in `build()` alone is unverified in CI and in every fresh clone. A
bug audit found three invariants on the wrong side of that line -- the badly-lost margin, the
mechanism/flag partition, and the rubric endpoints -- each demonstrated by a corrupted fixture
that the whole suite accepted. They are in `validate()` now.

Run:  python benchmarks/build_9l_temporal_fixture.py --recall-run PATH --mem0-run PATH
      python benchmarks/build_9l_temporal_fixture.py --check           (re-derive; needs the runs)
      python benchmarks/build_9l_temporal_fixture.py --validate-only   (CI: no runs needed)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import statistics
import sys
from datetime import date
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS = REPO_ROOT / "benchmarks" / "results"
FIXTURE_PATH = REPO_ROOT / "results" / "beam_9l_temporal.json"

QUESTION_TYPE = "temporal_reasoning"

#: The two runs §9l was drawn from, pinned by content.
#:
#: The Mem0 side is not a free choice: of the three re-judged artifacts, only this one covers all
#: 30 paired ids (the other two carry 599 rows and cover 20). Pairing is therefore forced by
#: coverage rather than picked, which is worth recording — "we chose the artifact that agreed with
#: us" is exactly the objection a pinned hash cannot answer on its own.
RECALL_RUN_SHA256 = "6cdf7a8c90c9dc090c4c6c42aedf3987b1a5d0c9166ac4b2fc4c8207a2ae512d"
MEM0_RUN_SHA256 = "815f4ac878473e0f351b4bf74f44737f11ad950233d0bfafe4acaa596bbfefdd"
RECALL_RUN_NAME = "beam1M_recall_20260726T201947Z.json"
MEM0_RUN_NAME = "beam1M_mem0-rejudged_20260726T113417Z.json"

#: A question "loses badly" when Mem0 scores a full point above us. §9l's own wording is
#: "badly-lost", and a full point on a 0..1 rubric mean is the only threshold that yields its
#: published 7 — recorded here so the set is reproducible rather than eyeballed.
BADLY_LOST_DELTA = -1.0

#: Hand classification. See the module docstring for what "judged" means here.
#:
#: `gold` / `ours` are the interval endpoints, transcribed from the rubric and from our own answer
#: text. They are not decoration: the rebuild recomputes the day counts from them, so a
#: transcription slip fails the build instead of shipping.
CLASSIFICATION: dict[str, dict[str, Any]] = {
    "1M_0_q18_temporal_reasoning": {
        "mechanism": "supersession",
        "gold": ("2024-03-25", "2024-04-01"),
        "ours": ("2024-04-01", "2024-04-15"),
        "reasoning": (
            "Our own answer names the revision: 'Translation API integration deadline: April 15, "
            "2024 (updated on March 26, 2024)'. The question's interval ends before that update, "
            "so the ORIGINAL deadline is the covering instance. This is the row §9l cites as "
            "ruling out newest-wins, and it is the ONLY one of the five a supersession edge "
            "between two assertions of the same field could reach."
        ),
    },
    "1M_1_q18_temporal_reasoning": {
        "mechanism": "role_value_vs_assertion_time",
        "gold": ("2024-02-15", "2024-02-20"),
        "ours": ("2024-01-10", "2024-01-10"),
        "reasoning": (
            "Gold reads 'when I set the Sprint 1 deadline' as the deadline's VALUE (15 Feb) and "
            "'when I adjusted the sprint deadline to mid-February' as the adjusted VALUE (20 "
            "Feb). We answered with the date on which both were ASSERTED (10 Jan), collapsing "
            "the interval to zero. Neither instance is the newer one — they are the same turn — "
            "so no rule that orders instances in time can separate them."
        ),
    },
    "1M_8_q18_temporal_reasoning": {
        "mechanism": "instance_disambiguation",
        "gold": ("2024-02-01", "2024-04-25"),
        "ours": ("2024-01-26", "2024-04-18"),
        "reasoning": (
            "Both endpoints are a different instance of the same recurring kind of event (a "
            "study-prep session). No revision and no value/assertion split is involved: the "
            "failure is which of several similar events the question names."
        ),
    },
    "1M_12_q18_temporal_reasoning": {
        "mechanism": "instance_disambiguation",
        "gold": ("2024-03-25", "2024-04-10"),
        "ours": ("2024-03-15", "2024-04-10"),
        "reasoning": (
            "Second endpoint correct. The first is a different property viewing (15 March) from "
            "the first-with-Mehmet the question names (25 March)."
        ),
    },
    "1M_13_q18_temporal_reasoning": {
        "mechanism": "event_vs_utterance_time",
        "gold": ("2023-03-15", "2023-08-01"),
        "ours": ("2023-03-15", "2024-01-10"),
        "reasoning": (
            "Our answer CONTAINS the gold endpoint — '(referencing that you met her during "
            "clinic onboarding on August 1, 2023)' — and then computes from the date the turn "
            "was SAID (10 Jan 2024) instead. Retrieval was sufficient; choosing between "
            "utterance time and event time is what failed. That is the distinction "
            "docs/REFERENCE_TIME_DESIGN.md identifies as the reason `valid_from` cannot serve as "
            "a question's reference time, here costing a full point on its own."
        ),
    },
    "1M_13_q19_temporal_reasoning": {
        "mechanism": "abstained_with_retrieval",
        "gold": ("2024-02-04", "2024-04-09"),
        "ours": None,
        "reasoning": (
            "Abstained although retrieval was NOT empty. No date was selected, so this is not a "
            "date-instance failure and no selector can be scored on it. It belongs in the "
            "abstention lane (§9m, §12), not this one."
        ),
    },
    "1M_14_q18_temporal_reasoning": {
        "mechanism": "retrieval_empty",
        "gold": ("2024-03-15", "2024-04-06"),
        "ours": None,
        "reasoning": (
            "Retrieval returned nothing, so the temporal layer was never reached. §9l already "
            "sets this one aside; it is carried here so the seven are enumerable rather than "
            "described."
        ),
    },
}

#: Mechanisms that describe a wrong DATE CHOICE, as opposed to no choice having been made.
DATE_SELECTION_MECHANISMS = frozenset(
    {"supersession", "role_value_vs_assertion_time", "instance_disambiguation", "event_vs_utterance_time"}
)


class BuildError(RuntimeError):
    """A self-check failed. Never raised for a missing file — that is a skip, not a failure."""


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_rows(path: Path, expected_sha256: str, label: str) -> dict[str, dict]:
    actual = sha256_of(path)
    if actual != expected_sha256:
        raise BuildError(
            f"{label} run at {path} has sha256 {actual}, expected {expected_sha256}. "
            f"This fixture pins the run §9l was drawn from; rebuilding it from a DIFFERENT run "
            f"would silently re-target the cases. Point --{label}-run at the pinned file, or "
            f"update the constant deliberately and say so in the commit."
        )
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {r["question_id"]: r for r in payload["rows"] if r["question_type"] == QUESTION_TYPE}


def _days(span: tuple[str, str]) -> int:
    start, end = (date.fromisoformat(s) for s in span)
    return (end - start).days


def _stated_days(answer: str) -> int | None:
    """The day count our answer leads with, e.g. '14 days.' -> 14.

    Read from the text rather than taken on trust: the structured `ours` endpoints below are a
    hand transcription, and this is what ties them back to the verbatim answer.
    """
    match = re.match(r"\s*(-?\d+)\s+days?\b", answer)
    return int(match.group(1)) if match else None


MONTHS = (
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
)

#: The rubric's second line always names the interval in prose, e.g.
#: "LLM response should state: from March 25, 2024 till April 1, 2024". The year is sometimes
#: absent ("from February 15 till February 20"), so it is optional here and compared only when
#: present -- requiring it would reject TWO of the seven (1M_1_q18 and 1M_14_q18) for being
#: transcribed correctly. Counted, because the first draft of this comment said three.
#: The year those two rubrics omit is pinned by S8 against `CLASSIFICATION` instead.
_RUBRIC_SPAN_RE = re.compile(r"\bfrom\s+(.+?)\s+till\s+(.+?)\s*$", re.I | re.M)


def _parse_prose_date(text: str) -> tuple[int, int, int | None]:
    """'March 25, 2024' -> (3, 25, 2024); 'February 15' -> (2, 15, None)."""
    match = re.match(r"\s*([A-Za-z]+)\s+(\d{1,2})(?:\s*,\s*(\d{4}))?\s*$", text)
    if match is None or match.group(1).capitalize() not in MONTHS:
        raise BuildError(f"cannot parse a date out of {text!r}")
    month = MONTHS.index(match.group(1).capitalize()) + 1
    return month, int(match.group(2)), int(match.group(3)) if match.group(3) else None


def _date_is_mentioned(text: str, iso: str) -> bool:
    """True when `text` names this date in prose, e.g. '2024-04-01' in 'April 1, 2024'.

    `\\b` after the day is load-bearing: without it 'April 1' matches inside 'April 15', and the
    endpoints of `1M_0_q18` are exactly April 1 and April 15.
    """
    day = date.fromisoformat(iso)
    pattern = rf"\b{MONTHS[day.month - 1]}\s+{day.day}\b(?:\s*,\s*{day.year})?"
    return re.search(pattern, text, re.I) is not None


def build(recall_rows: dict[str, dict], mem0_rows: dict[str, dict]) -> dict:
    paired = sorted(set(recall_rows) & set(mem0_rows))
    if not paired:
        raise BuildError(
            "no question ids are common to the two runs, so there is nothing to pair. "
            "S1's diagnostic below cannot run; check QUESTION_TYPE and the id scheme."
        )
    recall_mean = statistics.mean(recall_rows[i]["score"] for i in paired)
    mem0_mean = statistics.mean(mem0_rows[i]["score"] for i in paired)

    # ---- S1 --------------------------------------------------------------------------------
    if not (f"{recall_mean:.3f}" == "0.408" and f"{mem0_mean:.3f}" == "0.567"):
        raise BuildError(
            f"S1: paired means are {recall_mean:.4f}/{mem0_mean:.4f}, but §9l published "
            f"0.408/0.567. The runs no longer reproduce the published section."
        )

    lost = [i for i in paired if recall_rows[i]["score"] - mem0_rows[i]["score"] <= BADLY_LOST_DELTA]

    # ---- S2 --------------------------------------------------------------------------------
    if len(lost) != 7:
        raise BuildError(f"S2: {len(lost)} questions lose by a full point, §9l says 7")
    if set(lost) != set(CLASSIFICATION):
        raise BuildError(
            f"S2: the badly-lost set moved. runs={sorted(lost)} classified={sorted(CLASSIFICATION)}"
        )

    cases = []
    for question_id in lost:
        row = recall_rows[question_id]
        hand = CLASSIFICATION[question_id]
        gold_span = hand["gold"]
        ours_span = hand["ours"]
        case: dict[str, Any] = {
            "question_id": question_id,
            "conversation_idx": row["conversation_idx"],
            "question": row["question"],
            "gold_rubric": row["rubric"],
            "gold_interval": {"start": gold_span[0], "end": gold_span[1], "days": _days(gold_span)},
            "our_answer": row["generated_answer"],
            "our_interval": (
                None
                if ours_span is None
                else {"start": ours_span[0], "end": ours_span[1], "days": _days(ours_span)}
            ),
            "recall_score": row["score"],
            "mem0_score": mem0_rows[question_id]["score"],
            "retrieval_empty": row["retrieval_empty"],
            "abstained": row["abstained"],
            "mechanism": hand["mechanism"],
            "reasoning": hand["reasoning"],
        }
        cases.append(case)

    empty = [c for c in cases if c["retrieval_empty"]]
    confident_wrong = [c for c in cases if not c["abstained"] and not c["retrieval_empty"]]
    tally: dict[str, int] = {}
    for case in cases:
        tally[case["mechanism"]] = tally.get(case["mechanism"], 0) + 1

    fixture = {
        "_note": (
            "The seven temporal_reasoning questions §9l's conclusion was drawn from, re-derived "
            "from the two pinned BEAM runs. Regenerate with "
            "benchmarks/build_9l_temporal_fixture.py; do not hand-edit. `mechanism` and "
            "`reasoning` are a hand classification made from these runs alone — the BEAM corpus "
            "was not consulted, because it is not cached locally."
        ),
        "provenance": {
            "recall_run": RECALL_RUN_NAME,
            "recall_run_sha256": RECALL_RUN_SHA256,
            "mem0_run": MEM0_RUN_NAME,
            "mem0_run_sha256": MEM0_RUN_SHA256,
            "mem0_run_choice": (
                "Forced, not selected: of the three re-judged Mem0 artifacts only this one covers "
                "all 30 paired ids; the other two carry 599 rows and cover 20."
            ),
            "badly_lost_delta": BADLY_LOST_DELTA,
            "question_type": QUESTION_TYPE,
        },
        "summary": {
            "paired_n": len(paired),
            "recall_mean": recall_mean,
            "mem0_mean": mem0_mean,
            "badly_lost": len(cases),
            "retrieval_empty": len(empty),
            "abstained_with_retrieval": len([c for c in cases if c["abstained"] and not c["retrieval_empty"]]),
            "confident_wrong": len(confident_wrong),
            "arithmetic_correct_on_own_operands": len(confident_wrong),
            "date_selection_failures": len([c for c in cases if c["mechanism"] in DATE_SELECTION_MECHANISMS]),
            "mechanism_tally": tally,
            "supersession_reachable": tally.get("supersession", 0),
        },
        "cases": cases,
    }
    validate(fixture)
    return fixture


#: Fields every case must carry, with the type `validate` will treat them as.
_CASE_SHAPE: tuple[tuple[str, type | tuple[type, ...]], ...] = (
    ("question_id", str), ("question", str), ("gold_rubric", list), ("our_answer", str),
    ("conversation_idx", int), ("recall_score", (int, float)), ("mem0_score", (int, float)),
    ("retrieval_empty", bool), ("abstained", bool), ("mechanism", str), ("reasoning", str),
)


def _require_shape(case: object) -> None:
    """Raise `BuildError` -- not `KeyError`/`TypeError` -- for a malformed case.

    `main` catches only `BuildError`, so without this a hand-typed slip in `CLASSIFICATION`
    surfaced as a traceback instead of the `FAIL` line the CLI promises. A `gold_rubric` given
    as a bare string was worse than a crash: `" ".join(...)` spaced out every character and the
    old rubric regex still matched, so it was silently ACCEPTED.
    """
    if not isinstance(case, dict):
        raise BuildError(f"case is {type(case).__name__}, expected an object")
    where = case.get("question_id", "<no question_id>")
    for key, kind in _CASE_SHAPE:
        if key not in case:
            raise BuildError(f"{where} is missing {key!r}")
        if isinstance(case[key], bool) != (kind is bool) or not isinstance(case[key], kind):
            raise BuildError(f"{where}.{key} is {type(case[key]).__name__}, expected {kind}")
    if not all(isinstance(line, str) for line in case["gold_rubric"]):
        raise BuildError(f"{where}.gold_rubric holds a non-string line")
    for key in ("gold_interval", "our_interval"):
        span = case.get(key, ...)
        if span is ...:
            raise BuildError(f"{where} is missing {key!r}")
        if span is None:
            # Only `our_interval` may legitimately be null (S6 decides where): a case with no
            # gold interval has no answer to be scored against at all.
            if key == "gold_interval":
                raise BuildError(f"{where}.gold_interval is null; every case has a gold answer")
            continue
        if not isinstance(span, dict) or {"start", "end", "days"} - set(span):
            raise BuildError(f"{where}.{key} must hold start, end and days")
        if isinstance(span["days"], bool) or not isinstance(span["days"], int):
            raise BuildError(f"{where}.{key}.days is not an int")
        for bound in ("start", "end"):
            try:
                date.fromisoformat(span[bound])
            except (TypeError, ValueError) as exc:
                raise BuildError(f"{where}.{key}.{bound} is not an ISO date: {exc}") from exc


def validate(fixture: dict) -> None:
    """S2b, S3-S8 plus summary consistency, checked against the FIXTURE alone.

    Split out from `build` on purpose. The two runs are gitignored, so CI never sees them; if
    these checks lived only in `build`, the committed artifact would ship with nothing verifying
    it and the guard would be one that cannot fire where it matters. Everything here is derivable
    from the fixture, so `tests/test_9l_temporal_fixture.py` runs the SAME code on the committed
    file rather than a second implementation of the same rules.
    """
    try:
        cases = fixture["cases"]
        summary = fixture["summary"]
        delta_gate = fixture["provenance"]["badly_lost_delta"]
        if not isinstance(summary, dict):
            raise BuildError(f"summary is {type(summary).__name__}, expected an object")
        if isinstance(delta_gate, bool) or not isinstance(delta_gate, (int, float)):
            raise BuildError(f"provenance.badly_lost_delta is {type(delta_gate).__name__}")
        for case in cases:
            _require_shape(case)
    # No `AttributeError`: nothing inside this block can raise one. The `summary: []` case that
    # motivated adding it is rejected by the isinstance check above, and its actual raise site
    # (`summary.get`) is outside this try. An except arm that cannot fire reads as handling.
    except (KeyError, TypeError, ValueError) as exc:
        raise BuildError(f"malformed fixture: {type(exc).__name__}: {exc}") from exc

    # The threshold is a LABEL inside the file being checked, so it is pinned to the module
    # constant the same way the run hashes are. Read as authoritative, a two-token edit to
    # `provenance.badly_lost_delta` weakens S2b below while every control stays green —
    # demonstrated with 0.5, which admits a TIE as a "loss".
    if delta_gate != BADLY_LOST_DELTA:
        raise BuildError(
            f"provenance.badly_lost_delta is {delta_gate}, but this module builds the set at "
            f"{BADLY_LOST_DELTA}. The artifact does not get to move its own gate."
        )

    ids = [c["question_id"] for c in cases]
    if len(set(ids)) != len(ids):
        raise BuildError(f"duplicate question_id in cases: {ids}")
    if set(ids) != set(CLASSIFICATION):
        raise BuildError(
            f"cases do not match the classification table. "
            f"fixture={sorted(ids)} table={sorted(CLASSIFICATION)}"
        )

    # ---- S2b: the defining property, checked where CI can see it ----------------------------
    # `build` applies this to pick the seven, but `build` needs the gitignored runs. Without it
    # here, a committed fixture could publish seven questions RE-call did not lose -- confirmed
    # by mutation: setting every case to recall 1.0 / mem0 0.0 was accepted before this existed.
    for case in cases:
        margin = case["recall_score"] - case["mem0_score"]
        if margin > delta_gate:
            raise BuildError(
                f"S2b: {case['question_id']} scores {case['recall_score']} vs "
                f"{case['mem0_score']} (margin {margin:+}), which is not a loss of "
                f"{delta_gate}. This fixture is the badly-lost set."
            )

    empty = [c for c in cases if c["retrieval_empty"]]
    abstained_only = [c for c in cases if c["abstained"] and not c["retrieval_empty"]]
    confident_wrong = [c for c in cases if not c["abstained"] and not c["retrieval_empty"]]

    # ---- S3, S4 ----------------------------------------------------------------------------
    if len(empty) != 1:
        raise BuildError(f"S3: {len(empty)} had empty retrieval, §9l says 1")
    if len(confident_wrong) != 5:
        raise BuildError(f"S4: {len(confident_wrong)} answered without abstaining, §9l says 5")

    # ---- S6: operands exist exactly where a date was actually chosen -------------------------
    for case in cases:
        chose_dates = not case["abstained"] and not case["retrieval_empty"]
        if chose_dates != (case["our_interval"] is not None):
            raise BuildError(
                f"S6: {case['question_id']} has operands={case['our_interval'] is not None} but "
                f"abstained={case['abstained']} retrieval_empty={case['retrieval_empty']}. "
                f"Operands belong to cases where we chose dates, and only those."
            )

    # ---- S6b: the mechanism label agrees with the flags it claims to describe -----------------
    # Without this the tally is self-consistent by construction: `mechanism_tally` is recomputed
    # from whatever labels are present, so a mislabelled case publishes a wrong count straight
    # through a claim-gate marker into §9l's table with nothing objecting.
    for case in cases:
        mechanism = case["mechanism"]
        if case["retrieval_empty"]:
            required = "retrieval_empty"
        elif case["abstained"]:
            required = "abstained_with_retrieval"
        else:
            required = "<a date-selection mechanism>"
        wrong = (
            mechanism != required
            if required != "<a date-selection mechanism>"
            else mechanism not in DATE_SELECTION_MECHANISMS
        )
        if wrong:
            raise BuildError(
                f"S6b: {case['question_id']} is labelled {mechanism!r} but abstained="
                f"{case['abstained']} retrieval_empty={case['retrieval_empty']} requires "
                f"{required}"
            )

    # ---- S7: the transcribed gold interval agrees with the rubric that defines it -------------
    for case in cases:
        rubric = " ".join(case["gold_rubric"])
        gold = case["gold_interval"]
        if _days((gold["start"], gold["end"])) != gold["days"]:
            raise BuildError(f"S7: {case['question_id']} gold days disagree with its own endpoints")
        # Anchored to the rubric's own phrasing. A bare `\b{days}\b` matched ANY integer in the
        # rubric, and the rubric spells out both endpoint dates -- so a wrong gold interval whose
        # length happened to equal a day-of-month or the year passed. Confirmed by mutation:
        # 1M_12's gold shortened to 10 days matched the "April 10" in its own rubric.
        if not re.search(rf"state:\s*{gold['days']}\s+days?\b", rubric, re.I):
            raise BuildError(
                f"S7: {case['question_id']} gold interval spans {gold['days']} days but the "
                f"rubric does not state that as its answer: {rubric!r}"
            )
        # And the endpoints themselves, not only their difference: a delta-preserving slip
        # (shifting both ends by the same amount) survives every check that looks at length.
        span = _RUBRIC_SPAN_RE.search(rubric)
        if span is None:
            raise BuildError(f"S7: {case['question_id']} rubric states no 'from X till Y' span")
        for label, text, iso in (
            ("start", span.group(1), gold["start"]),
            ("end", span.group(2), gold["end"]),
        ):
            month, day, year = _parse_prose_date(text)
            actual = date.fromisoformat(iso)
            if (month, day) != (actual.month, actual.day) or (year is not None and year != actual.year):
                raise BuildError(
                    f"S7: {case['question_id']} gold {label} is {iso} but the rubric says "
                    f"{text!r}"
                )

    # ---- S5: our arithmetic was right; our operands were not --------------------------------
    for case in confident_wrong:
        interval = case["our_interval"]
        stated = _stated_days(case["our_answer"])
        if stated is None:
            raise BuildError(f"S5: {case['question_id']} answer does not lead with a day count")
        if _days((interval["start"], interval["end"])) != interval["days"]:
            raise BuildError(f"S5: {case['question_id']} our days disagree with its own endpoints")
        if stated != interval["days"]:
            raise BuildError(
                f"S5: {case['question_id']} answered {stated} days but "
                f"{interval['start']}..{interval['end']} is {interval['days']} days. Either the "
                f"transcribed operands are wrong, or the arithmetic claim in §9l is."
            )
        if interval["days"] == case["gold_interval"]["days"]:
            raise BuildError(f"S5: {case['question_id']} matches gold — it is not a loss")
        # The operands are the fixture's whole payload — a future selector is scored on them —
        # and until now only their DIFFERENCE was checked, so shifting both ends by the same
        # amount shipped clean. Our answers name their dates in prose, so require that.
        for label in ("start", "end"):
            if not _date_is_mentioned(case["our_answer"], interval[label]):
                raise BuildError(
                    f"S5: {case['question_id']} {label} operand {interval[label]} is not named "
                    f"anywhere in our own answer, so it is not what we computed from"
                )

    # ---- S8: the payload agrees with the table it was generated from --------------------------
    # Last on purpose. The checks above read the endpoints against prose that lives in the same
    # file, which makes them informative -- they say WHICH rubric or answer the value contradicts
    # -- but prose cannot constrain what prose omits. Two rubrics state no year, and
    # `_date_is_mentioned`'s year group is optional and trailing, so a whole-year shift satisfied
    # every one of them: 1M_0's operands moved to 2020 with all tests green.
    #
    # `CLASSIFICATION` holds the authoritative pair in CODE. Running it last keeps the specific
    # diagnostics first and makes this the backstop for exactly what they cannot see.
    for case in cases:
        hand = CLASSIFICATION[case["question_id"]]
        # All FOUR fields build() takes from the table, not just the two intervals. `mechanism`
        # is the one §9l publishes through a claim-gate marker, so an unpinned relabel moves a
        # figure in a released document: swapping 1M_8 to `event_vs_utterance_time` with a
        # self-consistent tally was accepted, CI green. `reasoning` is the evidence a reader
        # checks the label against, so it is pinned beside it rather than left free to drift.
        for key in ("mechanism", "reasoning"):
            if case[key] != hand[key]:
                raise BuildError(
                    f"S8: {case['question_id']}.{key} does not match the classification table"
                )
        for key, expected_span in (("gold_interval", hand["gold"]), ("our_interval", hand["ours"])):
            actual = case[key]
            if expected_span is None:
                # No "and it had better be null" branch here. Reaching one would need a case
                # whose flags say confident-wrong (S6 lets it carry operands), whose mechanism is
                # therefore a date-selection label (S6b), and whose mechanism equals the table's
                # `abstained_with_retrieval` (the check above) -- a contradiction. The branch
                # that used to sit here could not fire, which is the third dead guard this file
                # has grown and removed.
                continue
            try:
                want = {"start": expected_span[0], "end": expected_span[1],
                        "days": _days(expected_span)}
            except (TypeError, ValueError) as exc:
                # Names the TABLE, not the artifact. `_days` here runs on module data, so a typo
                # in CLASSIFICATION would otherwise surface as "FAIL <date error>" against the
                # committed JSON, blaming the file that is correct.
                raise BuildError(
                    f"S8: CLASSIFICATION[{case['question_id']!r}][{key}] is not a valid date "
                    f"pair: {exc}"
                ) from exc
            if actual != want:
                raise BuildError(
                    f"S8: {case['question_id']}.{key} is {actual}, but the classification table "
                    f"says {want}"
                )

    # ---- the summary must describe the cases it ships beside --------------------------------
    tally: dict[str, int] = {}
    for case in cases:
        tally[case["mechanism"]] = tally.get(case["mechanism"], 0) + 1
    expected = {
        "badly_lost": len(cases),
        "retrieval_empty": len(empty),
        "abstained_with_retrieval": len(abstained_only),
        "confident_wrong": len(confident_wrong),
        "arithmetic_correct_on_own_operands": len(confident_wrong),
        "date_selection_failures": len([c for c in cases if c["mechanism"] in DATE_SELECTION_MECHANISMS]),
        "mechanism_tally": tally,
        "supersession_reachable": tally.get("supersession", 0),
    }
    for key, value in expected.items():
        if summary.get(key) != value:
            raise BuildError(f"summary.{key} is {summary.get(key)!r}, cases say {value!r}")

    # There was a trailing "unknown mechanism label" check here. S6b makes it unreachable -- it
    # constrains every case's mechanism to the vocabulary already -- so it was removed rather than
    # left in place. A branch that cannot fire still reads as a guard, and the audit that prompted
    # this pass found its negative control was passing on the summary check above instead.


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--recall-run", type=Path, default=DEFAULT_RUNS / RECALL_RUN_NAME)
    parser.add_argument("--mem0-run", type=Path, default=DEFAULT_RUNS / MEM0_RUN_NAME)
    parser.add_argument("--out", type=Path, default=FIXTURE_PATH)
    # Mutually exclusive: passing both used to run only the weaker check and return 0, the same
    # code a real re-derivation returns. That is precisely the conflation the two-mode split
    # exists to prevent, so argparse rejects it rather than silently preferring one.
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check", action="store_true",
        help="rebuild from the runs and compare against the committed fixture (needs the runs)",
    )
    mode.add_argument(
        "--validate-only", action="store_true",
        help="run validate() on the committed fixture; needs no runs, so CI can use it",
    )
    args = parser.parse_args(argv)

    # `--validate-only` is the mode CI can actually reach. It is separate from `--check` rather
    # than a fallback inside it, because "the runs were missing so I checked less" must not be
    # reported with the same exit code as "the fixture re-derives from the pinned runs".
    if args.validate_only:
        try:
            validate(json.loads(args.out.read_text(encoding="utf-8")))
        # ValueError, not json.JSONDecodeError: a non-UTF-8 file raises UnicodeDecodeError, which
        # is a ValueError sibling of JSONDecodeError rather than an OSError, and escaped as a
        # traceback with no FAIL line. ValueError subsumes both.
        except (BuildError, OSError, ValueError) as exc:
            print(f"FAIL  {exc}")
            return 1
        print(f"OK    {args.out} is internally coherent (not re-derived — that needs --check).")
        return 0

    for path, flag in ((args.recall_run, "--recall-run"), (args.mem0_run, "--mem0-run")):
        if not path.is_file():
            print(f"missing run artifact: {path}")
            print(f"benchmarks/results/ is gitignored, so pass {flag} explicitly, or use "
                  f"--validate-only, which needs no runs.")
            return 2

    # One handler over build AND the file I/O below. Widening only the --validate-only path left
    # the same defect open here: a non-UTF-8 --out raised UnicodeDecodeError out of --check, and
    # an unwritable path raised FileNotFoundError out of the write, both as tracebacks with no
    # FAIL line. ValueError subsumes UnicodeDecodeError and JSONDecodeError.
    try:
        fixture = build(
            load_rows(args.recall_run, RECALL_RUN_SHA256, "recall"),
            load_rows(args.mem0_run, MEM0_RUN_SHA256, "mem0"),
        )

        rendered = json.dumps(fixture, indent=2, ensure_ascii=False) + "\n"
        if args.check:
            if not args.out.is_file():
                print(f"FAIL  no committed fixture at {args.out}")
                return 1
            if args.out.read_text(encoding="utf-8") != rendered:
                print(f"FAIL  {args.out} differs from a fresh rebuild. Re-run without --check.")
                return 1
            print(f"OK    {args.out} matches a fresh rebuild from the pinned runs.")
            return 0

        args.out.write_text(rendered, encoding="utf-8")
    except (BuildError, OSError, ValueError) as exc:
        print(f"FAIL  {exc}")
        return 1

    summary = fixture["summary"]
    print(f"wrote {args.out}")
    print(f"  paired n={summary['paired_n']}  RE-call {summary['recall_mean']:.4f} "
          f"vs Mem0 {summary['mem0_mean']:.4f}")
    print(f"  badly-lost {summary['badly_lost']}  = {summary['confident_wrong']} confident-wrong "
          f"+ {summary['abstained_with_retrieval']} abstained + {summary['retrieval_empty']} empty")
    for mechanism, count in sorted(summary["mechanism_tally"].items(), key=lambda kv: -kv[1]):
        print(f"    {count}  {mechanism}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
