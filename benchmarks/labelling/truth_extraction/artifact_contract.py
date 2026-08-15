"""Validation for the truth-extraction artifacts, applied at the write site.

The census is the artifact every later recall number is read against. Its counts and its lists
are the same facts written twice — a summary a reader quotes and a body a reader recomputes from.
If they disagree, the artifact is not merely wrong, it is unfalsifiable: nothing in it says which
of the two is the typo. So the disagreement is refused at write time, when it costs nothing.

The same argument covers an arm result, plus one more that the census does not need: an arm result
must say what produced it. The first M1 run wrote a file reporting one proposal and 37 refusals,
and that file was indistinguishable from a careful arm. Every call had in fact failed on a
markdown fence, and the prompt has been rewritten twice since. A number whose prompt revision is
not recorded is attributable to nothing.

Pattern follows `benchmarks/artifact_contract.py`.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from datetime import datetime

_REQUIRED_PROVENANCE = ("peps_sha", "clone_date", "recall_commit", "generated_at", "invocation")

#: Every arm, whatever proposed it.
_REQUIRED_ARM_PROVENANCE = ("peps_sha", "recall_commit", "generated_at", "invocation",
                            "pack_digest")

#: A model arm only. Without these the score names no prompt, no model and no endpoint, and
#: rerunning it is guesswork.
_REQUIRED_MODEL_PROVENANCE = ("prompt_revision", "model_id", "model_revision", "engine_id",
                              "status_vocabulary")

#: Above this share of failed calls a result is an apparatus failure rather than a refusal rate.
#: Not zero, because some documents legitimately fail and that is worth reporting; not `all`,
#: which is what it was, and which let 29 of 30 failures publish a tier computed over the one
#: document that answered.
MAX_FAILED_CALL_FRACTION = 0.10

#: The pre-registration's decision rule, as a closed set. The runner computes the verdict rather
#: than a reader inferring one, so the rule is applied to the number instead of being reread
#: after seeing it.
ARM_VERDICTS = (
    "VACUOUS ARM",
    "UNDERPOWERED",
    # The tier is a property of the PAIR of arms: the underpowered clause reads "fewer than 10
    # proposals in EITHER arm". One artifact cannot answer it alone, and guessing from half the
    # inputs is how a rule gets quietly reinterpreted after seeing a number.
    "PENDING SIBLING ARM",
    # Above the tiers, not among them: its action is "run the four upper falsifier checks first",
    # which is a human step, and its conditions are a superset of high confidence's.
    "SUSPICIOUS",
    "LINT POINTER",
    "REVIEWING AID",
    "BATCH REVIEWABLE",
    # Reachable only after a human has run the four falsifier checks, never by the runner.
    "HIGH CONFIDENCE",
)


def validate_census(payload: Mapping[str, object]) -> None:
    """Raise `ValueError` unless `payload` is a self-consistent, attributable census."""
    provenance = payload.get("_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("census payload requires a _provenance block")
    for field in _REQUIRED_PROVENANCE:
        if not provenance.get(field):
            raise ValueError(f"census _provenance requires {field}")

    edges = payload.get("edges")
    if not isinstance(edges, Sequence) or isinstance(edges, (str, bytes)):
        raise ValueError("census edges must be an array")
    if payload.get("n_header_edges") != len(edges):
        raise ValueError(
            f"n_header_edges {payload.get('n_header_edges')!r} disagrees with "
            f"{len(edges)} entries in edges"
        )

    restatements = payload.get("restatements")
    if not isinstance(restatements, Mapping):
        raise ValueError("census restatements must be an object")
    if payload.get("n_restated_in_prose") != len(restatements):
        raise ValueError(
            f"n_restated_in_prose {payload.get('n_restated_in_prose')!r} disagrees with "
            f"{len(restatements)} entries in restatements"
        )

    # The recall ceiling is a proportion of the header edges. A value above 1.0 means the
    # restatement detector matched something that is not in the gold set at all.
    if len(restatements) > len(edges):
        raise ValueError(
            f"n_restated_in_prose ({len(restatements)}) cannot exceed n_header_edges "
            f"({len(edges)}) — the recall ceiling cannot exceed 100%"
        )


def validate_arm_result(payload: Mapping[str, object]) -> None:
    """Raise `ValueError` unless `payload` is a self-consistent, attributable arm result."""
    provenance = payload.get("_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("arm result requires a _provenance block")
    for field in _REQUIRED_ARM_PROVENANCE:
        if not provenance.get(field):
            raise ValueError(f"arm _provenance requires {field}")

    calls = payload.get("model_calls")
    if not isinstance(calls, int) or isinstance(calls, bool):
        raise ValueError("arm result requires an integer model_calls (0 for a rules arm)")
    if calls:
        for field in _REQUIRED_MODEL_PROVENANCE:
            if not provenance.get(field):
                raise ValueError(f"a model arm's _provenance requires {field}")

    items = payload.get("proposed_items")
    if not isinstance(items, Sequence) or isinstance(items, (str, bytes)):
        raise ValueError("arm result proposed_items must be an array")
    if payload.get("proposed") != len(items):
        raise ValueError(
            f"proposed {payload.get('proposed')!r} disagrees with {len(items)} proposed_items"
        )

    scored = payload.get("proposed_scored")
    excluded = payload.get("proposed_undecidable_excluded")
    if not isinstance(scored, int) or not isinstance(excluded, int):
        raise ValueError("proposed_scored and proposed_undecidable_excluded must be integers")
    if scored + excluded != len(items):
        raise ValueError(
            f"proposed_scored ({scored}) plus proposed_undecidable_excluded ({excluded}) must "
            f"account for all {len(items)} proposals"
        )

    tp, fp = payload.get("true_positive"), payload.get("false_positive")
    if not isinstance(tp, int) or not isinstance(fp, int):
        raise ValueError("true_positive and false_positive must be integers")
    if tp + fp != scored:
        raise ValueError(
            f"true_positive ({tp}) plus false_positive ({fp}) must equal proposed_scored ({scored})"
        )

    # `null`, not NaN: an arm that proposed nothing DECLINED to answer, and 0.0 would read as
    # "it was wrong". JSON has no NaN that a strict parser will accept, so the absence is encoded
    # as null and pinned in both directions — a null precision with a non-empty denominator is
    # just as wrong as a number with an empty one.
    precision = payload.get("precision")
    if scored == 0:
        if precision is not None:
            raise ValueError("precision must be null when nothing scored, not a number")
    else:
        if not isinstance(precision, (int, float)) or isinstance(precision, bool):
            raise ValueError(f"precision must be a number when {scored} proposals scored")
        if not math.isclose(float(precision), tp / scored, abs_tol=1e-9):
            raise ValueError(
                f"precision {precision!r} disagrees with true_positive/proposed_scored "
                f"({tp}/{scored} = {tp / scored})"
            )

    failures = payload.get("batch_failures")
    if not isinstance(failures, Mapping):
        raise ValueError("arm result batch_failures must be an object")
    if payload.get("documents_refused_whole") != len(failures):
        raise ValueError(
            f"documents_refused_whole {payload.get('documents_refused_whole')!r} disagrees with "
            f"{len(failures)} entries in batch_failures"
        )

    # The defect this whole validator exists for. An arm whose calls failed produces a file that
    # is arithmetically perfect and reads as a selective arm: few proposals, many refusals.
    #
    # ⚠️ `== calls` was one comparison too tight. It caught only the total wipeout, so 29 of 30
    # failures published, and 15 of 30 published a TIER computed over the half that answered.
    # A fraction is what makes it a gate rather than a tripwire.
    # The RULES arm's equivalent, which had no gate at all: `if calls` skips every check above
    # for an arm with `model_calls == 0`, so an R1 run that located none of the 38 sentences in
    # their source bodies published cleanly as a vacuous arm. That failure mode is not
    # hypothetical — `peps_header.paragraphs` exists because it hit 30 of 38.
    unread = payload.get("candidates_not_read")
    if not isinstance(unread, int) or isinstance(unread, bool) or unread < 0:
        raise ValueError("arm result requires a non-negative integer candidates_not_read")
    reasons = payload.get("refusal_reasons")
    if not isinstance(reasons, Mapping):
        raise ValueError("arm result refusal_reasons must be an object")
    scored_universe = unread + len(items) + sum(int(n) for n in reasons.values())
    if scored_universe and unread > MAX_FAILED_CALL_FRACTION * scored_universe:
        raise ValueError(
            f"{unread} of {scored_universe} candidates were never read by the arm, above the "
            f"{MAX_FAILED_CALL_FRACTION:.0%} ceiling: a candidate the arm could not look at is "
            f"an apparatus failure, and counting it as a refusal flatters precision"
        )

    if calls and len(failures) > MAX_FAILED_CALL_FRACTION * calls:
        raise ValueError(
            f"{len(failures)} of {calls} model calls failed at the batch rung, above the "
            f"{MAX_FAILED_CALL_FRACTION:.0%} ceiling: this is an apparatus failure, not a "
            f"refusal rate. Reasons: {sorted(set(map(str, failures.values())))}"
        )

    if payload.get("verdict") not in ARM_VERDICTS:
        raise ValueError(
            f"verdict {payload.get('verdict')!r} is not one of the pre-registered {ARM_VERDICTS}"
        )

    # The bounds the tier is KEYED ON. Unchecked, a payload could carry an inverted interval, or
    # none at all, and still be stamped with a tier: the two facts that decide the verdict were
    # the two nothing validated.
    bounds = (payload.get("precision_wilson_lower"), payload.get("precision_wilson_upper"))
    if scored == 0:
        if any(bound is not None for bound in bounds):
            raise ValueError("precision_wilson bounds must be null when nothing scored")
    else:
        if not all(isinstance(bound, (int, float)) and not isinstance(bound, bool)
                   for bound in bounds):
            raise ValueError("precision_wilson_lower and _upper must be numbers when scored > 0")
        lower, upper = float(bounds[0]), float(bounds[1])  # type: ignore[arg-type]
        if not lower <= float(precision) <= upper:  # type: ignore[arg-type]
            raise ValueError(
                f"precision {precision!r} is outside its own interval [{lower}, {upper}]"
            )


def validate_fixtures_result(payload: Mapping[str, object]) -> None:
    """Raise `ValueError` unless `payload` is a self-consistent P10 result.

    Separate from `validate_arm_result` because it is a different measurement: no adjudicated
    pack, no precision, no decision-rule tier. Sharing one validator would have meant loosening
    both, which is how a validator stops refusing anything.
    """
    provenance = payload.get("_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError("fixtures result requires a _provenance block")
    for field in _REQUIRED_ARM_PROVENANCE:
        if not provenance.get(field):
            raise ValueError(f"fixtures _provenance requires {field}")
    # Always a model arm: the fixtures exist to test a model against rules that already failed.
    for field in ("prompt_revision", "model_id", "model_revision", "engine_id"):
        if not provenance.get(field):
            raise ValueError(f"a fixtures run's _provenance requires {field}")

    total = payload.get("fixtures")
    refused, proposed = payload.get("refused"), payload.get("proposed")
    if not isinstance(total, int) or total <= 0:
        raise ValueError("fixtures result must count at least one fixture")
    if not isinstance(refused, int) or not isinstance(proposed, int):
        raise ValueError("refused and proposed must be integers")
    if refused + proposed != total:
        raise ValueError(
            f"refused ({refused}) plus proposed ({proposed}) must equal fixtures ({total})"
        )

    per_fixture = payload.get("per_fixture")
    if not isinstance(per_fixture, Mapping) or len(per_fixture) != total:
        raise ValueError(
            f"per_fixture must carry one entry per fixture ({total}), so a reader can see WHICH "
            f"refusal came from the language and which from a resolution rung"
        )

    if payload.get("p10_holds") != (refused == total):
        raise ValueError(
            f"p10_holds {payload.get('p10_holds')!r} disagrees with {refused} of {total} refused"
        )

    failures = payload.get("batch_failures")
    if not isinstance(failures, Mapping):
        raise ValueError("fixtures result batch_failures must be an object")
    # ANY failure, not all of them. P10 is the load-bearing public prediction and it has four
    # data points: a fixture the model never read cannot refuse, so it reads as a refusal and
    # inflates the score. `== total` let 3 of 4 failures publish `p10_holds: true`.
    if failures:
        raise ValueError(
            f"{len(failures)} of {total} fixtures failed at the batch rung: a fixture that was "
            f"never read cannot refuse, so an apparatus failure reads here as a perfect P10 "
            f"score. That is the one way this check can lie, and P10 decides shipping"
        )


#: The fields the pre-registration's `## Registration` block must carry. `registration_authored`
#: is the author date of the commit named by `registration_commit`, in UTC, and it is what I5 is
#: measured against. Author date rather than committer date because a rebase rewrites the second
#: and preserves the first, and this branch is expected to be rebased before it lands.
REGISTRATION_FIELDS = (
    "registration_commit",
    "registration_authored",
    "gold_manifest_digest",
    "gold_manifest_questions",
)

#: The `## Registration` section, bounded at the next heading of the same level. Bounded rather
#: than `.*?`-to-the-first-fence: unbounded, a section that had LOST its yaml block would reach
#: forward and silently adopt a fence from some later section as the registration.
_REGISTRATION_SECTION = re.compile(
    r"^##\s+Registration\s*$(?P<section>(?:(?!^##\s).)*)", re.MULTILINE | re.DOTALL
)

#: A fenced yaml block inside that section. Counted separately from the heading, because two
#: blocks under ONE heading is the shape an amended registration takes, and a single
#: heading-anchored match reports that as "found 1" and silently returns the stale first block.
_YAML_FENCE = re.compile(r"^```yaml\s*$(?P<body>.*?)^```\s*$", re.MULTILINE | re.DOTALL)

#: A fenced code block, stripped before prediction rows are scanned. The pre-registration already
#: carries fences (the provenance commands), and an illustrative table row inside one would
#: otherwise register as a duplicate prediction and take every caller down.
#:
#: The opening run is captured and the close must repeat it, because a fixed ``` would MIS-PAIR
#: on the standard markdown for showing a fenced block: a ```` outer fence containing a ```
#: inner one. Mis-paired, the match runs from the stray delimiter to the next fence anywhere in
#: the document and deletes every real prediction row between them. Tildes for the same reason.
_ANY_FENCE = re.compile(
    r"^(?P<fence>`{3,}|~{3,})[^\n]*\n.*?^(?P=fence)`*~*\s*$", re.MULTILINE | re.DOTALL
)

#: A prediction row in the pre-registration: `| P7 | targets naming a file ... | 0 | exactly 0 |`.
#: Anchored on the id sitting alone in the first cell, so the reasoning prose that mentions "P7"
#: in passing is not mistaken for a registration of it.
_PREDICTION_ROW = re.compile(r"^\|\s*\**(?P<id>[PO]\d+)\**\s*\|(?P<rest>.*)$", re.MULTILINE)

#: Per-field shape. Checked at parse time so a malformed registration is blamed on the
#: registration: unchecked, a quoted timestamp surfaces as "unparseable timestamp for
#: <artifact>", naming the artifact for a defect in this block, and a quoted digest surfaces as
#: "the gold manifest digest has moved", which is the worst possible false alarm for a check
#: whose entire job is to notice that the labels were regenerated.
_REGISTRATION_SHAPES = {
    "registration_commit": re.compile(r"^[0-9a-f]{40}$"),
    "registration_authored": re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:[+-]\d{2}:\d{2}|Z)$"),
    "gold_manifest_digest": re.compile(r"^[0-9a-f]{64}$"),
    "gold_manifest_questions": re.compile(r"^\d+$"),
}


def read_registration(text: str) -> dict[str, str]:
    """Parse the pre-registration's `## Registration` block, or raise `ValueError`.

    Deliberately not YAML-parsed. The block is four flat scalars, and adding a parser dependency
    to a validator that runs at every write site would buy nothing but a way for the block to
    grow structure nobody checks. The cost of that choice is that YAML spellings this does not
    implement (quoted scalars, trailing comments) must be REFUSED rather than half-read, which is
    what `_REGISTRATION_SHAPES` does.
    """
    sections = [m.group("section") for m in _REGISTRATION_SECTION.finditer(text)]
    if len(sections) != 1:
        raise ValueError(
            f"expected exactly one `## Registration` section, found {len(sections)}: I5 has no "
            f"anchor otherwise, and two means a reader cannot tell which one is registered"
        )
    bodies = [m.group("body") for m in _YAML_FENCE.finditer(sections[0])]
    if len(bodies) != 1:
        raise ValueError(
            f"expected exactly one yaml block under `## Registration`, found {len(bodies)}: an "
            f"amended registration must REPLACE the block, not sit beside the one it supersedes"
        )

    out: dict[str, str] = {}
    for line in bodies[0].splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        key, sep, value = line.partition(":")
        if not sep:
            raise ValueError(f"registration line is not `key: value`: {line!r}")
        key = key.strip()
        if key in out:
            raise ValueError(f"registration key {key!r} appears twice; which one is registered?")
        out[key] = value.strip()

    missing = [f for f in REGISTRATION_FIELDS if not out.get(f)]
    if missing:
        raise ValueError(f"registration block is missing {missing}")
    for field, shape in _REGISTRATION_SHAPES.items():
        if not shape.fullmatch(out[field]):
            raise ValueError(
                f"registration field {field}={out[field]!r} does not match {shape.pattern}. "
                f"Quoted scalars and trailing comments are refused rather than half-read, "
                f"because a half-read value fails later and blames the artifact"
            )
    return out


def read_prediction_ids(text: str) -> dict[str, str]:
    """Map every registered prediction id to the rest of its row, lowercased.

    The row text is what makes an id checkable. An artifact publishing `p10_holds` is only
    correct if the pre-registration's P10 is the fixtures prediction, and the row is the only
    place that says so.
    """
    ids: dict[str, str] = {}
    for match in _PREDICTION_ROW.finditer(_ANY_FENCE.sub("", text)):
        key = match.group("id")
        if key in ids:
            raise ValueError(f"{key} is registered twice; an id must name one prediction")
        ids[key] = match.group("rest").strip().lower()
    if not ids:
        raise ValueError("no prediction rows found; the pre-registration tables did not parse")
    return ids


def validate_gold_manifest_frozen(
    registration: Mapping[str, str], header: Mapping[str, object]
) -> None:
    """I5's other half: raise unless the gold labels are still the ones that were registered.

    The invariant is "labels frozen before arms", and an ordering check alone does not cover it.
    An arm generated after the pre-registration, against a gold set REGENERATED after it too,
    satisfies every timestamp and measures against labels the prediction never saw. The digest is
    what closes that, it is content rather than history, and so it survives the squash that makes
    the registration commit unreachable.
    """
    for field, key in (("gold_manifest_digest", "digest"), ("gold_manifest_questions", "n_questions")):
        registered = registration.get(field)
        if not registered:
            raise ValueError(f"I5 needs `{field}`; the mapping handed in is not a registration")
        actual = header.get(key)
        if str(actual) != str(registered):
            raise ValueError(
                f"I5 FAILED: the gold manifest's {key} is {actual!r}, but the pre-registration "
                f"froze {registered!r}. The labels have moved since the prediction was made, so "
                f"any precision measured against them is measured against a different instrument"
            )


def validate_registration_ordering(
    registration: Mapping[str, str], payload: Mapping[str, object], *, artifact: str
) -> None:
    """I5: raise unless `payload` was generated strictly after the pre-registration was authored.

    ⚠️ Scoped to ARM artifacts on purpose. `census.json` predates the pre-registration by three
    days and must: the census is the input the predictions were written against, and the commit
    subject says so. Holding it to this ordering would invert the experiment.

    Strictly after, not at-or-after. Equal timestamps mean the two were written in one action,
    which is the shape a backdated pre-registration takes.
    """
    provenance = payload.get("_provenance")
    if not isinstance(provenance, Mapping):
        raise ValueError(f"{artifact}: I5 needs a _provenance block to read `generated_at` from")
    raw = provenance.get("generated_at")
    if not isinstance(raw, str) or not raw:
        raise ValueError(f"{artifact}: I5 needs a `generated_at`, found {raw!r}")
    # `.get`, not `[...]`. This function is exported and is called from the runner's write site,
    # where an unhandled KeyError would crash the run instead of taking the refusal path.
    registered = registration.get("registration_authored")
    if not registered:
        raise ValueError(
            "I5 needs `registration_authored`; the mapping handed in is not a registration block"
        )
    try:
        generated = datetime.fromisoformat(raw)
    except ValueError as exc:
        raise ValueError(f"{artifact}: unparseable `generated_at` {raw!r}: {exc}") from exc
    try:
        authored = datetime.fromisoformat(registered)
    except ValueError as exc:
        # Named apart from the artifact's own timestamp on purpose: this one is a defect in the
        # pre-registration, and reporting it against the artifact sends the reader to the wrong file.
        raise ValueError(f"unparseable `registration_authored` {registered!r}: {exc}") from exc
    if generated.tzinfo is None or authored.tzinfo is None:
        raise ValueError(
            f"{artifact}: I5 timestamps must carry an offset; a naive one compares two clocks"
        )
    if generated <= authored:
        raise ValueError(
            f"I5 FAILED for {artifact}: generated {generated.isoformat()} at or before the "
            f"pre-registration was authored ({authored.isoformat()}). Either the result predates "
            f"its own prediction, or the pre-registration was written after the result existed"
        )


__all__ = [
    "ARM_VERDICTS",
    "REGISTRATION_FIELDS",
    "read_prediction_ids",
    "read_registration",
    "validate_arm_result",
    "validate_census",
    "validate_fixtures_result",
    "validate_gold_manifest_frozen",
    "validate_registration_ordering",
]
