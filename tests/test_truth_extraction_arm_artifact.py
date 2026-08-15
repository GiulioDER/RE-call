"""The arm-result boundary: what an extraction score must say before it may be written.

The census validator exists because an artifact whose summary and body disagree is
unfalsifiable. An arm result needs that and one thing more: it must say what produced it. The
first M1 run wrote a file reporting 1 proposal and 37 refusals, which is indistinguishable from a
careful arm; every call had failed on a markdown fence. The prompt has been rewritten twice since,
so a score with no prompt revision in it is attributable to nothing.

Properties, one test each:
  1. A payload missing `_provenance` is refused.
  2. Each required provenance field is individually required.
  3. A MODEL arm additionally requires prompt revision, model, engine and status vocabulary.
  4. A RULES arm is NOT held to those, because it has no prompt and no model.
  5. `proposed` disagreeing with `proposed_items` is refused.
  6. Scored plus undecidable must account for every proposal.
  7. True plus false positives must equal the scored proposals.
  8. `precision` must recompute from the counts.
  9. `precision` must be null exactly when nothing scored, in both directions.
  10. `documents_refused_whole` must agree with `batch_failures`.
  11. An arm whose EVERY call failed at the batch rung is refused as an apparatus failure.
  12. An arm where only SOME calls failed is accepted; the guard must not refuse a partial failure.
  13. The verdict must be one of the pre-registered tiers.
  14. A well-formed payload is not refused.
  15. The write site calls the validator, and a refused payload never lands on disk.
  16. The decision rule maps each region to the tier the pre-registration fixed.
  17. The fixtures (P10) result has its own validator, and it refuses the one shape that would let
      an apparatus failure read as a perfect score.
  18. An extractor identity that changes mid-run is refused rather than recorded as one identity.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.labelling.truth_extraction.artifact_contract import (
    ARM_VERDICTS,
    validate_arm_result,
    validate_fixtures_result,
)
from benchmarks.labelling.truth_extraction import run_arms
from benchmarks.labelling.truth_extraction.run_arms import (
    ArmResult,
    _emit,
    _record_identity,
    preregistered_verdict,
    score,
)


def _ok() -> dict:
    """A model arm: 12 proposals, 9 of them correct, one row the adjudicator left blank."""
    return {
        "arm": "M1-model",
        "proposed": 13,
        "proposed_scored": 12,
        "proposed_undecidable_excluded": 1,
        "true_positive": 9,
        "false_positive": 3,
        "precision": 0.75,
        "precision_wilson_lower": 0.4681,
        "precision_wilson_upper": 0.9112,
        "referral_rate": None,
        "recall_vs_adjudicated_positives": 0.9,
        "adjudicated_positives": 10,
        "refusal_reasons": {"no supersession claim naming this target": 25},
        "candidates_not_read": 0,
        "not_read_reasons": {},
        "sibling_proposals": 14,
        "verdict": "REVIEWING AID",
        "proposed_items": [str(i) for i in range(1, 14)],
        "corpus_files": 733,
        "model_calls": 30,
        "cache_hits": 0,
        "batch_failures": {},
        "documents_refused_whole": 0,
        "_provenance": {
            "peps_sha": "5981b2a292610104eb30735423504c52fe454650",
            "recall_commit": "1bf1d36",
            "recall_tree_dirty": False,
            "generated_at": "2026-08-15T10:00:00+00:00",
            "invocation": "python -m benchmarks.labelling.truth_extraction.run_arms --arm model",
            "pack_digest": "b" * 64,
            "prompt_revision": "truth-extraction-prompt-v2",
            "model_id": "a-model",
            "model_revision": "2026-08-01",
            "engine_id": "openai@openrouter.ai",
            "status_vocabulary": "final,rejected",
        },
    }


def _rules() -> dict:
    """The same shape from a deterministic arm: no model, so no model identity to require."""
    payload = _ok()
    payload["arm"] = "R1-rules"
    payload["model_calls"] = 0
    for field in ("prompt_revision", "model_id", "model_revision", "engine_id",
                  "status_vocabulary"):
        del payload["_provenance"][field]
    return payload


def _fixtures_ok() -> dict:
    return {
        "arm": "M1-fixtures",
        "fixtures": 4,
        "refused": 4,
        "proposed": 0,
        "proposed_items": [],
        "per_fixture": {
            "hedged": {"proposed": [], "rejection_rungs": []},
            "partial_scope_claim": {"proposed": [], "rejection_rungs": []},
            "partial_scope_scope": {"proposed": [], "rejection_rungs": []},
            "reported_speech": {"proposed": [], "rejection_rungs": []},
        },
        "p10_prediction": "4 of 4 refused",
        "p10_holds": True,
        "model_calls": 4,
        "cache_hits": 0,
        "batch_failures": {},
        "_provenance": _ok()["_provenance"],
    }


def test_the_scorer_reads_a_bom_the_way_the_packs_own_tests_do(tmp_path: Path):
    """The reader that consumes the labels for the published number was the one still exposed.

    A spreadsheet's "CSV UTF-8" writes a BOM. The pack's own tests moved to `utf-8-sig`; this
    scorer, one directory over, kept plain `utf-8`, so the same file the tests declare healthy
    would have raised `KeyError: 'item'` out of the arm run. Nothing measured that, because the
    committed pack carries no BOM: the corruption has to be applied.
    """
    from benchmarks.labelling.truth_extraction.run_arms import CSV_PATH, KEY_PATH, load_pack

    bom = b"\xef\xbb\xbf"
    csv_copy, key_copy = tmp_path / "adjudication.csv", tmp_path / "adjudication_key.json"
    csv_copy.write_bytes(bom + CSV_PATH.read_bytes())
    key_copy.write_bytes(bom + KEY_PATH.read_bytes())

    pack = load_pack(csv_copy, key_copy)
    assert len(pack) == len(load_pack()), "a BOM'd pack lost rows"
    assert pack[0].item == "1"
    assert pack[0].source_pep.startswith("pep-")


def test_missing_provenance_is_refused():
    payload = _ok()
    del payload["_provenance"]
    with pytest.raises(ValueError, match="_provenance"):
        validate_arm_result(payload)


@pytest.mark.parametrize(
    "field", ["peps_sha", "recall_commit", "generated_at", "invocation", "pack_digest"]
)
def test_each_required_provenance_field_is_required(field: str):
    payload = _ok()
    del payload["_provenance"][field]
    with pytest.raises(ValueError, match=field):
        validate_arm_result(payload)


@pytest.mark.parametrize(
    "field", ["prompt_revision", "model_id", "model_revision", "engine_id", "status_vocabulary"]
)
def test_a_model_arm_must_name_what_produced_it(field: str):
    payload = _ok()
    del payload["_provenance"][field]
    with pytest.raises(ValueError, match=field):
        validate_arm_result(payload)


def test_a_rules_arm_is_not_held_to_the_model_fields():
    """Over-rejection guard. A deterministic arm has no prompt to record and no model to pin."""
    validate_arm_result(_rules())  # must not raise


def test_proposed_disagreeing_with_the_item_list_is_refused():
    payload = _ok()
    payload["proposed"] = 12
    with pytest.raises(ValueError, match="proposed"):
        validate_arm_result(payload)


def test_scored_and_undecidable_must_account_for_every_proposal():
    payload = _ok()
    payload["proposed_undecidable_excluded"] = 0
    with pytest.raises(ValueError, match="proposed_undecidable_excluded"):
        validate_arm_result(payload)


def test_true_and_false_positives_must_equal_the_scored_proposals():
    payload = _ok()
    payload["false_positive"] = 2
    with pytest.raises(ValueError, match="false_positive"):
        validate_arm_result(payload)


def test_precision_must_recompute_from_the_counts():
    payload = _ok()
    payload["precision"] = 0.9
    with pytest.raises(ValueError, match="precision"):
        validate_arm_result(payload)


def test_a_null_precision_with_a_non_empty_denominator_is_refused():
    payload = _ok()
    payload["precision"] = None
    with pytest.raises(ValueError, match="precision"):
        validate_arm_result(payload)


def test_a_numeric_precision_with_an_empty_denominator_is_refused():
    """The other direction. An arm that scored nothing DECLINED; 0.0 would read as "it was wrong"."""
    payload = _ok()
    payload.update(
        proposed=0, proposed_items=[], proposed_scored=0, proposed_undecidable_excluded=0,
        true_positive=0, false_positive=0, precision=0.0, verdict="VACUOUS ARM",
        precision_wilson_lower=None, precision_wilson_upper=None,
    )
    with pytest.raises(ValueError, match="precision must be null"):
        validate_arm_result(payload)
    payload["precision"] = None
    validate_arm_result(payload)  # and with null it is accepted


def test_refused_document_count_must_agree_with_the_failure_list():
    payload = _ok()
    payload["documents_refused_whole"] = 2
    with pytest.raises(ValueError, match="documents_refused_whole"):
        validate_arm_result(payload)


@pytest.mark.parametrize("failed", [30, 29, 15, 4])
def test_an_arm_whose_calls_mostly_failed_is_refused_as_apparatus_failure(failed: int):
    """The defect this validator was written for, and the ceiling it needed to be a gate.

    30 calls, 30 batch rejections, 0 proposals, 38 refusals: arithmetically perfect and it reads
    as a selective arm. It happened, on a markdown fence, and the file was nearly published.

    ⚠️ `== calls` caught ONLY the total wipeout. 29 of 30 published; 15 of 30 published a TIER
    computed over the half that answered. A run is not sound because one document survived it.
    """
    payload = _ok()
    payload["model_calls"] = 30
    payload["batch_failures"] = {f"pep-{i:04d}": "reply was not JSON" for i in range(failed)}
    payload["documents_refused_whole"] = failed
    with pytest.raises(ValueError, match="apparatus failure"):
        validate_arm_result(payload)


def test_a_partial_batch_failure_within_the_ceiling_is_accepted():
    """Over-rejection guard: some files legitimately fail, and that is a number worth reporting."""
    payload = _ok()
    payload["model_calls"] = 30
    payload["batch_failures"] = {"pep-0001": "reply was not JSON"}
    payload["documents_refused_whole"] = 1
    validate_arm_result(payload)  # 1 of 30 is under the ceiling; must not raise


def test_an_arm_that_could_not_READ_its_candidates_is_refused():
    """The RULES arm's apparatus failure, which had no gate at all.

    The whole-run check is guarded by `if calls`, and a rules arm always has `model_calls == 0`,
    so an R1 run that located none of the 38 sentences in their source bodies published cleanly
    as a vacuous arm. That failure mode is not hypothetical: `peps_header.paragraphs` exists
    because it hit 30 of 38.
    """
    payload = _ok()
    payload["model_calls"] = 0
    for field in ("prompt_revision", "model_id", "model_revision", "engine_id",
                  "status_vocabulary"):
        del payload["_provenance"][field]
    payload.update(
        proposed=0, proposed_items=[], proposed_scored=0, proposed_undecidable_excluded=0,
        true_positive=0, false_positive=0, precision=None, precision_wilson_lower=None,
        precision_wilson_upper=None, verdict="VACUOUS ARM",
        refusal_reasons={}, candidates_not_read=38,
        not_read_reasons={"sentence not located in body": 38},
    )
    with pytest.raises(ValueError, match="never read"):
        validate_arm_result(payload)


def test_the_wilson_bounds_the_tier_is_keyed_on_are_validated():
    """They decide the verdict, and nothing checked them: a payload could carry an inverted
    interval, or none at all, and still be stamped with a tier."""
    payload = _ok()
    del payload["precision_wilson_lower"]
    with pytest.raises(ValueError, match="must be numbers"):
        validate_arm_result(payload)

    payload = _ok()
    payload["precision_wilson_lower"], payload["precision_wilson_upper"] = 0.99, 0.10
    with pytest.raises(ValueError, match="outside its own interval"):
        validate_arm_result(payload)


def test_the_verdict_must_be_a_pre_registered_tier():
    payload = _ok()
    payload["verdict"] = "looks good"
    with pytest.raises(ValueError, match="pre-registered"):
        validate_arm_result(payload)


def test_a_well_formed_arm_result_is_accepted():
    validate_arm_result(_ok())  # must not raise


def test_the_write_site_validates_before_it_writes(tmp_path: Path, monkeypatch):
    """And a refused payload must not be on disk afterwards, which is why validation is first.

    `_emit` also enforces I5 against the real pre-registration, which would silently couple this
    module's hardcoded `generated_at` to a date in `results/`: amending `registration_authored`
    to any instant after 2026-08-15T10:00:00Z would turn a test about validator ORDERING red
    with an I5 message naming a pytest temp path, and nothing here would point a reader at the
    cause. So the registration is stubbed to a fixed instant this module owns. The real wiring is
    covered where it belongs, in `tests/test_prereg_authority.py`.
    """
    monkeypatch.setattr(
        run_arms, "registration_block", lambda: {"registration_authored": "2020-01-01T00:00:00+00:00"}
    )
    payload = _ok()
    payload["verdict"] = "looks good"
    out = tmp_path / "arm.json"
    with pytest.raises(ValueError, match="pre-registered"):
        _emit(payload, out)
    assert not out.exists(), "a refused payload was written anyway"

    _emit(_ok(), out)
    assert json.loads(out.read_text(encoding="utf-8"))["arm"] == "M1-model"
    assert b"\r\n" not in out.read_bytes(), "this repo is eol=lf"


@pytest.mark.parametrize(
    ("scored", "precision", "lower", "upper", "expected"),
    [
        # proposed 0 is handled separately below; these all propose. `sibling` is 14 throughout
        # except where the row is about the cross-arm clause.
        (9, 1.0, 0.70, 1.00, "UNDERPOWERED"),           # n < 10 beats every precision
        (10, 0.90, 0.60, 0.98, "REVIEWING AID"),        # BOUNDARY: n exactly 10 is enough
        (12, 0.50, 0.24, 0.76, "REVIEWING AID"),        # BOUNDARY: exactly 0.50 is not "below"
        (12, 0.49, 0.24, 0.74, "LINT POINTER"),         # below 0.50: a net cost to a reviewer
        (12, 0.75, 0.47, 0.91, "REVIEWING AID"),        # 0.50 to 0.80
        (12, 0.80, 0.72, 0.95, "BATCH REVIEWABLE"),     # BOUNDARY: exactly 0.80 leaves the band
        (12, 0.85, 0.72, 0.95, "BATCH REVIEWABLE"),     # WITH the Wilson gate met
        (12, 0.85, 0.70, 0.95, "BATCH REVIEWABLE"),     # BOUNDARY: gate exactly 0.70
        (12, 0.85, 0.6999, 0.95, "REVIEWING AID"),      # just under the gate -> tier below
        (12, 0.85, 0.60, 0.95, "REVIEWING AID"),        # gate missed -> tier below
        (24, 0.96, 0.92, 0.99, "SUSPICIOUS"),           # >= 0.95 and n >= 20: too good to tier
        (20, 0.95, 0.90, 0.99, "SUSPICIOUS"),           # BOUNDARY: exactly 0.95 and exactly 20
        (19, 0.96, 0.92, 0.99, "BATCH REVIEWABLE"),     # BOUNDARY: n = 19 is not suspicious
        (12, 0.96, 0.92, 0.99, "BATCH REVIEWABLE"),     # >= 0.95 but n < 20 -> tier below
        # BOUNDARY: half-width exactly 0.30 is NOT above 0.30. Written as bounds a Wilson
        # interval can actually take (both inside [0, 1]), and the rule carries a float
        # tolerance because 0.30 has no exact binary representation.
        (12, 0.75, 0.40, 1.00, "REVIEWING AID"),
        (12, 0.75, 0.38, 1.00, "UNDERPOWERED"),         # half-width 0.31, above
    ],
)
def test_the_decision_rule_maps_each_region_to_its_tier(scored, precision, lower, upper, expected):
    """The rule as code, so it is applied to the number rather than reread after seeing it.

    ⚠️ Every row sits ON a boundary or one step off it. The first version placed every row
    strictly inside its band, so flipping ANY comparison in the rule left the suite green: eight
    separate operator mutations survived, including `< 10` to `<= 10` and both Wilson gates.

    The "tier below" rows are the reading stated in `preregistered_verdict`'s docstring: the
    pre-registration gives each tier a point band AND a Wilson gate and is silent on meeting one
    without the other. `SUSPICIOUS` is the pre-registration's own row, and it swallows
    `HIGH CONFIDENCE` because its conditions are a superset and its action is a human step.
    """
    report = {
        "proposed": scored,
        "proposed_scored": scored,
        "precision": precision,
        "precision_wilson_lower": lower,
        "precision_wilson_upper": upper,
    }
    assert preregistered_verdict(report, sibling_proposals=14) == expected
    assert expected in ARM_VERDICTS


def test_a_tier_is_not_chosen_from_one_arm_alone():
    """The underpowered clause is CROSS-arm: "fewer than 10 proposals in EITHER arm".

    Reading it per-arm is not the rule, and the difference is live: R1 proposed 9, so under the
    real clause M1 cannot be given a tier however it scores. With the sibling unknown the honest
    answer is that the tier is pending, not a tier chosen from half the inputs.
    """
    report = {
        "proposed": 24, "proposed_scored": 24, "precision": 0.85,
        "precision_wilson_lower": 0.72, "precision_wilson_upper": 0.95,
    }
    assert preregistered_verdict(report, sibling_proposals=None) == "PENDING SIBLING ARM"
    assert preregistered_verdict(report, sibling_proposals=9) == "UNDERPOWERED"
    assert preregistered_verdict(report, sibling_proposals=10) == "BATCH REVIEWABLE"


def test_the_runner_can_never_stamp_high_confidence_by_itself():
    """`HIGH CONFIDENCE` requires four upper-falsifier checks that are a human step.

    Its conditions are a strict subset of `SUSPICIOUS`'s, so any result that would reach it is
    caught above. The tier stays in the vocabulary because a human may reach it after running
    those checks; the runner may not.
    """
    reachable = {
        preregistered_verdict(
            {
                "proposed": scored, "proposed_scored": scored, "precision": precision,
                "precision_wilson_lower": lower, "precision_wilson_upper": min(1.0, lower + 0.2),
            },
            sibling_proposals=20,
        )
        for scored in (10, 19, 20, 24, 40)
        for precision in (0.0, 0.49, 0.5, 0.79, 0.8, 0.94, 0.95, 0.99, 1.0)
        for lower in (0.0, 0.69, 0.7, 0.89, 0.9, 0.99)
    }
    assert "HIGH CONFIDENCE" not in reachable, "the runner tiered a result it must first doubt"
    assert "SUSPICIOUS" in reachable, "nothing reached the row that stands in its place"


def test_an_arm_that_proposed_nothing_is_vacuous_not_zero():
    report = {"proposed": 0, "proposed_scored": 0, "precision": None,
              "precision_wilson_lower": None, "precision_wilson_upper": None}
    assert preregistered_verdict(report, sibling_proposals=14) == "VACUOUS ARM"


def test_a_wide_interval_is_underpowered_even_at_ten_proposals():
    report = {"proposed": 10, "proposed_scored": 10, "precision": 0.5,
              "precision_wilson_lower": 0.24, "precision_wilson_upper": 0.96}
    assert preregistered_verdict(report, sibling_proposals=14) == "UNDERPOWERED"


# ---------------------------------------------------------------------------------------------
# The MEASUREMENT itself. Every test above this line is about the artifact's shape; none of them
# touched `score`, and an audit found that 21 of 24 mutations of the scoring code survived the
# suite — including "count blank verdicts against the arm" and "divide by all proposals".
# ---------------------------------------------------------------------------------------------


def _pack(verdicts: str) -> list:
    """One candidate per character of `verdicts`, each with a distinct sentence and target."""
    from benchmarks.labelling.truth_extraction.run_arms import Candidate

    return [
        Candidate(
            item=str(i), sentence=f"sentence {i}", target=f"pep-{i:04d}",
            source_pep="pep-9999", verdict="" if v == "-" else v,
        )
        for i, v in enumerate(verdicts, 1)
    ]


def test_a_blank_verdict_leaves_the_denominator_rather_than_counting_against_the_arm():
    """Blank means the adjudicator could not tell, which is not evidence the arm was wrong.

    Counting it as a false positive would make an honest "undecidable" indistinguishable from a
    mistake, and the pack has one deliberate blank precisely so that distinction exists.
    """
    from benchmarks.labelling.truth_extraction.run_arms import ArmResult, score

    pack = _pack("YYNN--")
    result = ArmResult(arm="t", proposed=[c.item for c in pack])
    report = score(result, pack)

    assert report["proposed"] == 6
    assert report["proposed_scored"] == 4, "the blanks stayed in the denominator"
    assert report["proposed_undecidable_excluded"] == 2
    assert report["true_positive"] == 2 and report["false_positive"] == 2
    assert report["precision"] == 0.5, "2 of 4 decided, not 2 of 6"


def test_precision_is_null_and_the_arm_vacuous_when_it_proposed_nothing():
    from benchmarks.labelling.truth_extraction.run_arms import ArmResult, score

    report = score(ArmResult(arm="t"), _pack("YYNN"))
    assert report["precision"] is None, "0.0 would read as 'it was wrong' rather than 'it declined'"
    assert report["proposed_scored"] == 0


def test_recall_divides_by_every_adjudicated_positive_not_by_the_pack():
    """The denominator its name claims. Dividing by the whole pack would make an arm that found
    every true edge report 0.4, and the number is published beside precision."""
    from benchmarks.labelling.truth_extraction.run_arms import ArmResult, score

    pack = _pack("YYNNN")
    result = ArmResult(arm="t", proposed=["1", "2"])
    report = score(result, pack)

    assert report["adjudicated_positives"] == 2
    assert report["recall_vs_adjudicated_positives"] == 1.0


@pytest.mark.parametrize(
    ("sentence", "expected_rung"),
    [
        # The hedge form the pre-registration names as the residual class, in PEP citation
        # style. `recall.fix` refuses it; the arm proposed it, because `_is_hedged` was handed
        # the 40 characters AFTER the match instead of the text between marker and reference.
        ("This PEP supersedes/augments :pep:`324`.", "hedged"),
        # And the inverse: production ACCEPTS this and the arm refused it as hedged, because
        # "or something" fell inside the window it was wrongly given.
        ("This PEP supersedes :pep:`324` or something.", None),
        # Partial scope, `the <noun> in X`, was handed a span that includes the reference text,
        # which production deliberately excludes. The phrasing is `recall/fix.py`'s own measured
        # example ("Supersedes the scope in [[...]]"), not one invented here: my first guess,
        # "the scope of", does not match the rule at all, and asserting it would have pinned a
        # verdict production does not reach.
        ("This PEP supersedes the scope in :pep:`324`.", "partial scope"),
        ("This PEP supersedes :pep:`324`.", None),
    ],
)
def test_the_rules_arm_hands_the_imported_rules_productions_own_spans(
    tmp_path, sentence: str, expected_rung: str | None
):
    """Importing the RULES unmodified and then feeding them different text is a different arm.

    Each row is decided by `recall.fix`'s own rule, and the assertion is that this arm agrees
    with `recall.fix._accept` on the same sentence rather than that it reaches some particular
    verdict — so the test cannot drift from production by restating what production does.
    """
    from recall.fix import _ACTIVE_RE as _PROD_ACTIVE
    from recall.fix import _accept

    from benchmarks.labelling.truth_extraction.run_arms import Candidate, rules_arm

    source = tmp_path / "pep-9999.rst"
    source.write_text(f"PEP: 9999\nTitle: T\n\n\n{sentence}\n", encoding="utf-8")
    pack = [Candidate(item="1", sentence=sentence, target="pep-0324",
                      source_pep="pep-9999", verdict="N")]

    result = rules_arm(pack, tmp_path, {"pep-0324", "pep-9999"})
    arm_refused = result.refused.get("1")

    # Production's verdict on the same prose, in ITS reference style, via its own entry point.
    production = sentence.replace(":pep:`324`", "pep-0324.md")
    match = next(iter(_PROD_ACTIVE.finditer(production)), None)
    assert match is not None, "the production pattern must match, or the comparison is empty"
    production_accepts = _accept(production, match) is not None

    assert (arm_refused is None) == production_accepts, (
        f"the arm and recall.fix disagree on {sentence!r}: arm refused {arm_refused!r}, "
        f"production {'accepts' if production_accepts else 'refuses'}"
    )
    if expected_rung is not None:
        assert arm_refused == expected_rung, f"refused for the wrong reason: {arm_refused!r}"
    assert not result.not_read, "the sentence must be located, or this proves nothing"


def test_a_candidate_the_arm_could_not_read_is_not_counted_as_a_refusal(tmp_path):
    """The arm did not decline it, it never saw it, and the two must not share a bucket.

    Lumped together, an R1 run that located none of the 38 sentences published as a careful arm
    with 38 refusals and a clean precision of nothing. `peps_header.paragraphs` exists because
    that failure mode hit 30 of 38.
    """
    from benchmarks.labelling.truth_extraction.run_arms import Candidate, rules_arm

    (tmp_path / "pep-9999.rst").write_text(
        "PEP: 9999\nTitle: T\n\n\nSomething else entirely.\n", encoding="utf-8"
    )
    pack = [Candidate(item="1", sentence="This PEP supersedes :pep:`324`.", target="pep-0324",
                      source_pep="pep-9999", verdict="N")]

    result = rules_arm(pack, tmp_path, {"pep-0324", "pep-9999"})
    assert result.not_read == {"1": "sentence not located in body"}
    assert not result.refused, "a candidate that was never read was counted as a refusal"
    assert score(result, pack)["candidates_not_read"] == 1


def test_a_document_refused_whole_does_not_yield_semantic_refusals(tmp_path):
    """A batch rung takes the document, so the model never read those candidates.

    Counted as ordinary refusals they are indistinguishable from candidates the model read and
    declined, which is how half a broken run publishes a tier computed over the surviving half.
    """
    from recall.truth_extraction.types import STATUS_VOCABULARY

    from benchmarks.labelling.truth_extraction.run_arms import Candidate, model_arm, score

    (tmp_path / "pep-9999.rst").write_text(
        "PEP: 9999\nTitle: T\n\n\nThis PEP supersedes :pep:`324`.\n", encoding="utf-8"
    )

    class _Broken:
        engine_id, model_id, revision = "e", "m", "r"

        def run(self, prompt):
            return "this is not JSON"

    pack = [Candidate(item="1", sentence="x", target="pep-0324",
                      source_pep="pep-9999", verdict="N")]
    result = model_arm(pack, tmp_path, {"pep-0324", "pep-9999"}, _Broken(), None,
                       STATUS_VOCABULARY)

    assert result.batch_failures, "the fixture must produce a batch rejection"
    assert not result.refused, "a candidate from a refused document was scored as a refusal"
    assert list(result.not_read) == ["1"]
    assert score(result, pack)["candidates_not_read"] == 1


def test_a_corpus_that_is_not_the_census_corpus_is_refused(monkeypatch, tmp_path):
    """Invariant I9. Nothing checked it, so an artifact could record 52 files against a census
    of 733 and validate: a throwaway git repo holding copied PEP files was accepted, with its
    own HEAD recorded as the corpus version."""
    import benchmarks.labelling.truth_extraction.run_arms as arms

    monkeypatch.setattr(arms, "_git", lambda repo, *a: "" if a[0] == "status" else "a" * 40)
    with pytest.raises(SystemExit, match="not the corpus"):
        arms.build_provenance(tmp_path, "inv", ArmResult(arm="t"), corpus_files=52)


def test_a_corpus_with_uncommitted_changes_is_refused(monkeypatch, tmp_path):
    """As fatal as an unknown SHA, and for the identical reason: the recorded commit would name
    a corpus that is not the one the arm read. Nine uncommitted edits moved precision from
    0.375 to 0.4286 under a byte-identical provenance block."""
    import benchmarks.labelling.truth_extraction.run_arms as arms

    monkeypatch.setattr(
        arms, "_git",
        lambda repo, *a: " M peps/pep-0376.rst" if a[0] == "status" else "a" * 40,
    )
    with pytest.raises(SystemExit, match="uncommitted changes"):
        arms.build_provenance(tmp_path, "inv", ArmResult(arm="t"), corpus_files=733)


def test_any_fixture_failing_refuses_the_p10_result():
    """P10 has four data points and decides shipping.

    A fixture the model never read cannot refuse, so it reads as a refusal and inflates the
    score. `== total` let 3 of 4 failures publish `p10_holds: true`.
    """
    payload = _fixtures_ok()
    payload["batch_failures"] = {"hedged": "reply was not JSON"}
    with pytest.raises(ValueError, match="never read cannot refuse"):
        validate_fixtures_result(payload)


def test_the_referral_rate_is_reported_as_unmeasured_not_as_zero():
    """P6 predicts 0.15 [0.05, 0.40]. Nothing populates `referred`: no extraction path emits a
    review-required status, so 0.0 was structural. Publishing it would have scored a registered
    prediction as falsified using a measurement that was never taken."""
    from benchmarks.labelling.truth_extraction.run_arms import ArmResult, score

    report = score(ArmResult(arm="t", proposed=["1"]), _pack("YN"))
    assert report["referral_rate"] is None
    assert "NOT MEASURED" in report["referral_rate_note"]


def test_a_well_formed_fixtures_result_is_accepted():
    validate_fixtures_result(_fixtures_ok())  # must not raise


def test_the_fixtures_counts_must_add_up():
    payload = _fixtures_ok()
    payload["refused"] = 3
    with pytest.raises(ValueError, match="refused"):
        validate_fixtures_result(payload)


def test_p10_holds_must_agree_with_the_counts():
    payload = _fixtures_ok()
    payload["refused"], payload["proposed"] = 3, 1
    payload["p10_holds"] = True
    with pytest.raises(ValueError, match="p10_holds"):
        validate_fixtures_result(payload)


def test_a_fixtures_run_where_every_call_failed_is_refused():
    """The one way a refusal check can lie: nothing ran, so nothing proposed, so P10 "held"."""
    payload = _fixtures_ok()
    payload["batch_failures"] = {name: "reply was not JSON" for name in payload["per_fixture"]}
    with pytest.raises(ValueError, match="apparatus failure"):
        validate_fixtures_result(payload)


def test_a_fixtures_run_must_report_every_fixture_individually():
    payload = _fixtures_ok()
    payload["per_fixture"].pop("hedged")
    with pytest.raises(ValueError, match="per_fixture"):
        validate_fixtures_result(payload)


def test_a_fixtures_run_must_name_the_prompt_that_produced_it():
    payload = _fixtures_ok()
    payload["_provenance"] = dict(payload["_provenance"])
    del payload["_provenance"]["prompt_revision"]
    with pytest.raises(ValueError, match="prompt_revision"):
        validate_fixtures_result(payload)


def _extraction(**over):
    """A REAL `FileExtraction`, not a stub with the four fields this test happens to read.

    The stub version declared `prompt_revision`, `model_id`, `revision` and `engine_id` by hand,
    so renaming any of them on the dataclass would have broken `_record_identity` in production
    and left this test green.
    """
    from recall.truth_extraction.types import FileExtraction

    fields = dict(
        file="pep-0001.rst", claims=(), rejections=(), engine_id="openai@openrouter.ai",
        model_id="a-model", revision="2026-08-01",
        prompt_revision="truth-extraction-prompt-v2", status_vocabulary=("final", "rejected"),
    )
    return FileExtraction(**{**fields, **over})


@pytest.mark.parametrize(
    "field",
    ["prompt_revision", "model_id", "revision", "engine_id", "status_vocabulary"],
)
def test_an_identity_that_changes_mid_run_stops_the_run(field: str):
    """Otherwise the artifact names whichever extractor answered last.

    Parametrised over EVERY pinned field: the single-field version left three of them unpinned,
    and replacing each with the empty string was a surviving mutant. `status_vocabulary` is here
    because `FileExtraction` documents it as part of the audit identity for the same reason
    `prompt_revision` is: it is rendered into the prompt.
    """
    changed = {"status_vocabulary": ("different",)} if field == "status_vocabulary" else {
        field: "changed"
    }
    result = ArmResult(arm="M1-model")
    _record_identity(result, _extraction())
    _record_identity(result, _extraction())  # same: fine
    assert result.identity["prompt_revision"] == "truth-extraction-prompt-v2"
    with pytest.raises(SystemExit, match="identity changed"):
        _record_identity(result, _extraction(**changed))
