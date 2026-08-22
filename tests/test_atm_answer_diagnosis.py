"""The ATM answer-side decomposition, tested on hand-built records.

`benchmarks/atm_answer_diagnosis.py` needs three things this suite does not have: the ATM-Bench
corpus, an evaluator checkout, and an archived run package. That is exactly why the aggregation is
split out from the loading -- everything below operates on `QuestionRecord`s built here, so the
arithmetic that produces a published number is checked on every run rather than only when someone
happens to have the dataset on disk.

Each test names the defect it exists to catch. Mutation-tested: each assertion was watched to go
red with the corresponding line of the source broken on purpose.
"""
from __future__ import annotations

import json

import pytest

from benchmarks.atm_answer_diagnosis import (
    MODALITY_FLOOR_THRESHOLD,
    ON_SCREEN_THRESHOLD,
    QuestionRecord,
    abstention_breakdown,
    abstention_separability,
    answered_wrong_with_evidence,
    modality_floor,
    recoverable_with_complete_evidence,
    summarise,
    token_coverage,
)


def rec(
    qid: str = "q",
    qtype: str = "number",
    score: float = 0.0,
    model_abstained: bool = False,
    gold_is_abstention: bool = False,
    evidence_complete: bool = True,
    evidence_hit: bool = True,
    coverage: float | None = 1.0,
    top1_score: float | None = 0.5,
) -> QuestionRecord:
    return QuestionRecord(
        qid=qid,
        qtype=qtype,
        score=score,
        model_abstained=model_abstained,
        gold_is_abstention=gold_is_abstention,
        evidence_complete=evidence_complete,
        evidence_hit=evidence_hit,
        coverage=coverage,
        top1_score=top1_score,
    )


# --- token_coverage: the absent-input-becomes-a-clean-zero failure -------------------------


def _tok(text: str) -> list[str]:
    return text.lower().split()


def test_coverage_is_none_not_zero_when_the_gold_answer_has_no_content_tokens() -> None:
    """0.0 means "not on screen" and is counted as recoverable; `None` means "not measurable".

    Collapsing them is the `.get(k, default)` defect in measurement code: an absent input becomes a
    constant, and the constant scores as an ordinary negative result nobody looks at twice.
    """
    assert token_coverage("the of", "anything at all", _tok, {"the", "of"}) is None


def test_coverage_counts_only_content_tokens_present_in_the_evidence() -> None:
    assert token_coverage("alpha beta the", "alpha only here", _tok, {"the"}) == pytest.approx(0.5)


def test_coverage_is_case_insensitive_against_the_evidence() -> None:
    assert token_coverage("alpha", "ALPHA appears shouting", _tok, set()) == pytest.approx(1.0)


# --- recoverable_with_complete_evidence ----------------------------------------------------


def test_only_questions_with_complete_evidence_count_toward_the_recoverable_prize() -> None:
    """A question whose evidence was never retrieved cannot be won by a better reader.

    Counting it would inflate the prize, which is the number every answer-side proposal is sized
    against.
    """
    records = [
        rec(qid="a", score=0.0, evidence_complete=True),
        rec(qid="b", score=0.0, evidence_complete=False),
    ]
    out = recoverable_with_complete_evidence(records)
    assert out["points_lost"] == pytest.approx(1.0)
    assert out["qs_points_lost"] == pytest.approx(50.0)


def test_a_perfect_score_on_complete_evidence_loses_nothing() -> None:
    out = recoverable_with_complete_evidence([rec(score=1.0), rec(score=1.0)])
    assert out["points_lost"] == pytest.approx(0.0)


def test_partial_credit_is_counted_as_partial_loss() -> None:
    """`list_recall` is Jaccard, so scores are fractional; treating anything below 1.0 as a whole
    point lost would overstate the prize on the type where it matters most."""
    out = recoverable_with_complete_evidence([rec(score=0.25)])
    assert out["points_lost"] == pytest.approx(0.75)


def test_qs_points_lost_is_normalised_over_all_questions_not_the_complete_subset() -> None:
    """The figure is quoted as QS points, and QS is a mean over EVERY question. Dividing by the
    complete-evidence subset instead would inflate it by the reciprocal of the retrieval rate."""
    records = [rec(qid="a", score=0.0, evidence_complete=True)] + [
        rec(qid=f"b{i}", score=1.0, evidence_complete=False) for i in range(3)
    ]
    assert recoverable_with_complete_evidence(records)["qs_points_lost"] == pytest.approx(25.0)


# --- abstention_breakdown ------------------------------------------------------------------


def test_a_refusal_the_benchmark_wanted_is_not_counted_as_loss() -> None:
    records = [rec(model_abstained=True, gold_is_abstention=True, score=1.0)]
    out = abstention_breakdown(records)
    assert out["abstentions"] == 1
    assert out["wrong_abstentions"] == 0
    assert out["correct_abstentions"] == 1
    assert out["wrong_abstention_points_lost"] == pytest.approx(0.0)


def test_a_refusal_on_an_answerable_question_is_dead_loss() -> None:
    out = abstention_breakdown([rec(model_abstained=True, gold_is_abstention=False, score=0.0)])
    assert out["wrong_abstentions"] == 1
    assert out["wrong_abstention_qs_points_lost"] == pytest.approx(100.0)


def test_an_abstention_that_still_scored_is_not_called_wrong() -> None:
    """The judge can award credit to a hedged `open_end` answer the normalizer reads as a refusal.

    Counting it as dead loss would double-count points the run already banked.
    """
    out = abstention_breakdown([rec(model_abstained=True, gold_is_abstention=False, score=0.9)])
    assert out["wrong_abstentions"] == 0


def test_correct_abstentions_and_the_strict_count_are_reported_separately() -> None:
    """`correct_abstentions` is "not dead loss" and admits a credited refusal on an answerable
    question. That is the right denominator for sizing a rescue, and the wrong one for the claim
    "the benchmark wanted this refusal", so both are emitted and a divergence is visible."""
    records = [
        rec(qid="a", model_abstained=True, gold_is_abstention=True, score=1.0),
        rec(qid="b", model_abstained=True, gold_is_abstention=False, score=0.9),
    ]
    out = abstention_breakdown(records)
    assert out["correct_abstentions"] == 2
    assert out["abstentions_on_gold_abstention_questions"] == 1


def test_the_per_type_split_is_what_makes_a_rescue_safe() -> None:
    """The load-bearing claim in the document is that a rescue restricted to the deterministic
    types cannot lose points, and it holds only because every gold abstention is `open_end`.
    If a gold abstention ever appears under `number`, this must show it."""
    records = [
        rec(qid="a", qtype="number", model_abstained=True, score=0.0),
        rec(qid="b", qtype="open_end", model_abstained=True, gold_is_abstention=True, score=1.0),
    ]
    out = abstention_breakdown(records)
    assert out["by_qtype"]["number"]["gold_abstention_questions"] == 0
    assert out["by_qtype"]["open_end"]["gold_abstention_questions"] == 1


def test_answer_on_screen_uses_the_stated_threshold_and_ignores_unmeasurable_coverage() -> None:
    records = [
        rec(qid="a", model_abstained=True, score=0.0, coverage=ON_SCREEN_THRESHOLD),
        rec(qid="b", model_abstained=True, score=0.0, coverage=ON_SCREEN_THRESHOLD - 0.01),
        rec(qid="c", model_abstained=True, score=0.0, coverage=None),
    ]
    assert abstention_breakdown(records)["wrong_abstentions_with_answer_on_screen"] == 1


# --- abstention_separability ---------------------------------------------------------------


def test_separability_reports_a_coin_when_the_signal_does_not_separate() -> None:
    """This function exists to KILL "gate abstention on the trust layer", so its null case is the
    one that must be right: identical distributions have to come back at 0.5, not at 0 or 1."""
    records = [
        rec(qid="a", model_abstained=True, gold_is_abstention=True, score=1.0, top1_score=0.5),
        rec(qid="b", model_abstained=True, gold_is_abstention=False, score=0.0, top1_score=0.5),
    ]
    assert abstention_separability(records)["p_correct_below_wrong"] == pytest.approx(0.5)


def test_separability_reports_one_when_correct_refusals_score_strictly_lower() -> None:
    records = [
        rec(qid="a", model_abstained=True, gold_is_abstention=True, score=1.0, top1_score=0.1),
        rec(qid="b", model_abstained=True, gold_is_abstention=False, score=0.0, top1_score=0.9),
    ]
    assert abstention_separability(records)["p_correct_below_wrong"] == pytest.approx(1.0)


def test_separability_says_it_is_unmeasurable_rather_than_returning_a_number() -> None:
    """With no correct abstentions there is nothing to separate. Returning 0.0 or 0.5 here would
    read as a measured null and be quoted as one."""
    out = abstention_separability([rec(model_abstained=True, gold_is_abstention=False, score=0.0)])
    assert out["measurable"] is False
    assert "p_correct_below_wrong" not in out


# --- answered_wrong_with_evidence and modality_floor ---------------------------------------


def test_selection_failures_exclude_refusals() -> None:
    """A refusal and a wrong pick are different mechanisms with different fixes. Letting the
    refusals fall into this bucket would double-count them against the abstention total."""
    records = [
        rec(qid="a", score=0.0, model_abstained=True, coverage=1.0),
        rec(qid="b", score=0.0, model_abstained=False, coverage=1.0),
    ]
    assert answered_wrong_with_evidence(records)["count"] == 1


def test_the_modality_floor_excludes_questions_the_model_got_right_anyway() -> None:
    """The floor is a statement about LOSS. A question with low lexical coverage that still scored
    is not stuck on the modality ceiling, it is a limitation of the coverage proxy.

    Reproduction: omitting this filter counted 33 questions worth 1.78 QS on the real run, against
    20 worth 1.97 -- MORE questions and FEWER points, which is the signature of sweeping in rows
    that scored well. Both readings are plausible in isolation; only the pair gives it away.
    """
    records = [
        rec(qid="a", evidence_hit=True, coverage=0.1, score=0.0),
        rec(qid="b", evidence_hit=True, coverage=0.1, score=1.0),
    ]
    out = modality_floor(records)
    assert out["count"] == 1
    assert out["points_lost"] == pytest.approx(1.0)


def test_the_modality_floor_requires_the_gold_item_to_have_been_retrieved() -> None:
    """The floor is "we found it and its text does not carry the answer". A question whose gold was
    never retrieved is a retrieval miss and belongs to a different bucket entirely."""
    records = [
        rec(qid="a", evidence_hit=True, coverage=MODALITY_FLOOR_THRESHOLD - 0.1, score=0.0),
        rec(qid="b", evidence_hit=False, coverage=MODALITY_FLOOR_THRESHOLD - 0.1, score=0.0),
    ]
    assert modality_floor(records)["count"] == 1


# --- summarise -----------------------------------------------------------------------------


def test_summarise_refuses_an_empty_record_set() -> None:
    """Publishing a decomposition of nothing would divide by zero or, worse, emit a table of
    clean zeroes that reads exactly like a measured null."""
    with pytest.raises(ValueError, match="nothing"):
        summarise([])


def test_summarise_reproduces_the_mean_score_as_qs() -> None:
    records = [rec(qid="a", score=1.0), rec(qid="b", score=0.0)]
    out = summarise(records)
    assert out["qs"] == pytest.approx(0.5)
    assert out["qs_percent"] == pytest.approx(50.0)
    assert out["question_count"] == 2


# --- the run artifact is read through the publication check ---------------------------------


def test_a_quarantined_run_artifact_is_refused_as_the_apparatus_baseline(tmp_path) -> None:
    """`benchmarks.run` marks an artifact it REFUSED to publish in band, and a refused file is
    byte-identical to a real measurement to anything that ignores the mark.

    This decomposition validates itself against the run's published QS, so reading that file with a
    bare `json.loads` would let a quarantined run become the baseline the check passes against --
    turning the apparatus check into a rubber stamp exactly when it matters. Caught by
    `test_no_benchmark_tool_reads_a_run_artifact_without_the_publication_check`, which is a
    package-wide guard; this pins the behaviour rather than the import.
    """
    from benchmarks.atm_answer_diagnosis import load_published_artifact

    quarantined = tmp_path / "run.json"
    quarantined.write_text(
        json.dumps(
            {
                "unpublished": True,
                "unpublished_reason": "provider returned a partial run",
                "official_score": {"qs": 0.9999},
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(SystemExit, match="REFUSED publication"):
        load_published_artifact(quarantined)


def test_a_published_run_artifact_is_accepted(tmp_path) -> None:
    """The positive half: without the mark the same read must succeed, or the guard above would be
    satisfied by a function that refuses everything."""
    from benchmarks.atm_answer_diagnosis import load_published_artifact

    good = tmp_path / "run.json"
    good.write_text(json.dumps({"official_score": {"qs": 0.5}}), encoding="utf-8")
    assert load_published_artifact(good)["official_score"]["qs"] == 0.5
