"""Harness logic for the near-miss abstention evaluation (arms A/B/C), judged with fake judges.

These tests pin the ARM SEMANTICS deterministically (a real QNLI judge's quality is what the
actual evaluation run measures, not what CI asserts):
- with an accept-all judge the entailment arms cannot differ from the threshold arm on FCR;
- with a reject-all judge the entailment arms must abstain on everything.
The near-miss query set must also stay OUT of the calibration inputs — the challenge set must
not tune the threshold it is challenging.
"""
from __future__ import annotations

import math

from recall.embeddings import HashingEmbedder
from recall.eval.harness import ARMS, run_nearmiss_eval

from tests.conftest import TEST_DSN, requires_db


class AcceptAll:
    def judge(self, query: str, texts: list[str]) -> list[bool]:
        return [True] * len(texts)


class RejectAll:
    def judge(self, query: str, texts: list[str]) -> list[bool]:
        return [False] * len(texts)


def _by_arm(results, emb_name):
    return {r.arm: r for r in results if r.embedder == emb_name}


@requires_db
def test_arms_present_and_metrics_bounded():
    emb = HashingEmbedder(dim=64)
    results = run_nearmiss_eval(TEST_DSN, [emb], judge=AcceptAll())
    arms = _by_arm(results, emb.name)
    assert set(arms) == set(ARMS)
    for r in arms.values():
        for v in (r.nearmiss_fcr, r.gap_fcr, r.false_abstain, r.mrr_answerable):
            assert math.isnan(v) or 0.0 <= v <= 1.0
        assert r.entail_latency_ms_mean >= 0.0
        assert r.query_latency_ms_mean > 0.0


@requires_db
def test_accept_all_judge_cannot_change_the_threshold_arm():
    emb = HashingEmbedder(dim=64)
    arms = _by_arm(run_nearmiss_eval(TEST_DSN, [emb], judge=AcceptAll()), emb.name)
    # a judge that trusts everything degenerates arm B to arm A exactly
    assert arms["threshold+entail"].nearmiss_fcr == arms["threshold"].nearmiss_fcr
    assert arms["threshold+entail"].false_abstain == arms["threshold"].false_abstain
    assert arms["threshold+entail"].mrr_answerable == arms["threshold"].mrr_answerable
    # the threshold arm pays zero judge latency
    assert arms["threshold"].entail_latency_ms_mean == 0.0


def test_near_miss_distractor_ids_resolve_against_the_eval_corpus():
    # deferred DAT-001: distractor_ids were documentation-only — a renamed corpus file would
    # silently degrade the near-miss set into far-gap queries, flattering the threshold arm
    import json
    from pathlib import Path

    from recall.index import chunk_text

    eval_dir = Path("recall/eval")
    nearmiss = json.loads((eval_dir / "near_miss.json").read_text(encoding="utf-8"))
    assert nearmiss, "near-miss set must not be empty"
    for q in nearmiss:
        assert q["distractor_ids"], f"{q['id']}: no distractor declared"
        for did in q["distractor_ids"]:
            fname, ord_s = did.rsplit(":", 1)
            f = eval_dir / "corpus" / fname
            assert f.is_file(), f"{q['id']}: distractor file {fname} missing from corpus"
            n_chunks = len(chunk_text(f.read_text(encoding="utf-8-sig")))
            assert int(ord_s) < n_chunks, f"{q['id']}: chunk ord {ord_s} out of range"


@requires_db
def test_reject_all_judge_abstains_on_everything():
    emb = HashingEmbedder(dim=64)
    arms = _by_arm(run_nearmiss_eval(TEST_DSN, [emb], judge=RejectAll()), emb.name)
    for arm in ("threshold+entail", "entail-only"):
        assert arms[arm].nearmiss_fcr == 0.0   # nothing near-miss survives
        assert arms[arm].false_abstain == 1.0  # ...at the cost of abstaining on every answerable


def test_loo_calibrations_never_include_the_sample_they_will_score():
    """Each fold's threshold must be fitted without the sample it is about to judge.

    This is what makes `gap_fcr` / `false_abstain` out-of-sample. The check is behavioural, not
    structural: fold i's threshold must equal a threshold fitted on the remaining samples, and
    a fold whose held-out sample is an outlier must differ from the full-sample fit.
    """
    from recall.calibration import from_samples
    from recall.eval.harness import _loo_calibrations

    ans = [0.80, 0.82, 0.85, 0.88]
    unans = [0.10, 0.20, 0.30, 0.65]
    cals = _loo_calibrations("e", unans, ans, hold_out_unanswerable=True)

    assert len(cals) == len(unans)
    for i in range(len(unans)):
        expected = from_samples("e", ans, unans[:i] + unans[i + 1:])
        assert cals[i].threshold == expected.threshold


def test_holding_out_an_unanswerable_sample_cannot_move_the_threshold():
    """Documents WHY the unanswerable-side split leaves `gap_fcr` unchanged.

    `best_threshold` scans candidates ascending and takes the first minimum-error one. A
    candidate below min(answerable) costs one unanswerable error for every sample above it and
    saves nothing; a candidate above it costs answerable errors. So the optimum sits exactly at
    min(answerable) no matter where the unanswerable samples fall — the "answerable vs
    unanswerable" fit is ONE-SIDED, and the unanswerable class does not inform the threshold at
    all in the separable regime.

    Consequence for the published numbers: leave-one-out is still the correct protocol, but it
    cannot change `gap_fcr` — that column was never at risk of memorising its samples. The
    answerable side is where the split bites (see tests/test_eval_calibrate_heldout.py::
    test_loo_exposes_that_the_fitted_threshold_has_no_answerable_side_margin).
    """
    from recall.calibration import best_threshold

    ans = [0.80, 0.82, 0.85, 0.88]
    for unans in ([0.10, 0.20, 0.30, 0.65], [0.10, 0.20, 0.30, 0.90]):
        folds = [best_threshold(ans, unans[:i] + unans[i + 1:]) for i in range(len(unans))]
        assert folds == [min(ans)] * len(unans)


def test_loo_calibrations_refuse_to_split_a_class_too_small_to_split():
    """One sample cannot be held out — the fold would fit on an empty class. None, not a guess."""
    from recall.eval.harness import _loo_calibrations

    assert _loo_calibrations("e", [0.2], [0.8, 0.9], hold_out_unanswerable=True) == [None]
    assert _loo_calibrations("e", [], [0.8], hold_out_unanswerable=False) == []


def test_loo_calibrations_places_the_held_out_class_on_the_correct_side():
    """`hold_out_unanswerable` decides which argument of from_samples the refit class fills.

    Swapping them silently inverts the threshold, so pin it: holding out an ANSWERABLE sample
    must leave the unanswerable samples intact as the lower class.
    """
    from recall.calibration import from_samples
    from recall.eval.harness import _loo_calibrations

    ans = [0.80, 0.82, 0.85]
    unans = [0.10, 0.20]
    cals = _loo_calibrations("e", ans, unans, hold_out_unanswerable=False)
    assert cals[0].threshold == from_samples("e", ans[1:], unans).threshold
