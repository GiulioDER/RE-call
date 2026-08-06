"""Multi-query fusion must be RE-call's own fusion, generalised, not a lookalike.

The offload harness earned this standard the hard way: an ordering that merely looks reasonable
produces publishable-looking metrics that RE-call would never compute, and nothing about the
output looks wrong. So the single-variant case is pinned against `recall.retriever._rrf` here,
and against a live `HybridRetriever` by the `validate` subcommand on the real index.

Two of these tests are load bearing for the PREREGISTRATION rather than for the code:

  - `test_identical_variants_fuse_to_the_identical_ranking` pins the structural-zero claim the
    power analysis rests on (102 of 777 dev queries carry byte-identical variants, so their
    per-query delta is 0.0 by construction and buys no power).
  - `test_nesting_rewards_cross_variant_consensus_and_flat_rewards_raw_votes` keeps contrast T1
    from being vacuous. If the
    two topologies were algebraically identical, T1 would be a guaranteed zero dressed up as a
    finding.
"""

from __future__ import annotations

import pytest

from benchmarks.mtrag.multiquery import (
    CONTRASTS,
    MQ_ARMS,
    RRF_K,
    SHIP_BAR,
    MultiQueryArm,
    decide_verdict,
    fuse_arm,
    rrf_scores,
    sorted_by_score,
)
from recall.retriever import _rrf


def stats(mean: float, *, established: bool = True) -> dict[str, object]:
    """A `paired_stats`-shaped result with the fields the decision rule reads."""
    return {"mean_delta": mean, "ci_excludes_zero": established, "holm_significant": established,
            "ci_high": mean + (0.005 if established else 0.05)}


def veto(mean: float, ci_high: float) -> dict[str, object]:
    """A veto-metric result. Only `ci_high` decides whether the guard trips."""
    return {"mean_delta": mean, "ci_high": ci_high, "ci_excludes_zero": ci_high < 0}


def legs(**variants: dict[str, list[str]]) -> dict[str, dict[str, list[str]]]:
    return dict(variants)


def test_single_variant_fusion_is_byte_identical_to_recall_fusion() -> None:
    """The control arm must be the system under test, not a reimplementation of it.

    `mq_last` is configuration-identical to the archived `hybrid_splade`, and the whole contrast
    family is measured against it. If this module's fusion differed from `recall.retriever` by so
    much as a tie-break, every delta would be contaminated by that difference rather than by the
    query diversity the experiment is about.
    """
    dense = ["d1", "d2", "d3", "shared"]
    splade = ["s1", "shared", "s2"]

    expected = sorted_by_score(_rrf([dense, [], splade]))
    arm = MultiQueryArm("probe", ("last",), "nested")

    assert fuse_arm(arm, legs(last={"dense": dense, "splade": splade})) == expected


def test_identical_variants_fuse_to_the_identical_ranking() -> None:
    """Fusing a ranking with copies of itself is order-preserving.

    This is the claim the power analysis rests on. MTRAG-human's three query files are
    byte-identical on the first turn of every conversation (102 of 777 judged dev queries), so
    those queries produce identical leg rankings for every variant. If RRF perturbed the order
    anyway, those 102 queries would contribute noise rather than a structural zero and the
    deciding-cell arithmetic in the preregistration would be wrong.
    """
    dense = ["a", "b", "c"]
    splade = ["c", "d", "a"]
    one = {"dense": dense, "splade": splade}

    single = fuse_arm(MultiQueryArm("one", ("last",), "nested"), legs(last=one))
    tripled = fuse_arm(
        MultiQueryArm("three", ("last", "full", "rewrite"), "nested"),
        legs(last=dict(one), full=dict(one), rewrite=dict(one)),
    )

    assert tripled == single


def test_nesting_rewards_cross_variant_consensus_and_flat_rewards_raw_votes() -> None:
    """Contrast T1 must be capable of being non-zero, and this is the mechanism that makes it so.

    🔑 The inner level emits a RANKING, not scores, so re-ranking discards how strongly a variant
    liked a document. A document both legs of ONE variant put first arrives at the outer fusion
    as merely "rank 0 of one variant". A document that tops ONE leg in TWO variants arrives as
    "rank 0 of two variants" and wins.

    Flat has no such level: every leg ranking is one equal vote, so both documents collect two
    first-place votes and tie, and the tie falls to insertion order.

    That is exactly why rank 1 nested rather than fusing flat -- it stops three volatile
    reformulations from out-voting two stable ones by weight of numbers. `agree` here is the
    document one reformulation is sure about; `spread` is the one two reformulations agree on.
    """
    variants = legs(
        last={"dense": ["agree", "x"], "splade": ["agree", "y"]},
        full={"dense": ["spread", "p"], "splade": ["q", "r"]},
        rewrite={"dense": ["spread", "m"], "splade": ["n", "o"]},
    )

    nested = fuse_arm(MultiQueryArm("n", ("last", "full", "rewrite"), "nested"), variants)
    flat = fuse_arm(MultiQueryArm("f", ("last", "full", "rewrite"), "flat"), variants)

    assert nested != flat
    assert nested[0] == "spread"  # agreed on by two reformulations
    assert flat[0] == "agree"  # two first-place votes, both from one reformulation


def test_outer_weights_demote_the_variant_they_down_weight() -> None:
    """`mq_nested3_vw` down-weights `full` to 0.5, and that has to actually do something.

    The weight is fixed a priori because `full` is the only variant with a measured negative
    effect (-0.0972 nDCG@5 alone). A weight the fusion silently ignored would make the
    variance-aware arm a duplicate of `mq_nested3` reported under a second name.
    """
    variants = legs(
        last={"dense": ["from_last"], "splade": ["from_last"]},
        full={"dense": ["from_full"], "splade": ["from_full"]},
    )
    equal = MultiQueryArm("equal", ("last", "full"), "nested")
    weighted = MultiQueryArm("weighted", ("last", "full"), "nested", weights=(1.0, 0.5))

    assert fuse_arm(equal, variants) == ["from_last", "from_full"]  # tie, insertion order
    scores = rrf_scores(
        [["from_last"], ["from_full"]], k=RRF_K, weights=(1.0, 0.5)
    )
    assert scores["from_last"] > scores["from_full"]
    assert fuse_arm(weighted, variants) == ["from_last", "from_full"]


def test_weights_must_match_the_variant_count() -> None:
    """A silently recycled or truncated weight vector would mis-weight an arm invisibly."""
    with pytest.raises(ValueError, match="weights"):
        fuse_arm(
            MultiQueryArm("bad", ("last", "full"), "nested", weights=(1.0,)),
            legs(last={"dense": ["a"]}, full={"dense": ["b"]}),
        )


def test_leg_truncation_bounds_what_can_enter_the_fusion() -> None:
    """`mq_nested3_budget33` holds total retrieval budget fixed by truncating every leg.

    A deeper single-query control is impossible here: `sparse_ef_search` caps at pgvector's
    `hnsw.ef_search` ceiling of 1000, which was measured to yield exactly 100 rows on this shared
    sidecar. So the budget is matched downwards instead, and the truncation has to bite.
    """
    variants = legs(last={"dense": ["a", "b", "c", "d"], "splade": ["e", "f", "g", "h"]})
    arm = MultiQueryArm("trunc", ("last",), "nested", leg_truncate=2)

    fused = fuse_arm(arm, variants)

    assert set(fused) == {"a", "b", "e", "f"}


def test_a_missing_variant_raises_rather_than_fusing_what_is_present() -> None:
    """Silently fusing two variants under a three-variant arm's name is the worst failure here.

    It would report a diversity result for a diversity the run never had, and the arm name in the
    metrics file would still say three.
    """
    with pytest.raises(KeyError, match="rewrite"):
        fuse_arm(
            MultiQueryArm("three", ("last", "full", "rewrite"), "nested"),
            legs(last={"dense": ["a"]}, full={"dense": ["b"]}),
        )


def test_an_empty_leg_contributes_nothing_but_is_not_an_error() -> None:
    """A query of pure stopwords legitimately encodes to no SPLADE terms.

    `HybridRetriever` already treats that as "this leg contributes nothing" rather than as a
    failure, and the reconstruction must agree or it would diverge from the system on exactly
    the degenerate queries.
    """
    variants = legs(last={"dense": ["a", "b"], "splade": []})

    assert fuse_arm(MultiQueryArm("one", ("last",), "nested"), variants) == ["a", "b"]


def test_the_declared_arms_are_the_preregistered_ones() -> None:
    """The arms are frozen in the preregistration; this pins them against a later quiet edit.

    Adding an arm after seeing scores is the specific failure this guards: a post-hoc arm is a
    different experiment wearing a preregistered experiment's name.
    """
    assert [a.name for a in MQ_ARMS] == [
        "mq_last",
        "mq_rewrite",
        "mq_full",
        "mq_nested3",
        "mq_nested2",
        "mq_flat6",
        "mq_nested3_vw",
        "mq_nested3_budget33",
    ]
    by_name = {a.name: a for a in MQ_ARMS}
    assert by_name["mq_last"].variants == ("last",)
    assert by_name["mq_nested3"].variants == ("last", "full", "rewrite")
    assert by_name["mq_nested3_vw"].weights == (1.0, 0.5, 1.0)
    assert by_name["mq_nested3_budget33"].leg_truncate == 33
    assert by_name["mq_flat6"].topology == "flat"
    # Every arm fuses at RE-call's own damping constant, at BOTH levels. Rank 1 used k_internal=40
    # for their weak-consensus group; adopting it would change the control's fusion too.
    assert RRF_K == 60


def test_the_budget_contrast_is_not_reported_over_the_structural_zero_cell() -> None:
    """B1's delta is driven by TRUNCATION, so identical variants are not a structural zero for it.

    Every other contrast compares fusions of the same untruncated legs, so a query whose three
    variants are byte-identical contributes an exact zero and is excluded from the deciding-cell
    mean. B1 compares 33-deep legs against 100-deep ones, which changes the ranking whatever the
    query text is. Reporting it over the `variants_differ` cell would silently drop the 102
    turn-1 queries, which carry real signal for this contrast alone.
    """
    cells = {cid: cell for cid, _t, _c, _q, cell in CONTRASTS}

    assert cells == {"P1": "variants_differ", "M1": "variants_differ", "T1": "variants_differ",
                     "R1": "variants_differ", "B1": "all"}

    identical = {"dense": [f"d{i}" for i in range(100)],
                 "splade": [f"s{i}" for i in range(100)]}
    variants = legs(last=dict(identical), full=dict(identical), rewrite=dict(identical))
    by_name = {a.name: a for a in MQ_ARMS}

    assert fuse_arm(by_name["mq_nested3_budget33"], variants) != fuse_arm(
        by_name["mq_last"], variants
    )


def test_a_significant_regression_is_not_reported_as_a_near_miss() -> None:
    """The smoke run on synthetic data produced exactly this case and it was mislabelled.

    A P1 that is significantly NEGATIVE was landing under
    `INCONCLUSIVE_REAL_BUT_SUB_THRESHOLD`, which reads like an encouraging result one more push
    would clear. It is the opposite: a measured harm.
    """
    assert decide_verdict(stats(-0.058), stats(0.01))["verdict"] == "REGRESSES"


def test_a_positive_result_below_the_bar_is_inconclusive_not_a_ship() -> None:
    """The bar is preregistered at +0.020 and sits just above the ~0.019 detectable effect.

    Significance alone must not ship a change: with n=777 a trivially small delta can clear a
    permutation test while being worth nothing operationally.
    """
    assert decide_verdict(stats(0.012), stats(0.01))["verdict"] == (
        "INCONCLUSIVE_REAL_BUT_SUB_THRESHOLD"
    )
    assert decide_verdict(stats(SHIP_BAR), stats(0.01))["verdict"] == "SHIPS"


def test_an_unestablished_effect_does_not_ship_however_large_the_point_estimate() -> None:
    """A point estimate is not a result.

    A -0.0043 "regression" was reported on 2026-08-06 and then retracted when its CI turned out
    to span zero. The same rule has to hold in the flattering direction.
    """
    verdict = decide_verdict(stats(0.25, established=False), stats(0.2))

    assert verdict["verdict"] == "DOES_NOT_SHIP"
    assert verdict["gpu_rental_justified"] is False


def test_a_large_primary_with_a_null_mechanism_closes_the_lever() -> None:
    """The most informative failure the preregistration anticipates.

    If fusion beats the single last-turn query but cannot beat the GOLD rewrite, the gain came
    from rewriting quality rather than from diversity. Gold is a ceiling an LLM rewriter cannot
    reach, so there is no headroom to rent a GPU for -- even though the headline P1 looks great.
    """
    verdict = decide_verdict(stats(0.05), stats(0.001, established=False))

    assert verdict["verdict"] == "SHIPS"
    assert verdict["mechanism_verdict"] == "REWRITING_QUALITY_OR_UNRESOLVED"
    assert verdict["gpu_rental_justified"] is False


def test_coverage_bought_with_ranking_is_blocked_rather_than_shipped() -> None:
    """The one-sided rule this veto was added to close.

    R@100 is the decision metric because coverage is the bottleneck, but that made the rule
    one-sided: an arm that lifted coverage while degrading the ranking cleared the bar. Not
    hypothetical here. The archived SPLADE run is exactly that shape, and rank 1 reports the same
    trade. So an established ranking regression blocks, and says which metric blocked it.
    """
    verdict = decide_verdict(
        stats(0.04), stats(0.02),
        {"nDCG@5": veto(-0.018, -0.011), "R@10": veto(0.001, 0.02)},
    )

    assert verdict["verdict"] == "BLOCKED_BY_RANKING_REGRESSION"
    assert verdict["vetoes_tripped"] == ["nDCG@5"]


def test_a_merely_negative_ranking_point_estimate_does_not_block() -> None:
    """A point estimate is not a result in the unflattering direction either.

    The guard trips on an ESTABLISHED regression, meaning the whole CI sits below zero. Blocking
    on a negative mean whose interval spans zero would reproduce the retracted -0.0043
    "regression" of 2026-08-06, just with the sign of the mistake reversed.
    """
    verdict = decide_verdict(stats(0.04), stats(0.02), {"nDCG@5": veto(-0.006, 0.004)})

    assert verdict["verdict"] == "SHIPS"
    assert verdict["vetoes_tripped"] == []


def test_a_tripped_veto_still_leaves_the_mechanism_finding_intact() -> None:
    """A blocked ship is not a dead lever, and the two verdicts are reported independently.

    If fusion beats the gold rewrite, the mechanism is established whatever the ranking did; the
    right follow-up is to fix the ranking cost, not to abandon query diversity.
    """
    verdict = decide_verdict(stats(0.04), stats(0.02), {"nDCG@5": veto(-0.02, -0.01)})

    assert verdict["mechanism_verdict"] == "FUSION"
    assert verdict["gpu_rental_justified"] is True


def test_fusion_mechanism_with_a_shippable_primary_justifies_the_gpu() -> None:
    """The one path that opens the next stage of work, stated so it cannot be assumed."""
    verdict = decide_verdict(stats(0.03), stats(0.015))

    assert verdict["mechanism_verdict"] == "FUSION"
    assert verdict["gpu_rental_justified"] is True


def test_sorted_by_score_breaks_ties_by_first_appearance() -> None:
    """Ties resolve by insertion order, exactly as `HybridRetriever` resolves them.

    RRF ties are common, not exotic: any two documents appearing at the same rank in the same
    number of equally weighted rankings tie exactly. `recall.retriever` sorts the `_rrf` dict with
    a stable sort, so the tie-break is "whichever ranking listed it first". Any other rule makes
    the control diverge from the shipped system on the ties a cut at 5 is most sensitive to.
    """
    assert sorted_by_score({"a": 1.0, "b": 2.0, "c": 1.0}) == ["b", "a", "c"]
