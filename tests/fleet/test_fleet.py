"""The eval calibration fleet: see docs/EVAL_CALIBRATION_FLEET_DESIGN.md."""
from __future__ import annotations

import json
import math
from contextlib import nullcontext

import pytest

from recall.eval.harness import (
    ARM_ENTAIL_ONLY,
    ARM_STACKED,
    ARM_THRESHOLD,
    _score_config,
    run_nearmiss_eval,
    run_trust_eval,
)
from recall.eval.promotion.aggregate import decide
from tests.fleet.members import (
    DEPTH,
    SURFACE_A,
    SURFACE_B,
    SURFACE_C,
    SURFACE_D,
    FleetMember,
    GateExpectation,
    _rows,
)
from tests.fleet.scripted import (
    AlwaysEntailJudge,
    QueryKeyedStore,
    QueryKeyedTrustStore,
    ScriptedEmbedder,
)


def test_scripted_store_returns_the_scripted_order_through_the_real_retriever():
    """The whole fleet rests on this: with fusion="dense" the retrieved order IS the script.

    If RRF or the sparse leg ever reordered these, every closed form in members.py would be
    wrong while still looking plausible, so this is checked directly rather than assumed.
    """
    embedder = ScriptedEmbedder()
    script = {"q": [("a.md:0", 0.8), ("b.md:0", 0.7), ("gold.md:0", 0.6)]}
    store = QueryKeyedStore(embedder, script)
    queries = [{"id": "q1", "query": "q", "relevant_ids": ["gold.md:0"], "answerable": True}]

    result = _score_config(store, embedder, queries, "dense", None)

    # gold is at 1-based rank 3 of 3: MRR = 1/3, R@5 = 1.0 (it is inside the top 5).
    assert result.mrr == pytest.approx(1 / 3)
    assert result.r_at_5 == pytest.approx(1.0)


def test_scripted_store_refuses_an_unscripted_query():
    """An unscripted query must raise, not return zero rows.

    Zero rows scores as a total retrieval failure, which is a FIXTURE bug that reads exactly
    like a defect the fleet detected. `arm_check.EmptySampleError` sets the same precedent.
    """
    embedder = ScriptedEmbedder()
    store = QueryKeyedStore(embedder, {"scripted": [("a.md:0", 0.8)]})
    queries = [{"id": "q1", "query": "unscripted", "relevant_ids": ["a.md:0"], "answerable": True}]

    with pytest.raises(KeyError, match="no script for query"):
        _score_config(store, embedder, queries, "dense", None)


def test_a_member_without_a_declared_blind_spot_is_refused():
    """`does_not_catch` is required, not optional.

    An optional field would be empty on every member within a month, and a fleet that does not
    state what it misses invites being read as covering more than it does.
    """
    with pytest.raises(ValueError, match="does_not_catch"):
        FleetMember(
            name="blank",
            defect="none",
            build=lambda: None,
            expected={},
            does_not_catch="   ",
        )


def test_a_gate_expectation_with_raises_and_promoted_is_refused():
    """`raises` set together with `promoted` (or `failure_contains`) is refused.

    The `raises` branch of the runner returns before checking either, so a stale value left on
    one of them would silently never be checked. Modelled on
    `test_a_member_without_a_declared_blind_spot_is_refused`.
    """
    with pytest.raises(ValueError, match="raises is set together with promoted"):
        GateExpectation(raises=ValueError, promoted=True)


def test_a_gate_expectation_with_raises_contains_but_no_raises_is_refused():
    """`raises_contains` set without `raises` is refused: there is no exception to read from."""
    with pytest.raises(ValueError, match="raises_contains is set without raises"):
        GateExpectation(raises_contains="something")


def test_a_gate_expectation_asserting_nothing_is_refused():
    """None of `raises`, `promoted`, `failure_contains` set is refused.

    A `GateExpectation` that asserts nothing would pass no matter what the gate did, which is
    the exact silent-pass failure mode this fleet exists to catch.
    """
    with pytest.raises(ValueError, match="none of raises, promoted or failure_contains"):
        GateExpectation()


@pytest.mark.parametrize("rank", [0, -1, DEPTH + 1])
def test_rows_rejects_a_rank_outside_1_to_depth(rank: int):
    """`rank <= 0` must raise, not silently wrap around to `ids[-1]` via negative indexing.

    `rank > DEPTH` is included too, so both directions of an out-of-range rank fail the same
    explicit way rather than one of them being an accident of `IndexError`.
    """
    with pytest.raises(ValueError, match="rank must be None or within"):
        _rows(gold="gold.md:0", rank=rank, prefix="filler", score=0.8)


def run_surface_a(member: FleetMember) -> dict[str, float]:
    """Drive one member through the REAL scoring path and return the published fields."""
    queries, script = member.build()
    if not queries:
        raise ValueError(
            f"{member.name} built an empty query list. A fixture that supplies no queries at "
            f"all must not read like a defect that was absent; `arm_check.EmptySampleError` "
            f"sets the same precedent for an empty comparison. This is distinct from a "
            f"non-empty list that scores zero QUALITY questions, which `no-answerable-queries` "
            f"deliberately does and must keep passing."
        )
    embedder = ScriptedEmbedder()
    store = QueryKeyedStore(embedder, script)
    result = _score_config(store, embedder, queries, "dense", None)
    return {
        "p_at_5": result.p_at_5,
        "r_at_5": result.r_at_5,
        "mrr": result.mrr,
        "ndcg_at_10": result.ndcg_at_10,
        "fcr_with_guard": result.fcr_with_guard,
    }


@pytest.mark.parametrize("member", SURFACE_A, ids=lambda m: m.name)
def test_surface_a_member_reports_its_closed_form(member: FleetMember):
    actual = run_surface_a(member)
    for field, expected in member.expected.items():
        # `pytest.approx` does NOT match NaN unless `nan_ok=True` is passed, and `nan_ok=True`
        # on a plain `pytest.approx(expected)` would make the comparison pass for ANY actual
        # value once expected is NaN (NaN tolerates everything under that flag). So NaN gets its
        # own explicit branch: `math.isnan(actual[field])` is False for 0.0, which is exactly
        # the old bug this member exists to catch — it still fails loudly against a `mean(x) if
        # x else 0.0` regression.
        if math.isnan(expected):
            assert math.isnan(actual[field]), (
                f"{member.name} ({member.defect}): {field} should be NaN (no data measured), "
                f"got {actual[field]}. Investigate the harness before editing this expectation."
            )
        else:
            assert actual[field] == pytest.approx(expected, abs=1e-9), (
                f"{member.name} ({member.defect}): {field} should be {expected} by construction, "
                f"got {actual[field]}. Investigate the harness before editing this expectation."
            )


def test_surface_a_has_eight_members():
    assert len(SURFACE_A) == 8


@pytest.mark.parametrize("member", SURFACE_A, ids=lambda m: m.name)
def test_surface_a_member_is_not_vacuous(member: FleetMember):
    """A member matching the clean twin on every field it declares certifies nothing.

    `gold-at-rank-3` matches the twin on r_at_5 and earns its place only through mrr and ndcg.
    This is `feedback-a-mutation-sweep-cannot-see-a-fixture-built-from-the-passing-shape`
    enforced mechanically instead of trusted.
    """
    twin = SURFACE_A[0]
    if member is twin:
        return
    assert any(
        member.expected[field] != twin.expected.get(field)
        for field in member.expected
    ), f"{member.name} is indistinguishable from the clean twin on every field it declares"


def test_the_fleet_detects_a_broken_recall_metric(monkeypatch):
    """Mutate the code under test and require the fleet to notice.

    Green tests are evidence of nothing until they have been shown to go red, and a test
    written after a fix and never shown to fail is a hypothesis rather than a guard. If this
    passes, the surface A members are decorative and THAT is the finding.
    """
    import recall.eval.harness as harness

    monkeypatch.setattr(harness, "recall_at_k", lambda *args, **kwargs: 1.0)

    # boundary-rank-6 is the member whose r_at_5 is 0.0 by construction, so a recall_at_k
    # pinned to 1.0 must move it. Picking the member by name rather than by index keeps this
    # honest if the table is reordered.
    member = next(m for m in SURFACE_A if m.name == "boundary-rank-6")
    actual = run_surface_a(member)

    assert actual["r_at_5"] != pytest.approx(
        member.expected["r_at_5"], abs=1e-9
    ), (
        "a recall_at_k stubbed to 1.0 did not change boundary-rank-6's r_at_5, so the fleet "
        "is not actually reading this metric"
    )


def run_surface_b(member: FleetMember):
    baseline, candidate, frozen, kwargs = member.build()
    return decide(baseline, candidate, frozen, **kwargs)


@pytest.mark.parametrize("member", SURFACE_B, ids=lambda m: m.name)
def test_surface_b_member_reaches_its_declared_verdict(member: FleetMember):
    expectation = member.expected

    if expectation.raises is not None:
        with pytest.raises(expectation.raises) as excinfo:
            run_surface_b(member)
        if expectation.raises_contains is not None:
            assert expectation.raises_contains in str(excinfo.value), (
                f"{member.name}: raised {expectation.raises.__name__} but its message does not "
                f"contain {expectation.raises_contains!r}; message was {str(excinfo.value)!r}. "
                f"A type match alone cannot tell this failure mode apart from a sibling "
                f"ValueError raised earlier in the same call."
            )
        return

    decision, _document = run_surface_b(member)
    assert decision.promoted is expectation.promoted, (
        f"{member.name} ({member.defect}): expected promoted={expectation.promoted}, got "
        f"{decision.promoted} with failures {decision.failures}"
    )
    if expectation.failure_contains is not None:
        assert any(expectation.failure_contains in f for f in decision.failures), (
            f"{member.name}: no failure mentions {expectation.failure_contains!r}; "
            f"failures were {decision.failures}"
        )


def test_surface_b_has_six_members():
    assert len(SURFACE_B) == 6


def test_the_fleet_detects_a_disabled_safety_axis(monkeypatch):
    """Make the safety axis unable to register a regression and require the fleet to notice.

    `safety-regressed` is the member that proves the safety axis can veto a candidate winning
    on quality. If it still refuses with false_abstain_rate pinned to 0.0, it was refusing for
    some other reason and the member is decorative.
    """
    import recall.eval.promotion.aggregate as aggregate

    monkeypatch.setattr(aggregate, "false_abstain_rate", lambda *args, **kwargs: 0.0)

    member = next(m for m in SURFACE_B if m.name == "safety-regressed")
    decision, _document = run_surface_b(member)

    assert not any("false abstention regresses" in f for f in decision.failures), (
        "false_abstain_rate stubbed to 0.0 still produced a false-abstention failure, so "
        "safety-regressed is not actually driven by that metric"
    )
    assert decision.promoted, (
        "with the safety axis disabled this candidate wins on quality and should now be "
        f"promoted; it still failed on {decision.failures}"
    )


def run_surface_c(member: FleetMember, tmp_path) -> dict[str, float]:
    """Drive one SURFACE_C member through the REAL `run_trust_eval` and return its published
    fields.

    `run_trust_eval` reads `queries_path` off disk regardless of `store_factory` (Part A's seam
    only replaces how the STORE is built, not the query set), so the fixture's queries are
    written to `tmp_path` here rather than passed in-process.
    """
    fixture = member.build()
    embedder = ScriptedEmbedder()
    store = QueryKeyedTrustStore(
        embedder, fixture.script, supersession_edges=fixture.supersession_edges
    )
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(json.dumps(fixture.queries), encoding="utf-8")
    results = run_trust_eval(
        "unused",  # never read: store_factory replaces the DSN-driven throwaway store entirely
        [embedder],
        queries_path=queries_path,
        touch_stale=False,  # this store's chunks carry no indexed_at for a touch to move
        store_factory=lambda emb: nullcontext(store),
    )
    assert len(results) == 1
    r = results[0]
    return {
        "str_baseline": r.str_baseline,
        "str_recency": r.str_recency,
        "str_trust": r.str_trust,
        "trust_coverage": r.trust_coverage,
        "successor_acc": r.successor_acc,
        "abstain_acc": r.abstain_acc,
        "mrr_answerable_baseline": r.mrr_answerable_baseline,
        "mrr_answerable_trust": r.mrr_answerable_trust,
    }


@pytest.mark.parametrize("member", SURFACE_C, ids=lambda m: m.name)
def test_surface_c_member_reports_its_closed_form(member: FleetMember, tmp_path):
    actual = run_surface_c(member, tmp_path)
    for field, expected in member.expected.items():
        # nan_ok=True: abstain_acc on trust-misses-unscripted-supersession is NaN BY
        # CONSTRUCTION (no expect=="abstain" query in that fixture), and that NaN is itself part
        # of the asserted closed form, not a skipped comparison.
        assert actual[field] == pytest.approx(expected, abs=1e-9, nan_ok=True), (
            f"{member.name} ({member.defect}): {field} should be {expected} by construction, "
            f"got {actual[field]}. Investigate run_trust_eval before editing this expectation."
        )


def test_surface_c_has_two_members():
    assert len(SURFACE_C) == 2


@pytest.mark.parametrize("member", SURFACE_C, ids=lambda m: m.name)
def test_surface_c_member_is_not_vacuous(member: FleetMember):
    twin = SURFACE_C[0]
    if member is twin:
        return
    assert any(
        member.expected[field] != twin.expected.get(field) for field in member.expected
    ), f"{member.name} is indistinguishable from the clean twin on every field it declares"


def test_the_fleet_detects_a_broken_supersession_resolver(monkeypatch, tmp_path):
    """Mutate the code SURFACE_C's supersession detection depends on and require it to notice.

    `trust-catches-scripted-supersession` earns its place only if the trust layer's ACTUAL
    supersession resolution — not merely a score comparison — is what keeps the scripted stale
    hit out of `str_trust`. Stubbing `resolve_successor` to always return `None` removes exactly
    that mechanism, with the edge still scripted, so a str_trust that does not move proves the
    member was never reading it.
    """
    import recall.trust as trust

    monkeypatch.setattr(trust, "resolve_successor", lambda *args, **kwargs: None)

    member = next(m for m in SURFACE_C if m.name == "trust-catches-scripted-supersession")
    actual = run_surface_c(member, tmp_path)

    assert actual["str_trust"] != pytest.approx(member.expected["str_trust"], abs=1e-9), (
        "resolve_successor stubbed to always return None did not change str_trust, so "
        "trust-catches-scripted-supersession is not actually driven by supersession detection"
    )


def run_surface_d(member: FleetMember, tmp_path) -> dict[str, dict[str, float]]:
    """Drive one SURFACE_D member through the REAL `run_nearmiss_eval` and return every arm's
    published fields, keyed by `recall.eval.harness` ARM constant."""
    fixture = member.build()
    embedder = ScriptedEmbedder()
    store = QueryKeyedTrustStore(embedder, fixture.script)
    queries_path = tmp_path / "queries.json"
    queries_path.write_text(json.dumps(fixture.queries), encoding="utf-8")
    nearmiss_path = tmp_path / "near_miss.json"
    nearmiss_path.write_text(json.dumps(fixture.nearmiss), encoding="utf-8")
    results = run_nearmiss_eval(
        "unused",
        [embedder],
        AlwaysEntailJudge(),
        queries_path=queries_path,
        nearmiss_path=nearmiss_path,
        store_factory=lambda emb: nullcontext(store),
    )
    by_arm = {r.arm: r for r in results}
    assert set(by_arm) == {ARM_THRESHOLD, ARM_STACKED, ARM_ENTAIL_ONLY}
    return {
        arm: {
            "nearmiss_fcr": row.nearmiss_fcr,
            "gap_fcr": row.gap_fcr,
            "false_abstain": row.false_abstain,
            "mrr_answerable": row.mrr_answerable,
        }
        for arm, row in by_arm.items()
    }


@pytest.mark.parametrize("member", SURFACE_D, ids=lambda m: m.name)
def test_surface_d_member_reports_its_closed_form(member: FleetMember, tmp_path):
    actual = run_surface_d(member, tmp_path)
    for arm, fields in member.expected.items():
        for field, expected in fields.items():
            assert actual[arm][field] == pytest.approx(expected, abs=1e-9), (
                f"{member.name} ({member.defect}) [{arm}]: {field} should be {expected} by "
                f"construction, got {actual[arm][field]}. Investigate run_nearmiss_eval before "
                f"editing this expectation."
            )


def test_surface_d_has_two_members():
    assert len(SURFACE_D) == 2


@pytest.mark.parametrize("member", SURFACE_D, ids=lambda m: m.name)
def test_surface_d_member_is_not_vacuous(member: FleetMember):
    twin = SURFACE_D[0]
    if member is twin:
        return
    assert member.expected != twin.expected, (
        f"{member.name} is indistinguishable from the clean twin on every field it declares"
    )


def test_the_fleet_detects_a_broken_near_miss_metric(monkeypatch, tmp_path):
    """Mutate the code under test and require SURFACE_D to notice.

    Modelled on `test_the_fleet_detects_a_broken_recall_metric`: if this passes, the SURFACE_D
    members are decorative and THAT is the finding.
    """
    import recall.eval.harness as harness

    monkeypatch.setattr(harness, "near_miss_false_confident_rate", lambda *args, **kwargs: 0.0)

    member = next(
        m for m in SURFACE_D if m.name == "nearmiss-above-threshold-fools-the-cosine-guard"
    )
    actual = run_surface_d(member, tmp_path)

    expected = member.expected[ARM_THRESHOLD]["nearmiss_fcr"]
    assert actual[ARM_THRESHOLD]["nearmiss_fcr"] != pytest.approx(expected, abs=1e-9), (
        "near_miss_false_confident_rate stubbed to 0.0 did not change nearmiss_fcr, so the "
        "fleet is not actually reading this metric"
    )


def test_the_fleet_declares_what_it_does_not_cover():
    """Roll every member's blind spot into one visible list.

    Buried per-member, a `does_not_catch` is a comment. Printed together, it is the answer to
    "what does this fleet NOT certify", which is the question a reader actually has. Run with
    `-s` to see it.
    """
    members = SURFACE_A + SURFACE_B + SURFACE_C + SURFACE_D
    assert len(members) == 18

    print("\nThe eval calibration fleet does NOT catch:")
    for member in members:
        print(f"  - {member.name}: {member.does_not_catch}")
    print(
        "\nBeyond the members: indexing and embedding are stubbed, so nothing here speaks to "
        "real retrieval quality. run_trust_eval and run_nearmiss_eval are now REACHED, via an "
        "injected store_factory (SURFACE_C, SURFACE_D) — but only as far as a scripted store "
        "can honestly drive them:\n"
        "  - str_recency's real 'prefer the newest timestamp' tie-break is unexercised: every "
        "scripted hit carries indexed_at=None, so recency degenerates to 'first hit in the "
        "confident pool'; touch_stale is also passed False in every member, so the re-sync "
        "simulation run_trust_eval performs by default is never taken.\n"
        "  - The entailment DEMOTION mechanism (verdict ok -> not_entailed) is unexercised: "
        "SURFACE_D's judge (AlwaysEntailJudge) never disagrees, so ARM_STACKED is only shown "
        "to be inert when the judge agrees, never shown to demote a confident near-miss the "
        "way a real judge is supposed to.\n"
        "  - The Wilson-CI fields (*_ci) and n_* sample counts TrustEvalResult publishes are "
        "pure functions of flags SURFACE_C already drives to differing values, and are not "
        "independently asserted here.\n"
        "  - entail_latency_ms_mean and query_latency_ms_mean are wall-clock timings with no "
        "closed form to derive, so no member asserts on them.\n"
        "LOCOMO, BEAM, MTRAG and the ladder are out of scope. Generation and judging are "
        "untouched: the 2026-08-09 conditioning bug lived in an upstream IBM scorer and this "
        "fleet closes its CLASS, not that instance."
    )
