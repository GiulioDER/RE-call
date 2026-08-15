"""The eval calibration fleet: see docs/EVAL_CALIBRATION_FLEET_DESIGN.md."""
from __future__ import annotations

import pytest

from recall.eval.harness import _score_config
from recall.eval.promotion.aggregate import decide
from tests.fleet.members import DEPTH, SURFACE_A, SURFACE_B, FleetMember, GateExpectation, _rows
from tests.fleet.scripted import QueryKeyedStore, ScriptedEmbedder


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


def test_the_fleet_declares_what_it_does_not_cover():
    """Roll every member's blind spot into one visible list.

    Buried per-member, a `does_not_catch` is a comment. Printed together, it is the answer to
    "what does this fleet NOT certify", which is the question a reader actually has. Run with
    `-s` to see it.
    """
    members = SURFACE_A + SURFACE_B
    assert len(members) == 14

    print("\nThe eval calibration fleet does NOT catch:")
    for member in members:
        print(f"  - {member.name}: {member.does_not_catch}")
    print(
        "\nBeyond the members: indexing and embedding are stubbed, so nothing here speaks to "
        "real retrieval quality. run_trust_eval and run_nearmiss_eval are unreached (both "
        "build a real store from a dsn internally). LOCOMO, BEAM, MTRAG and the ladder are "
        "out of scope. Generation and judging are untouched: the 2026-08-09 conditioning bug "
        "lived in an upstream IBM scorer and this fleet closes its CLASS, not that instance."
    )
