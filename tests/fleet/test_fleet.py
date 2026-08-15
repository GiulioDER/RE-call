"""The eval calibration fleet: see docs/EVAL_CALIBRATION_FLEET_DESIGN.md."""
from __future__ import annotations

import pytest

from recall.eval.harness import _score_config
from recall.eval.promotion.aggregate import decide
from tests.fleet.members import SURFACE_A, SURFACE_B, FleetMember
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
    from tests.fleet.members import FleetMember

    with pytest.raises(ValueError, match="does_not_catch"):
        FleetMember(
            name="blank",
            defect="none",
            build=lambda: None,
            expected={},
            does_not_catch="   ",
        )


def run_surface_a(member: FleetMember) -> dict[str, float]:
    """Drive one member through the REAL scoring path and return the published fields."""
    queries, script = member.build()
    if not queries:
        raise ValueError(
            f"{member.name} scored zero questions. A fixture that measured nothing must not "
            f"read like a defect that was absent; `arm_check.EmptySampleError` sets the same "
            f"precedent for an empty comparison."
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
        with pytest.raises(expectation.raises):
            run_surface_b(member)
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
