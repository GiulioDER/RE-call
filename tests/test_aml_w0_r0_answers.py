"""W0 R0: the merge, the per-arm renderer, the verdict and the retrieval-reproduction check. Each
test names its red proof."""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aml_w0_r0_answers as r0  # noqa: E402


def _item(text: str, ident: str) -> dict:
    return {"id": ident, "content": text, "created_at": "2023-05-08T13:56:00+00:00", "source": "s",
            "session_id": "conv-1/session-2", "kind": "raw", "score": 1.0, "render_facts": None}


def _collect(variant: str, rows: list[dict]) -> dict:
    return {"dataset": "locomo", "variant": variant, "rows": rows}


def test_merge_keeps_each_arms_items_and_only_shared_questions() -> None:
    """Invariant: a merged row holds A's items under A and P's under P, only questions both
    collects answered survive, and a question routed differently is counted, not repaired.

    Red proof: storing A's items under P too (``"P": row["items"]`` in `merge`) fails the equality.
    """
    a = _collect("A", [
        {"id": "q1", "category": "1", "route": "context", "items": [_item("alpha", "a1")]},
        {"id": "q2", "category": "2", "route": "code", "items": [_item("beta", "a2")]},
    ])
    p = _collect("P", [
        {"id": "q1", "category": "1", "route": "code", "items": [_item("gamma", "p1")]},
        {"id": "q9", "category": "2", "route": "code", "items": [_item("delta", "p9")]},
    ])
    merged = r0.merge(a, p)
    assert [row["id"] for row in merged["rows"]] == ["q1"]
    by_arm = merged["rows"][0]["items_by_arm"]
    assert [i["id"] for i in by_arm["A"]] == ["a1"] and [i["id"] for i in by_arm["P"]] == ["p1"]
    assert merged["route_mismatches"] == 1


def test_each_arm_renders_its_own_retrieval_as_the_served_c9_does() -> None:
    """Invariant: A and A2 read A's retrieval, P reads P's, all through the served render (the date
    header).

    Red proof: mapping A2 to P's retrieval (`RETRIEVAL`) makes A2 differ from A and fails the first
    equality.
    """
    row = {"id": "q1", "category": "1", "route": "code",
           "items_by_arm": {"A": [_item("the voyage window", "a1")], "P": [_item("the proxy window", "p1")]}}
    a, a2, p = (r0.render(arm, row) for arm in ("A", "A2", "P"))
    assert [i.content for i in a2] == [i.content for i in a]
    assert a[0].content.startswith("[2023-05-08") and "voyage" in a[0].content
    assert "proxy" in p[0].content


def test_the_verdict_applies_the_rule_in_the_order_written() -> None:
    """Invariant: a lower bound above -1.5 means the loss does not reach the answer, even when the
    upper bound is below 0 (then flagged as a small loss); otherwise an upper bound below 0 means it
    does; anything else is inconclusive. A replicate outside two points voids the run.

    Red proof: checking the upper bound first in `verdict` reads [-1.2, -0.1] as "reaches" and
    fails the first equality.
    """
    ok = {"diff_points": 0.4}
    small = r0.verdict({"ci95_points": [-1.2, -0.1]}, ok)
    assert small["reach"] == "does_not_reach_the_answer" and small["small_loss_detected"]
    assert r0.verdict({"ci95_points": [-3.0, -0.5]}, ok)["reach"] == "reaches_the_answer"
    assert r0.verdict({"ci95_points": [-2.0, 1.0]}, ok)["reach"] == "inconclusive"
    assert r0.verdict({"ci95_points": [-1.0, 1.0]}, {"diff_points": -2.5})["void"]


def test_reranked_arms_hold_each_base_arms_items_in_the_recorded_order() -> None:
    """Invariant (R1 stage 2): RA holds A's items and RP holds P's, each in the order R1 stage 1
    recorded for that arm; a question missing either order is dropped.

    Red proof: building RP from A's items in `with_reranked` fails the second equality.
    """
    merged = {"rows": [
        {"id": "q1", "items_by_arm": {"A": [_item("a0", "a0"), _item("a1", "a1")], "P": [_item("p0", "p0"), _item("p1", "p1")]}},
        {"id": "q2", "items_by_arm": {"A": [_item("a", "a")], "P": [_item("p", "p")]}},
    ]}
    out = r0.with_reranked(merged, {("A", "q1"): [1, 0], ("P", "q1"): [1, 0], ("A", "q2"): [0]})
    assert [row["id"] for row in out["rows"]] == ["q1"]
    by_arm = out["rows"][0]["items_by_arm"]
    assert [i["id"] for i in by_arm["RA"]] == ["a1", "a0"] and [i["id"] for i in by_arm["RP"]] == ["p1", "p0"]


def test_reranked_arms_render_their_own_lists() -> None:
    """Invariant: RP renders RP's list and RA renders RA's, not their base arms' stored order.

    Red proof: mapping RP back to P in `RETRIEVAL` renders P's order and fails the equality.
    """
    row = {"id": "q1", "category": "1", "route": "code", "items_by_arm": {
        "A": [_item("a first", "a0")], "P": [_item("p first", "p0"), _item("p second", "p1")],
        "RA": [_item("a first", "a0")], "RP": [_item("p second", "p1"), _item("p first", "p0")],
    }}
    assert "p second" in r0.render("RP", row)[0].content


def test_stage2_verdicts_use_the_pre_registered_bars() -> None:
    """Invariant: RP recovers the proxy when RP minus A's lower bound is above -1.5; RA improves C9
    only when RA minus A's lower bound is above 0.

    Red proof: judging RA against -1.5 in `stage2` marks an RA with one extra right answer (lower
    bound exactly 0.0, not above 0) as an improvement and fails the second assertion.
    """
    # 40 questions: A right on 30; RP right on the same 30; RA right on 31 (one extra).
    labels = {f"q{i}": {"A": i < 30, "P": i < 25, "RP": i < 30, "RA": i < 31} for i in range(40)}
    out = r0.stage2(labels)
    assert out["verdict"]["RP_recovers_the_proxy"] is True
    assert out["verdict"]["RA_improves_c9"] is False


def test_reproduction_compares_turn_hit_at_10_on_questions_with_gold_turns() -> None:
    """Invariant: each arm's R0 turn_hit@10, scored on its stored items, is compared with the same
    arm's W0 turn_hit@10 over the draw's questions that carry gold turns only.

    Red proof: reading W0's ``turn_hit@5`` instead of ``turn_hit@10`` in `reproduction` makes W0's
    figure 0 and fails the equality.
    """
    from aml_w0_embedding_compare import Question

    gold = "I adopted a golden retriever named Max last week"
    questions = {
        "q1": Question("q1", "1", "u", "dog?", frozenset({"s"}), (gold,)),
        "q2": Question("q2", "5", "u", "none?", frozenset({"s"}), ()),
    }
    r0_rows = {"q1": {"items": [_item(gold, "x")]}, "q2": {"items": [_item("nothing", "y")]}}
    w0_rows = {"q1": {"served": {"turn_hit@10": 1.0, "turn_hit@5": 0.0}}, "q2": {"served": {}}}
    result = r0.reproduction({"A": r0_rows, "P": r0_rows}, {"A": w0_rows, "P": w0_rows}, questions, ["q1", "q2"])
    assert result["A"] == {"n": 1, "r0": 100.0, "w0": 100.0, "ok": True}
