"""`scripts/aml_w0_r1_coding.py`: the parts of W0 R1-Coding that decide a number, tested offline.

Each test states the invariant and the mutation it was watched to fail on (the red proof).
Nothing here reranks, embeds, or reaches a network.
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aml_w0_embedding_compare as w0  # noqa: E402
import aml_w0_r1_coding as r1c  # noqa: E402


def _row(sessions: list[str]) -> dict[str, object]:
    return {"served_facts": {"kept": [{"session_id": s, "content": f"text of {s}"} for s in sessions]}}


def _question(task: str, gold: str) -> w0.Question:
    return w0.Question(task, "coding", "u", "fix the bug", frozenset({gold}))


def test_reranked_arms_reorder_their_own_base_arms_items() -> None:
    """Invariant: RA shows A's items and RP shows P's, each in the recorded rerank order; A and P
    show their served order unchanged.

    Red proof: changing `base_of` to return ``"A"`` for every arm makes P show A's items in the
    rerank order (``['a3', 'a1', 'a2']``) and fails the P assertion.
    """
    rows = {"A": {"t": _row(["a1", "a2", "a3"])}, "P": {"t": _row(["p1", "p2", "p3"])}}
    orders = {("A", "t"): [2, 0, 1], ("P", "t"): [1, 2, 0]}
    sessions = {arm: [i["session_id"] for i in r1c.arm_items(arm, rows, orders, "t")] for arm in r1c.ARMS}
    assert sessions["A"] == ["a1", "a2", "a3"]
    assert sessions["P"] == ["p1", "p2", "p3"]
    assert sessions["RA"] == ["a3", "a1", "a2"]
    assert sessions["RP"] == ["p2", "p3", "p1"]


def test_score_reads_mrr_from_the_reordered_lists_and_pairs_by_task() -> None:
    """Invariant: a reranker that lifts the gold session from rank 3 to rank 1 on P's list moves
    RP's MRR to 1.0 while P stays at 1/3, and the contrasts pair the same tasks.

    Red proof: changing `arm_items` to return ``list(items)`` for every arm (ignoring the rerank
    order) leaves RP at 1/3 and fails the RP assertion.
    """
    rows = {
        "A": {"t1": _row(["g1", "x", "y"]), "t2": _row(["g2", "x", "y"])},
        "P": {"t1": _row(["x", "y", "g1"]), "t2": _row(["x", "y", "g2"])},
    }
    orders = {(arm, t): ([2, 0, 1] if arm == "P" else [0, 1, 2]) for arm in ("A", "P") for t in ("t1", "t2")}
    questions = {"t1": _question("t1", "g1"), "t2": _question("t2", "g2")}
    result = r1c.score(rows, orders, questions)
    assert result["n"] == 2
    assert result["means"]["P"]["rr"] == round(1 / 3, 4)
    assert result["means"]["RP"]["rr"] == 1.0
    assert result["contrasts"]["RP-vs-RA"]["delta"] == 0.0


def test_the_blend_fuses_served_and_rerank_ranks_with_the_rerank_weighted() -> None:
    """Invariant: with the rerank order [2, 0, 1] over three served items, the RRF blend (k 60) at
    w 1 keeps item 0 first ([0, 2, 1]) and at w 2 lets the rerank's first choice win ([2, 0, 1]).

    Red proof: changing `blend_order` to ignore ``weight`` (``1 / (RRF_K + rerank_rank[i])``) gives
    [0, 2, 1] at w 2 and fails the second assertion.
    """
    assert r1c.blend_order(3, [2, 0, 1], 1.0) == [0, 2, 1]
    assert r1c.blend_order(3, [2, 0, 1], 2.0) == [2, 0, 1]


def test_blend_arms_reorder_their_base_arms_items_by_the_blend_not_the_rerank() -> None:
    """Invariant: BA and BP(2) show their base arm's items in the blend order, which differs from
    the pure rerank order (RA) on the same items.

    Red proof: changing ``if arm.startswith("B"):`` in `arm_items` to ``if False:`` makes BA show
    the pure rerank order ``['a3', 'a1', 'a2']`` and fails the BA assertion.
    """
    rows = {"A": {"t": _row(["a1", "a2", "a3"])}, "P": {"t": _row(["p1", "p2", "p3"])}}
    orders = {("A", "t"): [2, 0, 1], ("P", "t"): [2, 0, 1]}
    sessions = {arm: [i["session_id"] for i in r1c.arm_items(arm, rows, orders, "t")] for arm in ("RA", "BA", "BP(2)")}
    assert sessions["RA"] == ["a3", "a1", "a2"]
    assert sessions["BA"] == ["a1", "a3", "a2"]
    assert sessions["BP(2)"] == ["p3", "p1", "p2"]


def test_the_blend_verdict_judges_ba_and_bp_against_c9_as_served() -> None:
    """Invariant: the blend's two rules read BA minus A and BP minus A, each lower bound above -0.05.

    Red proof: changing `blend_verdict` to read ``contrasts["BP-vs-BA"]`` for the second rule passes
    the first case, whose BP-vs-BA bound is -0.01, and fails the second assertion.
    """
    contrasts = {"BA-vs-A": {"ci95_low": -0.02}, "BP-vs-A": {"ci95_low": -0.20}, "BP-vs-BA": {"ci95_low": -0.01}}
    result = r1c.blend_verdict(contrasts)
    assert result["blend_does_not_harm_c9"] is True
    assert result["proxy_blend_noninferior_to_c9"] is False


def test_the_verdict_judges_rp_against_reranked_c9_at_the_preregistered_bar() -> None:
    """Invariant: RP is non-inferior only when RP minus RA (not RP minus A) has a lower bound above
    -0.05; -0.05 exactly fails.

    Red proof: changing `verdict` to read ``contrasts["RP-vs-A"]`` passes the first case, whose
    RP-vs-A bound is -0.01, and fails the first assertion.
    """
    def contrasts(rp_ra: float, rp_a: float) -> dict[str, dict[str, float]]:
        return {"RP-vs-RA": {"ci95_low": rp_ra}, "RP-vs-A": {"ci95_low": rp_a}}

    assert r1c.verdict(contrasts(-0.20, -0.01))["RP_noninferior_to_RA"] is False
    assert r1c.verdict(contrasts(-0.04, -0.30))["RP_noninferior_to_RA"] is True
    assert r1c.verdict(contrasts(-0.05, 0.0))["RP_noninferior_to_RA"] is False
