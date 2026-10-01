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
