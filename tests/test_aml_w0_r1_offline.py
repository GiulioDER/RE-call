"""`scripts/aml_w0_r1_offline.py`: the parts of the offline ideas that decide a number, tested offline.

Each test states the invariant and the mutation it was watched to fail on (the red proof).
"""

from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aml_w0_r1_offline as off  # noqa: E402


def test_depth_limited_rerank_reorders_only_the_top_k() -> None:
    """Invariant: D(2) re-orders served items 0 and 1 by the rerank (which prefers 4, then 1, then 0)
    and leaves 2, 3, 4 in served order; D(0) is the served order.

    Red proof: changing `depth_order` to ``[i for i in rerank_order if i <= k]`` lets item 2 into
    the reranked head and fails the D(2) assertion.
    """
    rerank = [4, 1, 3, 0, 2]
    assert off.depth_order(5, rerank, 2) == [1, 0, 2, 3, 4]
    assert off.depth_order(5, rerank, 0) == [0, 1, 2, 3, 4]


def test_session_aggregates_differ_only_where_windows_are_spread() -> None:
    """Invariant: with windows [x, g, g, g, y], ``max`` ranks x first (first appearance) while
    ``sumrr`` and ``top20`` rank g first (three windows), and ``max2`` ranks x first (1 against 1/2
    plus 0.5/3).

    Red proof: changing ``sumrr`` to ``max(1.0 / (RRF_K + x) for x in r)`` makes it rank x first and
    fails the sumrr assertion.
    """
    sessions = ["x", "g", "g", "g", "y"]
    assert off.session_order(sessions, "max") == ["x", "g", "y"]
    assert off.session_order(sessions, "sumrr")[0] == "g"
    assert off.session_order(sessions, "top20")[0] == "g"
    assert off.session_order(sessions, "max2")[0] == "x"


def test_leave_one_out_never_selects_with_the_task_it_scores() -> None:
    """Invariant: V1 wins task 1 only and V2 task 2 only; each task is scored by the variant chosen
    on the OTHER task, so both tasks score 0, although V1 has the higher mean overall.

    Red proof: changing `loo_select` to rank by ``totals[v]`` (including the held-out task) picks V1
    for both and scores [1.0, 0.0], failing the assertion.
    """
    scores, chosen = off.loo_select({"V1": [1.0, 0.0], "V2": [0.0, 0.9]})
    assert scores == [0.0, 0.0]
    assert chosen == ["V2", "V1"]


def test_the_verdicts_apply_each_preregistered_bar_to_its_contrast() -> None:
    """Invariant: helps_proxy needs LOO(P) minus P above 0; harmless_to_c9 needs LOO(A) minus A above
    -0.05; closes_gap needs LOO(P) minus A above -0.05.

    Red proof: changing harmless_to_c9 to read ``LOO(P)-vs-A`` makes family D harmless (-0.01) and
    fails the D assertion.
    """
    contrasts = {}
    for f, (helps, harm, gap) in {"D": (0.01, -0.20, -0.01), "S": (-0.01, -0.04, -0.30), "L": (0.0, -0.05, -0.06)}.items():
        contrasts[f"{f}:LOO(P)-vs-P"] = {"ci95_low": helps}
        contrasts[f"{f}:LOO(A)-vs-A"] = {"ci95_low": harm}
        contrasts[f"{f}:LOO(P)-vs-A"] = {"ci95_low": gap}
    v = off.verdicts(contrasts)
    assert v["D"] == {"helps_proxy": True, "harmless_to_c9": False, "closes_gap": True}
    assert v["S"] == {"helps_proxy": False, "harmless_to_c9": True, "closes_gap": False}
    assert v["L"] == {"helps_proxy": False, "harmless_to_c9": False, "closes_gap": False}


def test_lexical_weight_zero_is_the_served_order_and_weight_moves_bm25_matches_up() -> None:
    """Invariant: L(0) keeps the served order; at L(2) the only item matching the query's words,
    served third, rises to the top (1/63 + 2/61 beats item 0's 1/61 + 2/62).

    Red proof: changing `lexical_order` to drop ``weight`` (``1 / (RRF_K + bm25_rank[i])``) moves
    the match to second at L(0) and fails the first assertion.
    """
    texts = ["alpha beta", "gamma delta", "refactor the parser module", "epsilon"]
    assert off.lexical_order("refactor parser", texts, 0.0) == [0, 1, 2, 3]
    assert off.lexical_order("refactor parser", texts, 2.0)[0] == 2
