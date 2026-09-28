"""`scripts/aml_w0_embedding_compare.py`: the parts of W0 that decide a number, tested offline.

Each test states the invariant, and the mutation of the harness it was watched to fail on (the
red proof). Nothing here builds C9, embeds, or reaches a network.
"""

from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import aml_w0_embedding_compare as w0  # noqa: E402

Q = w0.Question(
    question_id="q1",
    category="1",
    user_id="u",
    query="when?",
    gold_sessions=frozenset({"s2"}),
    gold_turns=("Caroline: I went to the lake with Mel on Sunday",),
)


def _items(sessions: list[str], texts: list[str] | None = None) -> list[dict[str, object]]:
    texts = texts or ["filler words here"] * len(sessions)
    return [{"session_id": s, "content": t} for s, t in zip(sessions, texts, strict=True)]


def test_session_hits_and_reciprocal_rank_follow_the_first_gold_session() -> None:
    """Invariant: the gold session at rank 12 counts at depth 20 and 100, not 5 or 10, and the
    reciprocal rank is 1/12.

    Red proof: changing ``sessions[:depth]`` to ``sessions[: depth + 5]`` in `score_items` makes
    session_hit@10 count rank 12 and fails the first assertion.
    """
    row = w0.score_items(_items(["s1"] * 11 + ["s2"]), Q)
    assert row["session_hit@10"] == 0.0
    assert row["session_hit@20"] == 1.0 and row["session_hit@100"] == 1.0
    assert row["rr"] == pytest.approx(1 / 12)


def test_turn_hits_need_the_labelled_turn_verbatim_in_a_returned_item() -> None:
    """Invariant: a turn hit needs the evidence turn's text inside an item; the right session
    with other text is a session hit only.

    Red proof: making `score_items` count a turn hit on any item from a gold session fails the
    second assertion.
    """
    hit = w0.score_items(_items(["s2"], ["[2023-05-08 13:56 UTC] Caroline: I went to the lake with Mel on Sunday and"]), Q)
    assert hit["turn_hit@5"] == 1.0
    miss = w0.score_items(_items(["s2"], ["Caroline: something else entirely"]), Q)
    assert miss["session_hit@5"] == 1.0 and miss["turn_hit@5"] == 0.0


def test_the_collect_refuses_an_environment_that_would_not_measure_the_embedder_alone() -> None:
    """Invariant: the compiler and T-1 must be off, and a query cache must be off for the
    no-instruction pass.

    Red proof: deleting the T-1 check in `check_environment` lets the second call pass and fails
    its ``pytest.raises``.
    """
    ok = {"RECALL_AML_COMPILER": "0", "RECALL_AML_RESOLVE_RELATIVE_TIMES": "0"}
    w0.check_environment(ok, no_instruction_pass=True, allow_compile=False)
    with pytest.raises(SystemExit, match="COMPILER"):
        w0.check_environment({**ok, "RECALL_AML_COMPILER": "1"}, no_instruction_pass=False, allow_compile=False)
    with pytest.raises(SystemExit, match="RELATIVE_TIMES"):
        w0.check_environment({"RECALL_AML_COMPILER": "0"}, no_instruction_pass=False, allow_compile=False)
    with pytest.raises(SystemExit, match="EMBED_CACHE"):
        w0.check_environment({**ok, "RECALL_AML_EMBED_CACHE_PATH": "/tmp/c.sqlite"},
                             no_instruction_pass=True, allow_compile=False)


def test_the_no_instruction_pass_swaps_the_query_encoder_and_restores_it() -> None:
    """Invariant: inside the context every DashScope query drops its instruction; after it, the
    served encoder is back, even when the body raised.

    Red proof: removing the ``finally`` restore in `without_query_instruction` leaves the class
    patched after the raise and fails the last assertion.
    """
    from recall.dashscope import DashScopeEmbedder

    served = DashScopeEmbedder.embed_query
    with pytest.raises(RuntimeError):
        with w0.without_query_instruction():
            assert DashScopeEmbedder.embed_query is DashScopeEmbedder.embed_query_without_instruction
            raise RuntimeError("boom")
    assert DashScopeEmbedder.embed_query is served


def _run(arm: str, dataset: str, values: list[float], *, metric: str, no_instruction: list[float] | None = None) -> dict:
    rows = []
    for i, value in enumerate(values):
        row = {"question_id": f"q{i}", "served": {metric: value}}
        if no_instruction is not None:
            row["no_instruction"] = {metric: no_instruction[i]}
        rows.append(row)
    return {"arm": arm, "dataset": dataset, "rows": rows, "no_instruction_pass": no_instruction is not None}


def test_the_report_applies_the_preregistered_margin_to_the_lower_bound() -> None:
    """Invariant: an arm passes only when its interval's lower bound is above minus the margin;
    a clear 30-point loss fails, an identical arm passes, and B0 is reported from the second pass.

    Red proof: comparing ``interval["delta"]`` instead of ``interval["ci95_low"]`` in `passes`
    makes the small, uncertain loss below pass and fails the `B-vs-A` assertion.
    """
    control = [1.0] * 60 + [0.0] * 40
    # -1 point on average, from 10 questions lost and 9 gained: the interval is wide.
    small_loss = list(control)
    for i in range(10):
        small_loss[i] = 0.0
    for i in range(60, 69):
        small_loss[i] = 1.0
    big_loss = [0.0] * 100
    runs = [
        _run("A", "locomo", control, metric="turn_hit@10"),
        _run("B", "locomo", small_loss, metric="turn_hit@10", no_instruction=big_loss),
        _run("C", "locomo", control, metric="turn_hit@10"),
    ]
    out = w0.report(runs, "A", margin_points=2.0, margin_mrr=0.05)
    assert out["locomo:C-vs-A"]["passes"] is True
    assert out["locomo:B0-vs-A"]["passes"] is False
    assert out["locomo:B-vs-A"]["delta"] > -2.0
    assert out["locomo:B-vs-A"]["passes"] is False


def test_coding_is_judged_on_mrr_with_its_own_margin() -> None:
    """Invariant: the coding comparison uses reciprocal rank, unscaled, against the MRR margin.

    Red proof: scaling ``rr`` by 100 in `report` turns the 0.01 MRR loss into -1 against a 0.05
    margin, which fails the first assertion.
    """
    control = [1.0, 0.5, 1.0, 1.0] * 25
    slightly = [1.0, 0.5, 1.0, 0.96] * 25  # -0.01 MRR on every question, no variance in sign
    worse = [0.96, 0.46, 0.96, 0.96] * 25  # -0.04 MRR
    out = w0.report(
        [_run("A", "coding", control, metric="rr"), _run("B", "coding", slightly, metric="rr"),
         _run("D", "coding", worse, metric="rr")],
        "A", margin_points=2.0, margin_mrr=0.05,
    )
    assert out["coding:B-vs-A"]["metric"] == "MRR" and out["coding:B-vs-A"]["passes"] is True
    assert out["coding:D-vs-A"]["passes"] is True
    assert out["coding:D-vs-A"]["delta"] == pytest.approx(-0.04)
