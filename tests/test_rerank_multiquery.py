"""The rerank stage's guards must be able to FAIL, and its claims must be measured.

A bug audit found that this module's original R@100 invariance check compared `set(raw)` against
`set(reranked)`. `rerank_order` returns a permutation of its input by construction, so that check
was unsatisfiable: a gate that cannot fire, writing `r@100_invariant: true` into the decision file
as though it had been verified. Under `--whole-pool` the rankings run deeper than the cutoff and
R@100 genuinely moves, which is exactly the state it claimed to rule out.

These tests pin the replacement, which derives the flag from the measured numbers.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from benchmarks.mtrag.multiquery import EVAL_K
from benchmarks.mtrag.rerank_multiquery import (
    RERANK_ARMS,
    cmd_apply,
    frozen_rankings,
    metric_fn,
)
from benchmarks.mtrag.run import DOMAINS


def write_fixture(root: Path, out: Path, depth: int) -> None:
    """A four-domain fixture whose rankings are `depth` documents deep.

    One gold document is planted at the very END of each ranking, so a reranker that reorders can
    push it past the R@100 cutoff when `depth > EVAL_K` and cannot when `depth <= EVAL_K`. That is
    the whole distinction under test.
    """
    qrels_lines = {domain: ["query-id\tcorpus-id\tscore"] for domain in DOMAINS}
    rankings: dict[str, dict[str, list[str]]] = {name: {} for name in RERANK_ARMS}
    scores = []
    for index, domain in enumerate(DOMAINS):
        task_id = f"task{index}"
        qrels_lines[domain].append(f"{task_id}\tgold{index}\t1")
        for name in RERANK_ARMS:
            ranked = [f"{name}_d{j}" for j in range(depth - 1)] + [f"gold{index}"]
            rankings[name][task_id] = ranked
            for rank, doc in enumerate(ranked):
                # Strictly increasing score, so reranking REVERSES the ranking. The planted gold
                # moves from last place to first, which changes R@100 only when depth > EVAL_K.
                scores.append({"qid": task_id, "doc_id": doc, "score": float(rank)})
    for domain in DOMAINS:
        path = root / "mtrag-human" / "retrieval_tasks" / domain / "qrels"
        path.mkdir(parents=True, exist_ok=True)
        (path / "dev.tsv").write_text("\n".join(qrels_lines[domain]) + "\n", encoding="utf-8")
    out.mkdir(parents=True, exist_ok=True)
    (out / "rankings_equal_width.json").write_text(json.dumps(rankings), encoding="utf-8")
    (out / "rankings_whole_pool.json").write_text(json.dumps(rankings), encoding="utf-8")
    # Also the raw run's own file, so the documented fallback path is exercisable.
    (out / "rankings.json").write_text(json.dumps(rankings), encoding="utf-8")
    with (out / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in scores:
            handle.write(json.dumps(row) + "\n")


def run_apply(root: Path, out: Path, whole_pool: bool) -> dict:
    cmd_apply(argparse.Namespace(
        mq_dir=out, output_dir=out, mtrag_root=root,
        scores=out / "scores.jsonl", whole_pool=whole_pool,
    ))
    return json.loads((out / "rerank_decision.json").read_text(encoding="utf-8"))


def test_r100_is_reported_invariant_only_when_the_ranking_is_capped(tmp_path: Path) -> None:
    """Capped at the cutoff, reordering cannot change which documents are inside it."""
    root, out = tmp_path / "root", tmp_path / "out"
    write_fixture(root, out, depth=EVAL_K)

    decision = run_apply(root, out, whole_pool=False)

    assert decision["r@100_invariant_for_every_arm"] is True
    assert decision["some_arm_ranks_deeper_than_the_cutoff"] is False
    for arm in decision["arms"]:
        assert arm["raw_R@100"] == arm["reranked_R@100"]


def test_a_deeper_pool_is_not_reported_as_invariant(tmp_path: Path) -> None:
    """The state the original check could never detect.

    With rankings deeper than the cutoff, reordering moves a gold document across the R@100
    boundary. The old `set(raw) != set(reranked)` comparison stayed silent here while the decision
    file still claimed invariance; the replacement measures it and reports False.
    """
    root, out = tmp_path / "root", tmp_path / "out"
    write_fixture(root, out, depth=EVAL_K + 60)

    decision = run_apply(root, out, whole_pool=True)

    assert decision["some_arm_ranks_deeper_than_the_cutoff"] is True
    assert decision["r@100_invariant_for_every_arm"] is False
    assert decision["whole_pool"] is True
    assert "CONFOUNDED" in decision["design"]
    # The planted gold sits last raw (outside the cut) and first reranked (inside it).
    for arm in decision["arms"]:
        assert arm["raw_R@100"] == 0.0
        assert arm["reranked_R@100"] == 1.0


def test_apply_creates_its_output_directory_and_says_when_it_falls_back(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Two things at once, because they only co-occur on this path.

    The final write used to crash on a fresh directory AFTER every bootstrap had run, discarding
    minutes of work. And pointing `apply` at a different `--output-dir` than `pairs` used means the
    frozen rankings are not there, so it re-derives; that is allowed, but it must SAY so rather
    than quietly measuring a ranking that may not be the one the scores were produced against.
    """
    root, out = tmp_path / "root", tmp_path / "out"
    write_fixture(root, out, depth=EVAL_K)
    fresh = tmp_path / "fresh"

    cmd_apply(argparse.Namespace(
        mq_dir=out, output_dir=fresh, mtrag_root=root,
        scores=out / "scores.jsonl", whole_pool=False,
    ))

    assert (fresh / "rerank_decision.json").exists()
    assert "frozen_rankings_absent" in capsys.readouterr().out


def test_frozen_rankings_prefers_the_file_the_pairs_stage_wrote(tmp_path: Path) -> None:
    """`apply` must rerank exactly what was scored.

    Re-deriving from the legs dump is silent when it merely REORDERS the same documents: the
    reranked numbers would be unchanged while the raw baselines describe a ranking that was never
    published.
    """
    out = tmp_path / "out"
    out.mkdir()
    frozen = {name: {"t0": ["a", "b"]} for name in RERANK_ARMS}
    (out / "rankings_equal_width.json").write_text(json.dumps(frozen), encoding="utf-8")

    # mq_dir deliberately does not exist: reading it would raise, so a pass proves the frozen
    # file was preferred rather than merely consulted first.
    assert frozen_rankings(out, tmp_path / "absent", whole_pool=False) == frozen


def test_metric_fn_maps_each_declared_name_to_the_right_measure() -> None:
    """A silently mismapped metric would report nDCG under a recall label."""
    ranked, relevant = ["a", "b", "c"], {"c"}

    assert metric_fn("R@5")(ranked, relevant) == 1.0
    assert metric_fn("R@1")(ranked, relevant) == 0.0
    assert metric_fn("nDCG@5")(ranked, relevant) == pytest.approx(0.5, abs=0.01)
