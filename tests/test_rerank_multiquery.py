"""The rerank stage's guards must be able to FAIL, and its claims must be measured.

Two successive audits found the same defect here in two disguises. The original R@100 invariance
check compared `set(raw)` against `set(reranked)`; the first fix compared raw against reranked
R@100. Both are unsatisfiable at depth <= the cutoff, because `rerank_order` returns a permutation
and `recall_at` takes `set(ranked[:EVAL_K])`, so the values are bit-identical whatever the scores
say. Each version wrote `r@100_invariant: true` as though it had been verified.

The property that actually carries the claim is the ranking DEPTH, which is read off directly. The
question the comparison was standing in for, "did the scores reorder anything at all", is now a
separate, deterministic gate.

These tests are written to survive mutation: each fails if the behaviour it names is removed.
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


def write_fixture(root: Path, out: Path, depth: int, whole_depth: int | None = None) -> None:
    """A four-domain fixture whose rankings are `depth` documents deep.

    One gold document is planted at the END of each ranking and the scores increase with rank, so
    reranking reverses the order and moves that gold to the front. It crosses the R@100 cutoff
    exactly when the ranking is deeper than the cutoff, which is the distinction under test.

    `whole_depth` writes a DEEPER ranking to `rankings_whole_pool.json`. Giving the two files
    identical content made every test pass whether or not `--whole-pool` selected the right one,
    which a mutation check exposed.
    """
    qrels_lines = {domain: ["query-id\tcorpus-id\tscore"] for domain in DOMAINS}
    rankings: dict[str, dict[str, list[str]]] = {name: {} for name in RERANK_ARMS}
    deep: dict[str, dict[str, list[str]]] = {name: {} for name in RERANK_ARMS}
    scores: list[dict] = []
    seen: set[tuple[str, str]] = set()

    def add_score(task_id: str, doc: str, rank: int) -> None:
        if (task_id, doc) not in seen:
            seen.add((task_id, doc))
            scores.append({"qid": task_id, "doc_id": doc, "score": float(rank)})

    for index, domain in enumerate(DOMAINS):
        task_id = f"task{index}"
        gold = f"gold{index}"
        qrels_lines[domain].append(f"{task_id}\t{gold}\t1")
        for name in RERANK_ARMS:
            shallow = [f"{name}_d{j}" for j in range(depth - 1)] + [gold]
            rankings[name][task_id] = shallow
            for rank, doc in enumerate(shallow):
                add_score(task_id, doc, rank)
            if whole_depth is not None:
                padded = ([f"{name}_d{j}" for j in range(depth - 1)]
                          + [f"{name}_pad{j}" for j in range(whole_depth - depth)]
                          + [gold])
                deep[name][task_id] = padded
                for rank, doc in enumerate(padded):
                    add_score(task_id, doc, rank)

    for domain in DOMAINS:
        path = root / "mtrag-human" / "retrieval_tasks" / domain / "qrels"
        path.mkdir(parents=True, exist_ok=True)
        (path / "dev.tsv").write_text("\n".join(qrels_lines[domain]) + "\n", encoding="utf-8")

    out.mkdir(parents=True, exist_ok=True)
    (out / "rankings_equal_width.json").write_text(json.dumps(rankings), encoding="utf-8")
    if whole_depth is not None:
        (out / "rankings_whole_pool.json").write_text(json.dumps(deep), encoding="utf-8")
    # The raw run's own file, so the documented fallback path stays exercisable.
    (out / "rankings.json").write_text(json.dumps(rankings), encoding="utf-8")
    with (out / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in scores:
            handle.write(json.dumps(row) + "\n")


def run_apply(root: Path, out: Path, whole_pool: bool, output_dir: Path | None = None) -> dict:
    cmd_apply(argparse.Namespace(
        mq_dir=out, output_dir=output_dir or out, mtrag_root=root,
        scores=out / "scores.jsonl", whole_pool=whole_pool,
    ))
    name = "rerank_decision_whole_pool.json" if whole_pool else "rerank_decision.json"
    return json.loads(((output_dir or out) / name).read_text(encoding="utf-8"))


def test_a_capped_ranking_is_reported_invariant_on_its_depth_not_on_its_metric(
    tmp_path: Path,
) -> None:
    """The basis must be the DEPTH, because the metric comparison cannot fail.

    An implementation that went back to inferring invariance from raw == reranked would still set
    the flag True here, so the flag alone cannot distinguish the two versions. The basis string is
    what pins which property was actually consulted.
    """
    root, out = tmp_path / "root", tmp_path / "out"
    write_fixture(root, out, depth=EVAL_K)

    decision = run_apply(root, out, whole_pool=False)

    assert decision["r@100_invariant_for_every_arm"] is True
    assert decision["some_arm_ranks_deeper_than_the_cutoff"] is False
    for arm in decision["arms"]:
        assert arm["max_ranking_length"] == EVAL_K
        assert arm["r@100_invariant"] is True
        assert "guaranteed" in arm["r@100_invariance_basis"]
        assert arm["queries_reordered"] == len(DOMAINS)


def test_a_deeper_pool_is_not_reported_as_invariant(tmp_path: Path) -> None:
    """The state neither earlier check could detect.

    Deeper than the cutoff, reranking moves the planted gold across the R@100 boundary. The flag
    must come out False and the numbers must show the movement.
    """
    root, out = tmp_path / "root", tmp_path / "out"
    write_fixture(root, out, depth=EVAL_K, whole_depth=EVAL_K + 60)

    decision = run_apply(root, out, whole_pool=True)

    assert decision["some_arm_ranks_deeper_than_the_cutoff"] is True
    assert decision["r@100_invariant_for_every_arm"] is False
    assert decision["whole_pool"] is True
    assert "CONFOUNDED" in decision["design"]
    for arm in decision["arms"]:
        assert arm["max_ranking_length"] == EVAL_K + 60
        assert "NOT invariant" in arm["r@100_invariance_basis"]
        assert arm["raw_R@100"] == 0.0
        assert arm["reranked_R@100"] == 1.0


def test_scores_that_reorder_nothing_are_refused(tmp_path: Path) -> None:
    """The deterministic gate that replaced two unsatisfiable ones.

    "R@100 moved" could not evidence that reranking did anything: capped it never moves, deeper it
    may legitimately not move. "The order changed on at least one query" is what was wanted. It
    cannot be satisfied by luck, and it cannot fire on real data where a genuine reordering happens
    not to cross the cutoff.
    """
    root, out = tmp_path / "root", tmp_path / "out"
    write_fixture(root, out, depth=EVAL_K)
    rankings = json.loads((out / "rankings_equal_width.json").read_text(encoding="utf-8"))
    with (out / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for per_query in rankings.values():
            for task_id, ranked in per_query.items():
                for doc in ranked:
                    handle.write(json.dumps({"qid": task_id, "doc_id": doc, "score": 1.0}) + "\n")

    with pytest.raises(RuntimeError, match="order of NONE"):
        run_apply(root, out, whole_pool=False)


def test_a_frozen_file_whose_arms_differ_is_refused(tmp_path: Path) -> None:
    """An EXTRA arm is the dangerous one, and it used to pass silently.

    `shared` intersected over every arm present in the file, so an extra arm covering fewer
    queries shrank the population every published number was computed on, with a verdict still
    written and no warning anywhere.
    """
    root, out = tmp_path / "root", tmp_path / "out"
    write_fixture(root, out, depth=EVAL_K)
    rankings = json.loads((out / "rankings_equal_width.json").read_text(encoding="utf-8"))
    rankings["mq_flat6"] = {"task0": ["x", "y"]}
    (out / "rankings_equal_width.json").write_text(json.dumps(rankings), encoding="utf-8")

    with pytest.raises(RuntimeError, match="unexpected"):
        run_apply(root, out, whole_pool=False)


def test_a_truncated_frozen_file_is_refused_rather_than_silently_re_derived(
    tmp_path: Path,
) -> None:
    """An emptied file is not an absent one, and absence must not read as agreement."""
    out = tmp_path / "out"
    out.mkdir()
    (out / "rankings_equal_width.json").write_text('{"mq_last": {', encoding="utf-8")

    with pytest.raises(RuntimeError, match="not valid JSON"):
        frozen_rankings(out, out, whole_pool=False)


def test_the_whole_pool_decision_cannot_overwrite_the_primary(tmp_path: Path) -> None:
    """Both analyses run from the directory holding the frozen rankings.

    Sharing one filename meant the width-confounded secondary replaced the preregistered primary,
    and the survivor looked like a complete result. `rerank_offload` already learned this.
    """
    root, out = tmp_path / "root", tmp_path / "out"
    write_fixture(root, out, depth=EVAL_K, whole_depth=EVAL_K + 60)

    primary = run_apply(root, out, whole_pool=False)
    secondary = run_apply(root, out, whole_pool=True)

    assert (out / "rerank_decision.json").exists()
    assert (out / "rerank_decision_whole_pool.json").exists()
    assert primary["whole_pool"] is False
    assert secondary["whole_pool"] is True
    # Re-read the primary from disk: the point is that it SURVIVED the second run.
    on_disk = json.loads((out / "rerank_decision.json").read_text(encoding="utf-8"))
    assert on_disk["whole_pool"] is False
    assert on_disk["r@100_invariant_for_every_arm"] is True


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

    run_apply(root, out, whole_pool=False, output_dir=fresh)

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
