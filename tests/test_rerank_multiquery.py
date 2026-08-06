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
    cmd_validate,
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
    stem = (out / "scores.jsonl").stem
    name = f"rerank_decision__{stem}{'_whole_pool' if whole_pool else ''}.json"
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
        assert "NOT guaranteed" in arm["r@100_invariance_basis"]
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

    assert (out / "rerank_decision__scores.json").exists()
    assert (out / "rerank_decision__scores_whole_pool.json").exists()
    assert primary["whole_pool"] is False
    assert secondary["whole_pool"] is True
    # Re-read the primary from disk: the point is that it SURVIVED the second run.
    on_disk = json.loads((out / "rerank_decision__scores.json").read_text(encoding="utf-8"))
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

    assert (fresh / "rerank_decision__scores.json").exists()
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


def test_depth_not_the_metric_decides_invariance(tmp_path: Path) -> None:
    """The regression test for the whole three-round saga, and the one that was missing.

    Rounds 1 and 2 both inferred invariance from a comparison that cannot fail at depth <= the
    cutoff. Nothing in the suite distinguished that from reading the depth, because no fixture
    produced the state where the two disagree: a ranking DEEPER than the cutoff whose reranking
    permutes only WITHIN the top 100, so R@100 does not move even though invariance is not
    guaranteed.

    Here the depth basis says False and the metric basis would say True. Reverting to either
    earlier version turns this red. It is also exactly the input round 2's raise fired on
    spuriously, so it pins that fix too.
    """
    root, out = tmp_path / "root", tmp_path / "out"
    write_fixture(root, out, depth=EVAL_K)
    rankings = json.loads((out / "rankings_equal_width.json").read_text(encoding="utf-8"))

    deep: dict[str, dict[str, list[str]]] = {}
    rows = []
    for name, per_query in rankings.items():
        deep[name] = {}
        for task_id, ranked in per_query.items():
            gold = ranked[-1]
            # Gold at rank 50, comfortably inside the cut; padding beyond it, outside.
            head = [f"{name}_h{j}" for j in range(50)] + [gold] + [
                f"{name}_h{j}" for j in range(50, EVAL_K - 1)]
            tail = [f"{name}_t{j}" for j in range(60)]
            deep[name][task_id] = head + tail
            # Reverse the top 100 only; the tail keeps its order and never crosses the cut.
            for rank, doc in enumerate(head):
                rows.append({"qid": task_id, "doc_id": doc, "score": float(rank)})
            for rank, doc in enumerate(tail):
                rows.append({"qid": task_id, "doc_id": doc, "score": -1.0 - rank})
    (out / "rankings_whole_pool.json").write_text(json.dumps(deep), encoding="utf-8")
    with (out / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")

    decision = run_apply(root, out, whole_pool=True)

    for arm in decision["arms"]:
        assert arm["max_ranking_length"] == EVAL_K + 60
        assert arm["queries_reordered"] > 0, "the reranking really did permute"
        # The metric did NOT move: an implementation inferring invariance from it says True.
        assert arm["raw_R@100"] == arm["reranked_R@100"]
        assert arm["r@100_delta"] == 0.0
        # The depth says otherwise, and the depth is what carries the claim.
        assert arm["r@100_invariant"] is False
        assert "NOT guaranteed" in arm["r@100_invariance_basis"]
    assert decision["r@100_invariant_for_every_arm"] is False


def test_a_frozen_file_missing_an_arm_is_refused(tmp_path: Path) -> None:
    """The reachable half of the arm-set guard: a payload frozen before an arm was declared."""
    root, out = tmp_path / "root", tmp_path / "out"
    write_fixture(root, out, depth=EVAL_K)
    rankings = json.loads((out / "rankings_equal_width.json").read_text(encoding="utf-8"))
    del rankings[RERANK_ARMS[-1]]
    (out / "rankings_equal_width.json").write_text(json.dumps(rankings), encoding="utf-8")

    with pytest.raises(RuntimeError, match="Missing"):
        run_apply(root, out, whole_pool=False)


def test_two_rerankers_cannot_overwrite_each_others_decision(tmp_path: Path) -> None:
    """The runbook scores MiniLM and BGE into one directory.

    A decision file named only for the design, not for the scores, meant the second model replaced
    the first and the survivor looked like a complete result while naming no model at all.
    """
    root, out = tmp_path / "root", tmp_path / "out"
    write_fixture(root, out, depth=EVAL_K)
    (out / "scores_bge.jsonl").write_bytes((out / "scores.jsonl").read_bytes())

    cmd_apply(argparse.Namespace(
        mq_dir=out, output_dir=out, mtrag_root=root,
        scores=out / "scores.jsonl", whole_pool=False))
    cmd_apply(argparse.Namespace(
        mq_dir=out, output_dir=out, mtrag_root=root,
        scores=out / "scores_bge.jsonl", whole_pool=False))

    first = json.loads((out / "rerank_decision__scores.json").read_text(encoding="utf-8"))
    second = json.loads((out / "rerank_decision__scores_bge.json").read_text(encoding="utf-8"))

    assert first["scores_file"] == "scores.jsonl"
    assert second["scores_file"] == "scores_bge.jsonl"


class _StubReranker:
    """A CrossEncoderReranker that scores from a lookup, so validate is testable with no model.

    Mirrors the real class's two used surfaces: `_model.predict(pairs)` and `rerank(query, hits)`
    with a stable descending sort, which is what `rerank_offload.rerank_order` reproduces.
    """

    def __init__(self, table: dict[str, float], **_kwargs: object) -> None:
        self._table = table
        self._model = self

    def predict(self, pairs: list[tuple[str, str]]) -> list[float]:
        return [self._table[text] for _query, text in pairs]

    def rerank(self, _query: str, hits: list) -> list:
        order = sorted(range(len(hits)),
                       key=lambda i: self._table[hits[i].chunk.text], reverse=True)
        return [hits[i] for i in order]


def write_validate_fixture(
    out: Path, mq_dir: Path, ranking: list[str], query_task: str = "t0"
) -> None:
    """The minimum `cmd_validate` reads: docs, queries via legs, and frozen rankings.

    `query_task` is separate from the ranking's task id so the mismatched --mq-dir case, where the
    legs dump describes different queries than the frozen rankings, is constructible.
    """
    out.mkdir(parents=True, exist_ok=True)
    (mq_dir / "legs").mkdir(parents=True, exist_ok=True)
    with (mq_dir / "legs" / "last.jsonl").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"task_id": query_task, "domain": "clapnq", "variant": "last",
                                 "query": "q", "dense": [], "splade": []}) + "\n")
    with (out / "docs.jsonl").open("w", encoding="utf-8") as handle:
        for doc in ranking:
            handle.write(json.dumps({"doc_id": doc, "text": f"text-{doc}"}) + "\n")
    (out / "rankings_equal_width.json").write_text(
        json.dumps({name: {"t0": list(ranking)} for name in RERANK_ARMS}), encoding="utf-8")


def validate_args(out: Path, mq_dir: Path, **over: object) -> argparse.Namespace:
    base = dict(mq_dir=mq_dir, output_dir=out, scores=out / "scores.jsonl",
                whole_pool=False, sample=5, model=None, revision=None)
    base.update(over)
    return argparse.Namespace(**base)


def test_validate_refuses_a_zero_sample(tmp_path: Path) -> None:
    """The gate the runbook makes the precondition for believing anything the rental produced."""
    with pytest.raises(ValueError, match="--sample must be >= 1"):
        cmd_validate(validate_args(tmp_path, tmp_path, sample=0))


def test_validate_names_a_missing_passage_instead_of_a_bare_keyerror(tmp_path: Path) -> None:
    """And it must do so BEFORE the cross-encoder loads, which on rented hardware is paid time."""
    out, mq_dir = tmp_path / "out", tmp_path / "mq"
    ranking = [f"d{i}" for i in range(5)]
    write_validate_fixture(out, mq_dir, ranking)
    with (out / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for rank, doc in enumerate(ranking):
            handle.write(json.dumps({"qid": "t0", "doc_id": doc, "score": float(rank)}) + "\n")
    # Drop one passage's text.
    lines = (out / "docs.jsonl").read_text(encoding="utf-8").splitlines()[:-1]
    (out / "docs.jsonl").write_text("\n".join(lines) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="have no text"):
        cmd_validate(validate_args(out, mq_dir))


def test_validate_names_an_unscored_candidate(tmp_path: Path) -> None:
    """A scores file that does not cover the rankings is a pipeline error, not a zero."""
    out, mq_dir = tmp_path / "out", tmp_path / "mq"
    ranking = [f"d{i}" for i in range(5)]
    write_validate_fixture(out, mq_dir, ranking)
    with (out / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for rank, doc in enumerate(ranking[:-1]):
            handle.write(json.dumps({"qid": "t0", "doc_id": doc, "score": float(rank)}) + "\n")

    with pytest.raises(RuntimeError, match="have no score"):
        cmd_validate(validate_args(out, mq_dir))


def test_validate_fails_when_a_near_tie_changes_the_top_100_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The check omitted as an equal-width assumption, which matters under --whole-pool.

    A near-tie straddling rank 100 leaves the top-10 ORDER identical, so the first branch passes,
    but it changes the top-100 SET, which is exactly what R@100 counts. `rerank_offload` records a
    real case of two candidates swapping on a 9.5e-07 gap. Without this branch the gate returns
    MATCH while certifying a reranked R@100 that RE-call would not compute.
    """
    import recall.rerank as rerank_module

    out, mq_dir = tmp_path / "out", tmp_path / "mq"
    ranking = [f"d{i}" for i in range(110)]
    write_validate_fixture(out, mq_dir, ranking)
    # Strictly decreasing, so the offloaded order IS the ranking order, with the rank 99/100 gap
    # collapsed to 1e-7 so a swap exactly there stays far inside SCORE_TOLERANCE.
    #
    # Two earlier versions of this fixture were wrong in instructive ways: the first used a gap of
    # ~1.0, which tripped the tolerance branch and returned 1 whatever the set check did; the
    # second overrode two scores to 500 while their neighbours sat near 900, which sorted both to
    # the bottom and moved the swap to ranks 108/109, nowhere near the cutoff. Both passed
    # vacuously and mutation runs caught them.
    offloaded = {doc: float(len(ranking) - rank) for rank, doc in enumerate(ranking)}
    offloaded[ranking[100]] = offloaded[ranking[99]] - 1e-7
    with (out / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for doc, score in offloaded.items():
            handle.write(json.dumps({"qid": "t0", "doc_id": doc, "score": score}) + "\n")

    # Local scores agree to well within tolerance, but swap the pair straddling rank 100.
    local = {f"text-{doc}": score for doc, score in offloaded.items()}
    local[f"text-{ranking[99]}"] = offloaded[ranking[100]]
    local[f"text-{ranking[100]}"] = offloaded[ranking[99]]
    monkeypatch.setattr(
        rerank_module, "CrossEncoderReranker", lambda **kw: _StubReranker(local, **kw))

    assert cmd_validate(validate_args(out, mq_dir, sample=1)) == 1


def test_validate_names_a_mismatched_mq_dir_before_loading_the_model(tmp_path: Path) -> None:
    """The legs dump describing different queries than the frozen rankings.

    This is the state `frozen_rankings_absent` warns about, and it used to die on a bare KeyError
    AFTER the cross-encoder had been constructed, which on rented hardware is paid minutes.
    """
    out, mq_dir = tmp_path / "out", tmp_path / "mq"
    ranking = [f"d{i}" for i in range(5)]
    write_validate_fixture(out, mq_dir, ranking, query_task="t9")
    with (out / "scores.jsonl").open("w", encoding="utf-8") as handle:
        for rank, doc in enumerate(ranking):
            handle.write(json.dumps({"qid": "t0", "doc_id": doc, "score": float(rank)}) + "\n")

    with pytest.raises(RuntimeError, match="no query text"):
        cmd_validate(validate_args(out, mq_dir))
