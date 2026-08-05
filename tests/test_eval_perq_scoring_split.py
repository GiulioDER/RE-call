"""`longmemeval_perq` and `labelled` must mean the same thing by `hit_at_k`.

Both harnesses publish a key of that name. Once one of them could be widened to every answerable
question and the other could not, the key named two different populations depending on which
module wrote the JSON, with nothing in either artifact saying which — and a reader comparing a
merged-corpus number against a per-question number would be comparing denominators, not systems.

So `score_retrieval_on` is threaded through here too, with the same two values, the same `"held"`
default and the same rule: abstention stays on the held half in BOTH modes, because its threshold
is fitted on the other one.

THE FIXTURE IS THE TEST. `_QUESTIONS` is built so the two implementations return DIFFERENT counts
(5 answerable overall against 2 held answerable). A balanced set — the same number either way —
would pass under a module that ignored the flag entirely and would still read like a guard.
`test_fixture_can_discriminate` pins that property so a later edit cannot quietly remove it.
"""
from __future__ import annotations

import inspect
import uuid

import pytest

from recall.embeddings import HashingEmbedder
from recall.eval import labelled as labelled_mod
from recall.eval import longmemeval_perq as perq_mod
from recall.eval._scoring import DEFAULT_SCORE_RETRIEVAL_ON, scores_retrieval
from recall.eval.longmemeval_perq import evaluate
from recall.index import Indexer
from recall.store import PgVectorStore

from .conftest import TEST_DSN, requires_db

#: Index parity assigns fit/held, so ORDER HERE IS SIGNIFICANT.
#:   fit  = idx 0,2,4,6 -> answerable q1,q3,q5 (3) + unanswerable q7 (1)
#:   held = idx 1,3,5   -> answerable q2,q4 (2)   + unanswerable q6 (1)
#: 5 answerable overall against 2 held answerable, and both classes present in `fit`, which the
#: threshold fit needs.
_QUESTIONS = [
    {"id": "q1", "query": "deploy target", "haystack_files": ["s1.md", "s2.md"],
     "relevant_files": ["s1.md"], "answerable": True, "question_type": "single"},
    {"id": "q2", "query": "database backup", "haystack_files": ["s2.md", "s3.md"],
     "relevant_files": ["s2.md"], "answerable": True, "question_type": "single"},
    {"id": "q3", "query": "oncall rotation", "haystack_files": ["s3.md", "s4.md"],
     "relevant_files": ["s3.md"], "answerable": True, "question_type": "single"},
    {"id": "q4", "query": "incident review", "haystack_files": ["s4.md", "s5.md"],
     "relevant_files": ["s4.md"], "answerable": True, "question_type": "single"},
    {"id": "q5", "query": "release freeze", "haystack_files": ["s5.md", "s1.md"],
     "relevant_files": ["s5.md"], "answerable": True, "question_type": "single"},
    {"id": "q6", "query": "airspeed of a swallow", "haystack_files": ["s1.md", "s3.md"],
     "relevant_files": [], "answerable": False, "question_type": "single"},
    {"id": "q7", "query": "unrelated nonsense xyzzy", "haystack_files": ["s2.md", "s4.md"],
     "relevant_files": [], "answerable": False, "question_type": "single"},
]
#: Every question's `haystack_files` is a DISTINCT pair, which is what lets
#: `test_the_default_visits_exactly_the_held_half_in_order` name the visited question from the
#: argument `populate_haystack` was called with. An earlier version reused two pairs across the
#: answerable/unanswerable boundary, and that test's precondition assertion is what caught it.

_ANSWERABLE_TOTAL = 5
_ANSWERABLE_HELD = 2
_HELD_UNANSWERABLE = 1


@pytest.fixture()
def master(tmp_path):
    """A 5-session master index covering every `haystack_files` entry above."""
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in (
        ("s1.md", "the deploy target is staging"),
        ("s2.md", "the database backup runs nightly"),
        ("s3.md", "the oncall rotation is weekly"),
        ("s4.md", "the incident review is on friday"),
        ("s5.md", "the release freeze starts in december"),
    ):
        (corpus / name).write_text(text, encoding="utf-8")

    emb = HashingEmbedder(dim=64)
    table = "pqm_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=emb.dim, table=table)
    # try/finally, not a bare teardown after `yield`: `ensure_schema` has already created the
    # table and written its ledger rows, so an exception in `index_path` would skip teardown
    # entirely and leak a uuid-named table per run — one that no later run reclaims.
    try:
        store.ensure_schema()
        Indexer(store, emb).index_path(corpus)
        yield emb, table
    finally:
        store.drop_table()
        store.close()


def test_fixture_can_discriminate() -> None:
    """The two modes must return different numbers on this fixture, or nothing below pins anything.

    No database needed: this is a property of `_QUESTIONS` itself.
    """
    answerable = [q for q in _QUESTIONS if q["answerable"]]
    held_answerable = [q for q in _QUESTIONS[1::2] if q["answerable"]]
    fit = _QUESTIONS[::2]

    assert len(answerable) == _ANSWERABLE_TOTAL
    assert len(held_answerable) == _ANSWERABLE_HELD
    assert _ANSWERABLE_TOTAL != _ANSWERABLE_HELD, "fixture cannot separate the two modes"
    assert any(q["answerable"] for q in fit) and any(not q["answerable"] for q in fit)


def test_the_two_harnesses_declare_the_same_flag_and_the_same_default() -> None:
    """The divergence itself, pinned. No database.

    Not a style check: the default is what decides what an already-published `hit_at_k` MEANS, so
    the two modules drifting apart on it is the defect, and a shared constant plus a shared
    default is the fix. Read through the signature rather than by grepping the source, so it
    holds however the default is expressed — a rename of the parameter fails here, intentionally.
    """
    perq_p = inspect.signature(perq_mod.evaluate).parameters["score_retrieval_on"]
    labelled_p = inspect.signature(labelled_mod.evaluate).parameters["score_retrieval_on"]

    assert perq_p.default == labelled_p.default == DEFAULT_SCORE_RETRIEVAL_ON == "held"
    # ONE list of legal values and ONE default object, not two of each that are equal today.
    assert perq_mod.SCORE_RETRIEVAL_ON is labelled_mod.SCORE_RETRIEVAL_ON
    assert perq_mod.DEFAULT_SCORE_RETRIEVAL_ON is labelled_mod.DEFAULT_SCORE_RETRIEVAL_ON


@pytest.mark.parametrize("module", [labelled_mod, perq_mod], ids=["labelled", "perq"])
def test_the_cli_default_is_held_in_both_harnesses(module, monkeypatch, tmp_path) -> None:
    """The signature is not the leg the published reproduce commands go through. No database.

    `README.md` and `results/RESULTS.md` both ship a `python -m recall.eval.…` line, which reaches
    `evaluate` via argparse. Pinning only the Python default left `default="all"` in either
    `add_argument` shipping green on the one setting that decides what a published `hit_at_k`
    counts — a guard that reads as protection over the whole flag while covering half of it.
    """
    import json

    captured: dict = {}

    def fake_run(*args, **kwargs):
        captured.update(kwargs)
        return {}

    questions = tmp_path / "q.json"
    questions.write_text(json.dumps([
        {"id": "q1", "query": "x", "relevant_files": ["a.md"], "haystack_files": ["a.md"],
         "answerable": True},
    ]), encoding="utf-8")

    argv = ["prog", "--questions", str(questions), "--embedder", "hashing"]
    argv += (["--corpus", str(tmp_path)] if module is labelled_mod else ["--master", "m"])
    monkeypatch.setattr(module, "evaluate", fake_run)
    monkeypatch.setattr("sys.argv", argv)

    module.main()

    assert captured["score_retrieval_on"] == "held"


@pytest.mark.parametrize("module", [labelled_mod, perq_mod], ids=["labelled", "perq"])
def test_the_cli_refuses_an_unknown_mode_in_both_harnesses(module, monkeypatch, tmp_path) -> None:
    """`choices=` bound to the shared tuple, not to a hand-copied list. No database."""
    import json

    questions = tmp_path / "q.json"
    questions.write_text(json.dumps([{"id": "q1", "query": "x", "relevant_files": ["a.md"],
                                      "haystack_files": ["a.md"], "answerable": True}]),
                         encoding="utf-8")
    argv = ["prog", "--questions", str(questions), "--score-retrieval-on", "everything"]
    argv += (["--corpus", str(tmp_path)] if module is labelled_mod else ["--master", "m"])
    monkeypatch.setattr("sys.argv", argv)

    with pytest.raises(SystemExit) as exc:
        module.main()
    assert exc.value.code == 2, "argparse rejects an out-of-choices value with exit status 2"


@pytest.mark.parametrize(
    "is_held, mode, expected",
    [
        (True, "held", True),    # held answerable: retrieved in both modes
        (True, "all", True),
        (False, "held", False),  # fit answerable: the whole point of the flag
        (False, "all", True),
    ],
)
def test_the_membership_rule_is_one_function_both_harnesses_call(
    is_held, mode, expected
) -> None:
    """The flag's MEANING, pinned by execution rather than by two docstrings agreeing.

    The cross-harness test above pins the flag's name, default and legal values. None of those
    stops the two modules implementing "which questions are retrieved" differently — which they
    did, as a list selection in one and an inline loop skip in the other. The rule now lives in
    `recall.eval._scoring` and both call it, so this table is the definition of record. It is
    consulted only for ANSWERABLE questions; answerability is the caller's to supply.
    """
    assert scores_retrieval(is_held=is_held, mode=mode) is expected


def test_rejects_an_unknown_score_retrieval_on() -> None:
    """A typo must fail loudly rather than fall through to one of the modes. No database."""
    with pytest.raises(ValueError, match="score_retrieval_on"):
        evaluate(TEST_DSN, "no_such_master", _QUESTIONS, HashingEmbedder(dim=64),
                 score_retrieval_on="everything")


@requires_db
def test_default_mode_scores_retrieval_on_the_held_half(master) -> None:
    """The DEFAULT stays `held` — this arm's published LongMemEval numbers were computed under it.

    Denominators and the mode label only. WHICH questions were visited, and in what order, is a
    different claim and is pinned by `test_the_default_visits_exactly_the_held_half_in_order`
    below — counts would be equal for a loop that scored the wrong two questions.
    """
    emb, table = master
    rep = evaluate(TEST_DSN, table, _QUESTIONS, emb, k=3)

    assert rep["questions"]["score_retrieval_on"] == "held"
    assert rep["questions"]["retrieval_scored_on"] == _ANSWERABLE_HELD
    assert rep["hit_at_3"]["n"] == _ANSWERABLE_HELD
    assert rep["false_abstain"]["n"] == _ANSWERABLE_HELD
    assert rep["abstention_accuracy"]["n"] == _HELD_UNANSWERABLE
    # Each label equals the n of the metric it names — a count derived independently of the value
    # it labels can drift from it.
    assert rep["questions"]["false_abstain_scored_on"] == rep["false_abstain"]["n"]
    assert rep["questions"]["abstention_accuracy_scored_on"] == rep["abstention_accuracy"]["n"]
    # by_type is a partition of the retrieval sample, so it must add up to the same denominator.
    assert sum(v["n"] for v in rep["by_type"].values()) == _ANSWERABLE_HELD
    assert rep["haystack_chunks"]["n"] == _ANSWERABLE_HELD + _HELD_UNANSWERABLE


@requires_db
@pytest.mark.parametrize(
    "mode, expected_loop_ids",
    [
        ("held", ["q2", "q4", "q6"]),
        ("all", ["q1", "q2", "q3", "q4", "q5", "q6"]),
    ],
)
def test_the_default_visits_exactly_the_held_half_in_order(
    master, monkeypatch, mode, expected_loop_ids
) -> None:
    """The loop restructure, checked on the questions themselves rather than on their count.

    The widened mode required replacing `for q in held:` with an index-carrying pass over
    `questions`, and that pass was then rewritten again around a shared predicate. Under the
    default both rewrites must visit exactly `questions[1::2]`, in order — a property no
    denominator can distinguish from a loop that scored a different pair of the same size.

    Read off `populate_haystack`, which is called once per visited question with that question's
    own `haystack_files`, so the recorded sequence IS the visit order. The calibration pass runs
    first (every fit question, answerable then unanswerable), so the loop's own visits are the
    tail after that fixed prefix.
    """
    emb, table = master
    fit = _QUESTIONS[::2]
    calibration_prefix = ([q for q in fit if q["answerable"]]
                          + [q for q in fit if not q["answerable"]])
    by_haystack = {tuple(q["haystack_files"]): q["id"] for q in _QUESTIONS}
    assert len(by_haystack) == len(_QUESTIONS), "haystacks must identify questions uniquely"

    visited: list[str] = []
    real_populate = perq_mod.populate_haystack

    def recording_populate(dsn, dim, master_table, scratch, files, **kwargs):
        visited.append(by_haystack[tuple(files)])
        return real_populate(dsn, dim, master_table, scratch, files, **kwargs)

    monkeypatch.setattr(perq_mod, "populate_haystack", recording_populate)
    evaluate(TEST_DSN, table, _QUESTIONS, emb, k=3, score_retrieval_on=mode)

    assert visited[:len(calibration_prefix)] == [q["id"] for q in calibration_prefix], (
        "the calibration pass is the fixed prefix this assertion subtracts"
    )
    assert visited[len(calibration_prefix):] == expected_loop_ids


@requires_db
def test_all_mode_widens_retrieval_and_leaves_abstention_on_the_held_half(master) -> None:
    """`"all"` doubles the retrieval sample and moves NOTHING else.

    The second half of that sentence is the load-bearing one: widening `false_abstain` too would
    score the abstention threshold on the questions it was fitted on, which is the defect the
    fit/held split exists to prevent and which no downstream number would reveal.
    """
    emb, table = master
    rep = evaluate(TEST_DSN, table, _QUESTIONS, emb, k=3, score_retrieval_on="all")

    assert rep["questions"]["score_retrieval_on"] == "all"
    assert rep["questions"]["retrieval_scored_on"] == _ANSWERABLE_TOTAL
    assert rep["hit_at_3"]["n"] == _ANSWERABLE_TOTAL
    assert sum(v["n"] for v in rep["by_type"].values()) == _ANSWERABLE_TOTAL
    # `haystack_chunks` widens too, and must SAY so: its population is the retrieval-scored
    # questions plus the held unanswerable ones, so an artifact from each mode carries the same
    # key over a different set. The `n` is what makes the two comparable.
    assert rep["haystack_chunks"]["n"] == _ANSWERABLE_TOTAL + _HELD_UNANSWERABLE

    assert rep["false_abstain"]["n"] == _ANSWERABLE_HELD
    assert rep["abstention_accuracy"]["n"] == _HELD_UNANSWERABLE
    assert rep["questions"]["false_abstain_scored_on"] == _ANSWERABLE_HELD
    assert rep["questions"]["abstention_accuracy_scored_on"] == _HELD_UNANSWERABLE


@requires_db
def test_the_abstention_leg_is_identical_in_both_modes(master) -> None:
    """Same threshold, same questions, same verdicts — the apparatus invariant, not a re-count.

    Equal `n` on both sides would also hold if the widened mode scored a DIFFERENT pair of held
    questions, so the published RATE is compared and not merely the denominator. The harness does
    not expose the per-question abstain flags, so this is the strongest check available from the
    artifact; at n=2 there are only three distinct rates, which bounds how much it can prove.
    """
    emb, table = master
    held_mode = evaluate(TEST_DSN, table, _QUESTIONS, emb, k=3)
    all_mode = evaluate(TEST_DSN, table, _QUESTIONS, emb, k=3, score_retrieval_on="all")

    assert all_mode["threshold"] == held_mode["threshold"]
    assert all_mode["false_abstain"] == held_mode["false_abstain"]
    assert all_mode["abstention_accuracy"] == held_mode["abstention_accuracy"]
