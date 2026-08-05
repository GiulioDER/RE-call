"""`recall.eval.labelled`: which questions each metric family is scored on.

`evaluate` splits `fit, held = questions[::2], questions[1::2]` and fits the abstention threshold
on `fit`. That split is load-bearing for `false_abstain` — scoring it on `fit` too would fit and
score on the same data — and pure cost for `hit_at_k` / `mrr`, which never read the calibration at
all. Scoring retrieval on the held half alone halves its sample for nothing: on the PEPs arm that
is n=44 of 88, wide enough (CI [0.534, 0.800] around 0.682) that one question moves the rate by
0.023 and a real four-point effect is indistinguishable from noise.

`score_retrieval_on="all"` is therefore available and `"held"` is the DEFAULT, because the default
decides what every already-published figure means. Three metric families, three denominators, and
each is pinned here. There are three ways to break it:

  * flip the DEFAULT to "all"                -> published README/gap numbers silently change
  * ignore the flag                          -> arm `n` does not move between the two modes
  * "fix" the split by widening EVERYTHING   -> `false_abstain` n rises to the retrieval count

THE FIXTURE IS THE TEST. `_QUESTIONS` is built so the two counts DISAGREE (5 retrieval vs 2
abstention). A balanced fixture where both implementations return the same number would assert
happily and discriminate nothing — the failure mode recorded after a benchmark shipped a test
named for the definition it was not measuring. `test_fixture_can_discriminate` guards that
property directly, so a later edit to `_QUESTIONS` cannot quietly remove it.
"""
from __future__ import annotations

import uuid
from pathlib import Path

from recall.embeddings import HashingEmbedder
from recall.eval.labelled import evaluate
from recall.store import PgVectorStore

from .conftest import TEST_DSN, requires_db

#: Index parity is what assigns questions to fit/held, so ORDER HERE IS SIGNIFICANT.
#:   fit  = idx 0,2,4,6 -> answerable q1,q3,q5 (3) + unanswerable q7 (1)
#:   held = idx 1,3,5   -> answerable q2,q4 (2)   + unanswerable q6 (1)
#: giving 5 answerable overall against 2 held answerable. Both classes appear in `fit`, which the
#: threshold fit needs, and 5 != 2, which is what makes the assertions below discriminate.
_QUESTIONS = [
    {"id": "q1", "query": "deploy target", "relevant_files": ["a.md"], "answerable": True},
    {"id": "q2", "query": "database backup", "relevant_files": ["b.md"], "answerable": True},
    {"id": "q3", "query": "oncall rotation", "relevant_files": ["c.md"], "answerable": True},
    {"id": "q4", "query": "incident review", "relevant_files": ["d.md"], "answerable": True},
    {"id": "q5", "query": "release freeze", "relevant_files": ["e.md"], "answerable": True},
    {"id": "q6", "query": "airspeed of a swallow", "relevant_files": [], "answerable": False},
    {"id": "q7", "query": "unrelated nonsense xyzzy", "relevant_files": [], "answerable": False},
]

_ANSWERABLE_TOTAL = 5
_ANSWERABLE_HELD = 2
_HELD_UNANSWERABLE = 1


def _corpus(tmp_path):
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in (
        ("a.md", "the deploy target is staging"),
        ("b.md", "the database backup runs nightly"),
        ("c.md", "the oncall rotation is weekly"),
        ("d.md", "the incident review is on friday"),
        ("e.md", "the release freeze starts in december"),
    ):
        (corpus / name).write_text(text, encoding="utf-8")
    return corpus


def test_fixture_can_discriminate() -> None:
    """The fixture must make the two implementations return DIFFERENT numbers.

    No database needed: this is a property of `_QUESTIONS` itself. If a later edit balances the
    fixture so that all-answerable and held-answerable coincide, every assertion below would pass
    under either implementation and the file would go on reading like a guard.
    """
    answerable = [q for q in _QUESTIONS if q["answerable"]]
    held_answerable = [q for q in _QUESTIONS[1::2] if q["answerable"]]
    fit = _QUESTIONS[::2]
    assert len(answerable) == _ANSWERABLE_TOTAL
    assert len(held_answerable) == _ANSWERABLE_HELD
    assert _ANSWERABLE_TOTAL != _ANSWERABLE_HELD, "fixture cannot separate the two implementations"
    # The threshold fit needs both classes, or `evaluate` is measuring something else entirely.
    assert any(q["answerable"] for q in fit) and any(not q["answerable"] for q in fit)


@requires_db
def test_retrieval_scores_every_answerable_question_abstention_only_the_held_half(
    tmp_path,
) -> None:
    corpus = _corpus(tmp_path)
    emb = HashingEmbedder(dim=64)
    table = "lab_split_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=emb.dim, table=table)
    try:
        rep = evaluate(TEST_DSN, corpus, _QUESTIONS, emb, k=3, table=table,
                       score_retrieval_on="all")

        # THREE denominators, each named after the metric it divides.
        assert rep["questions"]["score_retrieval_on"] == "all"
        assert rep["questions"]["retrieval_scored_on"] == _ANSWERABLE_TOTAL
        assert rep["questions"]["false_abstain_scored_on"] == _ANSWERABLE_HELD
        assert rep["questions"]["abstention_accuracy_scored_on"] == _HELD_UNANSWERABLE

        # Retrieval: every arm, not just `hybrid`. An arm scored on a different set than its
        # siblings would make the between-arm delta unattributable, which is the whole point of
        # running them against one index.
        for name, arm in rep["arms"].items():
            assert arm["hit_at_3"]["n"] == _ANSWERABLE_TOTAL, f"{name} scored on the wrong set"

        # Abstention: still the held half. This is the assertion that fails if someone widens
        # the split everywhere instead of only where it was free.
        assert rep["false_abstain"]["n"] == _ANSWERABLE_HELD
        # Each label must equal the n of the metric it names — a label computed independently of
        # its value can drift from it, which is the defect this whole block exists to prevent.
        assert rep["false_abstain"]["n"] == rep["questions"]["false_abstain_scored_on"]
        assert rep["abstention_accuracy"]["n"] == rep["questions"]["abstention_accuracy_scored_on"]
        assert rep["abstention_accuracy"]["n"] == _HELD_UNANSWERABLE
    finally:
        store.drop_table()
        store.close()


@requires_db
def test_default_mode_scores_retrieval_on_the_held_half(tmp_path) -> None:
    """The DEFAULT must stay `held`, because the default decides what a published number means.

    `README.md` states a PEPs table over "44 held-out answerable questions" and ships the
    reproduce command in the same file; `gap_run` records per-corpus rates for 17 BEIR corpora.
    Flipping this default silently changes what all of those figures mean, with no diff in their
    provenance. So the widened sample is opt-in, and this test is what stops it drifting back.
    """
    corpus = _corpus(tmp_path)
    emb = HashingEmbedder(dim=64)
    table = "lab_split_def_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=emb.dim, table=table)
    try:
        rep = evaluate(TEST_DSN, corpus, _QUESTIONS, emb, k=3, table=table)
        assert rep["questions"]["score_retrieval_on"] == "held"
        assert rep["questions"]["retrieval_scored_on"] == _ANSWERABLE_HELD
        for name, arm in rep["arms"].items():
            assert arm["hit_at_3"]["n"] == _ANSWERABLE_HELD, f"{name} scored on the wrong set"
        # Abstention is identical in BOTH modes — that is the apparatus invariant the PEPs
        # re-run is checked against, so it is pinned here rather than merely remembered.
        assert rep["false_abstain"]["n"] == _ANSWERABLE_HELD
        assert rep["abstention_accuracy"]["n"] == _HELD_UNANSWERABLE

        # Same index, same questions, twice: the abstention leg must be reproducible.
        again = evaluate(TEST_DSN, corpus, _QUESTIONS, emb, k=3, table=table)
        assert again["false_abstain"] == rep["false_abstain"]
    finally:
        store.drop_table()
        store.close()


def test_rejects_an_unknown_score_retrieval_on() -> None:
    """A typo must fail loudly, not silently fall through to one of the two modes.

    No database: the guard is argument validation and runs before any connection is opened.
    """
    import pytest

    with pytest.raises(ValueError, match="score_retrieval_on"):
        evaluate(TEST_DSN, Path("."), _QUESTIONS, HashingEmbedder(dim=64),
                 score_retrieval_on="everything")
