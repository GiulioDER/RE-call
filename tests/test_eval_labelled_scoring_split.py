"""`recall.eval.labelled`: retrieval is scored on ALL answerable questions, abstention on the held half.

`evaluate` splits `fit, held = questions[::2], questions[1::2]` and fits the abstention threshold
on `fit`. That split is load-bearing for `false_abstain` — scoring it on `fit` too would fit and
score on the same data — and pure cost for `hit_at_k` / `mrr`, which never read the calibration at
all. Scoring retrieval on the held half alone halved its sample for nothing: on the PEPs arm that
was n=44 of 88, wide enough (CI [0.534, 0.800] around 0.682) that one question moved the rate by
0.023 and a real four-point effect was indistinguishable from noise.

So the two metric families have DIFFERENT denominators on purpose, and this pins both. There are
two ways to break it and each has its own assertion:

  * revert retrieval to the held half        -> arm `n` drops to the abstention count
  * "fix" the split by widening EVERYTHING   -> `false_abstain` n rises to the retrieval count

THE FIXTURE IS THE TEST. `_QUESTIONS` is built so the two counts DISAGREE (5 retrieval vs 2
abstention). A balanced fixture where both implementations return the same number would assert
happily and discriminate nothing — the failure mode recorded after a benchmark shipped a test
named for the definition it was not measuring. `test_fixture_can_discriminate` guards that
property directly, so a later edit to `_QUESTIONS` cannot quietly remove it.
"""
from __future__ import annotations

import uuid

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
        rep = evaluate(TEST_DSN, corpus, _QUESTIONS, emb, k=3, table=table)

        # The report NAMES both denominators rather than leaving them to be inferred.
        assert rep["questions"]["retrieval_scored_on"] == _ANSWERABLE_TOTAL
        assert rep["questions"]["abstention_scored_on"] == _ANSWERABLE_HELD

        # Retrieval: every arm, not just `hybrid`. An arm scored on a different set than its
        # siblings would make the between-arm delta unattributable, which is the whole point of
        # running them against one index.
        for name, arm in rep["arms"].items():
            assert arm["hit_at_3"]["n"] == _ANSWERABLE_TOTAL, f"{name} scored on the wrong set"

        # Abstention: still the held half. This is the assertion that fails if someone widens
        # the split everywhere instead of only where it was free.
        assert rep["false_abstain"]["n"] == _ANSWERABLE_HELD
    finally:
        store.drop_table()
        store.close()


@requires_db
def test_widening_retrieval_did_not_change_the_abstention_denominator(tmp_path) -> None:
    """The rerank arm must not move either denominator.

    `hybrid_results` is collected during the `hybrid` arm and reused for `false_abstain`, so the
    two are coupled through a mutable list. Passing `rerank=True` adds another `score_arm` call
    against the same collector-free path; this pins that it cannot perturb the abstention sample.
    """
    corpus = _corpus(tmp_path)
    emb = HashingEmbedder(dim=64)
    table = "lab_split_rr_" + uuid.uuid4().hex[:8]
    store = PgVectorStore(TEST_DSN, dim=emb.dim, table=table)
    try:
        plain = evaluate(TEST_DSN, corpus, _QUESTIONS, emb, k=3, table=table)
        assert plain["false_abstain"]["n"] == _ANSWERABLE_HELD
        assert plain["arms"]["hybrid"]["hit_at_3"]["n"] == _ANSWERABLE_TOTAL
        # The abstention rate itself must be reproducible across runs on one index — it is the
        # apparatus invariant the PEPs re-run is checked against.
        again = evaluate(TEST_DSN, corpus, _QUESTIONS, emb, k=3, table=table)
        assert again["false_abstain"] == plain["false_abstain"]
    finally:
        store.drop_table()
        store.close()
