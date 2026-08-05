"""`q["id"]` is load-bearing in `recall.eval.labelled`, so it is validated at entry.

`evaluate` scores retrieval over `scored` — under `score_retrieval_on="all"` that is EVERY
answerable question — and then selects the held-out answerable subset back out of those results
by id, so `false_abstain` keeps the split its threshold was fitted under::

    answerable_held_ids = {x["id"] for x in answerable_held}
    ... for qid, res in hybrid_results if qid in answerable_held_ids

Nothing about that expression fails loudly when the ids are wrong:

  * a DUPLICATE id spanning the fit/held boundary matches a FIT question, so abstention is scored
    on a question the threshold was fitted on. The n does not move. The rate is merely optimistic,
    and it is optimistic in exactly the direction the split exists to prevent.
  * a MISSING id raises `KeyError` from inside a set comprehension, after the corpus is indexed
    and the threshold fitted — hours of embedding on a real corpus, to report a typo.

THE FIXTURE IS THE TEST. `_DUPLICATE_ACROSS_THE_SPLIT` is built so that the duplicate straddles
the fit/held boundary and BOTH copies are answerable, which is the only shape that contaminates.
A duplicate sitting entirely inside one half would be refused by the guard and harmless without
it, so it would pin nothing while reading like a guard.
`test_the_duplicate_fixture_would_actually_contaminate` asserts that property against a local
re-implementation of the unguarded selection, so a later edit to the fixture cannot quietly
remove it.
"""
from __future__ import annotations

import pytest

from recall.embeddings import HashingEmbedder
from recall.eval.labelled import check_question_ids, evaluate

from .conftest import TEST_DSN, requires_db

#: idx 0 (fit) and idx 1 (held) share "q1", and both are answerable. That is the contaminating
#: shape: the held set contributes the id, and the fit question is selected by it.
_DUPLICATE_ACROSS_THE_SPLIT = [
    {"id": "q1", "query": "deploy target", "relevant_files": ["a.md"], "answerable": True},
    {"id": "q1", "query": "database backup", "relevant_files": ["b.md"], "answerable": True},
    {"id": "q3", "query": "oncall rotation", "relevant_files": ["c.md"], "answerable": True},
    {"id": "q4", "query": "incident review", "relevant_files": ["d.md"], "answerable": True},
    {"id": "q5", "query": "airspeed of a swallow", "relevant_files": [], "answerable": False},
    {"id": "q6", "query": "unrelated nonsense xyzzy", "relevant_files": [], "answerable": False},
]

_VALID = [dict(q, id=f"u{i}") for i, q in enumerate(_DUPLICATE_ACROSS_THE_SPLIT)]

#: These tests reach `evaluate` and rely on the guard raising BEFORE the corpus is read. The
#: corpus is therefore an empty `tmp_path`, not the working directory: with `Path(".")` a guard
#: that stopped firing would not merely fail the assertion, it would index every markdown file in
#: the checkout into the test database. A guard test's failure mode should be cheap.


def test_the_duplicate_fixture_would_actually_contaminate() -> None:
    """The fixture must make the guarded and unguarded implementations DISAGREE.

    No database: this is a property of the question list. The unguarded selection is rebuilt here
    from `labelled`'s own two lines rather than imported, so this test states what the defect IS
    instead of asserting that the code still contains it.
    """
    fit, held = _DUPLICATE_ACROSS_THE_SPLIT[::2], _DUPLICATE_ACROSS_THE_SPLIT[1::2]
    answerable = [q for q in _DUPLICATE_ACROSS_THE_SPLIT if q["answerable"]]
    answerable_held = [q for q in held if q["answerable"]]

    # The unguarded path, under `score_retrieval_on="all"`: score every answerable question,
    # then filter those results back down by held id.
    answerable_held_ids = {q["id"] for q in answerable_held}
    selected = [q for q in answerable if q["id"] in answerable_held_ids]

    fit_objects = {id(q) for q in fit}
    contaminating = [q for q in selected if id(q) in fit_objects]
    assert contaminating, "fixture does not straddle the fit/held boundary — it pins nothing"
    # And the contamination is invisible in the shape of the output: it inflates the abstention
    # sample past the held half without any other number moving.
    assert len(selected) > len(answerable_held)
    # Both copies answerable, or the held set never contributes the shared id in the first place.
    assert all(q["answerable"] for q in _DUPLICATE_ACROSS_THE_SPLIT
               if q["id"] == "q1")


def test_a_duplicate_id_raises_instead_of_contaminating_the_abstention_sample(tmp_path) -> None:
    """The whole point: it must FAIL, not run and report a quietly optimistic rate.

    No database — the guard runs before any connection is opened, which is also the only place
    it is worth anything on a corpus that costs hours to index.
    """
    with pytest.raises(ValueError, match="duplicate question id"):
        evaluate(TEST_DSN, tmp_path, _DUPLICATE_ACROSS_THE_SPLIT, HashingEmbedder(dim=64),
                 score_retrieval_on="all")

    # The default mode does not read `answerable_held_ids` any less, so it is refused too.
    with pytest.raises(ValueError, match="duplicate question id"):
        evaluate(TEST_DSN, tmp_path, _DUPLICATE_ACROSS_THE_SPLIT, HashingEmbedder(dim=64))


def test_the_duplicate_message_counts_distinct_ids_and_locates_every_copy() -> None:
    """The message must name what the reader has to go and fix.

    Counting OCCURRENCES while printing a de-duplicated sample makes the two disagree — three
    copies of one id read as "2 duplicate question id(s): ['q1']" and send the reader looking for
    two ids that do not exist. And the indexes are the fact that decides whether this duplicate is
    the contaminating kind: one even and one odd is the fit/held straddle.
    """
    thrice = [dict(_VALID[0], id="q1"), dict(_VALID[1], id="q1"), dict(_VALID[2], id="q1")]

    with pytest.raises(ValueError, match="duplicate question id") as exc:
        check_question_ids(thrice)

    message = str(exc.value)
    assert message.startswith("1 duplicate question id(s)"), "counted occurrences, not ids"
    assert "'q1': [0, 1, 2]" in message, "every copy must be located, not just counted"


@pytest.mark.parametrize(
    "broken, why",
    [
        ({"query": "no id at all", "relevant_files": ["a.md"], "answerable": True}, "absent"),
        ({"id": "", "query": "empty", "relevant_files": ["a.md"], "answerable": True}, "empty"),
        ({"id": "   ", "query": "blank", "relevant_files": ["a.md"], "answerable": True},
         "whitespace"),
        ({"id": 7, "query": "not a string", "relevant_files": ["a.md"], "answerable": True},
         "non-string"),
    ],
)
def test_an_unusable_id_is_refused_up_front(broken, why, tmp_path) -> None:
    """Present, non-empty and a string. A `KeyError` from inside a comprehension is not a report.

    `id: 7` is included because JSON hands back whatever was written: an integer id compares
    unequal to nothing and would pass a bare `"id" in q` check while making `sorted(...)` in the
    error paths and the miss report inconsistent with the rest of the file's ids.
    """
    questions = [broken, *_VALID[1:]]

    with pytest.raises(ValueError, match="no usable 'id'") as exc:
        evaluate(TEST_DSN, tmp_path, questions, HashingEmbedder(dim=64))
    assert "index [0]" in str(exc.value), f"the {why} id must be located, not just counted"


@pytest.mark.parametrize(
    "malformed",
    [
        ["q1", "q2"],                              # a list of strings
        [{"id": "q1", "answerable": True}, None],  # a null entry
        [{"id": "q1", "answerable": True}, 7],     # a bare number
    ],
)
def test_a_non_object_entry_is_refused_through_the_same_error(malformed) -> None:
    """`q.get` on a string raises AttributeError — the raw traceback this guard replaces.

    A validator that itself crashes on malformed input has handed the caller exactly what it
    promised to take away, one exception type further along.
    """
    with pytest.raises(ValueError, match="no usable 'id'"):
        check_question_ids(malformed)


def test_a_questions_value_that_is_not_a_list_is_refused() -> None:
    """A JSON object or string enumerates into keys or characters and validates nonsense."""
    with pytest.raises(ValueError, match="must be a list"):
        check_question_ids({"q1": {"query": "x"}})  # type: ignore[arg-type]


@requires_db
def test_the_abstention_sample_is_re_checked_even_if_the_id_guard_is_defeated(
    tmp_path, monkeypatch
) -> None:
    """Defence in depth, executed rather than asserted: disable the guard, see the second one fire.

    `check_question_ids` closes the one contamination route we know about. The invariant it
    protects — `false_abstain` is scored on the held answerable half, no more and no fewer — is
    re-checked in `evaluate` for every other route, because the failure is silent by construction:
    a fit question entering that sample does not move the n published beside it, it only makes the
    rate optimistic.

    Neutering the guard here is what makes this a test of the SECOND check rather than of the
    first. Without the monkeypatch the run stops at the entry guard and this proves nothing.
    """
    from recall.eval import labelled as labelled_mod

    corpus = tmp_path / "corpus"
    corpus.mkdir()
    for name, text in (("a.md", "the deploy target is staging"),
                       ("b.md", "the database backup runs nightly"),
                       ("c.md", "the oncall rotation is weekly"),
                       ("d.md", "the incident review is on friday")):
        (corpus / name).write_text(text, encoding="utf-8")

    monkeypatch.setattr(labelled_mod, "check_question_ids", lambda questions: None)

    with pytest.raises(ValueError, match="false_abstain was scored on"):
        evaluate(TEST_DSN, corpus, _DUPLICATE_ACROSS_THE_SPLIT, HashingEmbedder(dim=64), k=3,
                 score_retrieval_on="all")


def test_a_valid_question_set_passes_and_the_check_is_reachable_directly() -> None:
    """The negative control. A guard that refuses everything is not a guard."""
    assert check_question_ids(_VALID) is None
    assert len({q["id"] for q in _VALID}) == len(_VALID)


def test_an_unknown_mode_is_reported_as_such_even_when_the_ids_are_also_wrong(tmp_path) -> None:
    """Argument validation first, id validation second — both before any database work.

    Ordering matters only in that a bad `score_retrieval_on` must still be reported as such even
    when the ids are also wrong; otherwise a typo in the flag would be reported as an id problem.
    """
    with pytest.raises(ValueError, match="score_retrieval_on"):
        evaluate(TEST_DSN, tmp_path, _DUPLICATE_ACROSS_THE_SPLIT, HashingEmbedder(dim=64),
                 score_retrieval_on="everything")


def test_the_cli_refuses_a_bad_id_before_it_dereferences_one(tmp_path, monkeypatch) -> None:
    """The guard has to hold on the path the README tells people to run.

    `main()` builds `[q["id"] for q in questions if answerable and not relevant_files]` of its
    own, BEFORE calling `evaluate`. Until the guard was hoisted into `main()`, a question set
    matching that filter with no id died with a bare `KeyError` from inside a comprehension —
    which is the exact failure this whole file exists to replace, surviving in the entry point
    most users take.
    """
    import json

    from recall.eval import labelled as labelled_mod

    path = tmp_path / "questions.json"
    path.write_text(json.dumps([{"query": "no id, no relevant_files", "answerable": True}]),
                    encoding="utf-8")
    monkeypatch.setattr(
        "sys.argv",
        ["labelled", "--corpus", str(tmp_path), "--questions", str(path), "--embedder", "hashing"],
    )

    with pytest.raises(ValueError, match="no usable 'id'"):
        labelled_mod.main()
