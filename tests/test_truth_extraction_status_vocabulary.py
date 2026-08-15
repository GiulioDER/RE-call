"""The status vocabulary is configurable for MEASUREMENT and closed for WRITING.

The shipped set is memo-shaped: `active, draft, deprecated, superseded, withdrawn`. Measured on
`python/peps`, the model reads `Status: Final` and emits `final`, which is outside it. Because
`claim_shape` is a BATCH rung, one such claim refused the whole document, and 12 of 30 documents
went down that way, taking their supersession claims with them. The vocabulary was the mismatch;
refusing was correct behaviour against the wrong list.

So the vocabulary is a parameter. Three properties hold it together, and each has a test:

1. It reaches the PROMPT, so the model is told what it may say.
2. It reaches the CACHE KEY, so two vocabularies never share an entry. Rendered into the prompt
   but absent from the key, the second run would silently serve the first's answer, produced
   under instructions it never sent.
3. It does NOT reach `recall.rewrite`, which keeps validating against the shipped constant. A
   value the trust layer cannot act on must not land in a user's memo because a research run
   named it.
"""
from __future__ import annotations

import pytest

from recall.truth_extraction import (
    STATUS_VOCABULARY,
    ExtractionBatchRejected,
    build_extraction_prompt,
    normalize_extraction,
)
from recall.truth_extraction._cache import extraction_cache_key
from recall.truth_extraction.types import coerce_status_vocabulary

PEP_STATUSES = ("final", "rejected", "deferred", "accepted", "active", "draft", "superseded")
BODY = "Status: Final"
PAYLOAD = '{"claims": [{"kind": "status", "value": "final", "quote": "Status: Final"}]}'


def _prompt(vocabulary=None):
    kwargs = {} if vocabulary is None else {"status_vocabulary": vocabulary}
    return build_extraction_prompt(
        file="pep-0376.rst", human_body=BODY, corpus_names=("pep-0376",), **kwargs
    )


def test_the_default_is_the_shipped_vocabulary() -> None:
    """Unchanged behaviour for every existing caller."""
    assert _prompt().status_vocabulary == STATUS_VOCABULARY
    assert f"- `value` is one of {list(STATUS_VOCABULARY)}." in _prompt().user


def test_a_custom_vocabulary_reaches_the_prompt() -> None:
    """The model must be told what it may say, or it is being judged on unstated rules."""
    rendered = _prompt(PEP_STATUSES).user
    assert f"- `value` is one of {list(PEP_STATUSES)}." in rendered
    # The SHIPPED words must be gone, not merely the custom ones present. Written as a set
    # difference rather than as `"'withdrawn'" not in rendered or "withdrawn" in PEP_STATUSES`,
    # which self-disabled the moment the custom list happened to contain that word.
    leaked = sorted(
        word for word in STATUS_VOCABULARY if word not in PEP_STATUSES and repr(word) in rendered
    )
    assert not leaked, f"the shipped vocabulary is still in the prompt: {leaked}"


def test_a_status_is_matched_case_insensitively_and_stored_in_the_corpus_spelling() -> None:
    """The casing half of the same defect, which the vocabulary parameter alone did not fix.

    PEP bodies write `Status: Final`; the model answered `final`. Passing the corpus's own
    spelling (`Final,Rejected,...`), which is the natural thing for a caller to do, then refused
    the batch on an exact-membership test and took the file's supersession claims with it.
    """
    payload = PAYLOAD.replace('"value": "final"', '"value": "FiNaL"')
    claims, rejections = normalize_extraction(
        payload,
        file="pep-0376.rst",
        human_body=BODY,
        corpus_names=("pep-0376",),
        status_vocabulary=("Final", "Rejected"),
    )
    assert [c.value for c in claims] == ["Final"], (
        "the vocabulary's spelling is what reaches the store, however the model wrote it"
    )
    assert not rejections


def test_the_shipped_vocabulary_refuses_a_pep_status() -> None:
    """The measured failure, pinned. `final` is outside the shipped set and refuses the batch."""
    with pytest.raises(ExtractionBatchRejected) as caught:
        normalize_extraction(
            PAYLOAD, file="pep-0376.rst", human_body=BODY, corpus_names=("pep-0376",)
        )
    assert caught.value.rung == "claim_shape"
    assert "final" in caught.value.reason


def test_a_custom_vocabulary_admits_it() -> None:
    claims, rejections = normalize_extraction(
        PAYLOAD,
        file="pep-0376.rst",
        human_body=BODY,
        corpus_names=("pep-0376",),
        status_vocabulary=PEP_STATUSES,
    )
    assert [c.value for c in claims] == ["final"]
    assert not rejections


def test_one_out_of_vocabulary_status_still_refuses_the_whole_batch() -> None:
    """Not softened while making the list configurable.

    `claim_shape` is a batch rung on purpose: a malformed shape suggests the model misread the
    schema, so the whole answer is suspect. Making the vocabulary a parameter fixes the LIST, and
    deliberately does not turn a batch rung into a per-claim one — that would be a separate
    decision, taken on its own evidence rather than smuggled in here.
    """
    payload = (
        '{"claims": ['
        '{"kind": "status", "value": "final", "quote": "Status: Final"},'
        '{"kind": "status", "value": "invented", "quote": "Status: Final"}]}'
    )
    with pytest.raises(ExtractionBatchRejected) as caught:
        normalize_extraction(
            payload,
            file="p.rst",
            human_body=BODY,
            corpus_names=("p",),
            status_vocabulary=PEP_STATUSES,
        )
    assert caught.value.rung == "claim_shape"
    assert "invented" in caught.value.reason


@pytest.mark.parametrize(
    "vocabulary,expected",
    [
        ((), "status_vocabulary is empty"),
        # `Sequence[str]` accepts `str` and mypy will not flag it, so a single word handed to a
        # `--status-vocabulary` flag typechecks and then renders `['f', 'i', 'n', 'a', 'l']`
        # into the prompt. The emptiness check cannot see it: a non-empty string makes a
        # non-empty tuple.
        ("final", "one character at a time"),
        # Matching is case-insensitive, so these are one word to the ladder and two to the
        # model, and which spelling reaches the store depends on list order.
        (("Final", "FINAL"), "differ only by case"),
        # Stringified rather than refused, a bytes entry rendered `["b'Final'"]` into the prompt
        # and into the audit record, looking like a word the caller had chosen.
        ((b"Final", "Rejected"), "must be strings"),
        ((1, 2), "must be strings"),
        # Blank once stripped: rendered into the prompt and unmatchable by any answer.
        (("", "Rejected"), "blank once stripped"),
        (("   ", "Rejected"), "blank once stripped"),
    ],
)
def test_a_degenerate_vocabulary_is_refused_at_both_entry_points(vocabulary, expected) -> None:
    """Both, because they used to disagree about every degenerate value.

    `None` meant "the default" in one and `TypeError` in the other; `()` raised at render time
    and silently refused every status claim at a BATCH rung in the ladder. One coercion is what
    stops the next value being decided twice.
    """
    for call in (
        lambda: _prompt(vocabulary),
        lambda: normalize_extraction(
            PAYLOAD,
            file="pep-0376.rst",
            human_body=BODY,
            corpus_names=("pep-0376",),
            status_vocabulary=vocabulary,
        ),
    ):
        with pytest.raises(ValueError, match=expected) as caught:
            call()
        # ⚠️ `ExtractionBatchRejected` SUBCLASSES `ValueError`, so `pytest.raises(ValueError)`
        # alone is satisfied by the ladder refusing the batch — which is the very outcome this
        # test exists to say must not happen. Without this line the property is carried entirely
        # by the `match=` string.
        assert not isinstance(caught.value, ExtractionBatchRejected), (
            "the vocabulary was accepted and then refused the whole batch, rather than being "
            "refused as the argument error it is"
        )


def test_a_padded_vocabulary_word_still_matches() -> None:
    """The coercion strips, because `_shape` strips the model's value before comparing.

    Unstripped, a `" Final"` from a comma split that forgot to strip renders into the prompt,
    never matches the answer, and refuses the whole file at a BATCH rung: the exact failure the
    vocabulary parameter exists to remove, reintroduced by the validation added to prevent it.
    """
    claims, rejections = normalize_extraction(
        PAYLOAD.replace('"value": "final"', '"value": "Final"'),
        file="pep-0376.rst",
        human_body=BODY,
        corpus_names=("pep-0376",),
        status_vocabulary=(" Final ", "Rejected"),
    )
    assert [c.value for c in claims] == ["Final"], "the stripped vocabulary spelling is what reaches the store"
    assert not rejections


def test_an_exact_duplicate_is_not_reported_as_a_case_collision() -> None:
    """A repeated word collapses harmlessly; refusing it with the wrong diagnosis does not.

    The detector counted casefold-equal entries, which includes a word compared with its own
    duplicate, so `("Final", "Final")` aborted the run claiming the two "differ only by case".
    """
    assert coerce_status_vocabulary(("Final", "Final", "Rejected")) == (
        "Final", "Final", "Rejected",
    )


class _Engine:
    engine_id = "e"
    model_id = "m"
    revision = "r"


def test_two_vocabularies_do_not_share_a_cache_entry() -> None:
    """The correctness case. Absent from the key, the second run serves the first's answer.

    The vocabulary is rendered into the prompt, so the two runs send DIFFERENT instructions and
    can only get different answers. A shared key would make the difference invisible.
    """
    shipped = extraction_cache_key(engine=_Engine(), prompt=_prompt())
    custom = extraction_cache_key(engine=_Engine(), prompt=_prompt(PEP_STATUSES))
    assert shipped != custom, "the cache key must separate two vocabularies"


def test_the_deterministic_engine_answers_in_the_vocabulary_it_was_HANDED() -> None:
    """The reference engine reads `prompt.status_vocabulary`, not the module constant.

    Reading the constant meant it emitted the five shipped memo words whatever the caller asked
    for, so under a PEP vocabulary a document with a status line produced a claim the ladder then
    refused at `claim_shape`. That is a BATCH rung: the file's SUPERSESSION claim went down with
    it, which is the whole point of the run. The two arms would also have been measured against
    different vocabularies while the report recorded one.

    Asserted end to end through `extract_file_claims`, because the defect was invisible at every
    smaller grain: the engine's output looked fine, and the ladder's refusal looked correct.
    """
    from recall.truth_extraction._engine import DeterministicExtractionEngine
    from recall.truth_extraction.extract import extract_file_claims

    extraction = extract_file_claims(
        file="pep-0376.rst",
        text="Status: Final\n\nThis PEP supersedes pep-0345.rst.\n",
        corpus_names=("pep-0345.rst", "pep-0376.rst"),
        engine=DeterministicExtractionEngine(),
        status_vocabulary=("Final", "Rejected", "Deferred"),
    )
    assert extraction.batch_rejection is None, (
        f"the whole file was refused: {extraction.batch_rejection}"
    )
    kinds = {type(claim).__name__: claim for claim in extraction.claims}
    assert "SupersessionClaim" in kinds, (
        "the supersession claim went down with the status claim, which is the cost of a batch rung"
    )
    assert kinds["StatusClaim"].value == "Final", (
        "the engine emitted a word from the shipped set rather than from the one it was handed"
    )


def test_a_reordered_vocabulary_is_a_different_prompt_and_a_different_key() -> None:
    """The ordering claim, made falsifiable.

    The previous version of this test hashed the SAME input twice and asserted the two agreed,
    under a docstring reading "order matters". That is in-process hash determinism; it cannot
    fail for the property it names. The vocabulary is deliberately not sorted before hashing
    (unlike `corpus_names`) precisely because it is rendered as an ordered list, so reordering it
    asks the model a textually different question and must not reuse the answer.
    """
    forward = _prompt(PEP_STATUSES)
    reversed_ = _prompt(tuple(reversed(PEP_STATUSES)))
    assert forward.user != reversed_.user, "a reordering that changes no prompt text is not one"
    assert extraction_cache_key(engine=_Engine(), prompt=forward) != extraction_cache_key(
        engine=_Engine(), prompt=reversed_
    )


def test_a_list_and_a_tuple_of_the_same_words_are_one_key() -> None:
    """The other half: the container type is not part of the question the model was asked."""
    as_tuple = extraction_cache_key(engine=_Engine(), prompt=_prompt(PEP_STATUSES))
    as_list = extraction_cache_key(engine=_Engine(), prompt=_prompt(list(PEP_STATUSES)))
    assert as_tuple == as_list, "a tuple and a list of the same words asked the same question"


def test_the_write_path_stays_closed_to_a_custom_vocabulary() -> None:
    """Configurable for measurement, closed for writing.

    `recall.rewrite` validates the derived block against the SHIPPED constant. A research run may
    name `final`, and that value still must not reach a user's memo, because the trust layer has
    no meaning for it.

    Asserted on the API and on the WRITER'S OWN GUARD, not on `inspect.getsource`. The source
    form went red the moment anyone added a comment saying why the writer takes no vocabulary —
    the very note this closure invites — and stayed green if the writer ever forwarded one under
    another parameter name. The behavioural half now drives `route_relation` with the status a
    PEP-vocabulary run proposes, which no extraction-side parameter can widen.
    """
    import inspect

    from recall.truth_extraction.extract import extract_corpus_claims

    assert "status_vocabulary" not in inspect.signature(extract_corpus_claims).parameters, (
        "the corpus-wide entry point, which the writer calls, must not take a vocabulary: "
        "widening measurement must not widen what can be written into a corpus"
    )
    assert "final" not in STATUS_VOCABULARY

    # And the behaviour that closure exists to produce, asserted on the guard that a parameter
    # CANNOT widen. The previous version re-asserted that `normalize_extraction` refuses `final`
    # under the shipped default: true, but a fact about the ladder rather than about the writer,
    # and green even if `rewrite.py` began forwarding a vocabulary. `route_relation` is where the
    # value meets the corpus, and `plan_rewrite` reaches it.
    from recall.rewrite import RewriteRefused, route_relation

    with pytest.raises(RewriteRefused, match="unroutable_status"):
        route_relation("declares_status", "pep-0376.rst", "status:final")


class _RecordingEngine:
    """Records the vocabulary of every prompt it is handed, and answers nothing."""

    engine_id = "recording"
    model_id = "recording"
    revision = "r"

    def __init__(self) -> None:
        self.seen: list[tuple[str, ...]] = []

    def run(self, prompt) -> str:
        self.seen.append(tuple(prompt.status_vocabulary))
        return '{"claims": []}'


class _DownEngine:
    """Fails every call, to drive the consecutive-failure path.

    Same identity as `_RecordingEngine`, deliberately: `extraction_cache_key` hashes engine_id/
    model_id/revision (`_cache.py`), so an outage test that warms the cache under one identity
    and reads it back under another misses for THAT reason regardless of vocabulary threading,
    which would make the test pass or fail for a cause unrelated to the one it names. Every
    other cache-key test in this file holds engine identity fixed and varies only the thing
    under test (see `_Engine` above); this class follows the same discipline.
    """

    engine_id = "recording"
    model_id = "recording"
    revision = "r"

    def run(self, prompt) -> str:
        raise RuntimeError("endpoint down")


def test_the_report_entry_point_takes_a_vocabulary_and_the_writer_does_not() -> None:
    """The split, pinned on BOTH doors.

    Asserting only that the writer's door is closed lets the pair collapse back into one closed
    function: every other assertion here would still pass while the CLI silently lost the flag.
    """
    import inspect

    from recall.truth_extraction.extract import (
        extract_corpus_claims,
        extract_corpus_claims_for_report,
    )

    assert (
        "status_vocabulary"
        in inspect.signature(extract_corpus_claims_for_report).parameters
    ), "the report door must be the one that opens"
    assert "status_vocabulary" not in inspect.signature(extract_corpus_claims).parameters


def test_every_document_gets_the_custom_vocabulary_not_just_the_first() -> None:
    """Asserted across the whole corpus, because the loop is where a threading bug hides.

    A version that resolved the vocabulary once and then rebuilt the prompt per file from the
    module default would pass a one-document test.
    """
    from recall.truth_extraction.extract import extract_corpus_claims_for_report

    engine = _RecordingEngine()
    documents = {f"pep-{n:04d}.rst": "Status: Final\n" for n in (1, 2, 3)}
    extract_corpus_claims_for_report(
        documents, engine=engine, status_vocabulary=PEP_STATUSES
    )
    assert engine.seen == [PEP_STATUSES] * 3, (
        f"a document was asked under the wrong vocabulary: {engine.seen}"
    )


def test_the_writer_door_still_asks_under_the_shipped_words() -> None:
    from recall.truth_extraction.extract import extract_corpus_claims

    engine = _RecordingEngine()
    extract_corpus_claims({"a.md": "Status: Final\n"}, engine=engine)
    assert engine.seen == [tuple(STATUS_VOCABULARY)]


def test_the_outage_path_looks_up_the_key_the_custom_vocabulary_warmed() -> None:
    """`extract.py:171`. The outage path builds its OWN prompt for the cache lookup.

    Threaded only through the per-file call, that lookup computes the SHIPPED key, misses every
    entry a custom-vocabulary run warmed, and turns a warm cache into zero surviving claims —
    the same shape of defect the comment on `_refused` records as fixed once already.
    """
    from recall.truth_extraction._sqlite_cache import SqliteExtractionCache
    from recall.truth_extraction.extract import extract_corpus_claims_for_report
    from recall.truth_extraction.types import CONSECUTIVE_ENGINE_FAILURE_LIMIT

    import tempfile
    from pathlib import Path

    # Documents are processed in SORTED order, and the first `LIMIT` of them must be COLD so the
    # engine is actually called and actually fails. Only then does the counter trip and the
    # outage path's OWN cache lookup run. Warming everything defeats the test entirely:
    # `extract_file_claims` consults the cache before calling the engine, so every file would hit
    # there, no failure would ever be counted, and line 171 would never execute.
    cold = {
        f"a-cold-{n}.rst": "Status: Final\n" for n in range(CONSECUTIVE_ENGINE_FAILURE_LIMIT)
    }
    warm = {f"z-warm-{n}.rst": "Status: Final\n" for n in range(3)}
    names = tuple(sorted({**cold, **warm}))

    with tempfile.TemporaryDirectory() as tmp:
        # A `with` block, not a bare assignment: the connection must be closed before the
        # `TemporaryDirectory` above tries to remove it. POSIX tolerates deleting a file a
        # process still has open; Windows refuses, and cleanup raced the sqlite handle here.
        with SqliteExtractionCache(Path(tmp) / "c.sqlite3") as cache:
            # Warm ONLY the `z-` files, under the custom vocabulary, with an engine that works.
            # `corpus_names` is the full set in both runs: it is hashed into the key, so warming
            # under a narrower corpus would miss for a reason that has nothing to do with the
            # vocabulary and would make this test green for the wrong cause.
            extract_corpus_claims_for_report(
                warm,
                engine=_RecordingEngine(),
                corpus_names=names,
                cache=cache,
                status_vocabulary=PEP_STATUSES,
            )
            # Now the engine is down. The cold files fail, the counter trips, and the warm files
            # must be read through the outage path's lookup.
            results = extract_corpus_claims_for_report(
                {**cold, **warm},
                engine=_DownEngine(),
                corpus_names=names,
                cache=cache,
                status_vocabulary=PEP_STATUSES,
            )
    hits = sorted(r.file for r in results if r.cached)
    assert hits == sorted(warm), (
        f"the outage path missed the warm entries: hit {hits}, expected {sorted(warm)}. It "
        f"looked the cache up under a different vocabulary than the run that wrote them."
    )


def test_the_outage_refusal_names_the_vocabulary_the_run_used() -> None:
    """`extract.py:192`. The refusal record is the audit identity of a failed run.

    Filed under the shipped words, it says the run used a vocabulary it never used, and a failed
    run is exactly where that field gets read.
    """
    from recall.truth_extraction.extract import extract_corpus_claims_for_report
    from recall.truth_extraction.types import CONSECUTIVE_ENGINE_FAILURE_LIMIT

    documents = {
        f"pep-{n:04d}.rst": "Status: Final\n"
        for n in range(CONSECUTIVE_ENGINE_FAILURE_LIMIT + 3)
    }
    results = extract_corpus_claims_for_report(
        documents, engine=_DownEngine(), status_vocabulary=PEP_STATUSES
    )
    skipped = [r for r in results if r.batch_rejection and "skipped" in r.batch_rejection.reason]
    assert skipped, "the failure limit was never reached, so this proves nothing"
    for result in skipped:
        assert result.status_vocabulary == PEP_STATUSES, (
            f"{result.file} filed its refusal under {result.status_vocabulary}"
        )


def test_the_write_path_calls_the_closed_door_and_not_the_open_one() -> None:
    """The convention that makes the door split mean anything, pinned so it cannot go silent.

    `test_the_write_path_stays_closed_to_a_custom_vocabulary` and
    `test_the_report_entry_point_takes_a_vocabulary_and_the_writer_does_not` both assert on the
    two functions' SIGNATURES: one takes no vocabulary, the other does. Neither notices which one
    `recall/rewrite.py` actually calls. Re-point that call site at
    `extract_corpus_claims_for_report` and pass it a vocabulary, and both signature assertions
    stay green, `route_relation`'s own refusal is never exercised because nothing upstream would
    be proposing an out-of-vocabulary status yet, and every test on this branch still passes. The
    real guarantee that a custom vocabulary never reaches a user's frontmatter is
    `route_relation` (`recall/rewrite.py:196`), refusing any status outside `STATUS_VOCABULARY`.
    The door split is defence in depth on top of that guarantee, not a second copy of it, and
    defence in depth that nobody checks is just a comment.

    An AST walk, not `inspect.getsource` or a text search, for the reason
    `test_nothing_on_this_path_builds_an_extraction_engine` (`tests/test_mcp_rewrite_plan.py:266`)
    already is: a `Call` node only appears from an actual call, so a comment that happens to
    mention the open door's name cannot forge one and a reformatted call cannot hide one.
    """
    import ast
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "recall" / "rewrite.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    called = {
        getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
    }
    assert "extract_corpus_claims" in called, "the write path must call the closed door"
    assert "extract_corpus_claims_for_report" not in called, (
        "the write path calls the door that takes a vocabulary; the write path's door is "
        "supposed to have no knob to turn"
    )
