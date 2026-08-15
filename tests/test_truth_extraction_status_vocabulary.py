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

    Asserted on the API and on BEHAVIOUR, not on `inspect.getsource`. The source form went red
    the moment anyone added a comment saying why the writer takes no vocabulary — the very note
    this closure invites — and stayed green if the writer ever forwarded one under another
    parameter name or through `**kwargs`.
    """
    import inspect

    from recall.truth_extraction.extract import extract_corpus_claims

    assert "status_vocabulary" not in inspect.signature(extract_corpus_claims).parameters, (
        "the corpus-wide entry point, which the writer calls, must not take a vocabulary: "
        "widening measurement must not widen what can be written into a corpus"
    )
    assert "final" not in STATUS_VOCABULARY

    # And the behaviour that closure exists to produce: a status outside the shipped set does not
    # survive the path the writer actually runs.
    with pytest.raises(ExtractionBatchRejected) as caught:
        normalize_extraction(
            PAYLOAD, file="pep-0376.rst", human_body=BODY, corpus_names=("pep-0376",)
        )
    assert caught.value.rung == "claim_shape"
