"""End to end extraction for one document or one corpus.

This is the ingest path entry point. It is deliberately the only module here that composes
the others, and it composes them in one fixed order: strip frontmatter, render the prompt,
consult the cache, run the engine, apply the ladder, cache the survivors.

A refusal is a RESULT, never an exception that escapes. One memo whose output was malformed
must not abort ingesting the other 791, and a refusal a caller never sees is a refusal nobody
reviews — so a batch level rejection is recorded on the returned `FileExtraction`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from recall.truth_extraction._cache import ExtractionCache, extraction_cache_key
from recall.truth_extraction._engine import ExtractionEngine
from recall.truth_extraction._normalize import human_body_of, normalize_extraction
from recall.truth_extraction._prompt import ExtractionPrompt, build_extraction_prompt
from recall.truth_extraction.types import (
    CONSECUTIVE_ENGINE_FAILURE_LIMIT,
    ClaimRejection,
    ExtractionBatchRejected,
    FileExtraction,
    coerce_status_vocabulary,
)


def _refused(
    *,
    file: str,
    engine: ExtractionEngine,
    prompt: ExtractionPrompt,
    rung: str,
    reason: str,
) -> FileExtraction:
    """A whole-file refusal, recorded so a reviewer sees it rather than seeing nothing.

    Deliberately NOT cached. A cache entry exists to avoid re-paying for an answer, and a rate
    limit or a dropped connection is not an answer: caching it would make a transient failure
    permanent for as long as the entry survives, and the next run would report the same refusal
    without ever having asked again.

    The reason carries the exception's CLASS NAME and not its text, matching
    `reasoning_proposals/_providers._safe_failure_message`. Provider error strings routinely
    echo the request, and the request carries the API key and the memo body.
    """
    return FileExtraction(
        file=file,
        claims=(),
        rejections=(),
        engine_id=engine.engine_id,
        model_id=engine.model_id,
        revision=engine.revision,
        prompt_revision=prompt.revision,
        # The refusal record names the same extractor a success would. Omitted, an engine_error
        # was filed under the shipped memo vocabulary whatever the run had used, which is the
        # half of the audit identity a failed run most needs to keep.
        status_vocabulary=prompt.status_vocabulary,
        batch_rejection=ClaimRejection(index=-1, kind="*", rung=rung, reason=reason),
    )


def extract_file_claims(
    *,
    file: str,
    text: str,
    corpus_names: Sequence[str],
    engine: ExtractionEngine,
    cache: ExtractionCache | None = None,
    status_vocabulary: Sequence[str] | None = None,
) -> FileExtraction:
    """Extract claims from one document. `text` is the raw file, frontmatter included."""
    body = human_body_of(text)
    prompt = (
        build_extraction_prompt(file=file, human_body=body, corpus_names=corpus_names)
        if status_vocabulary is None
        else build_extraction_prompt(
            file=file,
            human_body=body,
            corpus_names=corpus_names,
            status_vocabulary=status_vocabulary,
        )
    )
    key = extraction_cache_key(engine=engine, prompt=prompt)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            # `replace`, not a field-by-field rebuild, and matching what `extract_corpus_claims`
            # already does. The rebuild listed the fields it knew about, so the NEXT field added
            # to `FileExtraction` silently took its dataclass default here: `status_vocabulary`
            # did, which meant a cache HIT reported the shipped memo words for a run that had
            # used a corpus's own. A hit is the path a re-run takes, so the audit field was
            # wrong exactly where it would be read.
            return replace(cached, cached=True)
    # The engine call is guarded SEPARATELY from normalization, and it is guarded broadly. A
    # model engine reaches the network, so it can raise a rate limit, a timeout, a connection
    # reset or a malformed-response error, none of which is an `ExtractionBatchRejected`. Left
    # to propagate, one such failure on memo 400 of 792 aborts `extract_corpus_claims` and
    # discards all 399 extractions already built, because it collects into a `tuple(...)` over a
    # generator. That directly contradicts this module's contract: "One memo whose output was
    # malformed must not abort ingesting the other 791."
    #
    # The broad catch is scoped to `engine.run` alone. Bugs inside `normalize_extraction` are
    # the library's own and must still surface as crashes rather than be recorded as refusals.
    try:
        answer = engine.run(prompt)
    except Exception as failure:  # noqa: BLE001 - see above; the engine is third party code
        return _refused(
            file=file,
            engine=engine,
            prompt=prompt,
            rung="engine_error",
            reason=f"engine failed: {failure.__class__.__name__}",
        )
    try:
        claims, rejections = normalize_extraction(
            answer,
            file=file,
            human_body=body,
            corpus_names=corpus_names,
            # The prompt's own vocabulary, never the module default: the model was told this
            # list, so judging its answer against a different one would refuse a claim it was
            # invited to make.
            status_vocabulary=prompt.status_vocabulary,
        )
        batch_rejection = None
    except ExtractionBatchRejected as refused:
        claims, rejections = (), ()
        batch_rejection = ClaimRejection(
            index=-1, kind="*", rung=refused.rung, reason=refused.reason
        )
    result = FileExtraction(
        file=file,
        claims=claims,
        rejections=rejections,
        engine_id=engine.engine_id,
        model_id=engine.model_id,
        revision=engine.revision,
        prompt_revision=prompt.revision,
        status_vocabulary=prompt.status_vocabulary,
        batch_rejection=batch_rejection,
    )
    if cache is not None:
        cache.put(key, result)
    return result


def _extract_corpus(
    documents: Mapping[str, str],
    *,
    engine: ExtractionEngine,
    corpus_names: Sequence[str] | None,
    cache: ExtractionCache | None,
    status_vocabulary: Sequence[str] | None,
) -> tuple[FileExtraction, ...]:
    """The one implementation behind both public entry points.

    ONE loop, deliberately, because the vocabulary has to reach THREE prompt builds here and only
    one of them is the per-file call. The outage path looks the cache up with its own prompt and
    files its refusal record with another. A second copy of this loop would thread the obvious
    call and miss those two: a cache warmed under a corpus's own words would then be looked up
    under the shipped key and miss every entry, and the refusal would name a vocabulary the run
    never used. That is the same defect the comment on `_refused` records as fixed once already.

    Coerced ONCE, here, rather than per document. `build_extraction_prompt` coerces too and the
    operation is idempotent, so this changes no result — it changes WHEN a degenerate vocabulary
    is refused, from once per file to once per run, before the first engine call is paid for.
    """
    names = tuple(corpus_names) if corpus_names is not None else tuple(sorted(documents))
    vocabulary = coerce_status_vocabulary(status_vocabulary)

    def _prompt_for(file: str, text: str) -> ExtractionPrompt:
        # The SAME prompt `extract_file_claims` will build for this file, in one place, so the
        # outage path's cache lookup computes the key the warm run actually wrote under.
        return build_extraction_prompt(
            file=file,
            human_body=human_body_of(text),
            corpus_names=names,
            status_vocabulary=vocabulary,
        )

    results: list[FileExtraction] = []
    consecutive_failures = 0
    for file, text in sorted(documents.items()):
        # The break guards the ENGINE, so it must not stand between a caller and the cache. A
        # cache hit costs no engine call, so skipping it saves nothing and throws away an
        # extraction that already succeeded: on a warm cache with a dead endpoint, refusing
        # cached files turned 16 surviving claims into zero for identical engine cost.
        if consecutive_failures >= CONSECUTIVE_ENGINE_FAILURE_LIMIT and cache is not None:
            cached = cache.get(
                extraction_cache_key(engine=engine, prompt=_prompt_for(file, text))
            )
            if cached is not None:
                results.append(replace(cached, cached=True))
                continue
        if consecutive_failures >= CONSECUTIVE_ENGINE_FAILURE_LIMIT:
            # The engine is down, not this memo. Making a per-file refusal out of a systemic
            # outage was the right call for ONE failure and is the wrong call for all of them:
            # every remaining memo would pay the full retry and timeout budget to learn the same
            # thing. At 3 attempts against a 60 second timeout that is minutes per file, so a
            # 792 memo corpus spends over a day producing nothing.
            #
            # Still recorded per file, so the extractions already built survive and a reviewer
            # sees why the rest are missing. That is the contract this guard defends, not an
            # exception to it.
            results.append(
                _refused(
                    file=file,
                    engine=engine,
                    prompt=_prompt_for(file, text),
                    rung="engine_error",
                    reason=(
                        f"skipped: the engine failed {consecutive_failures} times in a row, "
                        f"so it is treated as unavailable rather than re-asked per file"
                    ),
                )
            )
            continue
        result = extract_file_claims(
            file=file,
            text=text,
            corpus_names=names,
            engine=engine,
            cache=cache,
            status_vocabulary=vocabulary,
        )
        # Only a result that actually reached the engine is evidence about the engine. A cache
        # hit says nothing either way, and letting one RESET the counter defeated the break
        # entirely: with edited memos interleaved among cached ones, which is what adding files
        # to a corpus produces alphabetically, every failure was followed by a hit and the
        # engine was asked for all of them.
        if not result.cached:
            failed = (
                result.batch_rejection is not None
                and result.batch_rejection.rung == "engine_error"
            )
            consecutive_failures = consecutive_failures + 1 if failed else 0
        results.append(result)
    return tuple(results)


def extract_corpus_claims(
    documents: Mapping[str, str],
    *,
    engine: ExtractionEngine,
    corpus_names: Sequence[str] | None = None,
    cache: ExtractionCache | None = None,
) -> tuple[FileExtraction, ...]:
    """Extract claims from every document, resolving targets against the corpus itself.

    `corpus_names` defaults to the documents' own names: a supersession target that is not in
    the batch being ingested is refused, which is the same refusal `recall/fix.py` makes for
    a target that is not a file in the corpus.

    ⚠️ This is the door the WRITE path uses (`recall/rewrite.py`), and it takes NO status
    vocabulary on purpose. `recall.rewrite` validates the derived block against the shipped
    `STATUS_VOCABULARY`, so a value the trust layer has no meaning for must not reach a user's
    memo because a research run named it. Configurable for MEASUREMENT, closed for WRITING; the
    measurement door is `extract_corpus_claims_for_report`, below.
    """
    return _extract_corpus(
        documents,
        engine=engine,
        corpus_names=corpus_names,
        cache=cache,
        status_vocabulary=None,
    )


def extract_corpus_claims_for_report(
    documents: Mapping[str, str],
    *,
    engine: ExtractionEngine,
    corpus_names: Sequence[str] | None = None,
    cache: ExtractionCache | None = None,
    status_vocabulary: Sequence[str] | None = None,
) -> tuple[FileExtraction, ...]:
    """The same extraction, for output a human READS rather than output a writer acts on.

    `recall extract` lands here. `status_vocabulary=None` is the shipped memo set, so this is
    `extract_corpus_claims` plus one knob — but it is a SEPARATE function rather than a keyword
    on that one, because the writer's door having no knob is the property
    `test_the_write_path_stays_closed_to_a_custom_vocabulary` pins. Given the knob, that test
    could only assert on this module's call sites, which is source inspection wearing a
    different hat.

    The shipped set is memo-shaped (`active, draft, deprecated, superseded, withdrawn`). A corpus
    with its own words needs its own list: measured on `python/peps`, `Status: Final` produced
    `final`, and because `claim_shape` is a BATCH rung that refused 12 of 30 documents outright,
    taking their supersession claims down with them.
    """
    return _extract_corpus(
        documents,
        engine=engine,
        corpus_names=corpus_names,
        cache=cache,
        status_vocabulary=status_vocabulary,
    )


__all__ = ["extract_corpus_claims", "extract_corpus_claims_for_report", "extract_file_claims"]
