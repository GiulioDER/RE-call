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
    ClaimRejection,
    ExtractionBatchRejected,
    FileExtraction,
)


#: Consecutive engine failures after which the engine is treated as unavailable for the rest of
#: the run. Consecutive rather than total: a corpus with a few individually awkward memos should
#: not stop, while an endpoint that is simply down should stop being asked once per file.
CONSECUTIVE_ENGINE_FAILURE_LIMIT = 3


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
        batch_rejection=ClaimRejection(index=-1, kind="*", rung=rung, reason=reason),
    )


def extract_file_claims(
    *,
    file: str,
    text: str,
    corpus_names: Sequence[str],
    engine: ExtractionEngine,
    cache: ExtractionCache | None = None,
) -> FileExtraction:
    """Extract claims from one document. `text` is the raw file, frontmatter included."""
    body = human_body_of(text)
    prompt = build_extraction_prompt(file=file, human_body=body, corpus_names=corpus_names)
    key = extraction_cache_key(engine=engine, prompt=prompt)
    if cache is not None:
        cached = cache.get(key)
        if cached is not None:
            return FileExtraction(
                file=cached.file,
                claims=cached.claims,
                rejections=cached.rejections,
                engine_id=cached.engine_id,
                model_id=cached.model_id,
                revision=cached.revision,
                prompt_revision=cached.prompt_revision,
                batch_rejection=cached.batch_rejection,
                cached=True,
            )
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
            answer, file=file, human_body=body, corpus_names=corpus_names
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
        batch_rejection=batch_rejection,
    )
    if cache is not None:
        cache.put(key, result)
    return result


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
    """
    names = tuple(corpus_names) if corpus_names is not None else tuple(sorted(documents))
    results: list[FileExtraction] = []
    consecutive_failures = 0
    for file, text in sorted(documents.items()):
        # The break guards the ENGINE, so it must not stand between a caller and the cache. A
        # cache hit costs no engine call, so skipping it saves nothing and throws away an
        # extraction that already succeeded: on a warm cache with a dead endpoint, refusing
        # cached files turned 16 surviving claims into zero for identical engine cost.
        if consecutive_failures >= CONSECUTIVE_ENGINE_FAILURE_LIMIT and cache is not None:
            prompt = build_extraction_prompt(
                file=file, human_body=human_body_of(text), corpus_names=names
            )
            cached = cache.get(extraction_cache_key(engine=engine, prompt=prompt))
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
                    prompt=build_extraction_prompt(
                        file=file, human_body=human_body_of(text), corpus_names=names
                    ),
                    rung="engine_error",
                    reason=(
                        f"skipped: the engine failed {consecutive_failures} times in a row, "
                        f"so it is treated as unavailable rather than re-asked per file"
                    ),
                )
            )
            continue
        result = extract_file_claims(
            file=file, text=text, corpus_names=names, engine=engine, cache=cache
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


__all__ = ["extract_corpus_claims", "extract_file_claims"]
