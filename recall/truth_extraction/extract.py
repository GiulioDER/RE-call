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

from recall.truth_extraction._cache import ExtractionCache, extraction_cache_key
from recall.truth_extraction._engine import ExtractionEngine
from recall.truth_extraction._normalize import human_body_of, normalize_extraction
from recall.truth_extraction._prompt import build_extraction_prompt
from recall.truth_extraction.types import (
    ClaimRejection,
    ExtractionBatchRejected,
    FileExtraction,
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
    try:
        claims, rejections = normalize_extraction(
            engine.run(prompt), file=file, human_body=body, corpus_names=corpus_names
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
    return tuple(
        extract_file_claims(
            file=file, text=text, corpus_names=names, engine=engine, cache=cache
        )
        for file, text in sorted(documents.items())
    )


__all__ = ["extract_corpus_claims", "extract_file_claims"]
